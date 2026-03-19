import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Project Imports
from dataloaders.dataset import VideoDataset


class FlowDatasetWrapper(Dataset):
    """Two-stream wrapper that pairs RGB clips with pre-cached optical flow.

    On first use, calls ensure_flows() to compute any missing flow.npy files
    (idempotent — subsequent runs skip videos that already have flow cached).

    Spatial crop, temporal crop, and horizontal flip are computed once and
    applied identically to both streams, keeping RGB and flow spatially aligned.
    Flow is loaded from the full-resolution flow.npy stored alongside the frames,
    then cropped to match the RGB clip.
    """

    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.dataset.ensure_flows()

    def __getitem__(self, index):
        ds = self.dataset

        # Load full-resolution frames [T, H, W, C] — before any crop or normalize
        buffer = ds.load_frames(ds.fnames[index])

        # Compute shared crop indices applied to both RGB and flow
        if ds.augment:
            max_t = buffer.shape[0] - ds.clip_len
            t_idx = np.random.randint(max_t + 1) if max_t > 0 else 0
            h_idx = np.random.randint(max(1, buffer.shape[1] - ds.crop_size))
            w_idx = np.random.randint(max(1, buffer.shape[2] - ds.crop_size))
        else:
            t_idx = max(0, (buffer.shape[0] - ds.clip_len) // 2)
            h_idx = max(0, (buffer.shape[1] - ds.crop_size) // 2)
            w_idx = max(0, (buffer.shape[2] - ds.crop_size) // 2)

        do_flip = ds.augment and np.random.random() < 0.5

        # --- RGB stream ---
        rgb = buffer[t_idx:t_idx + ds.clip_len,
                     h_idx:h_idx + ds.crop_size,
                     w_idx:w_idx + ds.crop_size]
        rgb = ds.ensure_clip_len(rgb, ds.clip_len)
        if do_flip:
            rgb = ds.randomflip(rgb)
        if ds.augment:
            rgb = ds.color_jitter(rgb)
        rgb = ds.normalize(rgb)
        rgb = torch.from_numpy(ds.to_tensor(rgb))
        if not torch.isfinite(rgb).all():
            rgb = torch.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=-255.0)

        # --- Flow stream: load pre-cached [2, T_full, H_full, W_full] ---
        flow = torch.from_numpy(
            np.load(os.path.join(ds.fnames[index], "flow.npy"))
        )
        flow = flow[:, t_idx:t_idx + ds.clip_len,
                       h_idx:h_idx + ds.crop_size,
                       w_idx:w_idx + ds.crop_size]
        flow = self._ensure_flow_clip_len(flow, ds.clip_len)
        if do_flip:
            flow = torch.flip(flow, dims=[3])  # flip width dimension
            flow = flow.clone()
            flow[0] = -flow[0]                 # negate x-component (direction reverses)
        if not torch.isfinite(flow).all():
            flow = torch.nan_to_num(flow, nan=0.0, posinf=1.0, neginf=-1.0)
        flow = flow.clamp(-2.0, 2.0).float()

        label = torch.from_numpy(np.array(ds.label_array[index]))
        return [rgb, flow], label

    def _ensure_flow_clip_len(self, flow, clip_len):
        """Pad or trim flow tensor [2, T, H, W] to exactly clip_len frames."""
        T = flow.shape[1]
        if T == clip_len:
            return flow
        if T > clip_len:
            return flow[:, :clip_len]
        pad = flow[:, -1:].repeat(1, clip_len - T, 1, 1)
        return torch.cat([flow, pad], dim=1)

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(args):
    print(f"==> Preparing data for: {args.dataset}")

    loaders = {}

    # pin_memory only benefits CUDA; persistent_workers and prefetch_factor
    # are invalid (raise ValueError) when num_workers=0.
    pin_memory = torch.cuda.is_available()
    use_workers = args.num_workers > 0
    worker_kwargs = dict(persistent_workers=True, prefetch_factor=2) if use_workers else {}

    if args.task == "cifar":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        full_trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        full_trainset_val = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_test
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )

        val_split = getattr(args, "val_split", 0.1)
        if val_split <= 0 or val_split >= 1:
            raise ValueError("val_split must be in (0, 1)")

        val_size = int(len(full_trainset) * val_split)
        train_size = len(full_trainset) - val_size

        generator = torch.Generator()
        generator.manual_seed(getattr(args, "seed", 42))
        indices = torch.randperm(len(full_trainset), generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        trainset = torch.utils.data.Subset(full_trainset, train_indices)
        valset = torch.utils.data.Subset(full_trainset_val, val_indices)

        loaders["train"] = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            **worker_kwargs,
        )
        loaders["val"] = DataLoader(
            valset,
            batch_size=100,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            **worker_kwargs,
        )
        loaders["test"] = DataLoader(
            testset,
            batch_size=100,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            **worker_kwargs,
        )

    elif args.task == "video":
        # Dataset instantiation
        train_ds = VideoDataset(
            dataset=args.dataset,
            split="train",
            clip_len=16,
            preprocess=False,
            augment=True,
        )
        val_ds = VideoDataset(
            dataset=args.dataset,
            split="val",
            clip_len=16,
            preprocess=False,
            augment=False,
        )
        test_ds = VideoDataset(
            dataset=args.dataset,
            split="test",
            clip_len=16,
            preprocess=False,
            augment=False,
        )

        if args.model in ("vnn_fusion", "vnn_fusion_ho"):
            # Wrap for Flow
            train_ds = FlowDatasetWrapper(train_ds)
            val_ds = FlowDatasetWrapper(val_ds)
            test_ds = FlowDatasetWrapper(test_ds)

        loaders["train"] = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            **worker_kwargs,
        )
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            **worker_kwargs,
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            **worker_kwargs,
        )

    return loaders
