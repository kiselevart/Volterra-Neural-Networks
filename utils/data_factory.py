import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import warnings

# Project Imports
from dataloaders.dataset import VideoDataset
from utils.video_utils import calculate_video_flow

class FlowDatasetWrapper(Dataset):
    """
    Wraps the standard VideoDataset to compute Optical Flow on the fly.
    """
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __getitem__(self, index):
        # Get raw data from original dataset
        # Assuming original returns (video, label)
        rgb_video, label = self.dataset[index]
        
        # Compute Flow here
        flow_video = calculate_video_flow(rgb_video)
        
        # Return tuple (rgb, flow), label
        # In the training loop, we check if input is a list/tuple
        return [rgb_video, flow_video], label

    def __len__(self):
        return len(self.dataset)

def get_dataloaders(args):
    print(f"==> Preparing data for: {args.dataset}")
    
    loaders = {}

    if args.task == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        full_trainset_val = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        val_split = getattr(args, 'val_split', 0.1)
        if val_split <= 0 or val_split >= 1:
            raise ValueError("val_split must be in (0, 1)")

        val_size = int(len(full_trainset) * val_split)
        train_size = len(full_trainset) - val_size

        generator = torch.Generator()
        generator.manual_seed(getattr(args, 'seed', 42))
        indices = torch.randperm(len(full_trainset), generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        trainset = torch.utils.data.Subset(full_trainset, train_indices)
        valset = torch.utils.data.Subset(full_trainset_val, val_indices)

        loaders['train'] = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        loaders['val'] = DataLoader(valset, batch_size=100, shuffle=False, num_workers=args.num_workers)
        loaders['test'] = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    elif args.task == 'video':
        # Dataset instantiation
        train_ds = VideoDataset(dataset=args.dataset, split='train', clip_len=16, preprocess=False, augment=True)
        val_ds = VideoDataset(dataset=args.dataset, split='val', clip_len=16, preprocess=False, augment=False)
        test_ds = VideoDataset(dataset=args.dataset, split='test', clip_len=16, preprocess=False, augment=False)

        if args.model == 'vnn_fusion':
            # Wrap for Flow
            train_ds = FlowDatasetWrapper(train_ds)
            val_ds = FlowDatasetWrapper(val_ds)
            test_ds = FlowDatasetWrapper(test_ds)

        loaders['train'] = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        loaders['val'] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        loaders['test'] = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return loaders