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

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        loaders['train'] = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        loaders['val'] = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
        loaders['test'] = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    elif args.task == 'video':
        # Dataset instantiation
        train_ds = VideoDataset(dataset=args.dataset, split='train', clip_len=16, preprocess=False)
        val_ds = VideoDataset(dataset=args.dataset, split='val', clip_len=16, preprocess=False)
        test_ds = VideoDataset(dataset=args.dataset, split='test', clip_len=16, preprocess=False)

        if args.model == 'vnn_fusion':
            # Wrap for Flow
            train_ds = FlowDatasetWrapper(train_ds)
            val_ds = FlowDatasetWrapper(val_ds)
            test_ds = FlowDatasetWrapper(test_ds)

        loaders['train'] = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        loaders['val'] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        loaders['test'] = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return loaders