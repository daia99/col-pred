import os
import random

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvideotransforms import video_transforms, volume_transforms # custom library for consistent transforms for single video clip

# from pl_bolts.utils import _TORCHVISION_AVAILABLE
# from pl_bolts.utils.warnings import warn_missing_pkg

# if _TORCHVISION_AVAILABLE:
import torchvision.transforms as transforms
# else:  # pragma: no cover
#     warn_missing_pkg('torchvision')

from data.custom_dataset import RawDataset

class RawDataModule(LightningDataModule):
    def __init__(
            self,
            train_args,
            pin_memory: bool = False,
            val_split: float = 0.1,
            num_workers: int = 2,
            batch_size: int = 12,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        """
        Kitti train, validation and test dataloaders.

        Note:
            You need to have downloaded the Kitti dataset first and provide the path to where it is saved.
            You can download the dataset here:
            http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

        Specs:
            - 200 samples
            - Each image is (3 x 1242 x 376)

        In total there are 34 classes but some of these are not useful so by default we use only 19 of the classes
        specified by the `valid_labels` parameter.

        Example::

            from pl_bolts.datamodules import KittiDataModule

            dm = KittiDataModule(PATH)
            model = LitModel()

            Trainer().fit(model, dm)

        Args:
            data_dir: where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
            val_split: size of validation test (default 0.2)
            test_split: size of test set (default 0.1)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
        """
        # if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        #     raise ModuleNotFoundError(
        #         'You want to use `torchvision` which is not installed yet.'
        #     )
        #Define required parameters here
        super().__init__(*args, **kwargs)
        #self.download_dir = ''
        self.data_dir = train_args.data_path if train_args.data_path is not None else os.getcwd()
        self.batch_size = train_args.batch_size
        self.num_workers = num_workers
        self.seed = seed
        if train_args.augment_data:
            video_aug_transform_list = [
                video_transforms.Resize((128, 416), interpolation='bilinear'),
                video_transforms.RandomHorizontalFlip(p=0.5),
                video_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
            self.video_transforms = video_transforms.Compose(video_aug_transform_list)
            video_transform_list = [
                video_transforms.Resize((128, 416), interpolation='bilinear'),
                video_transforms.RandomHorizontalFlip(p=0.5),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.transform = video_transforms.Compose(video_transform_list)
        else:
            video_transform_list = [
                video_transforms.Resize((128, 416), interpolation='bilinear'),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.video_transforms = None
            self.transform = video_transforms.Compose(video_transform_list)
        self.val_split = val_split
        self.train_args = train_args
        self.pin_memory = pin_memory
    
    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        # self.kitti_dataset = RawDataset(*args, **kwargs)
        pass

    def setup(self, stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.
        kitti_dataset = RawDataset(train_args=self.train_args)
        num_data = kitti_dataset.__len__()
        val_split = int(num_data*self.val_split)
        train_split = num_data - val_split
        self.train_data, self.val_data = random_split(kitti_dataset,
                                                      lengths=[train_split, val_split],
                                                      generator=torch.Generator().manual_seed(self.seed))
        
        # disable data augmentation for validation dataset if augmentation used
        if self.train_args.augment_data:
            val_video_transform_list = [
                    video_transforms.Resize((128, 416), interpolation='bilinear'),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]  
            val_transform = video_transforms.Compose(val_video_transform_list)
            self.train_data = ApplyTransform(self.train_data, augment=True, transform=self.transform, color_transform=self.video_transforms)
            self.val_data = ApplyTransform(self.val_data, augment=False, transform=val_transform)
        else:
            self.train_data = ApplyTransform(self.train_data, augment=False, transform=self.transform)
            self.val_data = ApplyTransform(self.val_data, augment=False, transform=self.transform)

    def train_dataloader(self):
        # Return DataLoader for Training Data here
        loader = DataLoader(self.train_data,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)
        return loader
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        loader = DataLoader(self.val_data,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)
        return loader
    
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        loader = DataLoader(self.val_data,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, augment=True, transform=None, color_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.color_transform = color_transform
        self.augment = augment

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        context = sample['rgb_context'] # list of PIL images
        if self.transform is not None:
            if not self.augment:
                context = self.transform(context)
            else:
                color_aug = random.random() > 0.5
                # transform color
                if color_aug:
                    context = self.color_transform(context)
                else:
                    context = self.transform(context)
            sample['rgb_context'] = context
                
        return sample

    def __len__(self):
        return len(self.dataset)