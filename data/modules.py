import torch
from torchvision import datasets
from torchvision import transforms as transform_lib
from pytorch_lightning import LightningDataModule

from data.transforms import BYOLDataTransform

import os


class ImageNetDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_classes(self):
        return 1000

    @property
    def mean(self):
        return (0.485, 0.456, 0.406)

    @property
    def std(self):
        return (0.229, 0.224, 0.225)

    def prepare_data(self):
        pass

    def setup(self, stage=None): # called on every GPU
        # build tranforms
        train_transform = BYOLDataTransform(
            crop_size=224,
            mean=self.mean,
            std=self.std)
        val_transform = self.default_transform()
        
        # build datasets
        train_data_dir = os.path.join(self.data_dir, 'train')
        val_data_dir = os.path.join(self.data_dir, 'val')
        self.train = datasets.ImageFolder(train_data_dir, transform=train_transform)
        self.val = datasets.ImageFolder(val_data_dir, transform=val_transform)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False)
        return loader

    def default_transform(self):
        transform = transform_lib.Compose([
            transform_lib.Resize(256),
            transform_lib.CenterCrop(224),
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean=self.mean, std=self.std)])
        return transform


class CIFARDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers, download, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

    @property
    def dataset(self):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def std(self):
        raise NotImplementedError()

    def prepare_data(self): # called only on 1 GPU
        if self.download:
            self.dataset(self.data_dir, train=True, download=self.download)

    def setup(self, stage=None): # called on every GPU
        # build tranforms
        train_transform = BYOLDataTransform(
            crop_size=32,
            mean=self.mean,
            std=self.std,
            blur_prob=[.0, .0],
            solarize_prob=[.0, .2])
        val_transform = self.default_transform()
        
        # build datasets
        self.train = self.dataset(self.data_dir, train=True, transform=train_transform)
        self.val = self.dataset(self.data_dir, train=False, transform=val_transform)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False)
        return loader

    def default_transform(self):
        transform = transform_lib.Compose([
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean=self.mean, std=self.std)
        ])
        return transform


class CIFAR10DataModule(CIFARDataModule):

    def __init__(self, data_dir, batch_size, num_workers, download, **kwargs):
        super().__init__(data_dir, batch_size, num_workers, download)

    @property
    def dataset(self):
        return datasets.CIFAR10

    @property
    def num_classes(self):
        return 10

    @property
    def mean(self):
        return (0.491, 0.482, 0.447)
    
    @property
    def std(self):
        return (0.247, 0.243, 0.261)


class CIFAR100DataModule(CIFARDataModule):

    def __init__(self, data_dir, batch_size, num_workers, download, **kwargs):
        super().__init__(data_dir, batch_size, num_workers, download)

    @property
    def dataset(self):
        return datasets.CIFAR100

    @property
    def num_classes(self):
        return 100

    @property
    def mean(self):
        return (0.507, 0.487, 0.441)
    
    @property
    def std(self):
        return (0.268, 0.257, 0.276)
