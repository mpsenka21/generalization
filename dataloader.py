# coding: utf-8

import os
import numpy as np

import torch
import torch.utils.data
from torch.utils.data import TensorDataset

import torchvision
import torchvision.models
import torchvision.transforms

import transforms

import copy

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.dataset_dir = os.path.join('~/.torchvision/datasets',
                                        config['dataset'])

        self.use_cutout = (
            'use_cutout' in config.keys()) and config['use_cutout']

        self.use_random_erasing = ('use_random_erasing' in config.keys()
                                   ) and config['use_random_erasing']

    def get_datasets(self, num_per_class=None):
        dset = self.config['dataset']
        if dset == 'MiniMNIST':
            dset = 'MNIST'

        train_dataset = getattr(torchvision.datasets, dset)(
            self.dataset_dir, train=True, transform=self.train_transform, download=True)
        test_dataset = getattr(torchvision.datasets, dset)(
            self.dataset_dir, train=False, transform=self.test_transform, download=True)
        
        if num_per_class:
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000000, shuffle=False)
            for images, labels in loader:
                break
                
            for cls in range(torch.max(labels).item() + 1):
                clsimages = images[(labels == cls)][:num_per_class]
                clslabels = labels[(labels == cls)][:num_per_class]
                
                if cls == 0:
                    all_images = copy.deepcopy(clsimages)
                    all_labels = copy.deepcopy(clslabels)
                else:
                    all_images = torch.cat([all_images, clsimages])
                    all_labels = torch.cat([all_labels, clslabels])
            
            return TensorDataset(all_images, all_labels), test_dataset
        
        return train_dataset, test_dataset

    def _get_random_erasing_train_transform(self):
        raise NotImplementedError

    def _get_cutout_train_transform(self):
        raise NotImplementedError

    def _get_default_train_transform(self):
        raise NotImplementedError

    def _get_train_transform(self):
        if self.use_random_erasing:
            return self._get_random_erasing_train_transform()
        elif self.use_cutout:
            return self._get_cutout_train_transform()
        else:
            return self._get_default_train_transform()


class CIFAR(Dataset):
    def __init__(self, config):
        super(CIFAR, self).__init__(config)

        if config['dataset'] == 'CIFAR10':
            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2470, 0.2435, 0.2616])
        elif config['dataset'] == 'CIFAR100':
            self.mean = np.array([0.5071, 0.4865, 0.4409])
            self.std = np.array([0.2673, 0.2564, 0.2762])

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def _get_random_erasing_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.random_erasing(
                self.config['random_erasing_prob'],
                self.config['random_erasing_area_ratio_range'],
                self.config['random_erasing_min_aspect_ratio'],
                self.config['random_erasing_max_attempt']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_cutout_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.cutout(self.config['cutout_size'],
                              self.config['cutout_prob'],
                              self.config['cutout_inside']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_default_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

    def _get_test_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform


class MNIST(Dataset):
    def __init__(self, config):
        super(MNIST, self).__init__(config)

        self.mean = np.array([0.1307])
        self.std = np.array([0.3081])

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_default_transform()

    def _get_random_erasing_train_transform(self):
        transform = torchvision.transforms.Compose([
            transforms.normalize(self.mean, self.std),
            transforms.random_erasing(
                self.config['random_erasing_prob'],
                self.config['random_erasing_area_ratio_range'],
                self.config['random_erasing_min_aspect_ratio'],
                self.config['random_erasing_max_attempt']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_cutout_train_transform(self):
        transform = torchvision.transforms.Compose([
            transforms.normalize(self.mean, self.std),
            transforms.cutout(self.config['cutout_size'],
                              self.config['cutout_prob'],
                              self.config['cutout_inside']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_default_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

    def _get_default_train_transform(self):
        return self._get_default_transform()

    def _get_default_test_transform(self):
        return self._get_default_transform()


class FashionMNIST(Dataset):
    def __init__(self, config):
        super(FashionMNIST, self).__init__(config)

        self.mean = np.array([0.2860])
        self.std = np.array([0.3530])

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_default_transform()

    def _get_random_erasing_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(28, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.random_erasing(
                self.config['random_erasing_prob'],
                self.config['random_erasing_area_ratio_range'],
                self.config['random_erasing_min_aspect_ratio'],
                self.config['random_erasing_max_attempt']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_cutout_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(28, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.cutout(self.config['cutout_size'],
                              self.config['cutout_prob'],
                              self.config['cutout_inside']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_default_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

    def _get_default_train_transform(self):
        return self._get_default_transform()

    def _get_default_test_transform(self):
        return self._get_default_transform()

# return_full: whether to return a loader that will give you the whole dataset
def get_loader(config, return_full=False):
    batch_size = 200000 if return_full else config['batch_size']
    num_workers = config['num_workers']
    use_gpu = config['use_gpu']
    num_per_class = config['num_per_class']

    dataset_name = config['dataset']
    assert dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'MiniMNIST']

    if dataset_name in ['CIFAR10', 'CIFAR100']:
        dataset = CIFAR(config)
    elif dataset_name in ['MNIST', 'MiniMNIST']:
        dataset = MNIST(config)
    elif dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(config)

    train_dataset, test_dataset = dataset.get_datasets(num_per_class=num_per_class if dataset_name == 'MiniMNIST' else None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu, #False if return_full else use_gpu,
        drop_last=False, # True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu, #False if return_full else use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader
