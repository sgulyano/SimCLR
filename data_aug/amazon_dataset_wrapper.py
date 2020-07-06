import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
import glob
import os

np.random.seed(0)

class AmazonDataset(Dataset):
    """Amazon dataset."""

    def __init__(self, root_dir, split, img_size=128, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        
        if split == 'train+unlabeled':
            l1 = glob.glob(os.path.join(root_dir, 'train/**/*'))
            l2 = glob.glob(os.path.join(root_dir, 'unlabeled/**/*'))
            self.img_list = l1 + l2
        elif split == 'test':
            self.img_list = glob.glob(os.path.join(root_dir, '/test/**/*'))
        print(f'number of images = {len(self.img_list)}')
        self.img_type = [f.split('/')[-2] for f in self.img_list]
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_list[idx]
        img = Image.open(img_name).convert('RGB')
        target = self.img_type[idx] == 'agri'

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        train_dataset = AmazonDataset(root_dir='./data/satellite/amazon', split='train+unlabeled',
                                      transform=SimCLRDataTransform(data_augment))
        
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        print(f'train loader length: {len(train_loader)}, validation loader length: {len(valid_loader)}')
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        print(self.input_shape)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])-1),
                                              transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        print(f'number of split : {split}')
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
