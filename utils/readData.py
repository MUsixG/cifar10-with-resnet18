import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils.cutout import Cutout
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#--------------

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_custom_cifar10_data(data_dir, transform, train=True):
    data = []
    labels = []

    if train:
        for i in range(1, 6):  # 加载5个训练批次
            batch = load_cifar_batch(os.path.join(data_dir, f'data_batch_{i}'))
            data.append(batch[b'data'])
            labels.extend(batch[b'labels'])
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 调整形状为：N x H x W x C
    else:
        batch = load_cifar_batch(os.path.join(data_dir, 'test_batch'))
        data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 直接加载单个测试批次
        labels = batch[b'labels']

    labels = np.array(labels)

    # 应用转换（包括Cutout，但Cutout仅应用于训练数据）
    processed_data = []
    if transform is not None:
      for image in data:
        processed_image = transform(transforms.ToPILImage()(image))
        processed_data.append(processed_image)
    processed_data = torch.stack(processed_data)

    return TensorDataset(processed_data, torch.tensor(labels, dtype=torch.long))








#----------------------





def read_dataset(batch_size=16,valid_size=0.2,num_workers=0,pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), 
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])


    # make data to torch.FloatTensor
    # train_data = datasets.CIFAR10(pic_path, train=True,
    #                             download=True, transform=transform_train)
    # valid_data = datasets.CIFAR10(pic_path, train=True,
    #                             download=True, transform=transform_test)
    # test_data = datasets.CIFAR10(pic_path, train=False,
    #                             download=True, transform=transform_test)
    #______________________
    
    
    train_data = get_custom_cifar10_data(pic_path, transform=transform_train, train=True)
    test_data = get_custom_cifar10_data(pic_path, transform=transform_test, train=False)
    #_____________________  

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    valid_data = get_custom_cifar10_data(pic_path, transform=transform_test, train=True)
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader,valid_loader,test_loader