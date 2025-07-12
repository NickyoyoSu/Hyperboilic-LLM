import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np

def calculate_imagenet_var(dataloader, num_batches=50):
    """使用批处理计算数据集方差"""
    print("计算数据集方差...")
    running_var = 0.0
    count = 0
    
    for i, (batch, _) in enumerate(dataloader):
        if i >= num_batches:  # 只使用部分批次计算方差
            break
        # 计算当前批次的方差
        batch_var = torch.var(batch).item()
        running_var += batch_var
        count += 1
        if i % 10 == 0:
            print(f"已处理 {i}/{num_batches} 批次")
    
    final_var = running_var / count if count > 0 else 0
    print(f"方差计算完成: {final_var:.6f}")
    return final_var

def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val

import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_imagenet_(data_dir="/ext/imagenet"):
    transform = transforms.Compose([
        transforms.Resize(128),  
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    return train, val
import os


def load_imagenet(data_dir="/ext/work/ILSVRC2012_img_val"):
    print(f"尝试加载数据集，路径: {data_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集根目录不存在: {data_dir}")
    
    #train_dir = os.path.join(data_dir, "train")
    train_dir = "/ext/imagenet/train"
    #val_dir = os.path.join(data_dir, "val")
    val_dir = "/ext/imagenet/val"
    
    # 检查训练和验证集目录
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"验证集目录不存在: {val_dir}")
    
    # 检查目录内容
    train_classes = os.listdir(train_dir)
    print(f"训练集目录包含以下内容: {train_classes}")    
    # 检查第一个类别文件夹中的文件
    if train_classes:
        first_class = train_classes[0]
        first_class_path = os.path.join(train_dir, first_class)
        if os.path.isdir(first_class_path):
            files = os.listdir(first_class_path)
            print(f"第一个类别 {first_class} 包含的文件: {files[:5]} ...")
            print(f"总计文件数: {len(files)}")
    
    transform = transforms.Compose([
        transforms.Resize(128),  
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        train = datasets.ImageFolder(root=train_dir, transform=transform)
        print(f"成功加载训练集，包含 {len(train)} 张图片")
    except Exception as e:
        print(f"加载训练集失败: {str(e)}")
        raise
        
    try:
        val = datasets.ImageFolder(root=val_dir, transform=transform)
        print(f"成功加载验证集，包含 {len(val)} 张图片")
    except Exception as e:
        print(f"加载验证集失败: {str(e)}")
        raise
    
    return train, val

def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        #x_train_var = np.var(training_data.train_data / 255.0)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'IMAGENET':
        training_data, validation_data = load_imagenet()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = calculate_imagenet_var(training_loader)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
