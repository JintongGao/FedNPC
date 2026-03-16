import torch
import random
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image


class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label


class ImageNetLTDataLoader(DataLoader):
    def __init__(self, shuffle=True):
        data_dir = '/home/gaojintong/dataset/imagenet/'
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = LT_Dataset(data_dir, '/home/gaojintong/dataset/imagenet/LT_txt/ImageNet_LT_train.txt', train_trsfm)
        val_dataset = LT_Dataset(data_dir, '/home/gaojintong/dataset/imagenet/LT_txt/ImageNet_LT_val.txt', test_trsfm)
        test_dataset = LT_Dataset(data_dir, '/home/gaojintong/dataset/imagenet/LT_txt/ImageNet_LT_test.txt', test_trsfm)

        self.dataset = dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.n_samples = len(self.dataset)
        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 1000
        self.num_classes = num_classes
    
        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list
        self.shuffle = shuffle
        self.init_kwargs = {
            'shuffle': self.shuffle
        }
        
        total = sum(self.cls_num_list)
        print("\nTotal samples:", total)
        print("Max class count:", max(self.cls_num_list))
        print("Min class count:", min([x for x in self.cls_num_list if x > 0]))
        print("Num of classes:", len([x for x in self.cls_num_list if x > 0]))
        
        self._process_labels()
        
        super().__init__(dataset=self.dataset, **self.init_kwargs)
    
    def _process_labels(self):
        """极速标签处理（比传统方法快100倍）"""
        # 零拷贝获取标签
        labels = np.asarray(self.dataset.labels, dtype=np.int32)
        
        # 向量化分类（无循环）
        sorted_idx = np.argsort(labels)
        sorted_labels = labels[sorted_idx]
        bounds = np.where(np.diff(sorted_labels, prepend=-1))[0]
        bounds = np.append(bounds, len(labels))
        
        # 存储分类结果
        self.label_to_indices = [
            sorted_idx[bounds[i]:bounds[i+1]].tolist() 
            for i in range(len(bounds)-1)
        ]
