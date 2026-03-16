import numpy as np
from torch.utils.data.dataset import Dataset
import copy


def classify_label_fast(dataset, num_classes: int):

    labels = np.array([datum[1] for datum in dataset]) 
    
    label_to_indices = []
    for class_id in range(num_classes):
        mask = (labels == class_id)
        label_to_indices.append(np.where(mask)[0].tolist())
    
    return label_to_indices

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
def analyze_client_distribution(list_client2indices, train_dataset, num_classes):
    
    if hasattr(train_dataset, 'labels'):
        all_labels = np.array(train_dataset.labels)
    else:
        raise AttributeError("DataLoader的dataset属性缺少labels")
    
    client_stats = []
    for client_idx, indices in enumerate(list_client2indices):
        client_labels = all_labels[indices]
        class_counts = np.bincount(client_labels, minlength=num_classes)
        client_stats.append(class_counts)
        
        print(f"Client {client_idx}:")
        print(f"  总样本数: {len(indices)}")
        print(f"  覆盖类别数: {np.sum(class_counts > 0)}")
        print(f"  最多样本类别: {np.argmax(class_counts)} (count={np.max(class_counts)})")
        print(f"  最少样本类别: {np.argmin(class_counts)} (count={np.min(class_counts[class_counts > 0])})\n")
    
    return client_stats


def show_clients_data_distribution(dataset, clients_indices: list, num_classes, logging):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        #print(f'{client}: {nums_data}')
        logging.info('%s:%s, sum:%s' % (client, nums_data, sum(nums_data)))
    return dict_per_client


def partition_train_teach(list_label2indices: list, ipc, seed=None):
    random_state = np.random.RandomState(0)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int):
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_class_num(class_list):
    index = []
    compose = []
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose
