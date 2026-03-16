from torchvision import datasets
from torchvision.transforms import transforms
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset
from Dataset.sample_dirichlet import clients_indices
import numpy as np
from torch import max, eq, no_grad
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from Model.Resnet8 import ResNet_cifar
import copy
import torch
import random
import logging, os
from datetime import datetime
import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_cifar10', type=str, default='./data/cifar10')
    parser.add_argument('--path_cifar100', type=str, default='./data/cifar100')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)
    parser.add_argument('--batch_size_local_training', type=int, default=32)
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    
    # FedNPC
    parser.add_argument('--baseline_name', type=str, default='FedAvg', help='Stage1 method')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate of stage2')
    parser.add_argument('--E', type=int, default=28, help='Training epochs of stage2')
    parser.add_argument('--M', type=int, default=10000, help='Training sample size of stage2')
    
    args = parser.parse_args()
    return args


class Global(object):
    def __init__(self,
                 num_classes: int,
                 device: str,
                 args):
        self.device = device
        self.num_classes = num_classes
        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []
        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []
        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # fedavg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.syn_model.state_dict()


class Local(object):
    def __init__(self,
                 data_client,
                 class_list: int):
        args = args_parser()

        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        self.criterion = CrossEntropyLoss().to(args.device)

        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(
            args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def local_train(self, args, global_params):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        
        local_avg_loss = 0.0
        total_steps = 0
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)
                _, outputs = self.local_model(images)
                loss = self.criterion(outputs, labels)
                local_avg_loss += loss.item()
                total_steps += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return self.local_model.state_dict(), local_avg_loss / total_steps


def generate_noisy_batch(args, n_classes, dimention):
    fake_classes = torch.randint(0, n_classes, (args.M,))
    features = torch.randn((args.M, dimention))
    return features, fake_classes

def re_train_classifier(args, acc, global_model, re_global_model, data_global_test):
    re_global_model = copy.deepcopy(global_model.syn_model)
    
    n_classes = re_global_model.classifier.weight.shape[0]
    dimention = re_global_model.classifier.weight.shape[1] 
    
    logging.info("Initialize Optimizer: ......") 
    for name, param in re_global_model.named_parameters():
        if 'classifier' not in name:  
            param.requires_grad = False
    parameters = list(filter(lambda p: p.requires_grad, re_global_model.parameters()))
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9)
    criterion = CrossEntropyLoss().cuda() 
    
    best_acc = acc
    best_epoch = 0
    best_pth = None
    
    for epoch in range(1, args.E):
        re_global_model.train()
        features, targets = generate_noisy_batch(args, n_classes, dimention)
        features = features.cuda()
        targets = targets.cuda()
        
        output = re_global_model.classifier(features)
        loss = criterion(output, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = global_model.global_eval(re_global_model.state_dict(), data_global_test, args.batch_size_test)
    
        logging.info('>> After(Epoch:%d): Global Model Test accuracy: %f' % (epoch, acc))
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_pth = copy.deepcopy(re_global_model.state_dict())
    
    logging.info('>> Best Epoch:%d,  Global Model Best Test accuracy: %f' % (best_epoch, best_acc))
    return best_pth  


def FedNPC():
    args = args_parser()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    if args.num_classes == 10:
        log_dir = f"./FedAvg-FedNPC/cifar10/seed{args.seed}_ratio{args.imb_factor}_alpha{args.non_iid_alpha}"
    elif args.num_classes == 100:
        log_dir = f"./FedAvg-FedNPC/cifar100/seed{args.seed}_ratio{args.imb_factor}_alpha{args.non_iid_alpha}"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info('imb_factor:%s, non_iid:%s' % (args.imb_factor, args.non_iid_alpha))
    logging.info(args)
    
    random_state = np.random.RandomState(args.seed)
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.num_classes == 10:
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    elif args.num_classes == 100:
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=transform_all)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_all)
    
    list_label2indices = classify_label(data_local_training, args.num_classes)
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes, logging)
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args)
    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)
    
    # Stage1 Training
    best_acc = 0
    re_trained_acc = []
    
    for r in range(1, args.num_rounds+1):
        global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []
        avg_local_loss = 0.0  
        
        # Local Training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])
            
            local_params, local_loss = local_model.local_train(args, copy.deepcopy(global_params))
            list_dicts_local_params.append(copy.deepcopy(local_params))
            avg_local_loss += local_loss
        
        avg_local_loss /= len(online_clients)
        
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        one_train_acc = global_model.global_eval(fedavg_params, data_global_test, args.batch_size_test)
        re_trained_acc.append(one_train_acc)
        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
        logging.info('Round %s: Test Accuracy = %s, Avg Local Loss = %s' % (r, one_train_acc, avg_local_loss))
        
        save_name = log_filename.rstrip('.log')
        if one_train_acc > best_acc:
            best_acc = one_train_acc
            # logging.info('Best Acc = %s' % best_acc)
            
    logging.info('Stage1 All Test Accuracies: %s' % re_trained_acc)
    
    # save stage1_final.pth
    save_name = log_filename.rstrip('.log')
    save_name_log = f"stage1_final_{save_name}.pth"
    save_path = os.path.join(log_dir, save_name_log)
    checkpoint = {
        'args_all': args,
        'global_model': fedavg_params,
        'test_acc_recorder': re_trained_acc
    }
    torch.save(checkpoint, save_path)
    logging.info('Stage1: final save successful!')
    
    # Stage2 Training
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          args=args)
    checkpoint = torch.load(save_path, weights_only=False)
    ft_params = (checkpoint['global_model']) 
    acc = global_model.global_eval(ft_params, data_global_test, args.batch_size_test)
    logging.info("Stage 1 Final Accuracy: %s", acc)
    
    re_global_model = copy.deepcopy(global_model.syn_model)
    copy_global_model = copy.deepcopy(global_model)
    re_best_pth = re_train_classifier(args, acc, copy_global_model, re_global_model, data_global_test)
    
    second_acc = global_model.global_eval(re_best_pth, data_global_test, args.batch_size_test)
    logging.info('Stage2 Final Best Accuracy: %s' % (second_acc))
    
    # save stage2_best.pth
    stage2_best_path = os.path.join(log_dir, f"stage2_best_{save_name}.pth")
    torch.save(re_best_pth, stage2_best_path)


if __name__ == '__main__':
    FedNPC()