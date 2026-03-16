# FedNPC: Stochastic Noise-driven Post-hoc Classifier Calibration Method for Federated Long-tailed Learning (CVPR Findings, 2026)

by Jintong Gao<sup>1</sup>, He Zhao<sup>2</sup>, Yibo Yang<sup>3</sup>, Dandan Guo<sup>1</sup>, <sup>3</sup>

<sup>1</sup>Jilin University, <sup>2</sup>CSIRO's Data61, <sup>3</sup>King Abdullah University of Science and Technology

This is the official implementation of [FedNPC: Stochastic Noise-driven Post-hoc Classifier Calibration Method for Federated Long-tailed Learning](XXX) in PyTorch.

## Requirements:

All codes are written by Python 3.10.9 with 

```
PyTorch 2.5.1
torchvision 0.20.1
Numpy 1.23.5
```

## Training

To train and test the model(s) in the paper, run this command:

### CIFAR-LT

CIFAR-10-LT (FedAvg + FedNPC):

```
CUDA_VISIBLE_DEVICES=0 python FedAvg-FedNPC.py --num_classes 10
```

CIFAR-100-LT (FedAvg + FedNPC):

```
CUDA_VISIBLE_DEVICES=0 python FedAvg-FedNPC.py --num_classes 100
```

## Citation

If you find our paper and repo useful, please cite our paper.

```
@inproceedings{
Gao2024DisA,
title={Distribution Alignment Optimization through Neural Collapse for Long-tailed Classification},
author={Jintong Gao and He Zhao and Dandan Guo and Hongyuan Zha},
booktitle={International Conference on Machine Learning (ICML)},
year={2024}
}
```
## Contact

If you have any questions when running the code, please feel free to concat us by emailing

+ Jintong Gao ([gaojt23@mails.jlu.edu.cn](mailto:gaojt23.mails.jlu.edu.cn))
