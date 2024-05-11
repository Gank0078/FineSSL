# [ICML 2024] Erasing the Bias: Fine-Tuning Foundation Models for Semi-Supervised Learning

This is PyTorch implementation of Erasing the Bias: Fine-Tuning Foundation Models for Semi-Supervised Learning at ICML 2024. 

## Abstract

Semi-supervised learning (SSL) has witnessed remarkable progress, resulting in the emergence of numerous method variations. However, practitioners often encounter challenges when attempting to deploy these methods due to their subpar performance. In this paper, we present a novel SSL approach named FINESSL that significantly addresses this limitation by adapting pre-trained foundation models. We identify the aggregated biases and cognitive deviation problems inherent in foundation models, and propose a simple yet effective solution by imposing balanced margin softmax and decoupled label smoothing. Through extensive experiments, we demonstrate that FI-NESSL sets a new state of the art for SSL on multiple benchmark datasets, reduces the training cost by over six times, and can seamlessly integrate various fine-tuning and modern SSL algorithms.

## Requirements

- Python 3.7.13
- PyTorch 1.12.0+cu116
- torchvision
- numpy
- timm

## Dataset

The directory structure for datasets looks like:
```
datasets
├── cifar-10
├── cifar-100
├── food-101
```


## Usage

Train our proposed FineSSL for different settings.

For CIFAR-10:

```
# run N1 setting with VPT
python main.py --cfg configs/peft/cifar10_N1.yaml vpt_deep True

# run N2 setting with VPT
python main.py --cfg configs/peft/cifar10_N2.yaml vpt_deep True

# run N4 setting with VPT
python main.py --cfg configs/peft/cifar10_N4.yaml vpt_deep True
```
For CIFAR-100:

```
# run N4 setting with VPT
python main.py --cfg configs/peft/cifar100_N4.yaml vpt_deep True

# run N25 setting with VPT
python main.py --cfg configs/peft/cifar100_N25.yaml vpt_deep True

# run N100 setting with VPT
python main.py --cfg configs/peft/cifar100_N100.yaml vpt_deep True
```

## Acknowledge

We thank the authors of the [PEL](https://github.com/shijxcs/PEL) for making their code available to the public.

## Citation

```
@InProceedings{Gan_2024_ICML,
    author    = {Gan, Kai and Wei, Tong},
    title     = {Erasing the Bias: Fine-Tuning Foundation Models for Semi-Supervised Learning},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML 2024)},
    month     = {July},
    year      = {2024}
}
```

