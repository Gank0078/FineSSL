import logging
import math

import os
import sys
import pickle

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch
import torchvision
import torch.utils.data as data

from .randaugment import RandAugmentMC
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader

from torch.utils.data import Dataset

import json
import math
import PIL.Image
import copy
import csv

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

SVHN_mean = (0.4377, 0.4438, 0.4728)
SVHN_std = (0.1980, 0.2010, 0.1970)


def transpose(x, source='NCHW', target='NHWC'):
    return x.transpose([source.index(d) for d in target])


def read_txt_as_list(prefix, file_path, fsplit=0):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                data = line.split(' ')
                if fsplit == 1:
                    data[0] = prefix + '/' + data[0]
                else:
                    data[0] = prefix + data[0]
                data[1] = int(data[1])
                data = tuple(data)
                data_list.append(data)
    return data_list


def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def get_cifar10(cfg):
    resize_dim = 256
    crop_dim = 224
    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

    base_dataset = datasets.CIFAR10(cfg.DATA.DATAPATH, train=True, download=True)

    l_samples = make_imb_data(cfg.DATA.NUM_L, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_L)
    u_samples = make_imb_data(cfg.DATA.NUM_U, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_U)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, cfg)

    train_labeled_dataset = CIFAR10SSL(
        cfg.DATA.DATAPATH, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        cfg.DATA.DATAPATH, train_unlabeled_idxs, train_labeled_idxs, train=True,
        transform=TransformFixMatch_ws(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        cfg.DATA.DATAPATH, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(cfg):
    resize_dim = 256
    crop_dim = 224
    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim*0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        cfg.DATA.DATAPATH, train=True, download=True)

    l_samples = make_imb_data(cfg.DATA.NUM_L, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_L)
    u_samples = make_imb_data(cfg.DATA.NUM_U, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_U)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, cfg)

    train_labeled_dataset = CIFAR100SSL(
        cfg.DATA.DATAPATH, train_labeled_idxs, train=True,
        transform=transform_labeled, cls_list=l_samples)

    # train_unlabeled_dataset = CIFAR100SSL(
    #     cfg.DATA.DATAPATH, train_unlabeled_idxs, train_labeled_idxs, train=True,
    #     transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std), cls_list=u_samples)

    train_unlabeled_dataset = CIFAR100SSL(
        cfg.DATA.DATAPATH, train_unlabeled_idxs, train_labeled_idxs, train=True,
        transform=TransformFixMatch_ws(mean=cifar100_mean, std=cifar100_std), cls_list=u_samples)

    test_dataset = datasets.CIFAR100(
        cfg.DATA.DATAPATH, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, cfg):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(cfg.DATA.NUMBER_CLASSES):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs


def train_split_l(labels, n_labeled_per_class, cfg):
    labels = np.array(labels)
    train_labeled_idxs = []
    # train_unlabeled_idxs = []
    for i in range(cfg.DATA.NUMBER_CLASSES):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs


def testsplit(labels):
    labels = np.array(labels)
    test_idxs=[]
    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        test_idxs.extend(idxs[:1500])
    np.random.shuffle(test_idxs)
    return test_idxs


def make_imbalance(dataset, indexs):
    dataset.data = dataset.data[indexs]
    dataset.labels = dataset.labels[indexs]
    return dataset


def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    print(class_num_list)
    return list(class_num_list)


class TransformFixMatch(object):
    def __init__(self, mean, std, img_size=32):
        resize_dim = 256
        crop_dim = 224
        self.weak = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)


class TransformFixMatch_ws(object):
    def __init__(self, mean, std, img_size=32):
        resize_dim = 256
        crop_dim = 224
        self.weak = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])

        self.strong = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)


class TransformFixMatchPlaces365(object):
    def __init__(self, mean, std):
        # resize_dim = 256
        crop_dim = 224
        self.weak = transforms.Compose([
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixMatchSTL(object):
    def __init__(self, mean, std):
        resize_dim = 256
        crop_dim = 224
        self.weak = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, exindexs = [], train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, exindexs = [], train=True,
                 transform=None, target_transform=None, cls_list=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.cls_list = cls_list
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class Places365(VisionDataset):
    def __init__(self, root, train=False, transform=None, target_transform=None, indexs=None, cls_list=None, loader=default_loader):
        super(Places365, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.train = train
        if self.train:
            nowroot = root + '/train'
        else:
            nowroot = root + '/val/'
        self.extensions = ('.jpg', '.jpeg', '.png')

        if self.train:
            self.samples = read_txt_as_list(nowroot, root + '/places365_train_standard.txt')
        else:
            self.samples = read_txt_as_list(nowroot, root + '/places365_val.txt')

        self.samples = np.array(self.samples)
        self.targets = self.samples[:, 1].astype(int)

        self.cls_list = cls_list

        if indexs is not None:
            self.samples = self.samples[indexs]
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        target = self.targets[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

def get_places365(cfg):
    crop_dim = 224

    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim*0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    base_dataset = Places365(cfg.DATA.DATAPATH, True)

    labeled_ratio = cfg.DATA.LABEL_RATIO
    arr_num_per_cls = np.load(os.path.join(cfg.DATA.DATAPATH, 'num_per_cls.npy'))
    num_labeled_per_cls = [int(np.around(num * labeled_ratio)) for num in arr_num_per_cls]
    train_labeled_idxs = train_split_l(base_dataset.targets, num_labeled_per_cls, cfg)

    train_labeled_dataset = Places365(cfg.DATA.DATAPATH, True, transform=transform_labeled,
                                          indexs=train_labeled_idxs)
    train_unlabeled_dataset = Places365(cfg.DATA.DATAPATH, True, transform=TransformFixMatchPlaces365(mean=dataset_mean,
                                                                                                     std=dataset_std))
    test_dataset = Places365(cfg.DATA.DATAPATH, False, transform=transform_val)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class FOOD101SSL(datasets.Food101):
    def __init__(self, root, split="train",
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

    def __getitem__(self, index):
        image_file, label = self._image_files[index], self._labels[index]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, index


def get_food101(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    base_dataset = FOOD101SSL(root=cfg.DATA.DATAPATH, split='train', transform=None)
    labeled_ratio = cfg.DATA.LABEL_RATIO
    num_labeled_per_cls = [cfg.DATA.NUM_L for _ in range(cfg.DATA.NUMBER_CLASSES)]
    # num_labeled_per_cls = [int(np.around(750 * labeled_ratio)) for _ in range(cfg.DATA.NUMBER_CLASSES)]
    train_labeled_idxs = train_split_l(base_dataset._labels, num_labeled_per_cls, cfg)

    train_labeled_dataset = FOOD101SSL(root=cfg.DATA.DATAPATH, split='train', transform=transform_labeled)
    train_labeled_dataset._image_files = np.array(train_labeled_dataset._image_files)
    train_labeled_dataset._labels = np.array(train_labeled_dataset._labels)
    train_labeled_dataset._image_files = train_labeled_dataset._image_files[train_labeled_idxs]
    train_labeled_dataset._labels = train_labeled_dataset._labels[train_labeled_idxs]

    train_unlabeled_dataset = FOOD101SSL(root=cfg.DATA.DATAPATH, split='train',
                                               transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))
    test_dataset = FOOD101SSL(root=cfg.DATA.DATAPATH, split='test', transform=transform_val)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class ImageNet(ImageFolder):
    def __init__(self, root, train=False, transform=None, target_transform=None, indexs=None, loader=default_loader):
        self.train = train
        self.loader = loader
        if self.train:
            nowroot = root + '/train'
        else:
            nowroot = root + '/val'
        super(ImageNet, self).__init__(nowroot, transform=transform, target_transform=target_transform)

        self.samples = np.array(self.samples)
        self.imgs = np.array(self.imgs)
        self.targets = np.array(self.targets)

        with open('/ImageNet_LT/imagenet-simple-labels.json') as f:
            self.classes = json.load(f)

        if indexs is not None:
            self.samples = self.samples[indexs]
            self.imgs = self.imgs[indexs]
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        target = self.targets[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        return len(self.samples)


def get_imagenet(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    base_dataset = ImageNet(root=cfg.DATA.DATAPATH, train=True)

    labeled_ratio = cfg.DATA.LABEL_RATIO
    num_per_cls = np.load(cfg.DATA.DATAPATH+'/num_per_cls.npy')
    num_per_cls = num_per_cls.tolist()
    num_labeled_per_cls = [int(np.around(num_per_cls[idx] * labeled_ratio)) for idx in range(cfg.DATA.NUMBER_CLASSES)]
    train_labeled_idxs = train_split_l(base_dataset.targets, num_labeled_per_cls, cfg)

    train_labeled_dataset = ImageNet(root=cfg.DATA.DATAPATH, train=True, transform=transform_labeled,
                                                 indexs=train_labeled_idxs)

    train_unlabeled_dataset = ImageNet(root=cfg.DATA.DATAPATH, train=True,
                                                   transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = ImageNet(root=cfg.DATA.DATAPATH, train=False, transform=transform_val)

    print('-')

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class iNaturalist(VisionDataset):
    def __init__(self, root, train=False, transform=None, target_transform=None, indexs=None, cls_list=None, loader=default_loader):
        super(iNaturalist, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.train = train
        if self.train:
            jsonpath = root + '/train2018.json'
        else:
            jsonpath = root + '/val2018.json'

        with open(jsonpath) as f:
            data = f.read()

        result = json.loads(data)

        self.samples = [root + '/' + item['file_name'] for item in result['images']]
        self.targets = [item['category_id'] for item in result['annotations']]

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)

        if indexs is not None:
            self.samples = self.samples[indexs]
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        image_path = self.samples[index]
        target = self.targets[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)


def get_iNaturalist(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.466, 0.471, 0.380)
    dataset_std = (0.195, 0.194, 0.192)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    base_dataset = iNaturalist(root=cfg.DATA.DATAPATH, train=True)
    labeled_ratio = cfg.DATA.LABEL_RATIO

    num_per_cls = np.load(cfg.DATA.DATAPATH+'/num_per_cls.npy')
    num_per_cls = num_per_cls.tolist()
    num_labeled_per_cls = [min(int(np.around(num_per_cls[idx] * labeled_ratio)), 1) for idx in range(cfg.DATA.NUMBER_CLASSES)]
    train_labeled_idxs = train_split_l(base_dataset.targets, num_labeled_per_cls, cfg)

    train_labeled_dataset = iNaturalist(root=cfg.DATA.DATAPATH, train=True, transform=transform_labeled,
                                     indexs=train_labeled_idxs)

    train_unlabeled_dataset = iNaturalist(root=cfg.DATA.DATAPATH, train=True,
                                       transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = iNaturalist(root=cfg.DATA.DATAPATH, train=False, transform=transform_val)

    print('-')

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class Semi_Aves(VisionDataset):
    def __init__(self, root, train=False, lab=True, out_ulab=False, transform=None, target_transform=None, indexs=None, cls_list=None, loader=default_loader):
        super(Semi_Aves, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.train = train
        self.lab = lab
        self.out_ulab = out_ulab
        if self.train:
            if self.lab:
                jsonpath = root + '/annotation/anno_l_train.json'
            else:
                jsonpath = root + '/annotation/anno_u_train_in.json'
        else:
            jsonpath = root + '/annotation/anno_test.json'

        with open(jsonpath) as f:
            data = f.read()

        result = json.loads(data)

        if self.train and self.lab:
            namepath = root + '/annotation/semi_aves_class_names.txt'
            classes = np.loadtxt(namepath, dtype=str)
            classes = classes[:, 1]
            for i in range(len(classes)):
                classes[i] = classes[i].replace('_', ' ')
            # print(classes)
            self.classes = classes

        self.samples = [root + '/' + item['file_name'] for item in result['images']]
        if self.train:
            self.targets = [item['category_id'] for item in result['annotations']]
            self.targets = np.array(self.targets)
        else:
            self.targets = np.loadtxt(root + '/annotation/solution.csv', delimiter=',', skiprows=1)
            self.targets = self.targets[:, 1].astype(int)

        self.samples = np.array(self.samples)

        if self.out_ulab:
            jsonpath_out = root + '/annotation/anno_u_train_out.json'
            with open(jsonpath_out) as f:
                data = f.read()
            result_out = json.loads(data)

            self.samples_out = [root + '/' + item['file_name'] for item in result_out['images']]
            self.samples_out = np.array(self.samples_out)
            self.samples = np.concatenate((self.samples, self.samples_out))

            self.targets_out = [item['category_id'] for item in result_out['annotations']]
            self.targets_out = np.array(self.targets_out)
            self.targets = np.concatenate((self.targets, self.targets_out))

        if indexs is not None:
            self.samples = self.samples[indexs]
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        image_path = self.samples[index]
        target = self.targets[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        return len(self.samples)


def get_semi_aves(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    train_labeled_dataset = Semi_Aves(root=cfg.DATA.DATAPATH, train=True, lab=True, transform=transform_labeled)

    train_unlabeled_dataset = Semi_Aves(root=cfg.DATA.DATAPATH, train=True, lab=False, out_ulab=cfg.DATA.out_ulab,
                                          transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = Semi_Aves(root=cfg.DATA.DATAPATH, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class ImageNet_LT(ImageFolder):
    def __init__(self, root, train=False, transform=None, target_transform=None, indexs=None, loader=default_loader):
        self.train = train
        self.loader = loader
        if self.train:
            txtpath = '../datasets/ImageNet_LT/ImageNet_LT_train.txt'
        else:
            txtpath = '../datasets/ImageNet_LT/ImageNet_LT_test.txt'
        super(ImageNet_LT, self).__init__(root, transform=transform, target_transform=target_transform)

        self.samples = read_txt_as_list(root, txtpath, fsplit=1)

        self.samples = np.array(self.samples)
        self.targets = self.samples[:, 1].astype(int)


        if indexs is not None:
            self.samples = self.samples[indexs]
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        target = self.targets[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        return len(self.samples)


def get_imagenet_lt(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    base_dataset = ImageNet_LT(root=cfg.DATA.DATAPATH, train=True)

    labeled_ratio = 0.25

    num_per_cls = np.load('/ImageNet_LT/num_per_cls.npy')
    num_per_cls = num_per_cls.tolist()
    num_labeled_per_cls = [math.ceil(num_per_cls[idx] * labeled_ratio) for idx in range(cfg.DATA.NUMBER_CLASSES)]

    train_labeled_idxs = train_split_l(base_dataset.targets, num_labeled_per_cls, cfg)

    train_labeled_dataset = ImageNet_LT(root=cfg.DATA.DATAPATH, train=True, transform=transform_labeled,
                                                 indexs=train_labeled_idxs)

    train_unlabeled_dataset = ImageNet_LT(root=cfg.DATA.DATAPATH, train=True,
                                                   transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = ImageNet_LT(root=cfg.DATA.DATAPATH, train=False, transform=transform_val)

    print('-')

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_stl10(cfg):
    resize_dim = 256
    crop_dim = 224
    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim*0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    train_labeled_dataset = STL10SSL(cfg.DATA.DATAPATH, split="train", transform=transform_labeled, download=True)
    train_unlabeled_dataset = STL10SSL(cfg.DATA.DATAPATH, split="unlabeled",
                                             transform=TransformFixMatchSTL(mean=stl10_mean, std=stl10_std),
                                             download=True)
    test_dataset = STL10SSL(cfg.DATA.DATAPATH, split="test", transform=transform_val, download=True)

    l_samples = make_imb_data(cfg.DATA.NUM_L, cfg.DATA.NUMBER_CLASSES, cfg.DATA.IMB_L)
    train_labeled_idxs = train_split_l(train_labeled_dataset.labels, l_samples, cfg)
    train_labeled_dataset = make_imbalance(train_labeled_dataset, train_labeled_idxs)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class STL10SSL(datasets.STL10):
    def __init__(self, root, split, transform=None, target_transform=None, cls_list=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.cls_list = cls_list

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class semi_ImageNet(ImageFolder):
    def __init__(self, root, train=False, lab=False, ratio=0.01, transform=None, target_transform=None, loader=default_loader):
        self.train = train
        self.lab = lab
        self.loader = loader

        nowroot = root + '/train'

        super(semi_ImageNet, self).__init__(nowroot, transform=transform, target_transform=target_transform)

        if self.train:
            if self.lab:
                new_targets, new_samples = [], []
                if ratio == 0.01:
                    with open(root+'/99percent.txt', 'r') as rfile:
                        for line in rfile:
                            class_name = line.split('_')[0]
                            target = self.class_to_idx[class_name]
                            img = line.split('\n')[0]
                            new_samples.append(
                                (os.path.join(nowroot, class_name, img),
                                 target))
                            new_targets.append(target)
                elif ratio == 0.1:
                    with open(root+'/90percent.txt', 'r') as rfile:
                        for line in rfile:
                            class_name = line.split('_')[0]
                            target = self.class_to_idx[class_name]
                            img = line.split('\n')[0]
                            new_samples.append(
                                (os.path.join(nowroot, class_name, img),
                                 target))
                            new_targets.append(target)

                self.targets, self.samples = np.array(new_targets), np.array(new_samples)

            else:
                self.samples = np.array(self.samples)
                self.imgs = np.array(self.imgs)
                self.targets = np.array(self.targets)
        else:
            new_targets, new_samples = [], []
            with open(root + '/val.txt', 'r') as rfile:
                for line in rfile:
                    class_name = line.split('_')[0]
                    target = self.class_to_idx[class_name]
                    img = line.split('\n')[0]
                    new_samples.append(
                        (os.path.join(nowroot, class_name, img),
                         target))
                    new_targets.append(target)

            self.targets, self.samples = np.array(new_targets), np.array(new_samples)

        with open('/ImageNet_LT/imagenet-simple-labels.json') as f:
            self.classes = json.load(f)


    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        target = self.targets[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        return len(self.samples)


def get_semi_imagenet(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    train_labeled_dataset = semi_ImageNet(root=cfg.DATA.DATAPATH, train=True, lab=True, ratio=cfg.DATA.LABEL_RATIO, transform=transform_labeled)

    train_unlabeled_dataset = semi_ImageNet(root=cfg.DATA.DATAPATH, train=True,
                                                   transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))

    test_dataset = semi_ImageNet(root=cfg.DATA.DATAPATH, train=False, transform=transform_val)

    print('-')

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def read(file):
    x, y = [], []
    with open(file, 'r') as f:
        for _, line in enumerate(f):
            xi, yi = line.strip().split()
            x.append(xi)
            y.append(int(yi))
    return np.array(x), np.array(y)


def split_Nmax(labels, Nmax, num_cls):
    labels = np.array(labels)
    idxs = []
    for i in range(num_cls):
        idx = np.where(labels == i)[0]
        idxs.extend(idx[:Nmax])
    return idxs


class domainnet(ImageFolder):
    def __init__(self, root, train=True, Nmax=30, type='real', transform=None, target_transform=None, num_cls=100, loader=default_loader):
        self.train = train
        self.loader = loader
        self.type = type
        self.Nmax = Nmax
        self.ori_root = root

        dataroot = os.path.join(root, type)

        super(domainnet, self).__init__(dataroot, transform=transform, target_transform=target_transform)

        if self.train:
            samples, targets = read(root + f'/annotation/{type}_train.txt')
            # idxs = split_Nmax()
        else:
            samples, targets = read(root + f'/annotation/{type}_test.txt')

        idxs = split_Nmax(targets, Nmax, num_cls)

        self.samples = samples[idxs]
        self.targets = targets[idxs]
        self.imgs = None

        print('-')

    def __getitem__(self, index):
        # image_path = self.samples[index]
        image_path = os.path.join(self.ori_root, self.samples[index])
        target = self.targets[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        return len(self.samples)


def get_domainnet(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    train_labeled_dataset = domainnet(root=cfg.DATA.DATAPATH, train=True, Nmax=cfg.DATA.NUM_L, type=cfg.s_type,
                                                    transform=transform_labeled, num_cls=cfg.DATA.NUMBER_CLASSES)
    train_unlabeled_dataset = domainnet(root=cfg.DATA.DATAPATH, train=True, Nmax=cfg.DATA.NUM_U, type=cfg.t_type,
                                            transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std),
                                        num_cls=cfg.DATA.NUMBER_CLASSES)

    test_dataset = domainnet(root=cfg.DATA.DATAPATH, train=False, Nmax=cfg.DATA.NUM_Test, type=cfg.s_type,
                             transform=transform_val, num_cls=cfg.DATA.NUMBER_CLASSES)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def eurosat_split(labels, Nlab, Nulab, Ntest, cfg):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    test_idxs = []
    for i in range(cfg.DATA.NUMBER_CLASSES):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:Nlab])
        train_unlabeled_idxs.extend(idxs[:Nlab + Nulab])
        test_idxs.extend(idxs[Nlab + Nulab: Nlab + Nulab + Ntest])
    return train_labeled_idxs, train_unlabeled_idxs, test_idxs


class eurosat(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, indexs=None, loader=default_loader):
        # self.train = train
        self.loader = loader

        super(eurosat, self).__init__(root, transform=transform, target_transform=target_transform)

        if indexs is not None:
            indexs = np.array(indexs)
            self.samples = np.array(self.samples)
            self.targets = np.array(self.targets)
            self.samples = self.samples[indexs]
            self.targets = self.targets[indexs]

    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        target = self.targets[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        return len(self.samples)


def get_eurosat(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    base_dataset = eurosat(root=cfg.DATA.DATAPATH)

    train_labeled_idxs, train_unlabeled_idxs, test_idxs = \
        eurosat_split(base_dataset.targets, cfg.DATA.NUM_L, cfg.DATA.NUM_U, cfg.DATA.NUM_Test, cfg)

    train_labeled_dataset = eurosat(root=cfg.DATA.DATAPATH, transform=transform_labeled, indexs=train_labeled_idxs)
    train_unlabeled_dataset = eurosat(root=cfg.DATA.DATAPATH, transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std),
                                      indexs=train_unlabeled_idxs)
    test_dataset = eurosat(root=cfg.DATA.DATAPATH, transform=transform_val, indexs=test_idxs)

    print('-')
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class CIFAR100_C(VisionDataset):
    def __init__(self, root, type='brightness', sidx=0, eidx=-1, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.type = type
        self.transform = transform
        self.target_transform = target_transform

        dataroot = os.path.join(root, type)
        dataroot = dataroot + '.npy'
        data = np.load(dataroot)
        targets = np.load(os.path.join(root, 'labels.npy'))

        self.data = data[sidx: eidx]
        self.targets = targets[sidx: eidx]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


def get_cifar100_c(cfg, Ctype=None):
    # resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    test_dataset_list = []
    sidx = 0
    eidx = 10000
    addlen = 10000
    for i in range(5):
        if Ctype is not None:
            test_dataset = CIFAR100_C(root=cfg.DATA.DATAPATH, type=Ctype, sidx=sidx, eidx=eidx, transform=transform_val)
        else:
            test_dataset = CIFAR100_C(root=cfg.DATA.DATAPATH, type=cfg.Ctype, sidx=sidx, eidx=eidx, transform=transform_val)
        test_dataset_list.append(test_dataset)

        sidx = sidx + addlen
        eidx = eidx + addlen
    return test_dataset_list


class CUB(Dataset):
    def __init__(self, root, train=True, lab=False, lab_freq=None, transform=None, target_transform=None, loader=default_loader):
        self.root = root
        self.train = train
        self.lab = lab
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # super(eurosat, self).__init__(root, transform=transform, target_transform=target_transform)

        self.images_path = {}
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id

        self.data_id = []
        self.classes = []
        if self.train:
            if lab_freq is not None:
                freq = copy.deepcopy(lab_freq)
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        if self.lab:
                            target = int(self._get_class_by_id(image_id)) - 1
                            if freq[target] > 0:
                                self.data_id.append(image_id)
                                freq[target] -= 1
                        else:
                            self.data_id.append(image_id)

            if self.lab:
                with open(os.path.join(self.root, 'classes.txt')) as f:
                    for line in f:
                        _, name = line.split()
                        self.classes.append(" ".join(name.split('.')[1].split('_')))

        if not self.train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)

        print('-')

    def __getitem__(self, index):
        image_id = self.data_id[index]
        target = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image_path = os.path.join(self.root, 'images', path)
        # image = cv2.imread(os.path.join(self.root, 'images', path))

        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, index

    def __len__(self):
        # return len(self.samples)
        return len(self.data_id)

    def _get_path_by_id(self, image_id):
        return self.images_path[image_id]

    def _get_class_by_id(self, image_id):
        return self.class_ids[image_id]


def get_cub(cfg):
    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    base_dataset = CUB(root=cfg.DATA.DATAPATH, train=True)

    cls_freq = np.zeros(cfg.DATA.NUMBER_CLASSES, dtype=int)
    for id in base_dataset.data_id:
        target = int(base_dataset._get_class_by_id(id)) - 1
        cls_freq[target] += 1

    lab_freq = (cls_freq * cfg.DATA.LABEL_RATIO).astype(int)

    train_labeled_dataset = CUB(root=cfg.DATA.DATAPATH, train=True, lab=True, lab_freq=lab_freq, transform=transform_labeled)
    train_unlabeled_dataset = CUB(root=cfg.DATA.DATAPATH, train=True, lab=False,
                                      transform=TransformFixMatch_ws(mean=dataset_mean, std=dataset_std))
    test_dataset = CUB(root=cfg.DATA.DATAPATH, train=False, transform=transform_val)

    print('-')

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class Conceptual_Captions(Dataset):
    def __init__(self):
        self.data = []
        file_path = '/Conceptual_Captions/Train_GCC-training.tsv'
        with open(file_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            # next(reader, None)
            for row in reader:
                item = row[0]
                self.data.append(item.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


DATASET_GETTERS = {'CIFAR10': get_cifar10,
                   'CIFAR100': get_cifar100,
                   'STL10': get_stl10,
                   'PLACES365': get_places365,
                   'FOOD101': get_food101,
                   'IMAGENET': get_imagenet,
                   'iNaturalist': get_iNaturalist,
                   'Semi_Aves': get_semi_aves,
                   'ImageNet-LT': get_imagenet_lt,
                   'semi_ImageNet': get_semi_imagenet,
                   'DomainNet': get_domainnet,
                   'EuroSAT': get_eurosat,
                   'CIFAR100-C': get_cifar100_c,
                   'CUB': get_cub}

