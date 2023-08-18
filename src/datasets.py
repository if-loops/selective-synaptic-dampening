"""
Datasets used for the experiments (CIFAR and Celebrity Faces)
"""

from typing import Any, Tuple
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Improves model performance (https://github.com/weiaicunzai/pytorch-cifar100)
CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Cropping etc. to improve performance of the model (details see https://github.com/weiaicunzai/pytorch-cifar100)
transform_train_from_scratch = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

transform_unlearning = [
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

transform_test = [
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]


# www.kaggle.com/datasets/hereisburak/pins-face-recognition
class PinsFaceRecognition(ImageFolder):
    def __init__(self, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.insert(0, transforms.Resize((36, 36)))
        transform.append(transforms.Resize((img_size, img_size)))
        transform = transforms.Compose(transform)
        super().__init__(root, transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class Cifar100(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class Cifar20(CIFAR100):
    def __init__(self, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

        # This map is for the matching of subclases to the superclasses. E.g., rocket (69) to Vehicle2 (19:)
        # Taken from https://github.com/vikram2000b/bad-teaching-unlearning
        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, y, coarse_y


class Cifar10(CIFAR10):
    def __init__(self, root, train, unlearning, download, img_size=32):
        if train:
            if unlearning:
                transform = transform_unlearning
            else:
                transform = transform_train_from_scratch
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y
