"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate subclass forgetting.
Seperate file to allow for easy reuse.
"""

import random
import os
import wandb

# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import models
from unlearn import *
from utils import *
import forget_subclass_strategies
import datasets
import models
import conf
from training_utils import *


"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, required=True, help="net type")
parser.add_argument(
    "-weight_path",
    type=str,
    required=True,
    help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
    "-dataset",
    type=str,
    required=True,
    nargs="?",
    choices=["Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition"],
    help="dataset to train on",
)
parser.add_argument(
    "-superclasses", type=int, required=True, help="number of superclasses"
)
parser.add_argument("-subclasses", type=int, required=True, help="number of subclasses")
parser.add_argument("-gpu", action="store_true", default=False, help="use gpu or not")
parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    required=True,
    nargs="?",
    choices=[
        "baseline",
        "retrain",
        "finetune",
        "blindspot",
        "amnesiac",
        "UNSIR",
        "FisherForgetting",
        "ssd_tuning",
    ],
    help="select unlearning method from choice set",
)
parser.add_argument(
    "-forget_class",
    type=str,
    required=True,
    nargs="?",
    help="class to forget",
    choices=list(conf.class_dict),
)
parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-my_seed", type=int, default=0, help="seed for runs")
parser.add_argument("-seed", type=int, default=42, help="seed for runs")
args = parser.parse_args()

# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


assert args.forget_class in conf.cifar100_classes
forget_class = conf.class_dict[args.forget_class]

batch_size = args.b

# get network
net = getattr(models, args.net)(num_classes=args.superclasses)
net.load_state_dict(torch.load(args.weight_path))

unlearning_teacher = getattr(models, args.net)(num_classes=args.superclasses)

if args.gpu:
    net = net.cuda()
    unlearning_teacher = unlearning_teacher.cuda()

root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

img_size = 224 if args.net == "ViT" else 32
trainset = getattr(datasets, args.dataset)(
    root=root, download=True, train=True, unlearning=True, img_size=img_size
)
validset = getattr(datasets, args.dataset)(
    root=root, download=True, train=False, unlearning=True, img_size=img_size
)

trainloader = DataLoader(trainset, num_workers=4, batch_size=args.b, shuffle=True)
validloader = DataLoader(validset, num_workers=4, batch_size=args.b, shuffle=False)

classwise_train, classwise_test = forget_subclass_strategies.get_classwise_ds(
    trainset, args.subclasses
), forget_subclass_strategies.get_classwise_ds(validset, args.subclasses)

(
    retain_train,
    retain_valid,
    forget_train,
    forget_valid,
) = forget_subclass_strategies.build_retain_forget_sets(
    classwise_train, classwise_test, args.subclasses, forget_class
)
forget_valid_dl = DataLoader(forget_valid, batch_size)
retain_valid_dl = DataLoader(retain_valid, batch_size)

forget_train_dl = DataLoader(forget_train, batch_size)
retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)

full_train_dl = DataLoader(
    ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    batch_size=batch_size,
)


for idx, li in validset.coarse_map.items():
    if forget_class in li:
        forget_superclass = idx
        break


# Change alpha here
model_size_scaler = 1
if args.net == "ViT":
    model_size_scaler = 2.5
else:
    model_size_scaler = 1

kwargs = {
    "model": net,
    "unlearning_teacher": unlearning_teacher,
    "retain_train_dl": retain_train_dl,
    "retain_valid_dl": retain_valid_dl,
    "forget_train_dl": forget_train_dl,
    "forget_valid_dl": forget_valid_dl,
    "full_train_dl": full_train_dl,
    "valid_dl": validloader,
    "dampening_constant": 1,
    "selection_weighting": 10 * model_size_scaler,
    "forget_subclass": forget_class,
    "forget_superclass": forget_superclass,
    "num_superclasses": args.superclasses,
    "num_subclasses": args.subclasses,
    "dataset_name": args.dataset,
    "device": "cuda" if args.gpu else "cpu",
    "model_name": args.net,
}

wandb.init(
    project=f"R1_{args.net}_{args.dataset}_subclass_{args.forget_class}",
    name=f"{args.method}",
)


import time

start = time.time()


testacc, retainacc, zrf, mia, d_f = getattr(forget_subclass_strategies, args.method)(
    **kwargs
)

end = time.time()
time_elapsed = end - start

print(testacc, retainacc, zrf, mia)
wandb.log(
    {
        "TestAcc": testacc,
        "RetainTestAcc": retainacc,
        "ZRF": zrf,
        "MIA": mia,
        "Df": d_f,
        "MethodTime": time_elapsed,  # do not forget to deduct baseline time from it to remove results calc (acc, MIA, ...)
    }
)

wandb.finish()
