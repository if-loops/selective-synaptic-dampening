import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import datasets
import models
import conf
from training_utils import *

# Original code from https://github.com/weiaicunzai/pytorch-cifar100 <- refer to this repo for comments


def train(epochs):
    start = time.time()
    net.train()
    for batch_index, (images, _, labels) in enumerate(trainloader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        #     loss.item(),
        #     optimizer.param_groups[0]['lr'],
        #     epoch=epoch,
        #     trained_samples=batch_index * args.b + len(images),
        #     total_samples=len(trainloader.dataset)
        # ))

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, _, labels in testloader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print("GPU INFO.....")
        print(torch.cuda.memory_summary(), end="")
    print("Evaluating Network.....")
    print(
        "Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            epoch,
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
            finish - start,
        )
    )
    print()

    return correct.float() / len(testloader.dataset)


parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, required=True, help="net type")
parser.add_argument("-dataset", type=str, required=True, help="dataset to train on")
parser.add_argument("-classes", type=int, required=True, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=False, help="use gpu or not")
parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
args = parser.parse_args()


MILESTONES = (
    getattr(conf, f"{args.dataset}_MILESTONES")
    if args.net != "ViT"
    else getattr(conf, f"{args.dataset}_ViT_MILESTONES")
)
EPOCHS = (
    getattr(conf, f"{args.dataset}_EPOCHS")
    if args.net != "ViT"
    else getattr(conf, f"{args.dataset}_ViT_EPOCHS")
)
# get network
net = getattr(models, args.net)(num_classes=args.classes)
if args.gpu:
    net = net.cuda()

# dataloaders
root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
img_size = 224 if args.net == "ViT" else 32

trainset = getattr(datasets, args.dataset)(
    root=root, download=True, train=True, unlearning=False, img_size=img_size
)
testset = getattr(datasets, args.dataset)(
    root=root, download=True, train=False, unlearning=False, img_size=img_size
)

trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
testloader = DataLoader(testset, batch_size=args.b, shuffle=False)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=MILESTONES, gamma=0.2
)  # learning rate decay
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

checkpoint_path = os.path.join(conf.CHECKPOINT_PATH, args.net, conf.TIME_NOW)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, "{net}-{dataset}-{epoch}-{type}.pth")

best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    if epoch > args.warm:
        train_scheduler.step(epoch)

    train(epoch)
    acc = eval_training(epoch)

    # start to save best performance model after learning rate decay to 0.01
    if best_acc < acc:  # and epoch > MILESTONES[1]
        weights_path = checkpoint_path.format(
            net=args.net, dataset=args.dataset, epoch=epoch, type="best"
        )
        print("saving weights file to {}".format(weights_path))
        torch.save(net.state_dict(), weights_path)
        best_acc = acc
        continue

    # if not epoch % conf.SAVE_EPOCH:
    #     weights_path = checkpoint_path.format(net=args.net, dataset=args.dataset, epoch=epoch, type='regular')
    #     print('saving weights file to {}'.format(weights_path))
    #     torch.save(net.state_dict(), weights_path)
