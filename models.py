"""
From https://github.com/vikram2000b/bad-teaching-unlearning
And https://github.com/weiaicunzai/pytorch-cifar100 (better performance) <- Refer to this for comments
"""

from torch import nn
import numpy as np
import torch
from torchvision.models import resnet18
from transformers import ViTModel, ViTFeatureExtractor
from resnet import ResNet, BasicBlock


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvStandard(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        output_padding=0,
        w_sig=np.sqrt(1.0),
    ):
        super(ConvStandard, self).__init__(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.w_sig = w_sig
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.weight,
            mean=0,
            std=self.w_sig / (self.in_channels * np.prod(self.kernel_size)),
        )
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        output_padding=0,
        activation_fn=nn.ReLU,
        batch_norm=True,
        transpose=False,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            #             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            #                                 )]
            model += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=not batch_norm,
                )
            ]
        else:
            model += [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=not batch_norm,
                )
            ]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    def __init__(
        self,
        filters_percentage=1.0,
        n_channels=3,
        num_classes=10,
        dropout=False,
        batch_norm=True,
    ):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(
                n_filter1,
                n_filter2,
                kernel_size=3,
                stride=2,
                padding=1,
                batch_norm=batch_norm,
            ),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(
                n_filter2,
                n_filter2,
                kernel_size=3,
                stride=2,
                padding=1,
                batch_norm=batch_norm,
            ),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output


class ViT(nn.Module):
    def __init__(self, num_classes=20, **kwargs):
        super(ViT, self).__init__()
        self.base = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.final = nn.Linear(self.base.config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(outputs.last_hidden_state[:, 0])

        return logits
