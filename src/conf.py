""" 
configurations for this project
"""

# Imports here
import os
from datetime import datetime

CHECKPOINT_PATH = "checkpoint"

# Class correspondence as done in https://github.com/vikram2000b/bad-teaching-unlearning
class_dict = {
    "rocket": 69,
    "vehicle2": 19,
    "veg": 4,
    "mushroom": 51,
    "people": 14,
    "baby": 2,
    "electrical_devices": 5,
    "lamp": 40,
    "natural_scenes": 10,
    "sea": 71,
    "42": 42,
    "1": 1,
    "10": 10,
    "20": 20,
    "30": 30,
    "40": 40,
}

# Classes from https://github.com/vikram2000b/bad-teaching-unlearning
cifar20_classes = {"vehicle2", "veg", "people", "electrical_devices", "natural_scenes"}

# Classes from https://github.com/vikram2000b/bad-teaching-unlearning
cifar100_classes = {"rocket", "mushroom", "baby", "lamp", "sea"}

# total training epochs

# Training parameters for the tasks; milestones are when the learning rate gets lowered
PinsFaceRecognition_EPOCHS = 200
PinsFaceRecognition_MILESTONES = [60, 120, 160]

Cifar100_EPOCHS = 200
Cifar100_MILESTONES = [60, 120, 160]

Cifar10_EPOCHS = 20
Cifar10_MILESTONES = [8, 12, 16]

Cifar20_EPOCHS = 40
Cifar20_MILESTONES = [15, 30, 35]

Cifar100_EPOCHS = 200
Cifar100_MILESTONES = [60, 120, 160]


Cifar10_ViT_EPOCHS = 8
Cifar10_ViT_MILESTONES = [7]

Cifar20_ViT_EPOCHS = 9
Cifar20_ViT_MILESTONES = [8]

Cifar100_ViT_EPOCHS = 8
Cifar100_ViT_MILESTONES = [7]

DATE_FORMAT = "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
# time of script run
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# log dir
LOG_DIR = "runs"

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
