# Selective Synaptic Dampening

This is the code for the paper **Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening** (https://arxiv.org/abs/2308.07707).

## Usage

All experiments can be run via

```
./MAIN_run_experiments.sh 0 # to run experiments on GPU 0 (nvidia-smi)
```

## Setup

You will need to train ResNet18's and Vision Transformers. Use pretrain_model.py for this and then copy the paths of the models into the respecive .sh files.

```
# fill in _ with your desired parameters as described in pretrain_model.py
python pretrain_model.py -net _ -dataset _ -classes _ -gpu _
```

We used https://hub.docker.com/layers/tensorflow/tensorflow/latest-gpu-py3-jupyter/images/sha256-901b827b19d14aa0dd79ebbd45f410ee9dbfa209f6a4db71041b5b8ae144fea5 as our base image and installed relevant packages on top.

```
datetime
wandb
sklearn
torch
copy
tqdm
transformers
matplotlib
scipy
```

You will need a wandb.ai account to use the implemented logging. Feel free to replace with any other logger of your choice.

## Modifying SSD

SSD functions are in ssd.py. To change alpha and lambda, set them in the respective forget_..._main.py file per unlearning task.

## Citing this work

```
@misc{foster2023fast,
      title={Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening}, 
      author={Jack Foster and Stefan Schoepf and Alexandra Brintrup},
      year={2023},
      eprint={2308.07707},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```