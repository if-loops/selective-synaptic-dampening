# Selective Synaptic Dampening (AAAI + ICLR TP code)

![GitHub last commit (branch)](https://img.shields.io/github/last-commit/if-loops/selective-synaptic-dampening/main) ![GitHub Repo stars](https://img.shields.io/github/stars/if-loops/selective-synaptic-dampening) ![GitHub repo size](https://img.shields.io/github/repo-size/if-loops/selective-synaptic-dampening)




![SSD_heading](https://github.com/if-loops/selective-synaptic-dampening/assets/47212405/2abb0ef1-8646-479e-a00e-613960d27f9c)

This is the code for the paper **Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening** (https://arxiv.org/abs/2308.07707), accepted at The 38th Annual **AAAI Conference on Artificial Intelligence** (Main Track).

## Related research

| Paper  | Code | Venue/Status |
| ------------- | ------------- |  ------------- |
| [Potion: Towards Poison Unlearning](https://arxiv.org/abs/2406.09173) | [GitHub](https://github.com/if-loops/towards_poison_unlearning) |  Journal of Data-Centric Machine Learning Research (DMLR)  |
| [Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization](https://browse.arxiv.org/abs/2402.01401)  | [GitHub](https://github.com/jwf40/Zeroshot-Unlearning-At-Scale) |  Preprint  |
| [Parameter-Tuning-Free Data Entry Error Unlearning with Adaptive Selective Synaptic Dampening](https://arxiv.org/abs/2402.10098)  | [GitHub](https://github.com/if-loops/adaptive-selective-synaptic-dampening) |  Preprint  |
| [ Loss-Free Machine Unlearning](https://arxiv.org/abs/2402.19308) (i.e. Label-Free) -> LFSSD | see below |  ICLR 2024 Tiny Paper  |

### Implementing LFSSD:
Replace the following in the compute_importances function(s):

```
# Vanilla SSD:
criterion = nn.CrossEntropyLoss()
loss = criterion(out, y)
...
imp.data += p.grad.data.clone().pow(2)

# LFSSD:
loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
...
imp.data += p.grad.data.clone().abs()
```

## Usage

All experiments can be run via

```
./MAIN_run_experiments.sh 0 # to run experiments on GPU 0 (nvidia-smi)
```
You might encounter issues with executing this file due to different line endings with Windows and Unix. Use dos2unix "filename" to fix.

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
@article{Foster_Schoepf_Brintrup_2024,
      title={Fast Machine Unlearning without Retraining through Selective Synaptic Dampening},
      volume={38},
      url={https://ojs.aaai.org/index.php/AAAI/article/view/29092},
      DOI={10.1609/aaai.v38i11.29092},
      number={11},
      journal={Proceedings of the AAAI Conference on Artificial Intelligence},
      author={Foster, Jack and Schoepf, Stefan and Brintrup, Alexandra},
      year={2024},
      month={Mar.},
      pages={12043-12051} }
```

## Authors

For our newest research, feel free to follow our socials:

Jack Foster: [LinkedIn](https://www.linkedin.com/in/jackfoster-ml/), [Twitter](https://twitter.com/JackFosterML)  

Stefan Schoepf: [LinkedIn](https://www.linkedin.com/in/schoepfstefan/), [Twitter](https://twitter.com/S__Schoepf)  

Alexandra Brintrup: [LinkedIn](https://www.linkedin.com/in/alexandra-brintrup-1684171/)  

Supply Chain AI Lab: [LinkedIn](https://www.linkedin.com/company/supply-chain-ai-lab/)  
