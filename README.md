# Essential BYOL
A simple and complete implementation of [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733) in PyTorch + [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

<img src="https://pytorch.org/assets/images/pytorch-logo.png" height="100"> <img src="https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_images/logos/lightning_logo-name.png" height="100">


Good stuff:
* good performance (~67% linear eval accuracy on CIFAR100)
* minimal code, easy to use and extend
* multi-GPU / TPU and AMP support provided by PyTorch Lightning
* ImageNet support (needs testing)
* linear evaluation is performed during training without any additional forward pass
* logging with Wandb

# Environment
```
conda create --name essential-byol python=3.8
conda activate essential-byol
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=XX.X -c pytorch
pip install pytorch-lightning==1.1.6 pytorch-lightning-bolts==0.3 wandb opencv-python
```
The code has been tested using these versions of the packages, but it will probably work with slightly different environments as well. When your run the code (see below for commands), PyTorch Lightning will probably throw a warning, advising you to install additional packages as `gym`, `sklearn` and `matplotlib`. They are not needed for this implementation to work, but you can install them to get rid of the warnings.

# Datasets
Three datasets are supported:
* CIFAR10
* CIFAR100
* ImageNet

For imagenet you need to pass the appropriate `--data_dir`, while for CIFAR you can just pass `--download` to download the dataset.

# Commands
The repo comes with minimal model specific arguments, check `main.py` for info. We also support all the arguments of the [PyTorch Lightning trainer](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html). Default parameters are optimized for CIFAR100 but can also be used for CIFAR10.

Sample commands for running CIFAR100 on a single GPU setup:
```
python main.py \
    --gpus 1 \
    --dataset CIFAR100 \
    --batch_size 256 \
    --max_epochs 1000 \
    --arch resnet18 \
    --precision 16 \
    --comment wandb-comment
```
and multi-GPU setup:
```
python main.py \
    --gpus 2 \
    --distributed_backend ddp \
    --sync_batchnorm \
    --dataset CIFAR100 \
    --batch_size 256 \
    --max_epochs 1000 \
    --arch resnet18 \
    --precision 16 \
    --comment wandb-comment
```

# Logging
Logging is performed with [Wandb](https://wandb.ai/site), please create an account, and follow the configuration steps in the terminal. You can pass your username using `--entity`. Training and validation stats are logged at every epoch. If you want to completely disable logging use `--offline`.

# Contribute
Help is appreciated. Stuff that needs work:
- [ ] test ImageNet performance
- [ ] exclude bias and bn from LARS adaptation (see comments in the code)
