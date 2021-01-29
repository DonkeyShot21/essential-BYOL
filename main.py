import torch
import pytorch_lightning as pl


from byol.model import BYOL
import data.modules as data_modules

from argparse import ArgumentParser
from datetime import datetime


parser = ArgumentParser()
parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset')
parser.add_argument('--download', default=False, action='store_true', help='wether to download the dataset')
parser.add_argument('--data_dir', default='datasets', type=str, help='data directory')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_workers', default=5, type=int, help='number of workers')
parser.add_argument('--arch', default='resnet18', type=str, help='backbone architecture')
parser.add_argument('--base_lr', default=1.0, type=float, help='base learning rate')
parser.add_argument('--min_lr', default=1e-3, type=float, help='min learning rate')
parser.add_argument('--linear_loss_weight', default=0.03, type=float, help='weight for the linear loss')
parser.add_argument('--momentum_opt', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--weight_decay', default=1.0e-6, type=float, help='weight decay')
parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
parser.add_argument('--proj_dim', default=256, type=int, help='projected dim')
parser.add_argument('--hidden_dim', default=4096, type=int, help='hidden dim in proj/pred head')
parser.add_argument('--base_momentum', default=0.99, type=float, help='base momentum for byol')
parser.add_argument('--final_momentum', default=1.0, type=float, help='final momentum for byol')
parser.add_argument('--comment', default=datetime.now().strftime('%b%d_%H-%M-%S'), type=str, help='wandb comment')
parser.add_argument('--project', default='essential-byol', type=str, help='wandb project')
parser.add_argument('--entity', default=None, type=str, help='wandb entity')
parser.add_argument('--offline', default=False, action='store_true', help='disable wandb')


def main(args):

    dm_class = getattr(data_modules, args.dataset + 'DataModule')
    dm = dm_class(**args.__dict__)

    run_name = '-'.join(['byol', args.arch, args.dataset, args.comment])
    wandb_logger = pl.loggers.wandb.WandbLogger(
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline)

    model = BYOL(**args.__dict__, num_classes=dm.num_classes)

    trainer = pl.Trainer.from_argparse_args(args, 
        logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
