import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

from byol.nets import Encoder, MLP

import math


class BYOL(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.current_momentum = self.hparams.base_momentum

        # online encoder
        self.online_encoder = Encoder(
            arch=self.hparams.arch,
            hidden_dim=self.hparams.hidden_dim,
            proj_dim=self.hparams.proj_dim,
            low_res='CIFAR' in self.hparams.dataset)

        # momentum encoder
        self.momentum_encoder = Encoder(
            arch=self.hparams.arch,
            hidden_dim=self.hparams.hidden_dim,
            proj_dim=self.hparams.proj_dim,
            low_res='CIFAR' in self.hparams.dataset)
        self.initialize_momentum_encoder()

        # predictor
        self.predictor = MLP(
            input_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=self.hparams.proj_dim)

        # linear layer for eval
        self.linear = torch.nn.Linear(
            self.online_encoder.feat_dim, self.hparams.num_classes)

    @torch.no_grad()
    def initialize_momentum_encoder(self):
        params_online = self.online_encoder.parameters()
        params_momentum = self.momentum_encoder.parameters()
        for po, pm in zip(params_online, params_momentum):
            pm.data.copy_(po.data)
            pm.requires_grad = False

    def collect_params(self, models, exclude_bias_and_bn=True):
        param_list = []
        for model in models:
            for name, param in model.named_parameters():
                if exclude_bias_and_bn and any(
                    s in name for s in ['bn', 'downsample.1', 'bias']):
                    param_dict = {
                        'params': param,
                        'weight_decay': 0.,
                        'lars_exclude': True}
                    # NOTE: with the current pytorch lightning bolts
                    # implementation it is not possible to exclude 
                    # parameters from the LARS adaptation
                else:
                    param_dict = {'params': param}
                param_list.append(param_dict)
        return param_list

    def configure_optimizers(self):
        params = self.collect_params([
            self.online_encoder, self.predictor, self.linear])
        optimizer = LARSWrapper(torch.optim.SGD(
            params,
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay))
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.linear(self.online_encoder.encoder(x))

    def cosine_similarity_loss(self, preds, targets):
        preds = F.normalize(preds, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)
        return 2 - 2 * (preds * targets).sum(dim=-1).mean()

    def training_step(self, batch, batch_idx):
        views, labels = batch

        # forward online encoder
        input_online = torch.cat(views, dim=0)
        z, feats = self.online_encoder(input_online)
        preds = self.predictor(z)

        # forward momentum encoder
        with torch.no_grad():
            input_momentum = torch.cat(views[::-1], dim=0)
            targets, _ = self.momentum_encoder(input_momentum)
        
        # compute BYOL loss
        loss = self.cosine_similarity_loss(preds, targets)

        # train linear layer
        preds_linear = self.linear(feats.detach())
        loss_linear = F.cross_entropy(preds_linear, labels.repeat(2))

        # gather results and log stats
        logs = {
            'loss': loss,
            'loss_linear': loss_linear,
            'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'momentum': self.current_momentum}
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
        return loss + loss_linear * self.hparams.linear_loss_weight

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # update momentum encoder
        self.momentum_update(
            self.online_encoder, self.momentum_encoder, self.current_momentum)
        # update momentum value
        max_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        self.current_momentum = self.hparams.final_momentum - \
            (self.hparams.final_momentum - self.hparams.base_momentum) * \
            (math.cos(math.pi * self.trainer.global_step / max_steps) + 1) / 2

    @torch.no_grad()
    def momentum_update(self, online_encoder, momentum_encoder, m):
        online_params = online_encoder.parameters()
        momentum_params = momentum_encoder.parameters()
        for po, pm in zip(online_params, momentum_params):
            pm.data.mul_(m).add_(po.data, alpha=1. - m)

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # predict using online encoder
        preds = self(images)

        # calculate accuracy @k
        acc1, acc5 = self.accuracy(preds, labels)

        # gather results and log
        logs = {'val/acc@1': acc1, 'val/acc@5':acc5}
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def accuracy(self, preds, targets, k=(1,5)):
        preds = preds.topk(max(k), 1, True, True)[1].t()
        correct = preds.eq(targets.view(1, -1).expand_as(preds))

        res = []
        for k_i in k:
            correct_k = correct[:k_i].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / targets.size(0)))
        return res