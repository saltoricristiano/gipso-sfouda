import os
import numpy as np
import torch
import MinkowskiEngine as ME
from utils.losses import CELoss, SoftCELoss, DICELoss, SoftDICELoss
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score


class PLTOneDomainTrainer(pl.core.LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 training_dataset,
                 validation_dataset,
                 optimizer_name='SGD',
                 criterion='CELoss',
                 lr=1e-3,
                 batch_size=12,
                 weight_decay=1e-5,
                 momentum=0.9,
                 val_batch_size=6,
                 train_num_workers=10,
                 val_num_workers=10,
                 num_classes=7,
                 clear_cache_int=2,
                 scheduler_name=None):

        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        if criterion == 'CELoss':
            self.criterion = CELoss(ignore_label=self.training_dataset.ignore_label,
                                    weight=None)

        elif criterion == 'WCELoss':
            self.criterion = CELoss(ignore_label=self.training_dataset.ignore_label,
                                    weight=self.training_dataset.weights)

        elif criterion == 'SoftCELoss':
            self.criterion = SoftCELoss(ignore_label=self.training_dataset.ignore_label)

        elif criterion == 'DICELoss':
            self.criterion = DICELoss(ignore_label=self.training_dataset.ignore_label)
        elif criterion == 'SoftDICELoss':
            self.criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        else:
            raise NotImplementedError

        self.ignore_label = self.training_dataset.ignore_label

        self.save_hyperparameters(ignore='model')

    def training_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(stensor).F
        labels = batch['labels'].long()

        loss, per_class_loss = self.criterion(out, labels, return_class=True)

        _, preds = out.max(1)

        iou_tmp = jaccard_score(preds.detach().cpu().numpy(), labels.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=-0.1)

        present_labels, class_occurs = np.unique(labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join('training', p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        present_names = [os.path.join('training', p + '_loss') for p in present_names]
        results_dict.update(dict(zip(present_names, per_class_loss.tolist())))

        results_dict['training/loss'] = loss
        results_dict['training/iou'] = np.mean(iou_tmp[present_labels])
        results_dict['training/lr'] = self.trainer.optimizers[0].param_groups[0]["lr"]
        results_dict['training/epoch'] = self.current_epoch

        for k, v in results_dict.items():
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True
            )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        phase = ['validation', 'target']
        phase = phase[dataloader_idx]
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(stensor).F
        labels = batch['labels'].long()

        loss = self.criterion(out, labels)
        _, preds = out.max(1)

        iou_tmp = jaccard_score(preds.detach().cpu().numpy(), labels.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        present_labels, class_occurs = np.unique(labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict[f'{phase}/loss'] = loss
        results_dict[f'{phase}/iou'] = np.mean(iou_tmp[present_labels])

        for k, v in results_dict.items():
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False
            )
        return results_dict

    def configure_optimizers(self):
        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            if self.scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            elif self.scheduler_name == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            elif self.scheduler_name == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10000, max_lr=self.lr,
                                                              step_size_up=5, mode="triangular2")

            else:
                raise NotImplementedError

            return [optimizer], [scheduler]

