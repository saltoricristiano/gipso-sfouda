import os
import wandb
import time
import logging
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import numpy as np

from pipelines.base_pipeline import BasePipeline
from utils.metrics import filtered_accuracy, confusion_matrix, iou_from_confusion

try:
    import pytorch_lightning as plt
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install open3d pytorch_lightning`."
    )


class OneDomainTrainer(BasePipeline):

    def __init__(self,
                 model=None,
                 training_dataset=None,
                 validation_dataset=None,
                 loss=None,
                 optimizer=None,
                 scheduler=None,
                 save_dir=None):

        # init super
        super().__init__(model=model,
                         loss=loss,
                         optimizer=optimizer,
                         scheduler=scheduler)

        # datasets
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.training_loader = None
        self.validation_loader = None

        # dirs
        self.save_dir = save_dir
        # logs
        self.use_wandb = None

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # for saving
        self.best_acc = 0.

    def single_gpu_train(self,
                         epochs=100,
                         lr=1e-3,
                         batch_size=4,
                         use_wandb=False,
                         run_name=None,
                         save_every=10,
                         collation=None):

        # init logs
        run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
        if run_name is not None:
            run_name = run_time + '_' + run_name
        else:
            run_name = run_time

        self.save_dir = os.path.join(self.save_dir, run_name)
        self.use_wandb = use_wandb

        os.makedirs(os.path.join(self.save_dir, 'weights'), exist_ok=True)

        # logging
        log_path = os.path.join(self.save_dir, 'logs')
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, 'train.log')

        logging.basicConfig(filename=log_file, level=logging.INFO, force=True)
        logging.info(f'RUN_NAME {run_name}')
        logging.info(f'Logging in this file {log_file}')

        quick_configs = {'run_name': run_name,
                             'save_dir': self.save_dir,
                             'loss': str(type(self.loss)),
                             'epochs': epochs,
                             'lr': lr,
                             'lr_decay': self.scheduler.gamma,
                             'batchsize': batch_size}

        logging.info(f'CONFIGS: {quick_configs}')

        # use or not wandb
        if use_wandb:

            wandb.init(project="cvpr2022-online-seg", entity="unitn-mhug-csalto",
                       name=run_name)


            logging.info('WANDB enabled')

        # init dataloaders
        self.training_loader = DataLoader(self.training_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collation,
                                     shuffle=True)

        self.validation_loader = DataLoader(self.validation_dataset,
                                     batch_size=batch_size,
                                     collate_fn=collation,
                                     shuffle=False)

        logging.info(f'Training started at {time.strftime("%H:%M:%S", time.gmtime())}')

        for epoch in range(epochs):

            logging.info(f'=======> Epoch {epoch}')

            start_time = time.time()

            train_loss, train_acc, train_iou = self.train()

            ep_time = time.time() - start_time
            logging.info(f'=======> Epoch {epoch} ended')
            logging.info(f'=======> Training loss {train_loss}')
            logging.info(f'=======> Training acc {train_acc}')
            logging.info(f'=======> Training IoU {train_iou}')
            logging.info(f'=======> Time {ep_time}')

            if self.use_wandb:
                wandb.log({'Training loss': train_loss,
                           'Training accuracy': train_acc,
                           'Training iou': train_iou})

            if epoch % save_every == 0:

                val_loss, val_acc, val_iou = self.validate()

                logging.info(f'******* Validation ******')
                logging.info(f'=======> Validation IoU {val_iou}')
                logging.info(f'=======> Validation acc {val_acc}')
                logging.info(f'=======> Validation loss {val_loss}')
                logging.info('**************************')

                if self.use_wandb:
                    wandb.log({'Validation accuracy': val_acc,
                               'Validation loss': val_loss,
                               'Validation IoU': val_iou})

                self.save_model(epoch, val_acc)

            self.scheduler.step()

    def train(self):
        self.model.train()
        training_losses = []
        training_labels = []
        training_preds = []
        training_confusions = []

        for t_idx, train_data in enumerate(self.training_loader):
            loss, train_pred = self.train_step(train_data)

            training_losses.append(loss.cpu().detach().numpy())
            training_labels.append(train_data["labels"].cpu())
            training_preds.append(train_pred.cpu())
            conf_m = torch.from_numpy(confusion_matrix(train_pred.cpu(), (train_data["labels"].cpu()))).unsqueeze(0)
            training_confusions.append(conf_m)

        training_loss_mean = np.mean(training_losses)
        training_preds = torch.cat(training_preds).view(-1)
        training_labels = torch.cat(training_labels).view(-1)

        training_acc = filtered_accuracy(training_preds, training_labels)
        train_iou_per_class, training_iou = iou_from_confusion(torch.cat(training_confusions))

        return training_loss_mean, training_acc, training_iou

    def train_step(self, batch):
        self.optimizer.zero_grad()
        stensor = ME.SparseTensor(coordinates=batch['coordinates'], features=batch['features'])
        out = self.model(stensor).F

        loss = self.loss(out, batch['labels'].long())
        loss.backward()
        self.optimizer.step()

        _, preds = out.max(1)

        return loss, preds

    def validate(self):
        self.model.eval()
        validation_labels = []
        validation_preds = []
        validation_loss = []
        validation_confusions = []

        with torch.no_grad():

            for v_idx, val_data in enumerate(self.validation_loader):
                val_loss, pred = self.validation_step(val_data)

                validation_labels.append(val_data["labels"].cpu())
                validation_preds.append(pred.cpu())
                conf_m = torch.from_numpy(confusion_matrix(pred.cpu(), val_data["labels"].cpu())).unsqueeze(0)
                validation_confusions.append(conf_m)

                validation_loss.append(val_loss)

        validation_preds = torch.cat(validation_preds).view(-1)
        validation_labels = torch.cat(validation_labels).view(-1)
        val_acc = filtered_accuracy(validation_preds, validation_labels)
        val_loss = np.mean(validation_loss)
        val_iou_per_class, val_iou = iou_from_confusion(torch.cat(validation_confusions))

        return val_loss, val_acc, val_iou

    def validation_step(self, batch):
        stensor = ME.SparseTensor(coordinates=batch['coordinates'], features=batch['features'])

        out = self.model(stensor).F

        loss = self.loss(out, batch['labels'].long())
        _, preds = out.max(1)
        # torch.cuda.empty_cache()

        return loss, preds

    def save_model(self, epoch, acc):

        torch.save({"state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()},
                    os.path.join(self.save_dir, 'weights', f'checkpoint_{epoch}.pth'))

        if acc > self.best_acc:

            torch.save({"state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict()},
                        os.path.join(self.save_dir, 'weights', f'best.pth'))

            self.best_acc = acc
