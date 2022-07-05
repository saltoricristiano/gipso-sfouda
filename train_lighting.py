import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.logger import CSVLogger
import MinkowskiEngine as ME

import models
from utils.dataset import get_dataset
from utils.config import get_config
from utils.collation import CollateFN
from utils.callbacks import SourceCheckpoint
from pipelines import PLTOneDomainTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synth4dkitti_source.yaml",
                    type=str,
                    help="Path to config file")

# AUG_DICT = {'RandomDropout': [0.2, 0.5]}
AUG_DICT = None


def train(config):

    def get_dataloader(dataset, shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=config.pipeline.dataloader.batch_size,
                          collate_fn=CollateFN(),
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
    try:
        mapping_path = config.dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None

    training_dataset, validation_dataset, target_dataset = get_dataset(dataset_name=config.dataset.name,
                                                                       dataset_path=config.dataset.dataset_path,
                                                                       voxel_size=config.dataset.voxel_size,
                                                                       augment_data=config.dataset.augment_data,
                                                                       aug_parameters=AUG_DICT,
                                                                       version=config.dataset.version,
                                                                       sub_num=config.dataset.num_pts,
                                                                       get_target=config.dataset.validate_target,
                                                                       target_dataset_path=config.dataset.target_path,
                                                                       num_classes=config.model.out_classes,
                                                                       ignore_label=config.dataset.ignore_label,
                                                                       mapping_path=mapping_path)

    training_dataloader = get_dataloader(training_dataset, shuffle=True)
    validation_dataloader = get_dataloader(validation_dataset, shuffle=False)

    if target_dataset is not None:
        target_dataloader = get_dataloader(target_dataset, shuffle=False)
        validation_dataloader = [validation_dataloader, target_dataloader]
    else:
        validation_dataloader = [validation_dataloader]

    # coords = [N, [x, y, z]], feats=[N, f] -> f [i] ----- [x, y, z, i]
    # model = MinkUNet34C(1, 8)
    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    pl_module = PLTOneDomainTrainer(training_dataset=training_dataset,
                                    validation_dataset=validation_dataset,
                                    model=model,
                                    criterion=config.pipeline.loss,
                                    optimizer_name=config.pipeline.optimizer.name,
                                    batch_size=config.pipeline.dataloader.batch_size,
                                    val_batch_size=config.pipeline.dataloader.batch_size,
                                    lr=config.pipeline.optimizer.lr,
                                    num_classes=config.model.out_classes,
                                    train_num_workers=config.pipeline.dataloader.num_workers,
                                    val_num_workers=config.pipeline.dataloader.num_workers,
                                    clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                    scheduler_name=config.pipeline.scheduler.scheduler_name)

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time

    save_dir = os.path.join(config.pipeline.save_dir, run_name)

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=run_name,
                               offline=config.pipeline.wandb.offline)
    csv_logger = CSVLogger(save_dir=save_dir,
                           name=run_name,
                           version='logs')

    loggers = [wandb_logger, csv_logger]

    checkpoint_callback = [ModelCheckpoint(dirpath=os.path.join(save_dir, 'checkpoints'), save_top_k=-1),
                           SourceCheckpoint()]

    trainer = Trainer(max_epochs=config.pipeline.epochs,
                      gpus=config.pipeline.gpus,
                      accelerator="ddp",
                      default_root_dir=config.pipeline.save_dir,
                      weights_save_path=save_dir,
                      precision=config.pipeline.precision,
                      logger=loggers,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      val_check_interval=1.0,
                      num_sanity_val_steps=0,
                      resume_from_checkpoint=config.pipeline.lightning.resume_checkpoint,
                      callbacks=checkpoint_callback)

    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

    train(config)
