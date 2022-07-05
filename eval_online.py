import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

import models
from models import MinkUNet18_HEADS
from utils.config import get_config
from utils.collation import CollateSeparated, CollateFN, CollateStream
from utils.dataset_online import get_online_dataset
from utils.online_logger import OnlineWandbLogger, OnlineCSVLogger
from pipelines import OneDomainAdaptation, OnlineTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/",
                    type=str,
                    help="Path to config file")
parser.add_argument("--split_size",
                    default=4071,
                    type=int,
                    help="Num frames per sub sequence (SemanticKITTI only)")

# AUG_DICT = {'RandomDropout': [0.2, 0.5]}
AUG_DICT = None


def train(config):

    eval_dataset = get_online_dataset(dataset_name=config.dataset.name,
                                      dataset_path=config.dataset.dataset_path,
                                      voxel_size=config.dataset.voxel_size,
                                      augment_data=config.dataset.augment_data,
                                      max_time_wdw=config.dataset.max_time_window,
                                      aug_parameters=AUG_DICT,
                                      version=config.dataset.version,
                                      sub_num=config.dataset.num_pts,
                                      ignore_label=config.dataset.ignore_label,
                                      split_size=args.split_size)

    adapt_dataset = get_online_dataset(dataset_name=config.dataset.name,
                                      dataset_path=config.dataset.dataset_path,
                                      voxel_size=config.dataset.voxel_size,
                                      augment_data=config.dataset.augment_data,
                                      max_time_wdw=config.dataset.max_time_window,
                                      aug_parameters=AUG_DICT,
                                      version=config.dataset.version,
                                      sub_num=config.dataset.num_pts,
                                      ignore_label=config.dataset.ignore_label,
                                       split_size=args.split_size)

    # coords = [N, [x, y, z]], feats=[N, f] -> f [i] ----- [x, y, z, i]
    # model = MinkUNet34C(1, 8)
    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    if config.model.name == 'MinkUNet18':
        model = MinkUNet18_HEADS(model)

    module = OneDomainAdaptation(eval_dataset=eval_dataset,
                                 adapt_dataset=adapt_dataset,
                                 model=model,
                                 criterion=config.pipeline.loss,
                                 ssl_criterion=config.pipeline.ssl_loss,
                                 ssl_beta=config.pipeline.ssl_beta,
                                 seg_beta=config.pipeline.segmentation_beta,
                                 optimizer_name=config.pipeline.optimizer.name,
                                 adaptation_batch_size=config.pipeline.dataloader.adaptation_batch_size,
                                 stream_batch_size=config.pipeline.dataloader.stream_batch_size,
                                 lr=config.pipeline.optimizer.lr,
                                 clear_cache_int=config.pipeline.trainer.clear_cache_int,
                                 scheduler_name=config.pipeline.scheduler.scheduler_name,
                                 train_num_workers=config.pipeline.dataloader.num_workers,
                                 val_num_workers=config.pipeline.dataloader.num_workers,
                                 use_random_wdw=config.pipeline.random_time_window,
                                 freeze_list=config.pipeline.freeze_list)

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time

    save_dir = os.path.join(config.pipeline.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    wandb_logger = OnlineWandbLogger(project=config.pipeline.wandb.project_name,
                                     entity=config.pipeline.wandb.entity_name,
                                     name=run_name,
                                     offline=config.pipeline.wandb.offline)

    csv_logger = OnlineCSVLogger(save_dir=save_dir,
                                 version='logs')

    loggers = [wandb_logger, csv_logger]

    trainer = OnlineTrainer(pipeline=module,
                            collate_fn_eval=CollateFN(),
                            collate_fn_adapt=CollateSeparated(),
                            device=config.pipeline.gpu,
                            default_root_dir=config.pipeline.save_dir,
                            weights_save_path=os.path.join(save_dir, 'checkpoints'),
                            loggers=loggers,
                            save_checkpoint_every=config.pipeline.trainer.save_checkpoint_every,
                            source_checkpoint=config.pipeline.source_model,
                            save_predictions=False)

    trainer.eval(is_adapt=False)


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
