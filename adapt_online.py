import os
import time
import argparse
import numpy as np
import random
import torch

import models
from models import MinkUNet18_HEADS, MinkUNet18_MCMC
from utils.config import get_config
from utils.collation import CollateSeparated, CollateFN
from utils.dataset_online import get_online_dataset
from utils.online_logger import OnlineWandbLogger, OnlineCSVLogger
from utils.pseudo import PseudoLabel
from pipelines import OneDomainAdaptation, OnlineTrainer

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/deva/nuscenes_sequence.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--split_size",
                    default=4071,
                    type=int,
                    help="Num frames per sub sequence (SemanticKITTI only)")
parser.add_argument("--drop_prob",
                    default=None,
                    type=float,
                    help="Dropout prob MCMC")
parser.add_argument("--save_predictions",
                    default=False,
                    action='store_true')
parser.add_argument("--geometric_path",
                    default=None,
                    type=str)
parser.add_argument("--use_global",
                    default=False,
                    action='store_true')

AUG_DICT = None


def get_mini_config(main_c):
    return dict(time_window=main_c.dataset.max_time_window,
                mcmc_it=main_c.pipeline.num_mc_iterations,
                metric=main_c.pipeline.metric,
                cbst_p=main_c.pipeline.top_p,
                th_pseudo=main_c.pipeline.th_pseudo,
                top_class=main_c.pipeline.top_class,
                propagation_size=main_c.pipeline.propagation_size,
                drop_prob=main_c.model.drop_prob)


def train(config, split_size=250, save_preds=False):

    mapping_path = config.dataset.mapping_path

    geometric_path = args.geometric_path

    eval_dataset = get_online_dataset(dataset_name=config.dataset.name,
                                      dataset_path=config.dataset.dataset_path,
                                      voxel_size=config.dataset.voxel_size,
                                      augment_data=config.dataset.augment_data,
                                      max_time_wdw=config.dataset.max_time_window,
                                      aug_parameters=AUG_DICT,
                                      version=config.dataset.version,
                                      sub_num=config.dataset.num_pts,
                                      ignore_label=config.dataset.ignore_label,
                                      split_size=split_size,
                                      mapping_path=mapping_path,
                                      num_classes=config.model.out_classes,
                                      geometric_path=geometric_path)

    adapt_dataset = get_online_dataset(dataset_name=config.dataset.name,
                                       dataset_path=config.dataset.dataset_path,
                                       voxel_size=config.dataset.voxel_size,
                                       augment_data=config.dataset.augment_data,
                                       max_time_wdw=config.dataset.max_time_window,
                                       oracle_pts=config.dataset.oracle_pts,
                                       aug_parameters=AUG_DICT,
                                       version=config.dataset.version,
                                       sub_num=config.dataset.num_pts,
                                       ignore_label=config.dataset.ignore_label,
                                       split_size=split_size,
                                       mapping_path=mapping_path,
                                       num_classes=config.model.out_classes,
                                       noisy_odo=config.pipeline.add_odo_noise,
                                       odo_roto_bounds=config.pipeline.odo_roto_bounds,
                                       odo_tras_bounds=config.pipeline.odo_tras_bounds,
                                       geometric_path=geometric_path)

    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    if config.model.name == 'MinkUNet18':
        model = MinkUNet18_HEADS(model)

    if config.pipeline.is_double:
        source_model = Model(config.model.in_feat_size, config.model.out_classes)
        if config.pipeline.use_mcmc:
            if args.drop_prob is not None:
                config.model.drop_prob = args.drop_prob

            source_model = MinkUNet18_MCMC(source_model, p_drop=config.model.drop_prob)
    else:
        source_model = None

    if config.pipeline.delayed_freeze_list is not None:
        delayed_list = dict(zip(config.pipeline.delayed_freeze_list, config.pipeline.delayed_freeze_frames))
    else:
        delayed_list = None

    if config.pipeline.is_pseudo:
        pseudo_device = torch.device(f'cuda:0')

        pseudor = PseudoLabel(metric=config.pipeline.metric,
                              topk_pseudo=config.pipeline.topk_pseudo,
                              th_pseudo=config.pipeline.th_pseudo,
                              top_p=config.pipeline.top_p,
                              propagate=config.pipeline.propagate,
                              propagation_size=config.pipeline.propagation_size,
                              top_class=config.pipeline.top_class,
                              device=pseudo_device,
                              use_matches=config.pipeline.use_matches,
                              propagation_method=config.pipeline.propagation_method)
    else:
        pseudor = None

    module = OneDomainAdaptation(eval_dataset=eval_dataset,
                                 adapt_dataset=adapt_dataset,
                                 model=model,
                                 num_classes=config.model.out_classes,
                                 source_model=source_model,
                                 criterion=config.pipeline.loss,
                                 epsilon=config.pipeline.eps,
                                 ssl_criterion=config.pipeline.ssl_loss,
                                 ssl_beta=config.pipeline.ssl_beta,
                                 seg_beta=config.pipeline.segmentation_beta,
                                 centroids_beta=config.pipeline.centroids_beta,
                                 temperature=config.pipeline.temperature,
                                 optimizer_name=config.pipeline.optimizer.name,
                                 adaptation_batch_size=config.pipeline.dataloader.adaptation_batch_size,
                                 stream_batch_size=config.pipeline.dataloader.stream_batch_size,
                                 lr=config.pipeline.optimizer.lr,
                                 clear_cache_int=config.pipeline.trainer.clear_cache_int,
                                 scheduler_name=config.pipeline.scheduler.scheduler_name,
                                 train_num_workers=config.pipeline.dataloader.num_workers,
                                 val_num_workers=config.pipeline.dataloader.num_workers,
                                 pseudor=pseudor,
                                 use_random_wdw=config.pipeline.random_time_window,
                                 freeze_list=config.pipeline.freeze_list,
                                 delayed_freeze_list=delayed_list,
                                 num_mc_iterations=config.pipeline.num_mc_iterations,
                                 use_global=args.use_global)

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time

    mini_configs = get_mini_config(config)

    for k, v in mini_configs.items():
        run_name += f'_{str(k)}:{str(v)}'

    save_dir = os.path.join(config.pipeline.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    wandb_logger = OnlineWandbLogger(project=config.pipeline.wandb.project_name,
                                     entity=config.pipeline.wandb.entity_name,
                                     name=run_name,
                                     offline=config.pipeline.wandb.offline,
                                     config=mini_configs)

    csv_logger = OnlineCSVLogger(save_dir=save_dir,
                                 version='logs')

    loggers = [wandb_logger, csv_logger]

    try:
        is_spatiotemporal = config.pipeline.is_spatiotemporal
    except:
        is_spatiotemporal =False

    trainer = OnlineTrainer(pipeline=module,
                            collate_fn_eval=CollateFN(),
                            collate_fn_adapt=CollateSeparated(),
                            device=config.pipeline.gpu,
                            default_root_dir=config.pipeline.save_dir,
                            weights_save_path=os.path.join(save_dir, 'checkpoints'),
                            loggers=loggers,
                            save_checkpoint_every=config.pipeline.trainer.save_checkpoint_every,
                            source_checkpoint=config.pipeline.source_model,
                            student_checkpoint=config.pipeline.student_model,
                            is_double=config.pipeline.is_double,
                            is_shot=config.pipeline.is_shot,
                            is_centroids=config.pipeline.is_centroids,
                            is_pseudo=config.pipeline.is_pseudo,
                            is_proda=config.pipeline.is_proda,
                            is_spatiotemporal=is_spatiotemporal,
                            use_mcmc=config.pipeline.use_mcmc,
                            centroids_path=config.pipeline.centroids_path,
                            sub_epochs=config.pipeline.sub_epoch,
                            save_predictions=save_preds)

    trainer.adapt_double()


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    train(config, split_size=args.split_size, save_preds=args.save_predictions)
