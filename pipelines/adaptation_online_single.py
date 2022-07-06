import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import csv
import pickle
import open3d as o3d
from knn_cuda import KNN
from sklearn.metrics import davies_bouldin_score

from utils.losses import CELoss, SoftCELoss, DICELoss, SoftDICELoss, HLoss, SCELoss
from utils.collation import CollateSeparated, CollateStream
from utils.sampler import SequentialSampler
from utils.dataset_online import PairedOnlineDataset, FrameOnlineDataset
from models import MinkUNet18_HEADS, MinkUNet18_SSL, MinkUNet18_MCMC


class OneDomainAdaptation(object):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 eval_dataset,
                 adapt_dataset,
                 source_model=None,
                 optimizer_name='SGD',
                 criterion='CELoss',
                 epsilon=0.,
                 ssl_criterion='Cosine',
                 ssl_beta=0.5,
                 seg_beta=1.0,
                 temperature=0.5,
                 lr=1e-3,
                 stream_batch_size=1,
                 adaptation_batch_size=2,
                 weight_decay=1e-5,
                 momentum=0.8,
                 val_batch_size=6,
                 train_num_workers=10,
                 val_num_workers=10,
                 num_classes=7,
                 clear_cache_int=2,
                 scheduler_name='ExponentialLR',
                 pseudor=None,
                 use_random_wdw=False,
                 freeze_list=None,
                 delayed_freeze_list=None,
                 num_mc_iterations=10,
                 use_global=False):

        super().__init__()

        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)

        if self.use_global:
            print('--> USING GLOBAL FEATS IN CONTRASTIVE!')

        if criterion == 'CELoss':
            self.criterion = CELoss(ignore_label=self.adapt_dataset.ignore_label,
                                    weight=None)

        elif criterion == 'WCELoss':
            self.criterion = CELoss(ignore_label=self.adapt_dataset.ignore_label,
                                    weight=self.adapt_dataset.weights)

        elif criterion == 'SoftCELoss':
            self.criterion = SoftCELoss(ignore_label=self.adapt_dataset.ignore_label)

        elif criterion == 'DICELoss':
            self.criterion = DICELoss(ignore_label=self.adapt_dataset.ignore_label)
        elif criterion == 'SoftDICELoss':
            self.criterion = SoftDICELoss(ignore_label=self.adapt_dataset.ignore_label,
                                          neg_range=True, eps=self.epsilon)

        elif criterion == 'SCELoss':
            self.criterion = SCELoss(alpha=1, beta=0.1, num_classes=self.num_classes, ignore_label=self.adapt_dataset.ignore_label)
        else:
            raise NotImplementedError

        if self.ssl_criterion == 'CosineSimilarity':
            self.ssl_criterion = nn.CosineSimilarity(dim=-1)
        else:
            raise NotImplementedError

        self.ignore_label = self.eval_dataset.ignore_label

        self.configure_optimizers()

        self.global_step = 0

        self.device = None
        self.max_time_wdw = self.eval_dataset.max_time_wdw

        self.delayed_freeze_list = delayed_freeze_list

        self.entropy = HLoss()

        self.pseudor = pseudor

        self.topk_matches = 0

        self.dataset_name = self.adapt_dataset.name

        self.knn_search = KNN(k=200, transpose_mode=True)

        self.symmetric_ce = SCELoss(alpha=1, beta=0.1, num_classes=7)

    def freeze(self):
        # here we freeze parts that have to be frozen forever
        if self.freeze_list is not None:
            for name, p in self.model.named_parameters():
                for pf in self.freeze_list:
                    if pf in name:
                        p.requires_grad = False

    def delayed_freeze(self, frame):
        # here we freeze parts that have to be frozen only for a certain period
        if self.delayed_freeze_list is not None:
            for name, p in self.model.named_parameters():
                for pf, frame_act in self.delayed_freeze_list.items():
                    if pf in name and frame <= frame_act:
                        p.requires_grad = False

    def adaptation_double_pseudo_step(self, batch, frame=0):

        self.model.train()
        self.freeze()
        self.source_model.eval()

        coords = batch["coordinates_all"][0]

        batch_all = torch.zeros([coords.shape[0], 1])
        coords_all = torch.cat([batch_all, coords], dim=-1)
        feats_all = torch.ones([coords_all.shape[0], 1]).float()

        # we assume that data the loader gives frames in pairs
        stensor_all = ME.SparseTensor(coordinates=coords_all.int().to(self.device),
                                     features=feats_all.to(self.device),
                                     quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        # t0 = time.perf_counter()
        with torch.no_grad():

            if self.pseudor.metric in ['entropy', 'confidence']:
                out_source, source_feats, _ = self.source_model(stensor_all, is_train=False)
                out_source = out_source.cpu()
            elif self.pseudor.metric in ['mcmc', 'mcmc_cbst', 'mcmc_cbst_easy2hard']:
                self.source_model.eval()
                _, source_feats, _ = self.source_model(stensor_all, is_train=False)
                self.source_model.dropout.train()
                out_source = []
                for i in range(self.num_mc_iterations):
                    out_tmp, _, _ = self.source_model(stensor_all, is_train=False)
                    out_tmp = F.softmax(out_tmp.cpu(), dim=-1)
                    out_source.append(out_tmp.view([out_tmp.shape[0], 1, -1]))
                out_source = torch.cat(out_source, dim=1)
            else:
                raise NotImplementedError
        # t1 = time.perf_counter()
        # tot_time = t1 - t0
        batch['model_features0'] = source_feats.cpu()
        pseudo0, _ = self.pseudor.get_pseudo(out_source, batch, frame, return_metric=True)

        # print(f'--> TIME: {tot_time:0.4f}')
        mean_metric = out_source.std(dim=1).mean(dim=-1).mean()

        if (pseudo0 != -1).sum() > 0:
            # we assume that data the loader gives frames in pairs
            stensor0 = ME.SparseTensor(coordinates=batch["coordinates0"].int().to(self.device),
                                       features=batch["features0"].to(self.device),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            stensor1 = ME.SparseTensor(coordinates=batch["coordinates1"].int().to(self.device),
                                       features=batch["features1"].to(self.device),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            # Must clear cache at regular interval
            if self.global_step % self.clear_cache_int == 0:
                torch.cuda.empty_cache()

            self.optimizer.zero_grad()

            # forward in mink
            out_seg0, out_en0, out_pred0, out_bck0, _, out_seg1, out_en1, out_pred1, out_bck1, _ = self.model((stensor0, stensor1))

            # segmentation loss for t0
            labels0 = batch['labels0'].long()

            loss_seg_head = self.criterion(out_seg0, pseudo0)

            pseudo0 = pseudo0.cpu()
            labels0 = labels0.cpu()

            # get matches in t0 and t1 (used for selection)
            matches0 = batch['matches0'].to(self.device)
            matches1 = batch['matches1'].to(self.device)
            if not self.use_global:
                # 2FUTURE CONTRASTIVE
                # forward preds (t0 -> t1)
                future_preds = torch.index_select(out_pred0, 0, matches0)
                # forward gt feats and stop grad
                future_gt = torch.index_select(out_en1.detach(), 0, matches1)
                future_neg_cos_sim = -self.ssl_criterion(future_preds, future_gt).cpu()

                if self.topk_matches > 0:
                    # select top-k worst performing matches
                    future_neg_cos_sim = future_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
                else:
                    future_neg_cos_sim = future_neg_cos_sim.mean(dim=0)

                # 2PAST CONTRASTIVE
                # backward preds (t1 -> t0)
                past_preds = torch.index_select(out_pred1, 0, matches1)
                # backward gt feats and stop grad
                past_gt = torch.index_select(out_en0.detach(), 0, matches0)
                past_neg_cos_sim = -self.ssl_criterion(past_preds, past_gt).cpu()
                if self.topk_matches > 0:
                    # select top-k worst performing matches
                    past_neg_cos_sim = past_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
                else:
                    past_neg_cos_sim = past_neg_cos_sim.mean(dim=0)
            else:
                # 2FUTURE CONTRASTIVE
                # forward preds (t0 -> t1)
                future_preds = out_pred0.mean(dim=0)
                # forward gt feats and stop grad
                future_gt = out_en1.detach().mean(dim=0)
                future_neg_cos_sim = -self.ssl_criterion(future_preds, future_gt).cpu()

                # 2PAST CONTRASTIVE
                # backward preds (t1 -> t0)
                past_preds = out_pred1.mean(dim=0)
                # backward gt feats and stop grad
                past_gt = out_en0.detach().mean(dim=0)
                past_neg_cos_sim = -self.ssl_criterion(past_preds, past_gt).cpu()

            # sum up to total
            ssl_loss = (future_neg_cos_sim + past_neg_cos_sim) * self.ssl_beta
            total_loss = self.seg_beta * loss_seg_head + ssl_loss

            # backward and optimize
            total_loss.backward()
            self.optimizer.step()

        else:
            # if no pseudo we skip the frame (happens never basically)
            # we assume that data the loader gives frames in pairs
            stensor0 = ME.SparseTensor(coordinates=batch["coordinates0"].int().to(self.device),
                                       features=batch["features0"].to(self.device),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            stensor1 = ME.SparseTensor(coordinates=batch["coordinates1"].int().to(self.device),
                                       features=batch["features1"].to(self.device),
                                       quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            # Must clear cache at regular interval
            if self.global_step % self.clear_cache_int == 0:
                torch.cuda.empty_cache()

            self.model.eval()
            with torch.no_grad():
                # forward in mink
                out_seg0, out_en0, out_pred0, out_bck0, _, out_seg1, out_en1, out_pred1, out_bck1, _ = self.model((stensor0, stensor1))

            # segmentation loss for t0
            labels0 = batch['labels0'].long()

            loss_seg_head = self.criterion(out_seg0, pseudo0)

            pseudo0 = pseudo0.cpu()
            labels0 = labels0.cpu()

            # get matches in t0 and t1 (used for selection)
            matches0 = batch['matches0'].to(self.device)
            matches1 = batch['matches1'].to(self.device)

            # 2FUTURE CONTRASTIVE
            # forward preds (t0 -> t1)
            future_preds = torch.index_select(out_pred0, 0, matches0)
            # forward gt feats and stop grad
            future_gt = torch.index_select(out_en1.detach(), 0, matches1)
            future_neg_cos_sim = -self.ssl_criterion(future_preds, future_gt).cpu()

            if self.topk_matches > 0:
                # select top-k worst performing matches
                future_neg_cos_sim = future_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
            else:
                future_neg_cos_sim = future_neg_cos_sim.mean(dim=0)

            # 2PAST CONTRASTIVE
            # backward preds (t1 -> t0)
            past_preds = torch.index_select(out_pred1, 0, matches1)
            # backward gt feats and stop grad
            past_gt = torch.index_select(out_en0.detach(), 0, matches0)
            past_neg_cos_sim = -self.ssl_criterion(past_preds, past_gt).cpu()
            if self.topk_matches > 0:
                # select top-k worst performing matches
                past_neg_cos_sim = past_neg_cos_sim.topk(self.topk_matches, dim=0).values.mean()
            else:
                past_neg_cos_sim = past_neg_cos_sim.mean(dim=0)

            # sum up to total
            ssl_loss = (future_neg_cos_sim + past_neg_cos_sim) * self.ssl_beta

        # increase step
        self.global_step += self.stream_batch_size

        # additional metrics
        _, pred_seg0 = out_seg0.detach().max(1)
        # iou
        iou_tmp = jaccard_score(pred_seg0.cpu().numpy(), labels0.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=-0.1)

        # forward preds (t0 -> t1)
        future_preds = torch.index_select(out_bck0.detach(), 0, matches0)
        # forward gt feats and stop grad
        future_gt = torch.index_select(out_bck1.detach(), 0, matches1)
        frame_match_sim = -self.ssl_criterion(future_preds, future_gt).mean()

        # we check pseudo labelling accuracy, not IoU as union of points changes
        valid_idx_pseudo = torch.logical_and(pseudo0 != -1, labels0 != -1)
        pseudo_acc = (pseudo0[valid_idx_pseudo] == labels0[valid_idx_pseudo]).sum() / labels0[valid_idx_pseudo].shape[0]

        present_labels, class_occurs = np.unique(labels0.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.adapt_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join('training', p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        # check pseudo nums
        valid_pseudo = (pseudo0 != -1)
        pseudo_classes, pseudo_num = torch.unique(pseudo0[valid_pseudo], return_counts=True)
        pseudo_names = self.adapt_dataset.class2names[pseudo_classes].tolist()
        classes_count = dict(zip(pseudo_names, pseudo_num.int().tolist()))
        classes_print = dict()
        for c in self.adapt_dataset.class2names[pseudo_classes]:
            if c in classes_count.keys():
                classes_print[f'training/pseudo_number/{c}'] = classes_count[c]
            else:
                classes_print[f'training/pseudo_number/{c}'] = -1

        results_dict.update(classes_print)
        # degeneration check
        out_en0_dg = out_en0.detach().clone()
        out_en1_dg = out_en1.detach().clone()

        max_val = 1/np.sqrt(out_en0_dg.shape[-1])

        out_en0_dg = F.normalize(out_en0_dg, p=2, dim=-1).std(dim=-1).mean()
        out_en1_dg = F.normalize(out_en1_dg, p=2, dim=-1).std(dim=-1).mean()

        results_dict['training/seg_loss'] = loss_seg_head
        results_dict['training/future_ssl'] = future_neg_cos_sim
        results_dict['training/past_ssl'] = past_neg_cos_sim
        results_dict['training/frame_similarity'] = frame_match_sim
        results_dict['training/future_degeneration'] = out_en0_dg
        results_dict['training/past_degeneration'] = out_en1_dg
        results_dict['training/max_degeneration'] = max_val
        results_dict['training/iou'] = np.mean(iou_tmp[present_labels])
        results_dict['training/lr'] = self.optimizer.param_groups[0]["lr"]
        results_dict['training/pseudo_accuracy'] = pseudo_acc
        results_dict['training/pseudo_number'] = torch.sum(pseudo_num)
        results_dict['training/mean_mcmc'] = mean_metric
        # results_dict['training/source_similarity'] = source_sim

        return results_dict

    def validation_step(self, batch, is_source=False, save_path=None, frame=None):
        self.model.eval()
        # for multiple dataloaders
        phase = 'validation' if not is_source else 'source'
        coords_name = 'coordinates'
        feats_name = 'features'
        label_name = 'labels'

        if save_path is not None:
            save_path_tmp = os.path.join(save_path, phase)

        # clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        # sparsify
        stensor = ME.SparseTensor(coordinates=batch[coords_name].int().to(self.device),
                                  features=batch[feats_name].to(self.device))

        # get output
        out, out_bck, out_bottle = self.model(stensor, is_train=False)

        labels = batch[label_name].long()
        present_lbl = torch.unique(labels)

        loss = self.criterion(out, labels)
        _, preds = out.max(1)

        preds = preds.cpu()
        labels = labels.cpu()
        self.global_step += self.stream_batch_size

        # eval iou and log
        iou_tmp = jaccard_score(preds.numpy(), labels.numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        valid_feats_idx = torch.where(labels != -1)[0].view(-1).long()
        db_index = davies_bouldin_score(out_bck.cpu()[valid_feats_idx].numpy(), labels.cpu()[valid_feats_idx].numpy())

        present_labels, class_occurs = np.unique(labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.adapt_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict[f'{phase}/loss'] = loss.cpu().item()
        results_dict[f'{phase}/iou'] = np.mean(iou_tmp[present_labels])
        results_dict[f'{phase}/db_index'] = db_index

        if save_path is not None:

            self.save_pcd(batch, preds.cpu().numpy(),
                          labels.cpu().numpy(), save_path_tmp, frame,
                          is_global=True)

        return results_dict

    def configure_optimizers(self):

        parameters = self.model.parameters()

        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(parameters,
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(parameters,
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            self.optimizer = optimizer
            self.scheduler = None

        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(parameters,
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
            elif self.optimizer_name == 'Adam' or self.optimizer_name == 'ADAM':
                optimizer = torch.optim.Adam(parameters,
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
            self.optimizer = optimizer
            self.scheduler = scheduler

    def get_online_dataloader(self, dataset, is_adapt=False):
        if is_adapt:
            collate = CollateSeparated(torch.device('cpu'))
            sampler = SequentialSampler(dataset, is_adapt=True, adapt_batchsize=self.adaptation_batch_size,
                                        max_time_wdw=self.max_time_wdw)
            dataloader = DataLoader(dataset,
                                    collate_fn=collate,
                                    sampler=sampler,
                                    pin_memory=False,
                                    num_workers=self.train_num_workers)
        else:
            # collate = CollateFN(torch.device('cpu'))
            collate = CollateStream(torch.device('cpu'))
            sampler = SequentialSampler(dataset, is_adapt=False, adapt_batchsize=self.stream_batch_size)
            dataloader = DataLoader(dataset,
                                    collate_fn=collate,
                                    sampler=sampler,
                                    pin_memory=False,
                                    num_workers=self.train_num_workers)
        return dataloader

    def save_pcd(self, batch, preds, labels, save_path, frame, is_global=False):
        pcd = o3d.geometry.PointCloud()

        if not is_global:
            pts = batch['coordinates']
            pcd.points = o3d.utility.Vector3dVector(pts[:, 1:])
        else:
            pts = batch['global_points'][0]
            pcd.points = o3d.utility.Vector3dVector(pts)
        if self.num_classes == 7 or self.num_classes == 2:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[labels+1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[labels])

        os.makedirs(os.path.join(save_path, 'gt'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_path, 'gt', str(frame)+'.ply'), pcd)
        if self.num_classes == 7 or self.num_classes == 2:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds+1])
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.eval_dataset.color_map[preds])

        os.makedirs(os.path.join(save_path, 'preds'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_path, 'preds', str(frame)+'.ply'), pcd)


class OnlineTrainer(object):

    def __init__(self,
                 pipeline,
                 collate_fn_eval=None,
                 collate_fn_adapt=None,
                 device='cpu',
                 default_root_dir=None,
                 weights_save_path=None,
                 loggers=None,
                 save_checkpoint_every=2,
                 source_checkpoint=None,
                 student_checkpoint=None,
                 boost=True,
                 save_predictions=False,
                 is_double=True,
                 is_pseudo=True,
                 use_mcmc=True,
                 sub_epochs=0):

        super().__init__()

        if device is not None:
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cpu')

        self.default_root_dir = default_root_dir
        self.weights_save_path = weights_save_path
        self.loggers = loggers
        self.save_checkpoint_every = save_checkpoint_every
        self.source_checkpoint = source_checkpoint
        self.student_checkpoint = student_checkpoint

        self.pipeline = pipeline
        self.pipeline.device = self.device

        self.is_double = is_double
        self.use_mcmc = use_mcmc
        self.model = self.pipeline.model

        if self.is_double:
            self.source_model = self.pipeline.source_model

        self.eval_dataset = self.pipeline.eval_dataset
        self.adapt_dataset = self.pipeline.adapt_dataset

        self.max_time_wdw = self.eval_dataset.max_time_wdw

        self.eval_dataset.eval()
        self.adapt_dataset.train()

        self.online_sequences = np.arange(self.adapt_dataset.num_sequences())
        self.num_frames = len(self.eval_dataset)

        self.collate_fn_eval = collate_fn_eval
        self.collate_fn_adapt = collate_fn_adapt
        self.collate_fn_eval.device = self.device
        self.collate_fn_adapt.device = self.device

        self.sequence = -1

        self.adaptation_results_dict = {s: [] for s in self.online_sequences}
        self.source_results_dict = {s: [] for s in self.online_sequences}

        # for speed up
        self.eval_dataloader = None
        self.adapt_dataloader = None

        self.boost = boost

        self.save_predictions = save_predictions

        self.is_pseudo = is_pseudo
        self.sub_epochs = sub_epochs
        self.num_classes = self.pipeline.num_classes

        self.dataset_name = self.pipeline.dataset_name

    def adapt_double(self):

        self.load_source_model()

        # first we eval getting performance of source model
        self.eval(is_adapt=True)

        # adapt
        for sequence in tqdm(self.online_sequences, desc='Online Adaptation'):
            # load source model
            self.reload_model()
            # self.reload_model_from_scratch()
            # set sequence in dataset, in weight path and loggers
            self.set_sequence(sequence)
            # adapt on sequence
            sequence_dict = self.online_adaptation_routine()
            self.adaptation_results_dict[sequence] = sequence_dict
        self.save_final_results()

    def eval(self, is_adapt=False):
        # load model only once
        self.reload_model(is_adapt=False)
        for sequence in tqdm(self.online_sequences, desc='Online Evaluation', leave=True):
            # set sequence
            self.set_sequence(sequence)
            # evaluate
            sequence_dict = self.online_evaluation_routine()
            # store dict
            self.source_results_dict[sequence] = sequence_dict
        if not is_adapt:
            self.save_eval_results()

    def check_frame(self, fr):
        return (fr+1) >= self.pipeline.adaptation_batch_size and fr >= self.max_time_wdw

    def online_adaptation_routine(self):
        # move to device
        self.model.to(self.device)

        if self.is_double:
            self.source_model.to(self.device)

        # for storing
        adaptation_results = []

        if self.save_predictions:
            save_path = os.path.join(self.weights_save_path, 'pcd')
        else:
            save_path = None

        for f in tqdm(range(len(self.eval_dataset)), desc=f'Seq: {self.sequence}', leave=True):

            # get eval batch (1 frame at a time)
            val_batch = self.get_evaluation_batch(f)
            # eval
            with torch.no_grad():
                val_dict = self.pipeline.validation_step(val_batch, save_path=save_path, frame=f)
            val_dict['validation/frame'] = f
            # log
            self.log(val_dict)

            # if enough frames
            if self.check_frame(f):
                train_dict = {}
                # get adaptation batch (t-b, t)
                # print('FRAME', f)
                batch = self.get_adaptation_batch(f)

                for _ in range(self.sub_epochs):

                    if self.is_pseudo:
                        train_dict = self.pipeline.adaptation_double_pseudo_step(batch, f)
                    else:
                        raise NotImplementedError

                    if train_dict is not None:
                        train_dict.update(train_dict)
                    # log
                self.log(train_dict)

            if (f+1) % self.save_checkpoint_every == 0:
                # save weights
                self.save_state_dict(f)

            # append dict
            adaptation_results.append(val_dict)

        return adaptation_results

    def online_evaluation_routine(self):
        # move model to device
        self.model.to(self.device)
        # for store
        source_results = []

        if self.save_predictions:
            save_path = os.path.join(self.weights_save_path, 'pcd')
        else:
            save_path = None

        with torch.no_grad():
            for f in tqdm(range(len(self.eval_dataset)), desc=f'Seq: {self.sequence}', leave=True):
                # get eval batch
                val_batch = self.get_evaluation_batch(f)
                # eval
                val_dict = self.pipeline.validation_step(val_batch, is_source=True, save_path=save_path, frame=f)
                val_dict['source/frame'] = f
                # store results
                self.log(val_dict)
                source_results.append(val_dict)

        return source_results

    def set_loggers(self, sequence):
        # set current sequence in loggers, for logging purposes
        for logger in self.loggers:
            logger.set_sequence(sequence)

    def set_sequence(self, sequence):
        # update current weight saving path
        self.sequence = str(sequence)
        path, _ = os.path.split(self.weights_save_path)
        self.weights_save_path = os.path.join(path, self.sequence)
        os.makedirs(self.weights_save_path, exist_ok=True)

        self.eval_dataset.set_sequence(sequence)
        self.adapt_dataset.set_sequence(sequence)

        if self.boost:
            self.eval_dataloader = iter(self.pipeline.get_online_dataloader(FrameOnlineDataset(self.eval_dataset),
                                                                            is_adapt=False))
            self.adapt_dataloader = iter(self.pipeline.get_online_dataloader(PairedOnlineDataset(self.adapt_dataset,
                                                                                                 use_random=self.pipeline.use_random_wdw),
                                                                                 is_adapt=True))

        # set sequence in path of loggers
        self.set_loggers(sequence)

    def log(self, results_dict):
        # log in ach logger
        for logger in self.loggers:
            logger.log(results_dict)

    def save_state_dict(self, frame):
        # save stat dict of the model
        save_dict = {'frame': frame,
                     'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.pipeline.optimizer.state_dict()}
        torch.save(save_dict, os.path.join(self.weights_save_path, f'checkpoint-frame{frame}.pth'))

    def reload_model(self, is_adapt=True):
        # reloads model
        def clean_state_dict(state):
            # clean state dict from names of PL
            for k in list(ckpt.keys()):
                if "model" in k:
                    ckpt[k.replace("model.", "")] = ckpt[k]
                del ckpt[k]
            return state

        if self.student_checkpoint is not None and is_adapt:
            checkpoint_path = self.student_checkpoint
            print(f'--> Loading student checkpoint {checkpoint_path}')
        else:
            checkpoint_path = self.source_checkpoint
            print(f'--> Loading source checkpoint {checkpoint_path}')

        # in case of SSL pretraining
        if isinstance(self.model, MinkUNet18_SSL):
            if checkpoint_path.endswith('.pth'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(ckpt)

            elif checkpoint_path.endswith('.ckpt'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
                ckpt = clean_state_dict(ckpt)
                self.model.load_state_dict(ckpt, strict=True)

            else:
                raise NotImplementedError('Invalid source model extension (allowed .pth and .ckpt)')

        # in case of segmentation pretraining
        elif isinstance(self.model, MinkUNet18_HEADS):
            def clean_student_state_dict(ckpt):
                # clean state dict from names of PL
                for k in list(ckpt.keys()):
                    if "seg_model" in k:
                        ckpt[k.replace("seg_model.", "")] = ckpt[k]
                    del ckpt[k]
                return ckpt
            if checkpoint_path.endswith('.pth'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                ckpt = clean_student_state_dict(ckpt['model_state_dict'])
                self.model.seg_model.load_state_dict(ckpt)

            elif checkpoint_path.endswith('.ckpt'):
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
                ckpt = clean_state_dict(ckpt)
                self.model.seg_model.load_state_dict(ckpt, strict=True)

    def reload_model_from_scratch(self):

        # in case of SSL pretraining
        if isinstance(self.model, MinkUNet18_SSL):
            self.model.weight_initialization()

        # in case of segmentation pretraining
        elif isinstance(self.model, MinkUNet18_HEADS):
            seg_model = self.model.seg_model
            seg_model.weight_initialization()
            self.model = MinkUNet18_HEADS(seg_model=seg_model)

    def load_source_model(self):
        # reloads model
        def clean_state_dict(state):
            # clean state dict from names of PL
            for k in list(ckpt.keys()):
                if "model" in k:
                    ckpt[k.replace("model.", "")] = ckpt[k]
                del ckpt[k]
            return state

        print(f'--> Loading source checkpoint {checkpoint_path}')

        if self.source_checkpoint.endswith('.pth'):
            ckpt = torch.load(self.source_checkpoint, map_location=torch.device('cpu'))
            if isinstance(self.source_model, MinkUNet18_MCMC):
                self.source_model.seg_model.load_state_dict(ckpt)
            else:
                self.source_model.load_state_dict(ckpt)

        elif self.source_checkpoint.endswith('.ckpt'):
            ckpt = torch.load(self.source_checkpoint, map_location=torch.device('cpu'))["state_dict"]
            ckpt = clean_state_dict(ckpt)
            if isinstance(self.source_model, MinkUNet18_MCMC):
                self.source_model.seg_model.load_state_dict(ckpt, strict=True)
            else:
                self.source_model.load_state_dict(ckpt, strict=True)

        else:
            raise NotImplementedError('Invalid source model extension (allowed .pth and .ckpt)')

    def get_adaptation_batch(self, frame_idx):
        if self.adapt_dataloader is None:
            frame_idx += 1
            batch_idx = np.arange(frame_idx - self.pipeline.adaptation_batch_size, frame_idx)

            batch_data = [self.adapt_dataset.__getitem__(b) for b in batch_idx]
            batch_data = [self.adapt_dataset.get_double_data(batch_data[b-1], batch_data[b]) for b in range(1, len(batch_data))]
            batch = self.collate_fn_adapt(batch_data)
        else:
            batch = next(self.adapt_dataloader)

        return batch

    def get_evaluation_batch(self, frame_idx):
        if self.eval_dataloader is None:
            data = self.eval_dataset.__getitem__(frame_idx)
            data = self.eval_dataset.get_single_data(data)

            batch = self.collate_fn_eval([data])
        else:
            batch = next(self.eval_dataloader)

        return batch

    def save_final_results(self):
        # stores final results in a final dict
        # finally saves results in a csv file

        final_dict = {}

        for seq in self.online_sequences:
            source_results = self.source_results_dict[seq]
            adaptation_results = self.adaptation_results_dict[seq]

            assert len(source_results) == len(adaptation_results)
            num_frames = len(source_results)

            source_results = self.format_val_dict(source_results)
            adaptation_results = self.format_val_dict(adaptation_results)

            final_dict[seq] = {}

            for k in adaptation_results.keys():
                relative_tmp = adaptation_results[k] - source_results[k]
                final_dict[seq][f'relative_{k}'] = relative_tmp
                final_dict[seq][f'source_{k}'] = source_results[k]
                final_dict[seq][f'adapted_{k}'] = adaptation_results[k]

        self.write_csv(final_dict, phase='final')
        self.write_csv(final_dict, phase='source')
        self.save_pickle(final_dict)

    def save_eval_results(self):
        # stores final results in a final dict
        # finally saves results in a csv file

        final_dict = {}

        for seq in self.online_sequences:
            eval_results = self.source_results_dict[seq]

            eval_results = self.format_val_dict(eval_results)

            final_dict[seq] = {}

            for k in eval_results.keys():
                final_dict[seq][f'eval_{k}'] = eval_results[k]

        self.write_csv(final_dict, phase='eval')
        self.save_pickle(final_dict)

    def format_val_dict(self, list_dict):
        # input is a list of dicts for each frame
        # returns a dict with [miou, iou_per_frame, per_class_miou, per_class_iou_frame]

        def change_names(in_dict):
            for k in list(in_dict.keys()):
                if "validation/" in k:
                    in_dict[k.replace("validation/", "")] = in_dict[k]
                    del in_dict[k]
                elif "source/" in k:
                    in_dict[k.replace("source/", "")] = in_dict[k]
                    del in_dict[k]

            return in_dict

        list_dict = [change_names(list_dict[f]) for f in range(len(list_dict))]

        if self.num_classes == 7:
            classes = {'vehicle_iou': [],
                       'pedestrian_iou': [],
                       'road_iou': [],
                       'sidewalk_iou': [],
                       'terrain_iou': [],
                       'manmade_iou': [],
                       'vegetation_iou': []}
        elif self.num_classes == 3:
            classes = {'background_iou': [],
                       'vehicle_iou': [],
                       'pedestrian_iou': []}
        else:
            classes = {'vehicle_iou': [],
                       'pedestrian_iou': []}

        for f in range(len(list_dict)):
            val_tmp = list_dict[f]
            for key in classes.keys():
                if key in val_tmp:
                    classes[key].append(val_tmp[key])
                else:
                    classes[key].append(np.nan)

        all_iou = np.concatenate([np.asarray(v)[np.newaxis, ...] for k, v in classes.items()], axis=0).T

        per_class_iou = np.nanmean(all_iou, axis=0)
        miou = np.nanmean(per_class_iou)

        per_frame_miou = np.nanmean(all_iou, axis=-1)

        return {'miou': miou,
                'per_frame_miou': per_frame_miou,
                'per_class_iou': per_class_iou,
                'per_class_frame_iou': all_iou}

    def write_csv(self, results_dict, phase='final'):
        if self.num_classes == 7:
            if phase == 'final':
                headers = ['sequence', 'relative_miou', 'relative_vehicle_iou',
                           'relative_pedestrian_iou', 'relative_road_iou',
                           'relative_sidewalk_iou', 'relative_terrain_iou',
                           'relative_manmade_iou', 'relative_vegetation_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou', 'source_vehicle_iou',
                           'source_pedestrian_iou', 'source_road_iou',
                           'source_sidewalk_iou', 'source_terrain_iou',
                           'source_manmade_iou', 'source_vegetation_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence', 'miou', 'eval_vehicle_iou',
                           'eval_pedestrian_iou', 'eval_road_iou',
                           'eval_sidewalk_iou', 'eval_terrain_iou',
                           'eval_manmade_iou', 'eval_vegetation_iou']
                file_name = 'evaluation_main.csv'
            else:
                raise NotImplementedError
        elif self.num_classes == 3:
            if phase == 'final':
                headers = ['sequence', 'relative_miou',
                           'relative_background_iou',
                           'relative_vehicle_iou',
                           'relative_pedestrian_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou',
                           'source_background_iou',
                           'source_vehicle_iou',
                           'source_pedestrian_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence','miou',
                           'source_backround_iou',
                           'eval_vehicle_iou',
                           'eval_pedestrian_iou']
                file_name = 'evaluation_main.csv'
            else:
                raise NotImplementedError
        elif self.num_classes == 2:
            if phase == 'final':
                headers = ['sequence', 'relative_miou',
                           'relative_vehicle_iou',
                           'relative_pedestrian_iou']
                file_name = 'final_main.csv'
            elif phase == 'source':
                headers = ['sequence', 'miou',
                           'source_vehicle_iou',
                           'source_pedestrian_iou']
                file_name = 'source_main.csv'
            elif phase == 'eval':
                headers = ['sequence','miou',
                           'eval_vehicle_iou',
                           'eval_pedestrian_iou']
                file_name = 'evaluation_main.csv'
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.dataset_name == 'nuScenes':
            cumul = []

        results_dir = os.path.join(os.path.split(self.weights_save_path)[0], 'final_results')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, file_name), 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(headers)

            for seq in results_dict.keys():
                dict_tmp = results_dict[seq]
                if phase == 'final':
                    per_class = dict_tmp['relative_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]
                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]
                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['relative_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                elif phase == 'source':
                    per_class = dict_tmp['source_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]
                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]

                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['source_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                elif phase == 'eval':
                    per_class = dict_tmp['eval_per_class_iou']
                    if self.num_classes == 7:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100,
                                per_class[3]*100,
                                per_class[4]*100,
                                per_class[5]*100,
                                per_class[6]*100]

                    elif self.num_classes == 3:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100,
                                per_class[2]*100]
                    
                    elif self.num_classes == 2:
                        data = [seq,
                                dict_tmp['eval_miou']*100,
                                per_class[0]*100,
                                per_class[1]*100]

                # write the data
                writer.writerow(data)

                if self.dataset_name == 'nuScenes':
                    if phase == 'final':
                        first_iou = dict_tmp['relative_miou']
                    elif phase == 'source':
                        first_iou = dict_tmp['source_miou']
                    elif phase == 'eval':
                        first_iou = dict_tmp['eval_miou']

                    if self.num_classes == 7:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100,
                                      per_class[2]*100,
                                      per_class[3]*100,
                                      per_class[4]*100,
                                      per_class[5]*100,
                                      per_class[6]*100])
                    elif self.num_classes == 3:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100,
                                      per_class[2]*100])
                    elif self.num_classes == 2:
                        cumul.append([first_iou*100,
                                      per_class[0]*100,
                                      per_class[1]*100])

            if self.dataset_name == 'nuScenes':
                avg_cumul = np.array(cumul)
                avg_cumul_tmp = np.nanmean(avg_cumul, axis=0)
                if self.num_classes == 7:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2],
                            avg_cumul_tmp[3],
                            avg_cumul_tmp[4],
                            avg_cumul_tmp[5],
                            avg_cumul_tmp[6],
                            avg_cumul_tmp[7]]
                elif self.num_classes == 3:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2]]

                elif self.num_classes == 2:
                    data = ['Average',
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1]]

                # write cumulative results
                writer.writerow(data)
                seq_locs = np.array([self.adapt_dataset.names2locations[self.adapt_dataset.online_keys[s]] for s in results_dict.keys()])

                for location in ['singapore-queenstown', 'boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth']:
                    valid_sequences = seq_locs == location
                    avg_cumul_tmp = np.nanmean(avg_cumul[valid_sequences, :], axis=0)
                    if self.num_classes == 7:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2],
                            avg_cumul_tmp[3],
                            avg_cumul_tmp[4],
                            avg_cumul_tmp[5],
                            avg_cumul_tmp[6],
                            avg_cumul_tmp[7]]

                    elif self.num_classes == 3:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1],
                            avg_cumul_tmp[2]]

                    elif self.num_classes == 2:
                        data = [location,
                            avg_cumul_tmp[0],
                            avg_cumul_tmp[1]]

                    # write cumulative results
                    writer.writerow(data)

    def save_pickle(self, results_dict):
        results_dir = os.path.join(os.path.split(self.weights_save_path)[0], 'final_results')
        with open(os.path.join(results_dir, 'final_all.pkl'), 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
