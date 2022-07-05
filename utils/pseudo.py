import os
import torch
import numpy as np
import torch.nn.functional as F
from knn_cuda import KNN


color_map = np.array([(240, 240, 240),   # unlabelled - white
                                   (25, 25, 255),    # vehicle - blue
                                   (250, 178, 50),   # pedestrian - orange
                                   (0, 0, 0),    # road - black
                                   (255, 20, 60),   # sidewalk - red
                                   (78, 72, 44),   # terrain - terrain
                                   (233, 166, 250),  # manmade - pink
                                   (157, 234, 50)]) / 255.0   # vegetation - green


class PseudoLabel(object):
    def __init__(self,
                 metric: str = 'mcmc',
                 topk_pseudo: int = None,
                 top_p: float = 0.01,
                 th_pseudo: float = 0.01,
                 is_oracle: bool = False,
                 propagate: bool = False,
                 propagation_size: int = 10,
                 propagation_method: str = 'geometric_features',
                 top_class: int = 1,
                 use_matches: bool = True,
                 **kwargs):

        self.topk_pseudo = topk_pseudo
        self.th_pseudo = th_pseudo
        self.top_p = top_p
        self.is_oracle = is_oracle
        if is_oracle:
            print('--> USING ORACLE ANNOTATIONS!')
        self.metric = metric
        self.propagate = propagate
        self.propagation_size = propagation_size
        self.propagation_method = propagation_method
        self.top_class = top_class
        self.use_matches = use_matches

        try:
            self.device = kwargs['device']
        except KeyError:
            self.device = torch.device('cpu')

        self.knn_search = KNN(k=self.propagation_size, transpose_mode=True)

    def get_pseudo_confidence(self, out, batch, **kwargs):
        frame = kwargs['frame']

        sampled_idx = batch['sampled_idx'][0]
        matches0 = batch['matches0'].long()

        out = out[sampled_idx]
        out = out[matches0]

        pseudo = out.max(dim=-1).indices.cpu()
        new_pseudo = -torch.ones(pseudo.shape).long()

        if self.metric == 'entropy':
            m = -(F.softmax(out, dim=-1) * F.log_softmax(out, dim=-1)).sum(dim=-1)
            descending = False
        elif self.metric == 'confidence':
            m = F.softmax(out, dim=-1).max(dim=-1).values.cpu()
            descending = True
        else:
            raise NotImplementedError

        if self.topk_pseudo is not None:
            present_classes = torch.unique(pseudo)
        else:
            present_classes = torch.unique(pseudo[m > self.th_pseudo])

        for c in present_classes:
            c_idx = torch.where(pseudo == c)[0]
            if self.topk_pseudo is not None:
                m_sort = torch.argsort(m[c_idx], dim=0, descending=descending)
                c_idx = c_idx[m_sort.long()]
                m_idx = c_idx[:self.topk_pseudo]
            else:
                m_best = m[c_idx] > self.th_pseudo
                m_idx = c_idx[m_best]

            if not self.is_oracle:
                new_pseudo[m_idx] = torch.ones(m_idx.shape[0]).long() * c

        pseudo_labels = -torch.ones(sampled_idx.shape[0]).long()
        pseudo_labels[matches0] = new_pseudo
        return_metric = m.mean()

        return pseudo_labels, return_metric

    def get_pseudo_mcmc(self, out, batch):

        uncertainty = out.std(dim=1).mean(dim=-1)
        return_metric = torch.mean(uncertainty, dim=0)
        out = out.mean(dim=1)

        sampled_idx = batch['sampled_idx'][0]
        if self.use_matches:
            matches0 = batch['matches0'].long()
        else:
            matches0 = torch.arange(sampled_idx.shape[0])

        if self.is_oracle:
            oracle_gt = batch['labels0'].long()
        else:
            oracle_gt = None

        out = out[sampled_idx]
        out = out[matches0]

        uncertainty = uncertainty[sampled_idx]
        uncertainty = uncertainty[matches0]

        pseudo = out.max(dim=-1).indices.cpu()
        present_classes = torch.unique(pseudo)

        new_pseudo = -torch.ones(pseudo.shape[0]).long()
        main_idx = torch.arange(pseudo.shape[0])
        valid_pseudo = []
        for c in present_classes:
            c_idx = main_idx[pseudo == c]
            uncertainty_c = uncertainty[c_idx]
            valid_unc = uncertainty_c < self.th_pseudo

            c_idx = c_idx[valid_unc]
            uncertainty_c = uncertainty_c[valid_unc]

            if valid_unc.sum() > 0:
                min_u = torch.argsort(uncertainty_c, dim=0)
                if self.top_class is not None and valid_unc.sum() > self.top_class:
                    min_u = min_u[:self.top_class]

                min_idx = c_idx[min_u]
                new_pseudo[min_idx] = c
                valid_pseudo.append(min_idx)

        pseudo_labels = -torch.ones(sampled_idx.shape[0]).long()

        if len(valid_pseudo) == 0:
            return pseudo_labels, return_metric

        pseudo_labels[matches0] = new_pseudo
        valid_pseudo = torch.cat(valid_pseudo)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(batch['coordinates0'][:, 1:])
        # pcd.colors = o3d.utility.Vector3dVector(color_map[pseudo_labels+1])
        # o3d.io.write_point_cloud('trial_before_prop.ply', pcd)

        if self.propagate:
            for _ in range(1):
                valid_pseudo = torch.where(pseudo_labels != -1)
                pseudo_labels, gf = self.geometric_propagation(pseudo=pseudo_labels,
                                                               batch=batch,
                                                               point_idx=matches0,
                                                               query_idx=matches0[valid_pseudo],
                                                               oracle=oracle_gt)

        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(batch['coordinates0'][:, 1:])
        #     pcd.colors = o3d.utility.Vector3dVector(color_map[pseudo_labels+1])
        #     o3d.io.write_point_cloud('trial_after_prop.ply', pcd)

        return pseudo_labels, return_metric

    def get_pseudo_mcmc_cbst(self, out, batch, p=0.01):
        def get_cbst_th(preds, vals):
            pc = torch.unique(preds)
            c_th = torch.zeros(pc.max()+1)
            for c in pc:
                c_idx = preds == c
                vals_c, _ = torch.sort(vals[c_idx], descending=False)
                c_th[c] = vals_c[torch.floor(torch.tensor((vals_c.shape[0]-1)*p)).long()]
            return c_th

        uncertainty = out.std(dim=1).mean(dim=-1)
        return_metric = torch.mean(uncertainty, dim=0)
        out = out.mean(dim=1)

        sampled_idx = batch['sampled_idx'][0]
        if self.use_matches:
            matches0 = batch['matches0'].long()
        else:
            matches0 = torch.arange(sampled_idx.shape[0])

        if self.is_oracle:
            oracle_gt = batch['labels0'].long()
            new_oracle_gt = -torch.ones_like(oracle_gt)
            valid_oracle_idx = torch.where(oracle_gt != -1)[0]
            oracle_idx = np.random.choice(valid_oracle_idx, int(0.10 * valid_oracle_idx.shape[0]), replace=False)
            new_oracle_gt[oracle_idx] = oracle_gt[oracle_idx]
            oracle_gt = new_oracle_gt
        else:
            oracle_gt = None

        out = out[sampled_idx]
        out = out[matches0]

        uncertainty = uncertainty[sampled_idx]
        uncertainty = uncertainty[matches0]

        pseudo = out.max(dim=-1).indices.cpu()
        class_th = get_cbst_th(pseudo, uncertainty)
        present_classes = torch.unique(pseudo)
        new_pseudo = -torch.ones(pseudo.shape[0]).long()
        main_idx = torch.arange(pseudo.shape[0])
        valid_pseudo = []
        for c in present_classes:
            c_idx = main_idx[pseudo == c]
            uncertainty_c = uncertainty[c_idx]
            valid_unc = uncertainty_c < class_th[c]
            c_idx = c_idx[valid_unc]
            new_pseudo[c_idx] = c
            valid_pseudo.append(c_idx)

        pseudo_labels = -torch.ones(sampled_idx.shape[0]).long()
        valid_pseudo = torch.cat(valid_pseudo)

        if valid_pseudo.shape[0] == 0:
            return pseudo_labels, return_metric

        pseudo_labels[matches0] = new_pseudo

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(batch['coordinates0'][:, 1:])
        # pcd.colors = o3d.utility.Vector3dVector(color_map[pseudo_labels+1])
        # o3d.io.write_point_cloud('trial_before_prop.ply', pcd)

        if self.propagate:
            if self.propagation_method == 'geometric_features':
                for _ in range(1):
                    valid_pseudo = torch.where(pseudo_labels != -1)[0]
                    unlabelled_pseudo = torch.where(pseudo_labels == -1)[0]
                    pseudo_labels, gf = self.geometric_propagation(pseudo=pseudo_labels,
                                                                   batch=batch,
                                                                   point_idx=matches0,
                                                                   query_idx=matches0[valid_pseudo],
                                                                   oracle=oracle_gt)
            elif self.propagation_method == 'knn':

                pseudo_labels, _ = self.knn_propagation(pseudo=pseudo_labels,
                                                        batch=batch,
                                                        point_idx=matches0,
                                                        query_idx=matches0[valid_pseudo],
                                                        oracle=oracle_gt)

            elif self.propagation_method == 'model_features':

                pseudo_labels, _ = self.features_propagation(pseudo=pseudo_labels,
                                                             batch=batch,
                                                             point_idx=matches0,
                                                             query_idx=matches0[valid_pseudo],
                                                             oracle=oracle_gt)
            elif self.propagation_method == 'mixed_features':
                pseudo_labels, _ = self.mixed_propagation(pseudo=pseudo_labels,
                                                          batch=batch,
                                                          point_idx=matches0,
                                                          query_idx=matches0[valid_pseudo],
                                                          oracle=oracle_gt)

            elif self.propagation_method == 'union_output':
                pseudo_labels_gf, _ = self.geometric_propagation(pseudo=pseudo_labels,
                                                               batch=batch,
                                                               point_idx=matches0,
                                                               query_idx=matches0[valid_pseudo],
                                                               oracle=oracle_gt)

                pseudo_labels_fp, _ = self.features_propagation(pseudo=pseudo_labels,
                                                             batch=batch,
                                                             point_idx=matches0,
                                                             query_idx=matches0[valid_pseudo],
                                                             oracle=oracle_gt)

                union_pseudo = -torch.ones(pseudo_labels_fp.shape).long()
                eq_idx = torch.eq(pseudo_labels_gf, pseudo_labels_fp)
                union_pseudo[eq_idx] = pseudo_labels_fp[eq_idx]

                # we get idx where pseudo are different
                diff_idx = torch.logical_not(eq_idx)
                selected_idx = torch.arange(pseudo_labels_fp.shape[0])

                # we select them
                diff_gf = pseudo_labels_gf[diff_idx]
                diff_fp = pseudo_labels_fp[diff_idx]
                merged_diff = torch.cat([diff_gf.view(-1, 1), diff_fp.view(-1, 1)], dim=-1)
                selected_idx = selected_idx[diff_idx]
                # we check where one or the other is -1
                one_label_idx = ((merged_diff == -1).sum(dim=-1)) == 1
                selected_idx = selected_idx[one_label_idx]
                merged_diff = merged_diff.max(dim=-1).values

                union_pseudo[selected_idx] = merged_diff[one_label_idx]
                pseudo_labels = union_pseudo

            elif self.propagation_method == 'minkowski_features':
                pseudo_labels, _ = self.minkowski_propagation(pseudo=pseudo_labels,
                                                              batch=batch,
                                                              point_idx=matches0,
                                                              query_idx=matches0[valid_pseudo],
                                                              oracle=oracle_gt)
            else:
                raise NotImplementedError

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(batch['coordinates0'][:, 1:])
            # pcd.colors = o3d.utility.Vector3dVector(color_map[pseudo_labels+1])
            # o3d.io.write_point_cloud('trial_after_prop.ply', pcd)
            # exit(0)

        return pseudo_labels, return_metric

    def get_pseudo_mcmc_cbst_easy2hard(self, out, batch, p=0.01):
        def get_cbst_th(preds, vals, pp):
            pc = torch.unique(preds)
            c_th = torch.zeros(pc.max()+1)
            for c in pc:
                c_idx = preds == c
                vals_c, _ = torch.sort(vals[c_idx], descending=False)
                c_th[c] = vals_c[torch.floor(torch.tensor((vals_c.shape[0]-1)*pp)).long()]
            return c_th

        uncertainty = out.std(dim=1).mean(dim=-1)
        return_metric = torch.mean(uncertainty, dim=0)
        out = out.mean(dim=1)

        sampled_idx = batch['sampled_idx'][0]
        if self.use_matches:
            matches0 = batch['matches0'].long()
        else:
            matches0 = torch.arange(sampled_idx.shape[0])

        if self.is_oracle:
            oracle_gt = batch['labels0'].long()
        else:
            oracle_gt = None

        out = out[sampled_idx]
        out = out[matches0]

        uncertainty = uncertainty[sampled_idx]
        uncertainty = uncertainty[matches0]

        pseudo = out.max(dim=-1).indices.cpu()
        class_th_easy = get_cbst_th(pseudo, uncertainty, pp=p)
        class_th_hard = get_cbst_th(pseudo, uncertainty, pp=0.9)
        present_classes = torch.unique(pseudo)
        new_pseudo = -torch.ones(pseudo.shape[0]).long()
        main_idx = torch.arange(pseudo.shape[0])
        valid_pseudo_easy = []
        valid_pseudo_hard = []
        for c in present_classes:
            c_idx = main_idx[pseudo == c]
            uncertainty_c = uncertainty[c_idx]
            valid_unc_easy = uncertainty_c < class_th_easy[c]
            valid_unc_hard = uncertainty_c > class_th_hard[c]
            c_idx_easy = c_idx[valid_unc_easy]
            c_idx_hard = c_idx[valid_unc_hard]
            new_pseudo[c_idx_easy] = c
            valid_pseudo_easy.append(c_idx_easy)
            valid_pseudo_hard.append(c_idx_hard)

        pseudo_labels = -torch.ones(sampled_idx.shape[0]).long()
        valid_pseudo_easy = torch.cat(valid_pseudo_easy)
        valid_pseudo_hard = torch.cat(valid_pseudo_hard)

        if valid_pseudo_easy.shape[0] == 0 or valid_pseudo_hard.shape[0] == 0:
            return pseudo_labels, return_metric

        pseudo_labels[matches0] = new_pseudo

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(batch['coordinates0'][:, 1:])
        # pcd.colors = o3d.utility.Vector3dVector(color_map[pseudo_labels+1])
        # o3d.io.write_point_cloud('trial_before_prop_vote.ply', pcd)

        if self.propagate:
            if self.propagation_method == 'voted_features':
                pseudo_labels, gf = self.voted_propagation(pseudo=pseudo_labels,
                                                           batch=batch,
                                                           easy_idx=matches0[valid_pseudo_easy],
                                                           hard_idx=matches0[valid_pseudo_hard],
                                                           oracle=oracle_gt)
            else:
                raise NotImplementedError

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(batch['coordinates0'][:, 1:])
            # pcd.colors = o3d.utility.Vector3dVector(color_map[pseudo_labels+1])
            # o3d.io.write_point_cloud('trial_after_prop_vote.ply', pcd)
            # exit(0)

        return pseudo_labels, return_metric

    def voted_propagation(self, pseudo, batch, easy_idx, hard_idx, dist_th=None, oracle=None):
        '''
        :param pseudo: pseudo labels
        :param batch: training batch
        :param easy_idx: indices of pcd in which to search (with matches)
        :param hard_idx: indices of pcd with valid labels
        :param dist_th: KNN distance threshold, needed?
        :param oracle: for debugging purposes, if provided it will use GT pseudo
        :return:
            prop_pseudo: propagated pseudo labels according to geometric descriptors
        '''

        if oracle is not None:
            pseudo_idx = torch.where(pseudo != -1)
            pseudo[pseudo_idx] = oracle[pseudo_idx]

        geometric_feats = batch["geometric_features0"][0]

        easy_pseudo = pseudo[easy_idx]
        geometric_feats_easy = geometric_feats[easy_idx]
        geometric_feats_hard = geometric_feats[hard_idx]

        prop_pseudo = -torch.ones(pseudo.shape).long()
        prop_pseudo[easy_idx] = easy_pseudo.long()

        knn_dist, knn_idx = self.knn_search(geometric_feats_easy.unsqueeze(0).to(self.device),
                                            geometric_feats_hard.unsqueeze(0).to(self.device))

        knn_dist = knn_dist.cpu().squeeze(0)
        knn_idx = knn_idx.cpu().squeeze(0)

        hard_matches = easy_pseudo[knn_idx]
        hard_pseudo = torch.mode(hard_matches, dim=-1).values.long()

        multi_idx = torch.zeros(knn_idx.shape[0])
        for k in range(knn_idx.shape[0]):
            u_classes, u_counts = torch.unique(hard_matches[k], return_counts=True, sorted=True)
            if u_classes.shape[0] >= 2:
                multi_idx[k] = 1

        hard_pseudo[multi_idx.bool()] = -1

        # we remove multi-class points
        prop_pseudo[hard_idx] = hard_pseudo

        return prop_pseudo, geometric_feats

    def geometric_propagation(self, pseudo, batch, point_idx, query_idx, dist_th=None, oracle=None):
        '''

        :param pseudo: pseudo labels
        :param batch: training batch
        :param point_idx: indices of pcd in which to search (with matches)
        :param query_idx: indices of pcd with valid labels
        :param dist_th: KNN distance threshold, needed?
        :param oracle: for debugging purposes, if provided it will use GT pseudo
        :return:
            prop_pseudo: propagated pseudo labels according to geometric descriptors
        '''

        if oracle is not None:
            pseudo_idx = torch.where(pseudo != -1)
            pseudo[pseudo_idx] = oracle[pseudo_idx]

        geometric_feats = batch["geometric_features0"][0]

        geometric_feats_match = geometric_feats[point_idx]
        geometric_feats_query = geometric_feats[query_idx]

        prop_pseudo = pseudo.clone()

        # put on GPU
        geometric_feats_match = geometric_feats_match.unsqueeze(0).to(self.device)
        geometric_feats_query = geometric_feats_query

        present_classes = torch.unique(pseudo)
        if -1 in present_classes:
            present_classes = present_classes[1:]

        knn_idx_all = []
        classes_prop = []
        for c in present_classes:
            c_idx = pseudo[query_idx] == c

            c_tmp = geometric_feats_query[c_idx].view(1, c_idx.sum(), -1).cuda()
             # get distance and idx
            knn_dist, knn_idx = self.knn_search(geometric_feats_match, c_tmp)

            c_tmp = c_tmp.cpu()
            knn_idx = knn_idx.cpu().squeeze(0).long()
            knn_dist = knn_dist.cpu().squeeze(0)

            if dist_th is not None:
                knn_idx = knn_idx[knn_dist < dist_th]
            knn_idx_all.append(point_idx[knn_idx.view(-1)])
            classes_prop.append(torch.ones(knn_idx.view(-1).shape[0]) * c)

        geometric_feats_match = geometric_feats_match.cpu()

        knn_idx_all = torch.cat(knn_idx_all)
        classes_prop = torch.cat(classes_prop)

        prop_pseudo[knn_idx_all] = classes_prop.long()

        # we ensure to not overlap features
        unique_idx, unique_count = torch.unique(knn_idx_all, return_counts=True)
        multi_idx = unique_idx[unique_count > 1]
        multi_idx = multi_idx[pseudo[multi_idx] == -1]

        # we remove multi-class points
        prop_pseudo[multi_idx] = -1

        return prop_pseudo, geometric_feats

    def knn_propagation(self, pseudo, batch, point_idx, query_idx, dist_th=None, oracle=None):
        '''

        :param pseudo: pseudo labels
        :param batch: training batch
        :param point_idx: indices of pcd in which to search (with matches)
        :param query_idx: indices of pcd with valid labels
        :param dist_th: KNN distance threshold, needed?
        :param oracle: for debugging purposes, if provided it will use GT pseudo
        :return:
            prop_pseudo: propagated pseudo labels according to geometric descriptors
        '''

        if oracle is not None:
            pseudo_idx = torch.where(pseudo != -1)
            pseudo[pseudo_idx] = oracle[pseudo_idx]

        coords = batch["coordinates0"][:, 1:]

        geometric_feats_match = coords[point_idx]
        geometric_feats_query = coords[query_idx]

        prop_pseudo = pseudo.clone()

        # put on GPU
        geometric_feats_match = geometric_feats_match.unsqueeze(0).to(self.device)
        geometric_feats_query = geometric_feats_query

        present_classes = torch.unique(pseudo)
        if -1 in present_classes:
            present_classes = present_classes[1:]

        knn_idx_all = []
        classes_prop = []
        for c in present_classes:
            c_idx = pseudo[query_idx] == c

            c_tmp = geometric_feats_query[c_idx].view(1, c_idx.sum(), -1).cuda()

             # get distance and idx
            knn_dist, knn_idx = self.knn_search(geometric_feats_match, c_tmp)

            knn_idx = knn_idx.cpu().squeeze(0).long()
            knn_dist = knn_dist.cpu().squeeze(0)

            if dist_th is not None:
                knn_idx = knn_idx[knn_dist < dist_th]
            knn_idx_all.append(point_idx[knn_idx.view(-1)])
            classes_prop.append(torch.ones(knn_idx.view(-1).shape[0]) * c)

        geometric_feats_match = geometric_feats_match.cpu()

        knn_idx_all = torch.cat(knn_idx_all)
        classes_prop = torch.cat(classes_prop)

        prop_pseudo[knn_idx_all] = classes_prop.long()

        # we ensure to not overlap features
        unique_idx, unique_count = torch.unique(knn_idx_all, return_counts=True)
        multi_idx = unique_idx[unique_count > 1]
        multi_idx = multi_idx[pseudo[multi_idx] == -1]

        # we remove multi-class points
        prop_pseudo[multi_idx] = -1

        return prop_pseudo, coords

    def features_propagation(self, pseudo, batch, point_idx, query_idx, dist_th=None, oracle=None):
        '''

        :param pseudo: pseudo labels
        :param batch: training batch
        :param point_idx: indices of pcd in which to search (with matches)
        :param query_idx: indices of pcd with valid labels
        :param dist_th: KNN distance threshold, needed?
        :param oracle: for debugging purposes, if provided it will use GT pseudo
        :return:
            prop_pseudo: propagated pseudo labels according to geometric descriptors
        '''

        if oracle is not None:
            pseudo_idx = torch.where(pseudo != -1)
            pseudo[pseudo_idx] = oracle[pseudo_idx]

        geometric_feats = batch["model_features0"]

        geometric_feats_match = geometric_feats[point_idx]
        geometric_feats_query = geometric_feats[query_idx]

        prop_pseudo = pseudo.clone()

        # put on GPU
        geometric_feats_match = geometric_feats_match.unsqueeze(0).to(self.device)
        geometric_feats_query = geometric_feats_query

        present_classes = torch.unique(pseudo)
        if -1 in present_classes:
            present_classes = present_classes[1:]

        knn_idx_all = []
        classes_prop = []
        for c in present_classes:
            c_idx = pseudo[query_idx] == c

            c_tmp = geometric_feats_query[c_idx].view(1, c_idx.sum(), -1).cuda()

             # get distance and idx
            knn_dist, knn_idx = self.knn_search(geometric_feats_match, c_tmp)

            knn_idx = knn_idx.cpu().squeeze(0).long()
            knn_dist = knn_dist.cpu().squeeze(0)

            if dist_th is not None:
                knn_idx = knn_idx[knn_dist < dist_th]
            knn_idx_all.append(point_idx[knn_idx.view(-1)])
            classes_prop.append(torch.ones(knn_idx.view(-1).shape[0]) * c)

        geometric_feats_match = geometric_feats_match.cpu()

        knn_idx_all = torch.cat(knn_idx_all)
        classes_prop = torch.cat(classes_prop)

        prop_pseudo[knn_idx_all] = classes_prop.long()

        # we ensure to not overlap features
        unique_idx, unique_count = torch.unique(knn_idx_all, return_counts=True)
        multi_idx = unique_idx[unique_count > 1]
        multi_idx = multi_idx[pseudo[multi_idx] == -1]

        # we remove multi-class points
        prop_pseudo[multi_idx] = -1

        return prop_pseudo, geometric_feats

    def mixed_propagation(self, pseudo, batch, point_idx, query_idx, dist_th=None, oracle=None):
        '''
        :param pseudo: pseudo labels
        :param batch: training batch
        :param point_idx: indices of pcd in which to search (with matches)
        :param query_idx: indices of pcd with valid labels
        :param dist_th: KNN distance threshold, needed?
        :param oracle: for debugging purposes, if provided it will use GT pseudo
        :return:
            prop_pseudo: propagated pseudo labels according to geometric descriptors
        '''

        if oracle is not None:
            pseudo_idx = torch.where(pseudo != -1)
            pseudo[pseudo_idx] = oracle[pseudo_idx]

        geometric_feats = F.normalize(batch["geometric_features0"][0], dim=-1)
        model_feats = F.normalize(batch["model_features0"], dim=-1)
        geometric_feats = torch.cat([geometric_feats, model_feats], dim=-1)

        geometric_feats_match = geometric_feats[point_idx]
        geometric_feats_query = geometric_feats[query_idx]

        prop_pseudo = pseudo.clone()

        # put on GPU
        geometric_feats_match = geometric_feats_match.unsqueeze(0).to(self.device)
        geometric_feats_query = geometric_feats_query

        present_classes = torch.unique(pseudo)
        if -1 in present_classes:
            present_classes = present_classes[1:]

        knn_idx_all = []
        classes_prop = []
        for c in present_classes:
            c_idx = pseudo[query_idx] == c

            c_tmp = geometric_feats_query[c_idx].view(1, c_idx.sum(), -1).cuda()

             # get distance and idx
            knn_dist, knn_idx = self.knn_search(geometric_feats_match, c_tmp)

            knn_idx = knn_idx.cpu().squeeze(0).long()
            knn_dist = knn_dist.cpu().squeeze(0)

            if dist_th is not None:
                knn_idx = knn_idx[knn_dist < dist_th]
            knn_idx_all.append(point_idx[knn_idx.view(-1)])
            classes_prop.append(torch.ones(knn_idx.view(-1).shape[0]) * c)

        geometric_feats_match = geometric_feats_match.cpu()

        knn_idx_all = torch.cat(knn_idx_all)
        classes_prop = torch.cat(classes_prop)

        prop_pseudo[knn_idx_all] = classes_prop.long()

        # we ensure to not overlap features
        unique_idx, unique_count = torch.unique(knn_idx_all, return_counts=True)
        multi_idx = unique_idx[unique_count > 1]
        multi_idx = multi_idx[pseudo[multi_idx] == -1]

        # we remove multi-class points
        prop_pseudo[multi_idx] = -1

        return prop_pseudo, geometric_feats

    def minkowski_propagation(self, pseudo, batch, point_idx, query_idx, dist_th=None, oracle=None):
        '''

        :param pseudo: pseudo labels
        :param batch: training batch
        :param point_idx: indices of pcd in which to search (with matches)
        :param query_idx: indices of pcd with valid labels
        :param dist_th: KNN distance threshold, needed?
        :param oracle: for debugging purposes, if provided it will use GT pseudo
        :return:
            prop_pseudo: propagated pseudo labels according to geometric descriptors
        '''

        if oracle is not None:
            pseudo_idx = torch.where(pseudo != -1)
            pseudo[pseudo_idx] = oracle[pseudo_idx]

        geometric_feats = batch["minkowski_features0"]

        geometric_feats_match = geometric_feats[point_idx]
        geometric_feats_query = geometric_feats[query_idx]

        prop_pseudo = pseudo.clone()

        # put on GPU
        geometric_feats_match = geometric_feats_match.unsqueeze(0).to(self.device)
        geometric_feats_query = geometric_feats_query

        present_classes = torch.unique(pseudo)
        if -1 in present_classes:
            present_classes = present_classes[1:]

        knn_idx_all = []
        classes_prop = []
        for c in present_classes:
            c_idx = pseudo[query_idx] == c

            c_tmp = geometric_feats_query[c_idx].view(1, c_idx.sum(), -1).cuda()

             # get distance and idx
            knn_dist, knn_idx = self.knn_search(geometric_feats_match, c_tmp)

            c_tmp = c_tmp.cpu()
            knn_idx = knn_idx.cpu().squeeze(0).long()
            knn_dist = knn_dist.cpu().squeeze(0)

            if dist_th is not None:
                knn_idx = knn_idx[knn_dist < dist_th]
            knn_idx_all.append(point_idx[knn_idx.view(-1)])
            classes_prop.append(torch.ones(knn_idx.view(-1).shape[0]) * c)

        geometric_feats_match = geometric_feats_match.cpu()

        knn_idx_all = torch.cat(knn_idx_all)
        classes_prop = torch.cat(classes_prop)

        prop_pseudo[knn_idx_all] = classes_prop.long()

        # we ensure to not overlap features
        unique_idx, unique_count = torch.unique(knn_idx_all, return_counts=True)
        multi_idx = unique_idx[unique_count > 1]
        multi_idx = multi_idx[pseudo[multi_idx] == -1]

        # we remove multi-class points
        prop_pseudo[multi_idx] = -1

        return prop_pseudo, geometric_feats

    def get_pseudo(self, out, batch, frame, return_metric=False):
        if self.metric == 'mcmc':
            pseudo, metric = self.get_pseudo_mcmc(out, batch)
        elif self.metric == 'mcmc_cbst':
            pseudo, metric = self.get_pseudo_mcmc_cbst(out, batch, p=self.top_p)
        else:
            pseudo, metric = self.get_pseudo_confidence(out, batch, frame=frame)

        if return_metric:
            return pseudo,  metric
        else:
            return pseudo

