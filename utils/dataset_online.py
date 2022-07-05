import os
from abc import ABC
import copy
import torch
import yaml
import pickle
import numpy as np
import open3d as o3d
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils import data_classes as dc
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from functools import reduce

from torch.utils.data import Dataset
import MinkowskiEngine as ME
import utils
from utils.augmentations import Compose
from utils.voxelizer import Voxelizer
from utils.dataset import SynthDataset
from knn_cuda import KNN
import random
from scipy.spatial.transform import Rotation as rotation



np.random.seed(1234)

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_matching_indices(source, target, search_voxel_size=0.3, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)
    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
              idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


class OnlineBaseDataset(Dataset):
    def __init__(self,
                 version: str,
                 phase: str,
                 sequence_idx: int,
                 dataset_path: str,
                 voxel_size: float = 0.05,
                 max_time_wdw: int = 1,
                 oracle_pts: int = 0,
                 sub_num: int = 50000,
                 use_intensity: bool = False,
                 augment_data: bool = False,
                 input_transforms: Compose = None,
                 ignore_label: int = -1,
                 device: torch.device = None,
                 num_classes: int = 7):

        self.CACHE = {}
        self.version = version
        self.phase = phase
        self.sequence_idx = sequence_idx
        self.dataset_path = dataset_path
        self.voxel_size = voxel_size  # in meter
        self.max_time_wdw = max_time_wdw
        self.time_windows = [x for x in range(1, self.max_time_wdw+1)]
        self.sub_num = sub_num
        self.use_intensity = use_intensity
        self.augment_data = augment_data
        self.input_transforms = input_transforms

        self.ignore_label = ignore_label

        # for input augs
        # self.clip_bounds = ((-100, 100), (-100, 100), (-100, 100))
        self.clip_bounds = None
        self.scale_augmentation_bound = (0.9, 1.1)
        self.rotation_augmentation_bound = ((-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20))
        self.translation_augmentation_ratio_bound = ((-0.02, 0.02), (-0.05, 0.05), (-0.02, 0.02))

        self.voxelizer = Voxelizer(voxel_size=self.voxel_size,
                                   clip_bound=self.clip_bounds,
                                   use_augmentation=self.augment_data,
                                   scale_augmentation_bound=self.scale_augmentation_bound,
                                   rotation_augmentation_bound=self.rotation_augmentation_bound,
                                   translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                                   ignore_label=self.ignore_label)

        self.device = device

        self.split = {'train': [],
                      'validation': []}

        self.maps = None

        if num_classes == 7:
            # unique color map among all the datasets
            self.color_map = np.array([(240, 240, 240),   # unlabelled - white
                                       (25, 25, 255),    # vehicle - blue
                                       (250, 178, 50),   # pedestrian - orange
                                       (0, 0, 0),    # road - black
                                       (255, 20, 60),   # sidewalk - red
                                       (78, 72, 44),   # terrain - terrain
                                       (233, 166, 250),  # manmade - pink
                                       (157, 234, 50)]) / 255.0   # vegetation - green

            self.class2names = np.array(['vehicle',
                                         'pedestrian',
                                         'road',
                                         'sidewalk',
                                         'terrain',
                                         'manmade',
                                         'vegetation'])
        elif num_classes == 3:
            # unique color map among all the datasets
            self.color_map = np.array([(0, 0, 0),   # unlabelled - white
                                       (25, 25, 255),    # vehicle - blue
                                       (250, 178, 50)]) / 255.0   # pedestrian - orange

            self.class2names = np.array(['background',
                                         'vehicle',
                                         'pedestrian'])
        else:
            raise NotImplementedError

        self.oracle_pts = oracle_pts

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i: int):
        raise NotImplementedError

    def random_sample(self, points: np.ndarray, center: np.array = None) -> np.array:
        """
        :param points: input points of shape [N, 3]
        :param center: center to sample around, default is None, not used for now
        :return: np.ndarray of N' points sampled from input points
        """

        num_points = points.shape[0]
        if self.sub_num <= num_points:
            sampled_idx = np.random.choice(np.arange(num_points), self.sub_num, replace=False)
        else:
            over_idx = np.random.choice(np.arange(num_points), self.sub_num - num_points, replace=False)
            sampled_idx = np.concatenate([np.arange(num_points), over_idx])

        np.random.shuffle(sampled_idx)

        return sampled_idx

    def globalize(self, pts_temp, trans):
        points = np.ones((pts_temp.shape[0], 4))
        points[:, 0:3] = pts_temp[:, 0:3]
        tpoints = np.matmul(trans, points.T).T
        return torch.from_numpy(tpoints[:, :3])

    def get_double_data(self, data, next_data):
        points = data['points']
        colors = data['features']
        labels = data['labels']
        global_pts = data['global_points']
        geometric_feats = data['geometric_feats']
        points_all = copy.deepcopy(points)

        next_points = next_data['points']
        next_colors = next_data['features']
        next_labels = next_data['labels']
        global_next_pts = next_data['global_points']

        sampled_idx = None

        _, voxel_idx = ME.utils.sparse_quantize(points/self.voxel_size, return_index=True)

        _, next_voxel_idx = ME.utils.sparse_quantize(next_points/self.voxel_size, return_index=True)

        geometric_feats = geometric_feats[voxel_idx]

        points = points[voxel_idx]
        next_points = next_points[next_voxel_idx]
        points_all = points_all[voxel_idx]

        colors = colors[voxel_idx]
        next_colors = next_colors[next_voxel_idx]

        labels = labels[voxel_idx]
        next_labels = next_labels[next_voxel_idx]

        global_pts = global_pts[voxel_idx]
        global_next_pts = global_next_pts[next_voxel_idx]

        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)

            # sampled_main_idx = sampled_main_idx[sampled_idx]
            geometric_feats = geometric_feats[sampled_idx]
            # pca_gf = pca_gf[sampled_idx]
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]
            global_pts = global_pts[sampled_idx]

            next_sampled_idx = self.random_sample(next_points)

            next_points = next_points[next_sampled_idx]
            next_colors = next_colors[next_sampled_idx]
            next_labels = next_labels[next_sampled_idx]
            global_next_pts = global_next_pts[next_sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

            homo_coords = np.hstack((next_points, np.ones((next_points.shape[0], 1), dtype=next_points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            next_points = homo_coords @ rigid_transformation.T[:, :3]

            _, voxel_tr_idx = ME.utils.sparse_quantize(points/self.voxel_size, return_index=True)
            _, next_voxel_tr_idx = ME.utils.sparse_quantize(next_points/self.voxel_size, return_index=True)

            # sampled_main_idx = sampled_main_idx[voxel_tr_idx]
            geometric_feats = geometric_feats[voxel_tr_idx]
            # pca_gf = pca_gf[voxel_tr_idx]
            points = points[voxel_tr_idx]
            colors = colors[voxel_tr_idx]
            labels = labels[voxel_tr_idx]
            global_pts = global_pts[voxel_tr_idx]

            next_points = next_points[next_voxel_tr_idx]
            next_colors = next_colors[next_voxel_tr_idx]
            next_labels = next_labels[next_voxel_tr_idx]
            global_next_pts = global_next_pts[next_voxel_tr_idx]

        geometric_feats = torch.from_numpy(geometric_feats)
        points_all = torch.from_numpy(points_all)
        points = torch.from_numpy(points)
        colors = torch.from_numpy(colors)
        labels = torch.from_numpy(labels)

        next_points = torch.from_numpy(next_points)
        next_colors = torch.from_numpy(next_colors)
        next_labels = torch.from_numpy(next_labels)

        if sampled_idx is None:
            sampled_idx = np.arange(points_all.shape[0])
        else:
            sampled_idx = sampled_idx[voxel_tr_idx]

        sampled_idx = torch.from_numpy(sampled_idx)

        pcd0 = make_open3d_point_cloud(global_pts)
        next_pcd0 = make_open3d_point_cloud(global_next_pts)

        # Get matches between t and t+tw
        matches = get_matching_indices(pcd0, next_pcd0, K=1, search_voxel_size=0.7)
        matches = torch.tensor(matches)

        # if matches.shape[0] < 100:
        if matches.shape[0] < 100:
            print(f'Found only {matches.shape[0]} matches')
            matches = get_matching_indices(pcd0, next_pcd0, K=1, search_voxel_size=1.)
            matches = torch.tensor(matches)

        if matches.shape[0] < 10:
            raise ValueError(f'Found only {matches.shape[0]} matches')

        matches0 = matches[:, 0]
        matches1 = matches[:, 1]

        num_pts0 = points.shape[0]
        num_pts1 = next_points.shape[0]

        if self.input_transforms is not None:
            raise NotImplementedError

        coords = torch.floor(points / self.voxel_size)
        next_coords = torch.floor(next_points / self.voxel_size)
        coords_all = torch.floor(points_all / self.voxel_size)

        if self.oracle_pts > 0:
            # we select oracle_pts point per class
            present_c = torch.unique(labels)
            selected_oracle = []
            new_labels = -torch.ones(labels.shape).type(labels.type())
            for c in present_c:
                c_idx = torch.where(labels == c)[0]
                if c_idx.shape[0] > self.oracle_pts:
                    new_labels[c_idx[:self.oracle_pts]] = labels[c_idx[:self.oracle_pts]]
                else:
                    new_labels[c_idx] = labels[c_idx]

            labels = new_labels

        return {'coordinates_all': coords_all.int(),
                'coordinates': coords.int(),
                'features': colors.float(),
                'geometric_feats': geometric_feats.float(),
                'labels': labels,
                'next_coordinates': next_coords.int(),
                'next_features': next_colors.float(),
                'next_labels': next_labels,
                'matches0': matches0.int(),
                'matches1': matches1.int(),
                'num_pts0': num_pts0,
                'num_pts1': num_pts1,
                'sampled_idx': sampled_idx}

    def get_single_data(self, data):
        points = data['points']
        colors = data['features']
        labels = data['labels']
        global_pts = data['global_points']

        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)

            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]
            global_pts = global_pts[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        points = torch.from_numpy(points)

        _, voxel_idx = ME.utils.sparse_quantize(points/self.voxel_size, return_index=True)

        points = points[voxel_idx]
        global_pts = global_pts[voxel_idx]

        feats = torch.from_numpy(colors[voxel_idx])
        labels = torch.from_numpy(labels[voxel_idx])

        if self.input_transforms is not None:
            raise NotImplementedError

        coords = torch.floor(points / self.voxel_size)

        return {"coordinates": coords.int(),
                "features": feats.float(),
                "labels": labels,
                'global_points': global_pts}

    def eval(self):
        self.phase = 'eval'

    def train(self):
        self.phase = 'train'


class OnlineSemanticKITTIDataset(OnlineBaseDataset, ABC):
    def __init__(self,
                 version='full',
                 phase='eval',
                 dataset_path='/data/csaltori/SemanticKITTI/data/sequences',
                 mapping_path='./_resources/semantic-kitti.yaml',
                 sequence_idx=0,
                 voxel_size=0.05,
                 use_intensity=False,
                 augment_data=False,
                 input_transforms=None,
                 sub_num=50000,
                 max_time_wdw=1,
                 oracle_pts=0,
                 device=None,
                 ignore_label=-1,
                 clip_range=[[-50, 50], [-50, 50]],
                 split_size=250,
                 num_classes=7,
                 noisy_odo=False,
                 odo_roto_bounds=None,
                 odo_tras_bounds=None,
                 geometric_path='experiments/dip_features/08'):

        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         sequence_idx=sequence_idx,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         max_time_wdw=max_time_wdw,
                         oracle_pts=oracle_pts,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         input_transforms=input_transforms,
                         device=device,
                         ignore_label=ignore_label,
                         num_classes=num_classes)

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))
        self.name = 'SemanticKITTI'
        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        self.split = {'eval': '08',
                      'train': '08'}

        self.sequence = self.split[self.phase]
        self.split_size = split_size

        if self.split_size == 250:
            version_splits = {'full': 16,
                              'mini': 3,
                              'micro': 1}
        else:
            # version_splits = {'full': 4,
            #                   'mini': 1,
            #                   'micro': 1}
            version_splits = {'full': 1,
                              'mini': 1,
                              'micro': 1}

        self.num_frames = len(os.listdir(os.path.join(self.dataset_path, self.sequence, 'labels')))

        self.online_sequences = self.get_online_split(num_seq=version_splits[self.version], split_size=self.split_size)
        self.online_keys = list(self.online_sequences.keys())

        self.seq_path_list = {}
        self.seq_label_list = {}

        self.transforms = {}

        self.get_paths()

        self.sub_seq = None
        self.selected_sequence = None
        self.selected_transforms = None
        self.pcd_path = []
        self.label_path = []

        self.set_sequence(self.sequence_idx)

        if clip_range is not None:
            self.clip_range = np.array(clip_range)
        else:
            self.clip_range = None

        self.geometric_path = geometric_path

        self.noisy_odo = noisy_odo
        self.odo_roto_bounds = odo_roto_bounds
        self.odo_tras_bounds = odo_tras_bounds

    def num_sequences(self):
        return len(self.online_keys)

    def check_range(self, pts):
        range_x = np.logical_and(pts[:, 1] < self.clip_range[0, 1], pts[:, 1] > self.clip_range[0, 0])
        range_z = np.logical_and(pts[:, 0] < self.clip_range[1, 1], pts[:, 0] > self.clip_range[1, 0])
        range_idx = np.logical_and(range_x, range_z)
        return range_idx

    def get_paths(self):

        calibration = self.parse_calibration(os.path.join(self.dataset_path, self.sequence, "calib.txt"))
        poses = self.parse_poses(os.path.join(self.dataset_path, self.sequence, "poses.txt"), calibration)
        poses = np.asarray(poses)
        for sub_seq in self.online_keys:

            frames = self.online_sequences[sub_seq]
            self.transforms[sub_seq] = poses[frames]

            self.seq_path_list[sub_seq] = []
            self.seq_label_list[sub_seq] = []
            for f in frames:
                pcd_path = os.path.join(self.dataset_path, self.sequence, 'velodyne', f'{int(f):06d}.bin')
                label_path = os.path.join(self.dataset_path, self.sequence, 'labels', f'{int(f):06d}.label')
                self.seq_path_list[sub_seq].append(pcd_path)
                self.seq_label_list[sub_seq].append(label_path)

    def __len__(self):
        return self.split_size

    def get_online_split(self, num_seq, split_size=250):
        online_sequences = {x: None for x in range(num_seq)}

        for k, seq in enumerate(online_sequences):
            assert (k+1) * split_size <= self.num_frames, f'Error in frames indexing for {num_seq} sequences!'
            online_sequences[seq] = np.arange(k*split_size, (k+1) * split_size)

        return online_sequences

    @staticmethod
    def parse_calibration(filename):
        """ read calibration file with given filename
          Returns
          -------
          dict
              Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename
          Returns
          -------
          list
              list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        return poses

    def get_transform(self, path):
        idx = self.pcd_path.index(path)
        tr_tmp = self.selected_transforms[idx]

        return tr_tmp

    def get_sequence(self, path):
        prev, _ = os.path.split(path)
        _, seq = os.path.split(os.path.split(prev)[0])

        return seq

    def add_tr_noise(self, tr):

        rot_noise_mtx = rotation.from_euler('z', random.uniform(-self.odo_roto_bounds, self.odo_roto_bounds),
                                            degrees=True).as_matrix()


        noisy_mtx = np.zeros((4, 4))
        noisy_mtx[:3, :3] = rot_noise_mtx
        noisy_mtx[3, 3] = 1

        tras_noise_mtx = np.random.uniform(-self.odo_tras_bounds, self.odo_tras_bounds, 2)

        noisy_mtx[0, 3] += tras_noise_mtx[0]
        # noisy_mtx[1, 3] += tras_noise_mtx[1]

        noisy_tr = noisy_mtx @ tr
        return noisy_tr

    def get_frame(self, pcd_tmp, label_tmp):

        if pcd_tmp not in self.CACHE:
            pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
            label = self.load_label_kitti(label_tmp)
            points = pcd[:, :3]

            _, name = os.path.split(pcd_tmp)
            gf_path = os.path.join(self.geometric_path, name[:-4]+'.npz')
            geometric_feats = np.load(gf_path)['features']

            if self.clip_range is not None:
                range_idx = self.check_range(points)
                points = points[range_idx]
                label = label[range_idx]
                geometric_feats = geometric_feats[range_idx]

            pc_tr = self.get_transform(pcd_tmp)

            if self.noisy_odo:
                pc_tr = self.add_tr_noise(pc_tr)

            global_pts = self.globalize(points, pc_tr)

            if self.use_intensity:
                colors = points[:, 3][..., np.newaxis]
            else:
                colors = np.ones((points.shape[0], 1), dtype=np.float32)
            data = {'points': points,
                    'features': colors,
                    'labels': label,
                    'global_points': global_pts,
                    'geometric_feats': geometric_feats}
            # self.CACHE[pcd_tmp] = data
        else:
            data = self.CACHE[pcd_tmp]

        return data

    def __getitem__(self, i):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        data = self.get_frame(pcd_tmp, label_tmp)

        return data

    def set_sequence(self, idx):
        self.sub_seq = self.online_keys[idx]
        self.selected_sequence = self.online_sequences[self.sub_seq]
        self.selected_transforms = self.transforms[self.sub_seq]
        self.pcd_path = self.seq_path_list[self.sub_seq]
        self.label_path = self.seq_label_list[self.sub_seq]

    def load_label_kitti(self, label_path):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = self.remap_lut_val[sem_label]
        return sem_label.astype(np.int32)


class OnlineNuScenesDataset(OnlineBaseDataset, ABC):
    def __init__(self,
                 nusc=None,
                 version='full',
                 phase='eval',
                 dataset_path='/data/csaltori/nuScenes-lidarseg/',
                 sequence_idx=0,
                 mapping_path='_resources/nuscenes.yaml',
                 voxel_size=0.05,
                 sub_num=50000,
                 max_time_wdw=1,
                 oracle_pts=0,
                 device=None,
                 use_intensity=False,
                 augment_data=False,
                 input_transforms=None,
                 ignore_label=None,
                 clip_range=[[-100, 100], [-100, 100]],
                 num_sweeps=0,
                 num_classes=7,
                 geometric_path='experiments/dip_features/nuscenes'):

        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         sequence_idx=sequence_idx,
                         voxel_size=voxel_size,
                         max_time_wdw=max_time_wdw,
                         oracle_pts=oracle_pts,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         input_transforms=input_transforms,
                         device=device,
                         ignore_label=ignore_label,
                         num_classes=num_classes)

        self.version = 'v1.0-trainval' if self.version == 'full' else 'v1.0-mini'

        if not nusc:
            self.nusc = NuScenes(version=self.version,
                                 dataroot=self.dataset_path,
                                 verbose=True)
        else:
            self.nusc = nusc

        splits = create_splits_scenes()
        if self.version == 'v1.0-trainval':
            self.split = splits['val']
        else:
            self.split = splits['mini_val']

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))
        self.name = 'nuScenes'

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        self.token_list = []
        self.seq_token_list = {}
        self.names2tokens = {}
        self.names2locations = {}

        self.get_tokens()

        self.online_keys = list(self.seq_token_list.keys())

        self.selected_sequence = None

        self.set_sequence(self.sequence_idx)

        self.clip_range = np.array(clip_range)

        self.geometric_path = geometric_path

        self.location = None

        self.num_sweeps = num_sweeps

        if self.num_sweeps > 0:
            self.interpolate = True
            self.knn_search = KNN(k=1, transpose_mode=True)
            print(f'--> INTERPOLATION OF {self.num_sweeps}')
        else:
            print(f'--> INTERPOLATION OFF')
            self.interpolate = False

    def get_tokens(self):
        scenes_tokens = {}
        scenes = self.nusc.scene

        for s in scenes:
            # list tokens of the scenes
            scenes_tokens[s['name']] = s['token']

        self.names2tokens = scenes_tokens

        for scene in self.split:
            # init empty lists
            token_list_seq = []

            # get scene token and scene
            scene_token = scenes_tokens[scene]

            self.names2locations[scene] = self.get_location(scene_token)

            scene_temp = self.nusc.get('scene', scene_token)

            # get first sample(frame) token
            sample_token = scene_temp['first_sample_token']

            # iterate over samples given tokens
            while sample_token != '':
                # get sample record
                sample_record = self.nusc.get('sample', sample_token)
                # append token
                token_list_seq.append(sample_token)
                # update sample token with the next
                sample_token = sample_record['next']

            self.seq_token_list[scene] = token_list_seq

    def set_sequence(self, idx):
        self.selected_sequence = self.online_keys[idx]
        self.token_list = self.seq_token_list[self.selected_sequence]
        self.location = self.names2locations[self.selected_sequence]

    def num_sequences(self):
        return len(self.online_keys)

    def check_range(self, pts):
        range_x = np.logical_and(pts[:, 1] < self.clip_range[0, 1], pts[:, 1] > self.clip_range[0, 0])
        range_z = np.logical_and(pts[:, 0] < self.clip_range[1, 1], pts[:, 0] > self.clip_range[1, 0])
        range_idx = np.logical_and(range_x, range_z)
        return range_idx

    def __len__(self):
        return len(self.token_list)

    def globalize(self, pts_temp, sensor):
        pts_temp = np.copy(pts_temp).T

        # here starts nuScenes routines
        pc = dc.LidarPointCloud(pts_temp)

        point_sens = self.nusc.get('sample_data', sensor)

        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get('calibrated_sensor', point_sens['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame
        poserecord = self.nusc.get('ego_pose', point_sens['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))
        np_points = torch.from_numpy(pc.points[:3, :].transpose())

        return np_points

    def get_location(self, scene_token):
        scene_record = self.nusc.get('scene', scene_token)
        location = self.nusc.get('log', scene_record['log_token'])['location']
        return location

    def interpolate_labels(self, ref_xyz, query_xyz, ref_labels, min_dist=0.01):

        ref_xyz = torch.from_numpy(ref_xyz).unsqueeze(0).cuda()
        query_xyz = torch.from_numpy(query_xyz).unsqueeze(0).cuda()
        ref_labels = torch.from_numpy(ref_labels)

        knn_dist, knn_idx = self.knn_search(ref_xyz, query_xyz)

        knn_idx = knn_idx.cpu().squeeze(0).long()
        knn_dist = knn_dist.cpu().squeeze(0).long()
        valid_dist = torch.logical_not(knn_dist < min_dist)
        query_labels = ref_labels[knn_idx].float()
        query_labels[valid_dist] = 0

        return query_labels.view(1, -1).numpy()

    def from_file_multisweep(self,
                             sample_rec,
                             chan,
                             ref_chan,
                             nsweeps,
                             min_distance):
            """
            Return a point cloud that aggregates multiple sweeps.
            As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
            As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
            :param sample_rec: The current sample.
            :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
            :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
            :param nsweeps: Number of sweeps to aggregated.
            :param min_distance: Distance below which points are discarded.
            :return: (all_pc, all_labels, all_times). The aggregated point cloud and timestamps.
            """
            # Init.
            points = np.zeros((dc.LidarPointCloud.nbr_dims(), 0), dtype=np.float32)
            all_pc = dc.LidarPointCloud(points)
            all_times = np.zeros((1, 0))
            all_labels = np.zeros((1, 0))

            # Get reference pose and timestamp.
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
            # lidarseg filename
            ref_sd_rec_label = self.nusc.get('lidarseg', ref_sd_token)['filename']
            # lidarseg path
            ref_sd_rec_label_filename = os.path.join(self.nusc.dataroot, ref_sd_rec_label)

            ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
            ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
            ref_time = 1e-6 * ref_sd_rec['timestamp']

            # Homogeneous transform from ego car frame to reference frame.
            ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

            # Homogeneous transformation matrix from global to _current_ ego car frame.
            car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                               inverse=True)

            # Aggregate current and previous sweeps.
            sample_data_token = sample_rec['data'][chan]
            current_sd_rec = self.nusc.get('sample_data', sample_data_token)

            ref_sd_rec_filename = os.path.join(self.nusc.dataroot, current_sd_rec['filename'])
            ref_points = np.fromfile(ref_sd_rec_filename, dtype=np.float32)
            ref_points = ref_points.reshape((-1, 5))[:, :3]
            ref_labels = np.fromfile(ref_sd_rec_label_filename, dtype=np.uint8)

            for si in range(nsweeps):
                # Load up the pointcloud and remove points close to the sensor.
                current_pc = dc.LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))

                if not current_sd_rec['is_key_frame']:
                    current_pc.remove_close(min_distance)

                # Get past pose.
                current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(current_pose_rec['translation'],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)

                # Fuse four transformation matrices into one and perform transform.
                trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                current_pc.transform(trans_matrix)

                # Add time vector which can be used as a temporal feature.
                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
                times = time_lag * np.ones((1, current_pc.nbr_points()))
                all_times = np.hstack((all_times, times))

                # Merge with key pc.
                all_pc.points = np.hstack((all_pc.points, current_pc.points))

                if not current_sd_rec['is_key_frame']:
                    # current_labels = self.interpolate_labels(ref_points,
                    #                                          np.asarray(current_pc.points).T[:, :3],
                    #                                          ref_labels).reshape([1, -1])
                    current_labels = np.zeros([1, current_pc.nbr_points()])
                    # ref_points = np.asarray(current_pc.points).T[:, :3]
                    # ref_labels = current_labels.reshape(-1)
                else:
                    current_labels = ref_labels.reshape([1, -1])

                all_labels = np.hstack((all_labels, current_labels))

                # Abort if there are no previous sweeps.
                if current_sd_rec['next'] == '':
                    break
                else:
                    current_sd_rec = self.nusc.get('sample_data', current_sd_rec['next'])

            return np.asarray(all_pc.points).T, all_labels.T.reshape(-1), all_times.T.reshape(-1)

    def get_frame(self, sample_token):

        # get sample record
        sample_record = self.nusc.get('sample', sample_token)
        lidar = sample_record['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar)
        lidar_file = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        _, name = os.path.split(lidar_file)
        gf_path = os.path.join(self.geometric_path, name[:-8]+'.npz')

        if self.interpolate:

            ref_chan = 'LIDAR_TOP'
            chan = lidar_data['channel']
            points, points_label, times = self.from_file_multisweep(sample_record,
                                                                    chan,
                                                                    ref_chan,
                                                                    self.num_sweeps,
                                                                    min_distance=1.0)
            points = np.ascontiguousarray(points)
            points_label = self.remap_lut_val[points_label.astype(np.long)]
        else:
            scan = np.fromfile(lidar_file, dtype=np.float32)
            points = scan.reshape((-1, 5))[:, :4]

            # lidarseg filename
            lidar_label_file = self.nusc.get('lidarseg', lidar)['filename']
            # lidarseg path
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot, lidar_label_file)

            if not os.path.exists(lidarseg_labels_filename):
                points_label = np.zeros(np.shape(points)[0], dtype=np.int32)
            else:
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
                points_label = self.remap_lut_val[points_label]

        global_points = self.globalize(points, lidar)

        pcd = points[:, :3]
        geometric_feats = np.load(gf_path)['features']

        if self.clip_range is not None:
            range_idx = self.check_range(pcd)
            pcd = pcd[range_idx]
            points_label = points_label[range_idx]
            global_points = global_points[range_idx]
            geometric_feats = geometric_feats[range_idx]

        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)

        data = {'points': pcd,
                'features': colors,
                'labels': points_label,
                'global_points': global_points,
                'geometric_feats': geometric_feats}

        # x = o3d.geometry.PointCloud()
        # x.points = o3d.utility.Vector3dVector(pcd)
        # x.colors = o3d.utility.Vector3dVector(self.color_map[points_label+1])
        #
        # o3d.io.write_point_cloud('aggregated.ply', x)
        # exit(0)
        return data

    def __getitem__(self, i):
        # get token
        sample_token = self.token_list[i]

        data = self.get_frame(sample_token)

        return data


def get_online_dataset(dataset_name: str,
                       dataset_path: str,
                       voxel_size: float = 0.02,
                       sub_num: int = 50000,
                       max_time_wdw: int = 1,
                       oracle_pts: int = 0,
                       augment_data: bool = False,
                       aug_parameters: dict = None,
                       version: str = 'mini',
                       ignore_label: int = -1,
                       split_size: int = 250,
                       mapping_path: str = None,
                       num_classes: int = 7,
                       noisy_odo: bool = False,
                       odo_roto_bounds: int = None,
                       odo_tras_bounds: float = None,
                       geometric_path: str = None) -> OnlineBaseDataset:

    if aug_parameters is not None:
        input_transforms = get_augmentations(aug_parameters)
    else:
        input_transforms = None

    if dataset_name == 'SemanticKITTI':

        if mapping_path is None:
            mapping_path = '_resources/semantic-kitti.yaml'

        if geometric_path is None:
            geometric_path = 'experiments/dip_features/08'

        print(f'--> USING GEOM PATH: {geometric_path}')

        online_dataset = OnlineSemanticKITTIDataset(dataset_path=dataset_path,
                                                    mapping_path=mapping_path,
                                                    version=version,
                                                    phase='eval',
                                                    sequence_idx=0,
                                                    voxel_size=voxel_size,
                                                    augment_data=augment_data,
                                                    input_transforms=input_transforms,
                                                    max_time_wdw=max_time_wdw,
                                                    oracle_pts=oracle_pts,
                                                    ignore_label=ignore_label,
                                                    split_size=split_size,
                                                    num_classes=num_classes,
                                                    noisy_odo=noisy_odo,
                                                    odo_roto_bounds=odo_roto_bounds,
                                                    odo_tras_bounds=odo_tras_bounds,
                                                    geometric_path=geometric_path)

    elif dataset_name == 'nuScenes':
        _version = 'v1.0-trainval' if version == 'full' else 'v1.0-mini'

        nusc = NuScenes(version=_version,
                        dataroot=dataset_path,
                        verbose=True)

        if mapping_path is None:
            mapping_path = '_resources/nuscenes.yaml'

        if geometric_path is None:
            geometric_path = 'experiments/dip_features/nuscenes'

        print(f'--> USING GEOM PATH: {geometric_path}')

        online_dataset = OnlineNuScenesDataset(nusc=nusc,
                                               mapping_path=mapping_path,
                                               version=version,
                                               phase='eval',
                                               voxel_size=voxel_size,
                                               augment_data=augment_data,
                                               input_transforms=input_transforms,
                                               sub_num=sub_num,
                                               max_time_wdw=max_time_wdw,
                                               oracle_pts=oracle_pts,
                                               ignore_label=ignore_label,
                                               num_classes=num_classes,
                                               geometric_path=geometric_path)

    else:
        raise NotImplementedError

    return online_dataset


class PairedOnlineDataset(object):

    def __init__(self, dataset, use_random=False):

        self.dataset = dataset
        self.max_time_wdw = dataset.max_time_wdw
        self.ignore_label = self.dataset.ignore_label
        self.use_random = use_random
        self.num_sequences = dataset.num_sequences

    def __getitem__(self, idx):
        if self.use_random:
            time_wdw = np.random.randint(self.max_time_wdw, size=1)[0]
        else:
            time_wdw = self.max_time_wdw
        # data = self.dataset.__getitem__(idx-time_wdw)
        # next_data = self.dataset.__getitem__(idx)
        data = self.dataset.__getitem__(idx)
        next_data = self.dataset.__getitem__(idx-time_wdw)
        pair = self.dataset.get_double_data(data, next_data)
        return pair

    def __len__(self):
        return len(self.dataset)


class FrameOnlineDataset(object):
    def __init__(self, dataset):

        self.dataset = dataset
        self.num_sequences = dataset.num_sequences
        self.ignore_label = self.dataset.ignore_label

    def __getitem__(self, idx):

        data = self.dataset.__getitem__(idx)
        pair = self.dataset.get_single_data(data)
        return pair

    def __len__(self):
        return len(self.dataset)


def get_augmentations(aug_dict: dict) -> Compose:
    aug_list = []
    for aug_name in aug_dict.keys():
        aug_class = getattr(utils.augmentations, aug_name)

        aug_list.append(aug_class(*aug_dict[aug_name]))
    return Compose(aug_list)


if __name__ == '__main__':
    # dataset = nuScenesDataset()
    #
    # data = dataset.__getitem__(1)
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data['coordinates'])
    # pcd.colors = o3d.utility.Vector3dVector(dataset.color_map[data['labels']])
    #
    # o3d.io.write_point_cloud('trial.ply', pcd)
    dataset = SynthDataset()

    data = dataset.__getitem__(1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data['coordinates'])
    pcd.colors = o3d.utility.Vector3dVector(dataset.color_map[data['labels']])
    #
    # o3d.io.write_point_cloud('trial.ply', pcd)
