import os
import torch
import yaml
import pickle
import numpy as np
import open3d as o3d
import tqdm
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from torch.utils.data import Dataset
import MinkowskiEngine as ME
import utils
from utils.augmentations import Compose
from utils.voxelizer import Voxelizer

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class BaseDataset(Dataset):
    def __init__(self,
                 version: str,
                 phase: str,
                 dataset_path: str,
                 voxel_size: float = 0.05,
                 sub_num: int = 50000,
                 use_intensity: bool = False,
                 augment_data: bool = False,
                 input_transforms: Compose = None,
                 num_classes: int = 7,
                 ignore_label: int = None,
                 device: str = None):

        self.CACHE = {}
        self.version = version
        self.phase = phase
        self.dataset_path = dataset_path
        self.voxel_size = voxel_size  # in meter
        self.sub_num = sub_num
        self.use_intensity = use_intensity
        self.augment_data = augment_data and self.phase == 'train'
        self.input_transforms = input_transforms
        self.num_classes = num_classes

        self.ignore_label = ignore_label

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

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
                                   ignore_label=vox_ign_label)

        self.device = device

        self.split = {'train': [],
                      'validation': []}

        self.maps = None

        if self.num_classes == 7:
            # unique color map among all the datasets
            self.color_map = np.array([(255, 255, 255),   # unlabelled - white
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
        elif self.num_classes == 3:

                self.color_map = np.array([(0, 0, 0),   # unlabelled - white
                                           (25, 25, 255),    # vehicle - blue
                                           (250, 178, 50)]) / 255.0   # pedestrian - orange

                self.class2names = np.array(['background',
                                             'vehicle',
                                             'pedestrian'])
        else:
            # unique color map among all the datasets
            self.color_map = np.array([(25, 25, 255),    # vehicle - blue
                                       (250, 178, 50)]) / 255.0   # pedestrian - orange

            self.class2names = np.array(['vehicle',
                                         'pedestrian'])

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

        if self.sub_num is not None:
            if self.sub_num <= num_points:
                sampled_idx = np.random.choice(np.arange(num_points), self.sub_num, replace=False)
            else:
                over_idx = np.random.choice(np.arange(num_points), self.sub_num - num_points, replace=False)
                sampled_idx = np.concatenate([np.arange(num_points), over_idx])
        else:
            sampled_idx = np.arange(num_points)

        np.random.shuffle(sampled_idx)

        return sampled_idx


class SemanticKITTIDataset(BaseDataset):
    def __init__(self,
                 version='full',
                 phase='train',
                 dataset_path='/data/csaltori/SemanticKITTI/data/sequences',
                 mapping_path='_resources/semantic-kitti.yaml',
                 weights_path='_weights/',
                 voxel_size=0.05,
                 use_intensity=False,
                 augment_data=False,
                 input_transforms=None,
                 sub_num=50000,
                 device=None,
                 num_classes=7,
                 ignore_label=None):

        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         input_transforms=input_transforms,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label)

        if self.version == 'full':
            self.split = {'train': ['00', '01', '02', '03', '04', '05',
                                    '06', '07', '09', '10'],
                          'validation': ['08']}
        elif self.version == 'mini':
            self.split = {'train': ['00', '01'],
                          'validation': ['08']}
        else:
            raise NotImplementedError

        self.name = 'SemanticKITTIDataset'
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        for sequence in self.split[self.phase]:
            num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))

            for f in np.arange(num_frames):
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(f):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(f):06d}.label')
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)

        weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        self.weights_path = os.path.join(weights_path, 'semantic_kitti'+'_'+self.version+'.npy')

        if self.phase == 'train':
            if os.path.isfile(self.weights_path):
                self.weights = np.load(self.weights_path)
            else:
                os.makedirs(weights_path, exist_ok=True)
                self.weights = self.get_dataset_weights()
                np.save(self.weights_path, self.weights)

    def __len__(self):
        return len(self.pcd_path)

    def __getitem__(self, i):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        if pcd_tmp not in self.CACHE:
            pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
            label = self.load_label_kitti(label_tmp)
            points = pcd[:, :3]
            if self.use_intensity:
                colors = points[:, 3][..., np.newaxis]
            else:
                colors = np.ones((points.shape[0], 1), dtype=np.float32)
            data = {'points': points, 'colors': colors, 'labels': label}
            self.CACHE[pcd_tmp] = data

        data = self.CACHE[pcd_tmp]

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)

            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats, labels = ME.utils.sparse_quantize(points,
                                                                   colors,
                                                                   labels=labels,
                                                                   ignore_label=vox_ign_label,
                                                                   quantization_size=self.voxel_size)

        if self.input_transforms is not None:
            quantized_coords, feats, labels = self.input_transforms(quantized_coords, feats, labels)

        return {"coordinates": quantized_coords,
                "features": torch.from_numpy(feats),
                "labels": torch.from_numpy(labels)}

    def load_label_kitti(self, label_path):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = self.remap_lut_val[sem_label]
        return sem_label.astype(np.int32)

    def get_dataset_weights(self):
        weights = np.zeros(self.remap_lut_val.max()+1)
        for l in tqdm.tqdm(range(len(self.label_path)), desc='Loading weights', leave=True):
            label_tmp = self.label_path[l]
            label = self.load_label_kitti(label_tmp)
            lbl, count = np.unique(label, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count

        return weights


class nuScenesDataset(BaseDataset):

    def __init__(self,
                 nusc=None,
                 version='full',
                 phase='train',
                 dataset_path='/data/csaltori/nuScenes-lidarseg/',
                 weight_path='_weights',
                 mapping_path='_resources/nuscenes.yaml',
                 voxel_size=0.05,
                 sub_num=50000,
                 device=None,
                 use_intensity=False,
                 augment_data=False,
                 input_transforms=None,
                 num_classes=7,
                 ignore_label=None):

        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         input_transforms=input_transforms,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label)

        self.version = 'v1.0-trainval' if self.version == 'full' else 'v1.0-mini'
        self.name = 'nuScenesDataset'
        if not nusc:
            self.nusc = NuScenes(version=self.version,
                                 dataroot=self.dataset_path,
                                 verbose=True)
        else:
            self.nusc = nusc

        splits = create_splits_scenes()
        if self.version == 'v1.0-trainval':
            self.split = {'train': splits['train'],
                          'validation': splits['val']}
        else:
            self.split = {'train': splits['mini_train'],
                          'validation': splits['mini_val']}

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        self.token_list = []

        scenes_tokens = {}
        scenes = self.nusc.scene

        for s in scenes:
            # list tokens of the scenes
            scenes_tokens[s['name']] = s['token']

        for scene in self.split[self.phase]:
            # init empty lists
            token_list_seq = []

            # get scene token and scene
            scene_token = scenes_tokens[scene]
            scene_temp = self.nusc.get('scene', scene_token)

            # get first sample(frame) token
            sample_token = scene_temp['first_sample_token']

            # iterate over samples given tokens
            while sample_token != '':

                # append token
                token_list_seq.append(sample_token)

                # get sample record
                sample_record = self.nusc.get('sample', sample_token)

                # update sample token with the next
                sample_token = sample_record['next']

            self.token_list.extend(token_list_seq)

        weight_path = os.path.join(ABSOLUTE_PATH, weight_path)
        self.weight_path = os.path.join(weight_path, 'nuscenes_'+self.version+'.npy')

        if self.phase == 'train':
            if os.path.isfile(self.weight_path):
                self.weights = np.load(self.weight_path)
            else:
                os.makedirs(weight_path, exist_ok=True)
                self.weights = self.get_dataset_weights()
                np.save(self.weight_path, self.weights)

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, i):
        # get token
        sample_token = self.token_list[i]

        # get sample record
        sample_record = self.nusc.get('sample', sample_token)

        # get sensor token for given sample record
        lidar = sample_record['data']['LIDAR_TOP']

        # get sample data of the lidar sensor
        lidar_data = self.nusc.get('sample_data', lidar)

        # get lidar path
        lidar_file = os.path.join(self.nusc.dataroot, lidar_data['filename'])

        # we load points and get numpy ndarray
        scan = np.fromfile(lidar_file, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :4]

        # lidarseg filename
        lidar_label_file = self.nusc.get('lidarseg', lidar)['filename']
        # lidarseg path
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot, lidar_label_file)

        if sample_token not in self.CACHE:
            if not os.path.exists(lidarseg_labels_filename):
                points_label = np.zeros(np.shape(points)[0], dtype=np.int32)
            else:
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
                points_label = self.remap_lut_val[points_label]

            pcd = points[:, :3]
            if self.use_intensity:
                colors = points[:, 3][..., np.newaxis]
            else:
                colors = np.ones((points.shape[0], 1), dtype=np.float32)

            data = {'points': pcd, 'colors': colors, 'labels': points_label}
            self.CACHE[sample_token] = data

        data = self.CACHE[sample_token]

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)

            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        # quantized_coords, feats, labels = self.voxelizer.voxelize(points, colors, labels)

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats, labels = ME.utils.sparse_quantize(points,
                                                                   colors,
                                                                   labels=labels,
                                                                   ignore_label=vox_ign_label,
                                                                   quantization_size=self.voxel_size)

        return {"coordinates": quantized_coords,
                "features": feats,
                "labels": labels}

    def get_dataset_weights(self):
        weights = np.zeros(self.remap_lut_val.max()+1)

        for l in tqdm.tqdm(range(len(self.token_list)), desc='Loading weights', leave=True):
            # get token
            sample_token = self.token_list[l]

            # get sample record
            sample_record = self.nusc.get('sample', sample_token)

            # get sensor token for given sample record
            lidar = sample_record['data']['LIDAR_TOP']
            # lidarseg filename
            lidar_label_file = self.nusc.get('lidarseg', lidar)['filename']
            # lidarseg path
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot, lidar_label_file)

            if os.path.exists(lidarseg_labels_filename):
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
                points_label = self.remap_lut_val[points_label]
                lbl, count = np.unique(points_label, return_counts=True)
                if self.ignore_label is not None:
                    if self.ignore_label in lbl:
                        count = count[lbl != self.ignore_label]
                        lbl = lbl[lbl != self.ignore_label]
                weights[lbl] += count

        return weights


class SynthDataset(BaseDataset):
    def __init__(self,
                 version='full',
                 sensor='hdl64e',
                 phase='train',
                 dataset_path='/data/csaltori/CARLA/',
                 weight_path='_weights',
                 split_path='/data/csaltori/CARLA/splits/nuscenes_synth/',
                 mapping_path='_resources/synthetic.yaml',
                 voxel_size=0.05,
                 sub_num=50000,
                 use_intensity=False,
                 augment_data=False,
                 input_transforms=None,
                 device=None,
                 num_classes=7,
                 ignore_label=None):

        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         input_transforms=input_transforms,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label)

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        split = 'training_split' if phase == 'train' else 'validation_split'
        split_path = os.path.join(split_path, split)

        self.split = load_obj(split_path)
        self.sensor = sensor

        if self.sensor == 'hdl64e':
            self.name = 'SyntheticKITTIDataset'
            self.dataset_path = os.path.join(self.dataset_path, 'kitti_synth')
        elif self.sensor == 'hdl32e':
            self.name = 'SyntheticNuScenesDataset'
            self.dataset_path = os.path.join(self.dataset_path, 'nuscenes_synth')
        else:
            raise NotImplementedError

        if self.version == 'mini':
            _split = {}
            for town in self.split.keys():
                _split[town] = np.random.choice(self.split[town], 100)
            self.split = _split

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        self.path_list = []

        for town in self.split.keys():
            pc_path = os.path.join(self.dataset_path, town, 'velodyne')
            self.path_list.extend([os.path.join(pc_path, str(f)+'.npy') for f in np.sort(self.split[town])])

        weight_path = os.path.join(ABSOLUTE_PATH, weight_path)
        self.weight_path = os.path.join(weight_path, 'synthetic'+self.sensor+'_'+self.version+'.npy')

        if self.phase == 'train':
            if os.path.isfile(self.weight_path):
                self.weights = np.load(self.weight_path)
            else:
                os.makedirs(weight_path, exist_ok=True)
                self.weights = self.get_dataset_weights()
                np.save(self.weight_path, self.weights)

    def __getitem__(self, i):
        pc_path = self.path_list[i]
        points = np.load(pc_path).astype(np.float32)

        dir, file = os.path.split(pc_path)
        label_path = os.path.join(dir, '../labels', file[:-4] + '.npy')

        if pc_path not in self.CACHE:

            if not os.path.exists(label_path):
                labels = np.zeros(np.shape(points)[0], dtype=np.int32)
            else:
                labels = np.load(label_path).astype(np.int32).reshape([-1])
                labels = self.remap_lut_val[labels]

            pcd = points[:, :3]

            if self.use_intensity:
                colors = points[:, 3][..., np.newaxis]
            else:
                colors = np.ones((points.shape[0], 1), dtype=np.float32)

            data = {'points': pcd, 'colors': colors, 'labels': labels}
            self.CACHE[pc_path] = data

        data = self.CACHE[pc_path]

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)

            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats, labels = ME.utils.sparse_quantize(points,
                                                                   colors,
                                                                   labels=labels,
                                                                   ignore_label=vox_ign_label,
                                                                   quantization_size=self.voxel_size)

        if self.input_transforms is not None:
            quantized_coords, feats, labels = self.input_transforms(quantized_coords, feats, labels)

        return {"coordinates": quantized_coords,
                "features": torch.from_numpy(feats),
                "labels": torch.from_numpy(labels)}

    def __len__(self):
        return len(self.path_list)

    def get_dataset_weights(self):
        weights = np.zeros(self.remap_lut_val.max()+1)

        for l in tqdm.tqdm(range(len(self.path_list)), desc='Loading weights', leave=True):
            pc_path = self.path_list[l]

            dir, file = os.path.split(pc_path)
            label_path = os.path.join(dir, '../labels', file[:-4] + '.npy')

            if pc_path not in self.CACHE:

                if os.path.exists(label_path):
                    labels = np.load(label_path).astype(np.int32).reshape([-1])
                    labels = self.remap_lut_val[labels]
                    lbl, count = np.unique(labels, return_counts=True)
                    if self.ignore_label is not None:
                        if self.ignore_label in lbl:
                            count = count[lbl != self.ignore_label]
                            lbl = lbl[lbl != self.ignore_label]
                    weights[lbl] += count

        return weights


class SynLiDARDataset(BaseDataset):
    def __init__(self,
                 version='full',
                 phase='train',
                 dataset_path='/data/csaltori/SynLiDAR/',
                 mapping_path='_resources/synlidar.yaml',
                 weights_path='_weights/',
                 voxel_size=0.05,
                 use_intensity=False,
                 augment_data=False,
                 input_transforms=None,
                 sub_num=50000,
                 device=None,
                 num_classes=7,
                 ignore_label=None):

        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         input_transforms=input_transforms,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label)

        self.name = 'SynLiDARDataset'
        if self.version == 'full':
            self.split = {'train': ['01', '02', '03', '05',
                                    '06', '08', '09', '10', '11', '12'],
                          'validation': ['00', '04', '07']}
        elif self.version == 'subset':
            self.split = {'train': ['01', '03', '06', '09'],
                          'validation': ['12']}
        elif self.version == 'mini':
            self.split = {'train': ['03'],
                          'validation': ['07']}
        else:
            raise NotImplementedError

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val
        skipped = 0
        for sequence in self.split[self.phase]:
            num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))

            for f in np.arange(num_frames):

                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(f):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(f):06d}.label')
                if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                    self.pcd_path.append(pcd_path)
                    self.label_path.append(label_path)
                else:
                    skipped += 1
                    print(f'--> Skipping {pcd_path} not found!')
        print(f'--> Total skipped {skipped} !!')
        # weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        # self.weights_path = os.path.join(weights_path, 'synlidar'+'_'+self.version+'.npy')
        #
        # if self.phase == 'train':
        #     if os.path.isfile(self.weights_path):
        #         self.weights = np.load(self.weights_path)
        #     else:
        #         os.makedirs(weights_path, exist_ok=True)
        #         self.weights = self.get_dataset_weights()
        #         np.save(self.weights_path, self.weights)

    def __len__(self):
        return len(self.pcd_path)

    def __getitem__(self, i):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        if pcd_tmp not in self.CACHE:
            pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
            label = np.fromfile(label_tmp, dtype=np.uint32)
            label = label.reshape((-1))
            label = self.remap_lut_val[label].astype(np.int32)
            points = pcd[:, :3]
            if self.use_intensity:
                colors = pcd[:, 3][..., np.newaxis]
            else:
                colors = np.ones((points.shape[0], 1), dtype=np.float32)
            data = {'points': points, 'colors': colors, 'labels': label}
            self.CACHE[pcd_tmp] = data

        data = self.CACHE[pcd_tmp]

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)

            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats, labels = ME.utils.sparse_quantize(points,
                                                                   colors,
                                                                   labels=labels,
                                                                   ignore_label=vox_ign_label,
                                                                   quantization_size=self.voxel_size)

        if self.input_transforms is not None:
            quantized_coords, feats, labels = self.input_transforms(quantized_coords, feats, labels)

        return {"coordinates": quantized_coords,
                "features": torch.from_numpy(feats),
                "labels": torch.from_numpy(labels)}


def get_dataset(dataset_name: str,
                dataset_path: str,
                voxel_size: float = 0.02,
                sub_num: int = 50000,
                augment_data: bool = False,
                aug_parameters: dict = None,
                version: str = 'mini',
                num_classes: int = 7,
                ignore_label: int = -1,
                get_target: bool = True,
                target_dataset_path: str = None,
                mapping_path: str = None) -> (BaseDataset, BaseDataset):

    if aug_parameters is not None:
        input_transforms = get_augmentations(aug_parameters)
    else:
        input_transforms = None

    target_dataset = None

    if dataset_name == 'SemanticKITTI':
        if mapping_path is None:
            mapping_path = '_resources/semantic-kitti.yaml'

        training_dataset = SemanticKITTIDataset(dataset_path=dataset_path,
                                                mapping_path=mapping_path,
                                                version=version,
                                                phase='train',
                                                voxel_size=voxel_size,
                                                augment_data=augment_data,
                                                input_transforms=input_transforms,
                                                sub_num=sub_num,
                                                num_classes=num_classes,
                                                ignore_label=ignore_label)
        validation_dataset = SemanticKITTIDataset(dataset_path=dataset_path,
                                                  mapping_path=mapping_path,
                                                  version=version,
                                                  phase='validation',
                                                  voxel_size=voxel_size,
                                                  augment_data=False,
                                                  num_classes=num_classes,
                                                  ignore_label=ignore_label)
    elif dataset_name == 'nuScenes':
        version = 'v1.0-mini' if version == 'mini' else 'v1.0-trainval'

        if mapping_path is None:
            mapping_path = '_resources/nuscenes.yaml'

        nusc = NuScenes(version=version,
                        dataroot=dataset_path,
                        verbose=True)

        training_dataset = nuScenesDataset(nusc=nusc,
                                           version=version,
                                           mapping_path=mapping_path,
                                           phase='train',
                                           voxel_size=voxel_size,
                                           augment_data=augment_data,
                                           input_transforms=input_transforms,
                                           sub_num=sub_num,
                                           num_classes=num_classes,
                                           ignore_label=ignore_label)

        validation_dataset = nuScenesDataset(nusc=nusc,
                                             version=version,
                                             mapping_path=mapping_path,
                                             phase='validation',
                                             voxel_size=voxel_size,
                                             augment_data=False,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label)
    elif dataset_name == 'SyntheticKITTI':
        split_path = os.path.join(dataset_path, 'splits', 'kitti_synth')

        if mapping_path is None:
            mapping_path = '_resources/synthetic.yaml'
            target_mapping_path = '_resources/semantic-kitti.yaml'
        else:
            if ignore_label is None:
                target_mapping_path = '_resources_three/semantic-kitti.yaml'
            else:
                target_mapping_path = '_resources_three_noback/semantic-kitti.yaml'

        training_dataset = SynthDataset(dataset_path=dataset_path,
                                        version=version,
                                        split_path=split_path,
                                        mapping_path=mapping_path,
                                        phase='train',
                                        sensor='hdl64e',
                                        voxel_size=voxel_size,
                                        augment_data=augment_data,
                                        input_transforms=input_transforms,
                                        sub_num=sub_num,
                                        num_classes=num_classes,
                                        ignore_label=ignore_label)

        validation_dataset = SynthDataset(dataset_path=dataset_path,
                                          version=version,
                                          split_path=split_path,
                                          mapping_path=mapping_path,
                                          phase='validation',
                                          sensor='hdl64e',
                                          voxel_size=voxel_size,
                                          augment_data=False,
                                          num_classes=num_classes,
                                          ignore_label=ignore_label)
        if get_target:

            target_dataset = SemanticKITTIDataset(dataset_path=target_dataset_path,
                                                  mapping_path=target_mapping_path,
                                                  version=version,
                                                  phase='validation',
                                                  voxel_size=voxel_size,
                                                  augment_data=False,
                                                  num_classes=num_classes,
                                                  ignore_label=ignore_label)

    elif dataset_name == 'SyntheticNuScenes':
        split_path = os.path.join(dataset_path, 'splits', 'nuscenes_synth')

        if mapping_path is None:
            mapping_path = '_resources/synthetic.yaml'
            target_mapping_path = '_resources/nuscenes.yaml'
        else:
            if ignore_label is None:
                target_mapping_path = '_resources_three/nuscenes.yaml'
            else:
                target_mapping_path = '_resources_three_noback/nuscenes.yaml'

        training_dataset = SynthDataset(dataset_path=dataset_path,
                                        split_path=split_path,
                                        mapping_path=mapping_path,
                                        version=version,
                                        phase='train',
                                        sensor='hdl32e',
                                        voxel_size=voxel_size,
                                        augment_data=augment_data,
                                        input_transforms=input_transforms,
                                        sub_num=sub_num,
                                        num_classes=num_classes,
                                        ignore_label=ignore_label)

        validation_dataset = SynthDataset(dataset_path=dataset_path,
                                          split_path=split_path,
                                          mapping_path=mapping_path,
                                          version=version,
                                          phase='validation',
                                          sensor='hdl32e',
                                          voxel_size=voxel_size,
                                          augment_data=False,
                                          num_classes=num_classes,
                                          ignore_label=ignore_label)

        if get_target:
            version = 'v1.0-mini' if version == 'mini' else 'v1.0-trainval'

            nusc = NuScenes(version=version,
                            dataroot=target_dataset_path,
                            verbose=True)

            target_dataset = nuScenesDataset(nusc=nusc,
                                             version=version,
                                             mapping_path=target_mapping_path,
                                             phase='validation',
                                             voxel_size=voxel_size,
                                             augment_data=False,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label)
    elif dataset_name == 'SynLiDAR':

        if mapping_path is None:
            mapping_path = '_resources/synlidar.yaml'
            target_mapping_path = '_resources/semantic-kitti.yaml'
        else:
            if ignore_label is None:
                target_mapping_path = '_resources_three/semantic-kitti.yaml'
            else:
                target_mapping_path = '_resources_three_noback/semantic-kitti.yaml'

        training_dataset = SynLiDARDataset(dataset_path=dataset_path,
                                           version=version,
                                           phase='train',
                                           voxel_size=voxel_size,
                                           augment_data=augment_data,
                                           num_classes=num_classes,
                                           ignore_label=ignore_label,
                                           mapping_path=mapping_path)

        validation_dataset = SynLiDARDataset(dataset_path=dataset_path,
                                             version=version,
                                             phase='validation',
                                             voxel_size=voxel_size,
                                             augment_data=False,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label,
                                             mapping_path=mapping_path)
        if get_target:

            target_dataset = SemanticKITTIDataset(dataset_path=target_dataset_path,
                                                  mapping_path=target_mapping_path,
                                                  version=version,
                                                  phase='validation',
                                                  voxel_size=voxel_size,
                                                  augment_data=False,
                                                  num_classes=num_classes,
                                                  ignore_label=ignore_label)

    else:
        raise NotImplementedError

    return training_dataset, validation_dataset, target_dataset


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
    # dataset = SynthDataset()
    #
    # data = dataset.__getitem__(1)
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data['coordinates'])
    # pcd.colors = o3d.utility.Vector3dVector(dataset.color_map[data['labels']])
    #
    # o3d.io.write_point_cloud('trial.ply', pcd)
    dataset = SynLiDARDataset()

    data = dataset.__getitem__(1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data['coordinates'])
    pcd.colors = o3d.utility.Vector3dVector(dataset.color_map[data['labels']+1])

    o3d.io.write_point_cloud('trial_synlidar.ply', pcd)
