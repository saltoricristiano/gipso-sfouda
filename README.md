# Official implementation for **GIPSO**

## [**[ECCV-2022] GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation** :fire:LINK COMING SOON!:fire:]()
![teaser](https://user-images.githubusercontent.com/56728964/177330335-83c056b8-141f-461f-9c7a-4f1948256b80.jpg)

## News

- 7/2022: GIPSO is accepted to ECCV 2022!:fire:
- 7/2022: GIPSO repository has been created! Our work is the first allowing source-free online and unsupervised adaptation for 3D semantic segmentation!



## Installation
To run GIPSO you will nedd to first install:

- PyTorch 1.8.0
- Pytorch-Lighting 1.4.1
- [MinkowskiEnginge](https://github.com/NVIDIA/MinkowskiEngine)
- [Open3D 0.13.0](http://www.open3d.org)
- [KNN-CUDA](https://github.com/unlimblue/KNN_CUDA)

Docker ready-to-use container coming soon!


## Synth4D Dataset
You can find our proposed Synth4D dataset at the following Drive link [Synth4D]().


## Data preparation

### Synth4D
Download the Synth4D dataset following the above instructions and prepare the dataset paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──kitti_synth
        |   ├──Town03
        |   |     ├── calib
        |   |     |    ├── 000000.npy
        |   |     |    └── ... 
        |   |     ├── labels
        |   |     |    ├── 000000.npy
        |   |     |    └── ...
        |   |     └── velodyne
        |   |         ├── 000000.npy
        |   |         └── ...
        |   ├──Town06
        |   ├──Town07
        |   └──Town10HD
        |
		└──nuscenes_synth
```


### SynLiDAR
Download SynLiDAR dataset from [here](), then prepare data folders as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        └── 12/
```

### SemanticKITTI
To download SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   ├── labels/ 
        |   |   ├── 000000.label
        |   |   ├── 000001.label
        |   |   └── ...
            ├── calib.txt
            ├── poses.txt
            └── times.txt
        └── 08/ # for validation
```

### nuScenes
Follow the instructions [here](https://www.nuscenes.org/nuscenes#download) to download the data and paths will be already like that:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──v1.0-trainval
		├──v1.0-test
		├──samples
		├──sweeps
		├──maps
		└──lidarseg
```
If you need to save space on your server you can remove ``sweeps`` as they are not used.


## Source training

To train the source model on Synth4D
```
python train_lighting.py --config_file configs/source/synth4dkitti_source.yaml
```
In the case of SynLiDAR use ``--config_file configs/source/synlidar_source.yaml`` and nuScenes ``--config_file configs/source/synth4dnusc_source.yaml``

**NB:** we provide pretrained models in ```pretrained_models```, so you can skip this time consuming step!:rocket:

## Preprocess geometric features
First we need to pre-compute geometric features by using [DIP](https://github.com/fabiopoiesi/dip). This step will use the pretrained model in ```pretrained_models/dip_model```.

To compute geometric features on SemanticKITTI

```
python compute_dip_features_kitti.py --source_path PATH/TO/SEMANTICKITTI/IN/CONFIGS
```
while to compute geometric features on nuScenes
```
python compute_dip_features_nuscenes.py --source_path PATH/TO/NUSCENES/IN/CONFIGS
```

This will save geometric features in ```experiments/dip_features/semantickitti``` and ```experiments/dip_features/nuscenes```, respectively.
If you want to change features path add ```---save_path PATH/TO/SAVE/FEATURES```.

## Adaptation to target

To adapt the source model Synth4DKITTI to the target domain SemanticKITTI

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python adapt_online_lighting.py --config_file configs/adaptation/synth4d2kitti_adaptation.yaml --geometric_path experiments/dip_features/semantickitti 
```
The adapted model will be saved following config file in ```pipeline.save_dir``` together with evaluation results.
If you want to save point cloud for future visualization you will need to add ``--save_predictions``.

## References
Reference will be uploaded after publication !:rocket:

## Acknowledgments

We thanks the open source projects [DIP](https://github.com/fabiopoiesi/dip), [Minkowski-Engine](https://github.com/NVIDIA/MinkowskiEngine), and [KNN-KUDA](https://github.com/unlimblue/KNN_CUDA)!







