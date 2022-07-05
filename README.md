# Official implementation for **GIPSO**

## [**[ECCV-2022] GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation** :fire:LINK COMING SOON!:fire:]()
![teaser](https://user-images.githubusercontent.com/56728964/177330335-83c056b8-141f-461f-9c7a-4f1948256b80.jpg)

## News

- 7/2022: GIPSO is accepted to ECCV 2022!:fire:
- 7/2022: GIPSO repository has been created! Our work is the first allowing source-free online and unsupervised adaptation for 3D semantic segmentation!

![video](assets/gipso_video.mp4)


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
If you need to save space you can remove ``sweeps`` as they are not used


## Source training

To train the source model on Synth4D run
```
python train_lighting.py --config_file configs/source/synthkitti_source.yaml
```
In the case of SynLiDAR use ``--config_file configs/source/synlidar_source.yaml`` and nuScenes ``--config_file configs/source/synthnusc_source.yaml``

**NB:** we provide pretrained models so you can skip this time consuming step!:rocket:

## Target adaptation
First we need to pre-compute geometric features by using [DIP](https://github.com/fabiopoiesi/dip). Download teh 3DMatch pretrained model from [here]() and put it in `` pretrained_models/dip_model/``.

Then, compute geometric features running

```
python
```
**NOTE:** geometric features could be computed also online but experiencing a slow down during adaptation!

To adapt the source model XXX to the target domain YYY you need to prepare the configuration file.

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python 
```
The adapted model will be saved in ``` ``` while evaluation results will be saved in ``` ```.
If you want also to save point cloud for future visualization you will need to add `` ``.

## Pretrained models

### Source
We provide pretrained source models on Synth4D-KITTI, Synth4D-nuScenes and SynLiDAR:
- [Synth4D-KITTI]()
- [Synth4D-nuScenes]()
- [SynLiDAR]()


## References
Reference will be uploaded after publication !:rocket:


## Acknowledgments

We thanks the open source projects [DIP](), [Minkowski-Engine](), [Open3D]() and [KNN-KUDA]()!







