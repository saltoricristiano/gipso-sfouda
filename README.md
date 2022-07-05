# Official implementation for **GIPSO**

## [**[ECCV-2022] GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation** :fire:LINK COMING SOON!:fire:]()

## Installation
To run GIPSO you will nedd to first install:

- PyTorch 1.8.0
- Pytorch-Lighting 1.4.1
- [MinkowskiEnginge](https://github.com/NVIDIA/MinkowskiEngine)
- [Open3D 0.13.0](http://www.open3d.org)
- [KNN-CUDA](https://github.com/unlimblue/KNN_CUDA)

Docker ready-to-use container **COMING SOON**!


## Synth4D Dataset
You can find the dataset at the following Drive link [Synth4D]()


## Data preparation

### Synth4D
Download the Synth4D dataset following the above instructions and prepare the dataset paths as follows:
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

```


### SynLIDAR
Download SynLiDAR dataset from [here](), then prepare data folders as follows:
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

```
### SemanticKITTI
To donwload SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
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
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
Follow the instructions [here]() to download the data and paths will be already like that:
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
		├──
		

```


## Source training

To train the source model on Synth4D run
```
python 
```
In the case of SynLiDAR use ``--config_file `` and nuScenes ``--config_file ``
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
python 
```


## Evaluation
To evaluate the source model run 
```
python 
```

To evaluate the target model run
```
python 
```

If you want also to save the results for visualization you will need to add `` ``

## Pretrained models

### Source
We provide pretrained source models on Synth4D and SynLiDAR:
- [Synth4D]()
- [SynLiDAR]()


## References
Reference will be uploaded after publication !:rocket:


## Acknowledgments

We thanks the open source projects [DIP](), [Minkowski-Engine](), [Open3D]() and [KNN-KUDA]()!







