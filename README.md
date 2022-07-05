# Official implementation for **GIPSO**

## [**[ECCV-2022] GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation** :fire:LINK COMING SOON!:fire:]()

NEWS:


## Installation
To run GIPSO you will nedd to first install:

PyTorch >=
Pytorch-Lighting 
[MinkowskiEnginge](https://github.com/NVIDIA/MinkowskiEngine)
[Open3D 0.13.0](http://www.open3d.org)
[KNN-CUDA](https://github.com/unlimblue/KNN_CUDA)

Docker ready-to-use container :fire: COMING SOON!:fire:!


## Synth4D Dataset
You can find the dataset at the following Drive link [Synth4D]()


## Data preparation

### SemanticKITTI
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

## Source training


## Target adaptation


## Evaluation


## Results


## Adapted models


## References
Reference will be uploaded after publication !:rocket:


## Acknowledgments








