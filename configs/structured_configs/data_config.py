from typing import Tuple, Optional, Any, List
from dataclasses import dataclass, field
from omegaconf import MISSING
import torch


@dataclass
class VisVolume:
    """Volume extents and resolution used to visualize dataset samples in 3D."""
    X_RANGE: Tuple[float, float] = (-9, 9)
    Y_RANGE: Tuple[float, float] = (.0, .75)
    Z_RANGE: Tuple[float, float] = (21, 3)
    # Resolution along the X, Y and Z axes
    X_RES: int = 256
    Y_RES: int = 64
    Z_RES: int = 256
    # Points-per-meter used to define PPM-specific resolutions.
    PPM: int = 10
    @property
    def X_RES_PPM(self):
        return int((self.X_RANGE[1]-self.X_RANGE[0]) * self.PPM)
    @property
    def Y_RES_PPM(self):
        return int((self.Y_RANGE[1]-self.Y_RANGE[0]) * self.PPM)
    @property
    def Z_RES_PPM(self):
        return int((self.Z_RANGE[1]-self.Z_RANGE[0]) * self.PPM)


@dataclass
class DatasetConfig:
    """Fields that are shared accross all datasets."""
    # The dataset is registered in the Hydra config store under the type string.
    type: str = MISSING
    data_path: str = MISSING
    skip: int = 0
    image_size: Tuple[int, int] = MISSING
    # How many consecutive frames to return.
    frame_count: int = 1
    # Whether to return depth.
    return_depth: bool = False
    # Whether to return predicted depths. Not implemented in all datasets 
    # and needs preprocessing.
    return_predicted_depth: bool = False
    predicted_depth_path: Optional[str] = None
    # SE(4) matrix to transform an inclined camera coordinate system to an un-inclined one.
    # Cannot add `torch.Tensor` or `Any` type annotation because Hydra will complain.
    CAM_INCL_ADJUST = torch.eye(4)
    # Dataset shortening arguments.
    MAX_TRAIN_DATASET_LEN: Optional[int] = None
    MAX_VAL_DATASET_LEN: Optional[int] = 256
    MAX_TEST_DATASET_LEN: Optional[int] = None
    USE_TRAIN_FOR_VAL: bool = False 
    USE_VAL_FOR_TEST: bool = False
    TRAIN_SAMPLES_POLICY: str = "slice"
    VAL_SAMPLES_POLICY: str = "eqspaced"
    TEST_SAMPLES_POLICY: str = "eqspaced"
    # Augmentations
    color_aug: bool = False
    # We may want to visualize different volumes in different datasets.
    VIS_VOLUME: VisVolume = VisVolume()
    
    
@dataclass
class ConcatDatasetConfig:
    type: str = MISSING
    DATASETS: List[DatasetConfig] = MISSING
    
    
@dataclass
class DataConfigCO3D(DatasetConfig):
    type: str = "CO3D"
    data_path: str = "data/CO3D"
    image_size: Tuple[int, int] = field(default_factory=lambda: [192, 192])
    MAX_VAL_DATASET_LEN = 64
     
    CATEGORY_NAME: str = "hydrant"
    SUBSET_NAME: str = "fewview_train"
    BOX_CROP_CONTEXT: float = 1.0
    BOX_CROP_POLICY: str = "fg_mask"
    APPLY_ADJUST_UPRIGHT: bool = True
    RETURN_POINTCLOUD: bool = False
    RETURN_MASK: bool = False
    MAX_POINTS: Optional[int] = 10_000
    RETURN_POLICY: str = "frame"
    VIS_VOLUME: VisVolume = VisVolume(
        X_RANGE=(-1, 1), 
        Y_RANGE=(-1., 1.), 
        Z_RANGE=(0.5, 3),
        PPM=32,    
    )
    

@dataclass
class DataConfigConcatCO3D(ConcatDatasetConfig):
    type: str = "ConcatCO3D"
    DATASETS: List[DatasetConfig] = field(default_factory=lambda: [
        DataConfigCO3D(CATEGORY_NAME="hydrant"),
        DataConfigCO3D(CATEGORY_NAME="motorcycle"),
    ])
    
    
@dataclass   
class DataConfigKitti360(DatasetConfig):
    type: str = "KITTI_360"
    data_path: str = "data/KITTI-360"
    pose_path: str = "data/KITTI-360/data_poses"
    split_path: str = "datasets/kitti_360/splits/seg"
    image_size: Tuple[int, int] = field(default_factory=lambda: [ 192, 640 ])
    # KITTI 360 cameras have a 5 degrees negative inclination.
    CAM_INCL_ADJUST = torch.tensor([
        [1.0000000,  0.0000000,  0.0000000, 0],
        [0.0000000,  0.9961947, -0.0871557, 0],
        [0.0000000,  0.0871557,  0.9961947, 0],
        [0.0000000,  000000000,  0.0000000, 1]
    ])
    
    dilation: int = 1
    
    return_stereo: bool = False
    return_fisheye: bool = False
    # Whether to load a pre-computed depth image (e.g. from a depth predictor) per stereo-image 
    # from the given path. Requires the pre-computing step to be run beforehand.
    return_predicted_depth: bool = False
    predicted_depth_path: Optional[str] = None
    # Int ot List of ints
    fisheye_rotation: Any = (0, -15)
    fisheye_offset: int = 0
    # Whether the images have been pre-sampled to the correct resolution and labelled, e.g. data_192x640.
    is_preprocessed: bool = False
    
    color_aug: bool = False
    
    VIS_VOLUME: VisVolume = VisVolume(X_RANGE=(-12, 12), Y_RANGE=(0, .75), Z_RANGE=(0, 30), PPM=10)


@dataclass
class DataConfigDAVIS(DatasetConfig):
    type: str = "DAVIS"
    data_path: str = "data/DAVIS"
    image_size: Tuple[int, int] = field(default_factory=lambda: [192, 640])
    MAX_VAL_DATASET_LEN = 64
     
    VIS_VOLUME: VisVolume = VisVolume()


@dataclass   
class DataConfigWaymo(DatasetConfig):
    type: str = "Waymo"
    data_path: str = "data/waymo"
    split_path: str = "datasets/waymo/splits/mvs_dayonly"
    image_size: Tuple[int, int] = field(default_factory=lambda: [ 320, 480 ])
    mode: str = "training"
    return_45: bool = False
    return_90: bool = False
    dilation: int = 1
    offset_45: int = 0
    offset_90: int = 0
    VIS_VOLUME: VisVolume = VisVolume(X_RANGE=(-12, 12), Y_RANGE=(0.5, 1.25), Z_RANGE=(0, 30), PPM=10)


@dataclass   
class DataConfigKittiRaw(DatasetConfig):
    type = "KITTI_Raw"
    data_path = "data/KITTI-Raw"
    pose_path = "datasets/kitti_raw/orb-slam_poses"
    split_path = "datasets/kitti_raw/splits/eigen_zhou"
    image_size = [ 128, 384 ]
    data_stereo: bool = True
    data_fc: int = 2


@dataclass   
class DataConfigKittiRawTulsiani(DatasetConfig):
    type = "KITTI_Raw"
    data_path = "data/KITTI-Raw"
    pose_path = "datasets/kitti_raw/out"
    split_path = "datasets/kitti_raw/splits/tulsiani"
    image_size = [ 128, 384 ]
    data_stereo = True
    data_fc = 1
    return_depth = True
    
    
@dataclass   
class DataConfigRe10k(DatasetConfig):
    type = "RealEstate10k"
    data_path = "/usr/wiss/wimbauer/storage/group/dataset_mirrors/01_incoming/realestate10k"
    split_path = "/usr/wiss/wimbauer/storage/user/unsup-objects/unsup-objects/datasets/realestate10k/splits/mine"
    image_size = [ 256, 384 ]
    data_fc = 3
    dilation = 5