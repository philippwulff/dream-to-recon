import os
import torch

from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from datasets.kitti_odom.kitti_odometry_dataset import KittiOdometryDataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset
from datasets.nyu_depth_v2.nyu_depth_v2_dataset import NYUDepthV2Dataset
from datasets.realestate10k.realestate10k_dataset import RealEstate10kDataset
from datasets.davis.dataset import DavisDataset
from datasets.waymo.waymo_dataset import WaymoDataset
from datasets.co3d.co3d_dataset import CO3DDataset
from configs.structured_configs.data_config import DatasetConfig, DataConfigKitti360, DataConfigCO3D, DataConfigDAVIS, ConcatDatasetConfig, DataConfigWaymo
from torch.utils.data import Subset, ConcatDataset
from copy import copy


def make_datasets(config: DatasetConfig | ConcatDatasetConfig):
    
    match config:
        case ConcatDatasetConfig():
            # Case is also true for any subclass.
            train_dataset = []
            val_dataset = []
            test_dataset = []
            for d_cfg in config.DATASETS:
                train, val, test = make_datasets(d_cfg)
                train_dataset.append(train)
                val_dataset.append(val)
                test_dataset.append(test)
            
            train_dataset = ConcatDataset(train_dataset)
            val_dataset = ConcatDataset(val_dataset)
            test_dataset = ConcatDataset(test_dataset)
                
        case "KITTI_Odometry":
            train_dataset = KittiOdometryDataset(
                base_path=config["data_path"],
                frame_count=config.get("data_fc", 1),
                target_image_size=config.get("image_size", (128, 256)),
                return_stereo=config.get("data_stereo", False),
                sequences=config.get("train_sequences", ("00",)),
                custom_pose_path=config.get("custom_pose_path", None),
                keyframe_offset=0 #-(config.get("data_fc", 1) // 2)
            )
            val_dataset = KittiOdometryDataset(
                base_path=config["data_path"],
                frame_count=config.get("data_fc", 1),
                target_image_size=config.get("image_size", (128, 256)),
                return_stereo=config.get("data_stereo", False),
                sequences=config.get("val_sequences", ("00",)),
                custom_pose_path=config.get("custom_pose_path", None),
                keyframe_offset=0 #-(config.get("data_fc", 1) // 2)
            )

        case "KITTI_Raw":
            train_dataset = KittiRawDataset(
                data_path=config["data_path"],
                pose_path=config["pose_path"],
                split_path=os.path.join(config["split_path"], "train_files.txt"),
                target_image_size=config.get("image_size", (192, 640)),
                frame_count=config.get("data_fc", 1),
                return_stereo=config.get("data_stereo", False),
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
                color_aug=config.get("color_aug", False)
            )
            val_dataset = KittiRawDataset(
                data_path=config["data_path"],
                pose_path=config["pose_path"],
                split_path=os.path.join(config["split_path"], "val_files.txt"),
                target_image_size=config.get("image_size", (192, 640)),
                frame_count=config.get("data_fc", 1),
                return_stereo=config.get("data_stereo", False),
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
            )

        case DataConfigKitti360():
            if config.split_path is None:
                train_split_path = None
                val_split_path = None
                test_split_path = None
            else:
                train_split_path = os.path.join(config.split_path, "train_files.txt")
                val_split_path = os.path.join(config.split_path, "val_files.txt")
                test_split_path = os.path.join(config.split_path, "test_files.txt")
                
            train_dataset = Kitti360Dataset.from_conf(config, split_path=train_split_path)
            val_dataset = Kitti360Dataset.from_conf(config, split_path=val_split_path)
            test_dataset = Kitti360Dataset.from_conf(config, split_path=test_split_path)
        
        case DataConfigDAVIS():
            train_dataset = DavisDataset.from_conf(config, split="train")
            val_dataset = DavisDataset.from_conf(config, split="val")
            test_dataset = None

        case "RealEstate10k":
            train_dataset = RealEstate10kDataset(
                data_path=config["data_path"],
                split_path=None,
                target_image_size=config.get("image_size", (256, 384)),
                frame_count=config.get("data_fc", 2),
                keyframe_offset=0, #-(config.get("data_fc", 1) // 2),
                dilation=config.get("dilation", 10),
                color_aug=config.get("color_aug", False)
            )
            val_dataset = RealEstate10kDataset(
                data_path=config["data_path"],
                split_path=os.path.join(config["split_path"], "val_files.txt"),
                target_image_size=config.get("image_size", (256, 384)),
                frame_count=config.get("data_fc", 2),
                keyframe_offset=0, #-(config.get("data_fc", 1) // 2),
                dilation=config.get("dilation", 10),
                color_aug=False
            )
        case DataConfigCO3D():
            train_dataset = None
            val_dataset = None
            train_dataset = CO3DDataset.from_conf(config, split="train")
            val_dataset = CO3DDataset.from_conf(config, split="val")
            test_dataset = CO3DDataset.from_conf(config, split="test")

        case DataConfigWaymo():
            
            if config.split_path:
                train_split_path = os.path.join(config.split_path, "train_files.txt")
                val_split_path = os.path.join(config.split_path, "val_files.txt")
                test_split_path = os.path.join(config.split_path, "test_files.txt")
            else:
                train_split_path = None
                val_split_path = None
                test_split_path = None

            train_dataset = WaymoDataset.from_conf(config, split_path=train_split_path, mode="training")
            val_dataset = WaymoDataset.from_conf(config, split_path=val_split_path, mode="validation")
            test_dataset = WaymoDataset.from_conf(config, split_path=test_split_path, mode="testing")
        case _:
            raise NotImplementedError(f"Unsupported dataset type: {type}")
        
    if config.MAX_TRAIN_DATASET_LEN:
        max_len_train = min(config.MAX_TRAIN_DATASET_LEN, len(train_dataset))
        match config.TRAIN_SAMPLES_POLICY:
            case "slice":
                idxs = list(range(max_len_train))
            case "random":
                idxs = torch.randperm(len(train_dataset))[:max_len_train].tolist()
            case "eqspaced":
                idxs = torch.linspace(0, len(train_dataset)-1, steps=max_len_train, dtype=torch.int).tolist()
            case _:
                raise ValueError(f"Policy {config.TRAIN_SAMPLES_POLICY} not available.")
        # train_dataset = Subset(train_dataset, range(min(config.MAX_TRAIN_DATASET_LEN, len(train_dataset))))
        train_dataset = Subset(train_dataset, idxs)

    if config.USE_TRAIN_FOR_VAL:
        val_dataset = copy(train_dataset)
    else:
        if config.MAX_VAL_DATASET_LEN and val_dataset is not None:
            max_len_val = min(config.MAX_VAL_DATASET_LEN, len(val_dataset))
            match config.VAL_SAMPLES_POLICY:
                case "slice":
                    idxs = list(range(max_len_val))
                case "random":
                    idxs = torch.randperm(len(val_dataset))[:max_len_val].tolist()
                case "eqspaced":
                    idxs = torch.linspace(0, len(val_dataset)-1, steps=max_len_val, dtype=torch.int).tolist()
                case _:
                    raise ValueError(f"Policy {config.VAL_SAMPLES_POLICY} not available.")
            val_dataset = Subset(val_dataset, idxs)

    if config.USE_VAL_FOR_TEST:
        test_dataset = copy(val_dataset)
    else:
        if config.MAX_TEST_DATASET_LEN and test_dataset is not None:
            max_len_test = min(config.MAX_TEST_DATASET_LEN, len(test_dataset))
            match config.TEST_SAMPLES_POLICY:
                case "slice":
                    idxs = list(range(test_dataset))
                case "random":
                    idxs = torch.randperm(len(test_dataset))[:max_len_test].tolist()
                case "eqspaced":
                    idxs = torch.linspace(0, len(test_dataset)-1, steps=max_len_test, dtype=torch.int).tolist()
                case _:
                    raise ValueError(f"Policy {config.VAL_SAMPLES_POLICY} not available.")
            test_dataset = Subset(test_dataset, idxs)
    
    return train_dataset, val_dataset, test_dataset


def make_test_dataset(config):
    raise NotImplementedError()
    type = config.get("type", "KITTI_Raw")
    if type == "KITTI_Raw":
        test_dataset = KittiRawDataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config["split_path"], "test_files.txt"),
            target_image_size=config.get("image_size", (192, 640)),
            return_depth=True,
            frame_count=1,
            return_stereo=config.get("data_stereo", False),
            keyframe_offset=0
        )
        return test_dataset
    elif type == "KITTI_360":
        test_dataset = Kitti360Dataset(
            data_path=config["data_path"],
            pose_path=config["pose_path"],
            split_path=os.path.join(config.get("split_path", None), "test_files.txt"),
            target_image_size=tuple(config.get("image_size", (192, 640))),
            frame_count=config.get("data_fc", 1),
            return_stereo=config.get("data_stereo", False),
            return_fisheye=config.get("data_fisheye", False),
            return_3d_bboxes=config.get("data_3d_bboxes", False),
            return_segmentation=config.get("data_segmentation", False),
            keyframe_offset=0,
            fisheye_rotation=config.get("fisheye_rotation", 0),
            fisheye_offset=config.get("fisheye_offset", 1),
            dilation=config.get("dilation", 1),
            is_preprocessed=config.get("is_preprocessed", False)
        )
        return test_dataset
    elif type == "RealEstate10k":
        test_dataset = RealEstate10kDataset(
            data_path=config["data_path"],
            split_path=os.path.join(config["split_path"], "test_files.txt"),
            target_image_size=config.get("image_size", (256, 384)),
            frame_count=config.get("data_fc", 2),
            keyframe_offset=0,
            dilation=config.get("dilation", 10),
            color_aug=False
        )
        return test_dataset
    elif type == "NYU_Depth_V2":
        test_dataset = NYUDepthV2Dataset(
            data_path=config["data_path"],
            target_image_size=config.get("image_size", (256, 384)),
        )
        return test_dataset
    else:
        raise NotImplementedError(f"Unsupported dataset type: {type}")