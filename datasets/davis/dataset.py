import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from configs.structured_configs.data_config import DataConfigDAVIS
from utils.utils import asdict_lowercase_keys_override


class DavisDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        image_size = (192, 640),
        **kwargs,
    ):
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self._datapoints = []

        train_split = os.path.join(self.data_path, "ImageSets/2017/train.txt")
        val_split = os.path.join(self.data_path, "ImageSets/2017/val.txt")
        
        match self.split:
            case "train":
                with open(train_split, "r") as f:
                    self.folders = f.readlines()
            case "val":
                with open(val_split, "r") as f:
                    self.folders = f.readlines()
            case "train_val":
                with open(train_split, "r") as f:
                    self.folders = f.readlines()
                with open(val_split, "r") as f:
                    self.folders.extend(f.readlines())
            case _:
                raise ValueError(f"Invalid split: {self.split}")
            
        for folder in self.folders:
            folder = folder.strip()
            path = os.path.join(data_path, "JPEGImages", "480p", folder)
            for file in os.listdir(path):
                if file.endswith(".jpg"):
                    self._datapoints.append(os.path.join(path, file))

    def __getitem__(self, index: int):
        _start_time = time.time()

        if index >= len(self._datapoints):
            raise IndexError()

        filepath = self._datapoints[index]

        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img) * 2 - 1

        # Resize such that the smaller edge is aligned to image_size
        _, original_h, original_w = img.shape
        target_h, target_w = self.image_size
        
        scale = min(target_h / original_h, target_w / original_w)
        new_h, new_w = int(original_h * scale), int(original_w * scale)
        img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        
        _, h, w = img.shape
        fx, fy = 1000, 1000
        fx = (fx / w) * 2 - 1
        fy = (fy / h) * 2 - 1

        proj = torch.eye(3)
        proj[0, 0] = fx
        proj[1, 1] = fy

        _proc_time = np.array(time.time() - _start_time)

        data = {
            "imgs": [img],                               # list of [3, H, W]       
            "projs": [proj],                             # list of [3, 3] 
            "poses": [torch.eye(4)],                             # list of [3, 3]
            "hw_unnorm": torch.tensor([h, w]),
            "ts": [index],
            "t__get_item__": np.array([_proc_time]),
            "idxs": np.array([index])
        }

        return data

    def __len__(self) -> int:
        return len(self._datapoints)

    @classmethod
    def from_conf(cls, conf: DataConfigDAVIS, **kwargs):
        return cls(
            **asdict_lowercase_keys_override(conf, **kwargs)
        )
