import os
import time
import json
import time
import logging
import warnings
from copy import copy
from PIL import Image
from typing import Optional, List, Tuple, Literal, Dict
from collections import defaultdict
from iopath.common.file_io import PathManager

import numpy as np
from plyfile import PlyData
import torch
from torch.utils.data import Dataset

from datasets.co3d.data_types import load_dataclass_jgzip, SequenceAnnotation, FrameAnnotation
from torch.utils.data import Dataset
from configs.structured_configs.data_config import DataConfigCO3D
from utils.utils import asdict_lowercase_keys_override
from utils.transformation_ops import rodrigues_rotation_matrix

logger = logging.getLogger(__name__)

# NOTE: Pieced together using code from
# - CO3D dataloader implementation: https://github.com/ayushtewari/DFM/blob/main/data_io/co3d/json_index_dataset.py
# - CO3Dv2 version: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/implicitron/dataset/json_index_dataset.py#L156


class CO3DDataset(Dataset):
    """
        Dataset for CO3Dv2.
        https://github.com/facebookresearch/co3d/tree/main
    
        # Dataset format
        The dataset is organized in the filesystem as follows:

        ```
        CO3DV2_DATASET_ROOT
            ├── <category_0>
            │   ├── <sequence_name_0>
            │   │   ├── depth_masks
            │   │   ├── depths
            │   │   ├── images
            │   │   ├── masks
            │   │   └── pointcloud.ply
            │   ├── <sequence_name_1>
            │   │   ├── depth_masks
            │   │   ├── depths
            │   │   ├── images
            │   │   ├── masks
            │   │   └── pointcloud.ply
            │   ├── ...
            │   ├── <sequence_name_N>
            │   ├── set_lists
            │       ├── set_lists_<subset_name_0>.json
            │       ├── set_lists_<subset_name_1>.json
            │       ├── ...
            │       ├── set_lists_<subset_name_M>.json
            │   ├── eval_batches
            │   │   ├── eval_batches_<subset_name_0>.json
            │   │   ├── eval_batches_<subset_name_1>.json
            │   │   ├── ...
            │   │   ├── eval_batches_<subset_name_M>.json
            │   ├── frame_annotations.jgz
            │   ├── sequence_annotations.jgz
            ├── <category_1>
            ├── ...
            ├── <category_K>
        ```

        The dataset contains sequences named `<sequence_name_i>` from `K` categories with
        names `<category_j>`. Each category comprises sequence folders `<category_k>/<sequence_name_i>` containing the list of sequence images, depth maps, foreground masks, and valid-depth masks `images`, `depths`, `masks`, and `depth_masks` respectively. Furthermore, `<category_k>/<sequence_name_i>/set_lists/` stores `M` json files `set_lists_<subset_name_l>.json`, each describing a certain sequence subset.

        Users specify the loaded dataset subset by setting `self.subset_name` to one of the
        available subset names `<subset_name_l>`.

        `frame_annotations.jgz` and `sequence_annotations.jgz` are gzipped json files containing the list of all frames and sequences of the given category stored as lists of `FrameAnnotation` and `SequenceAnnotation` objects respectivelly.


        ## Set lists

        Each `set_lists_<subset_name_l>.json` file contains the following dictionary:   
        ```
        {
            "train": [
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
            "val": [
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
            "test": [
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
        }
        ```
        defining the list of frames (identified with their `sequence_name` and `frame_number`) in the "train", "val", and "test" subsets of the dataset.

        <i>Note that `frame_number` can be obtained only from `frame_annotations.jgz` and does not necesarrily correspond to the numeric suffix of the corresponding image file name (e.g. a file `<category_0>/<sequence_name_0>/images/frame00005.jpg` can have its frame number set to 20, not 5).</i>


        ### Available subset names in CO3Dv2

        In CO3DV2, by default, each category contains a _subset_ of the following set lists:
        ```
        "set_lists_fewview_test.json"  # Few-view task on the "test" sequence set.
        "set_lists_fewview_dev.json"  # Few-view task on the "dev" sequence set.
        "set_lists_manyview_test.json"  # Many-view task on the "test" sequence of a category.
        "set_lists_manyview_dev_0.json"  # Many-view task on the 1st "dev" sequence of a category.
        "set_lists_manyview_dev_1.json"  # Many-view task on the 2nd "dev" sequence of a category.
        ```

        ## Eval batches

        Each `eval_batches_<subset_name_l>.json` file contains a list of evaluation examples in the following form:
        ```
        [
            [  # batch 1
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
            [  # batch 1
                (sequence_name: str, frame_number: int, image_path: str),
                ...
            ],
        ]
        ```
        Note that the evaluation examples always come from the `"test"` part of the corresponding set list `set_lists_<subset_name_l>.json`.

        <b>The evaluation task</b> then consists of generating the first image in each batch given the knowledge of the other ones. Hence, the first image in each batch represents the (unseen) target frame, for which only the camera parameters are known, while the rest of the images in the batch are the known source frames whose cameras and colors are given.

        Note that for the Many-view task, where a user is given many known views of a particular sequence and the goal is to generate held-out views from the same sequence, `eval_batches_manyview_<sequence_set>_<sequence_id>.json` contain a single (target) frame per evaluation batch. Users can obtain the known views from the corresponding `"train"` list of frames in the set list `set_lists_manyview_<sequence_set>_<sequence_id>.json`.
            
    """
    def __init__(
        self,
        data_path: str,
        category_name: str,
        subset_name: str,
        split: Literal["train", "val", "test"] = 'train',
        image_size: Tuple[int, int] | None = (400, 400),
        return_depth=False,
        return_mask=False,
        return_pointcloud=False,
        return_policy: Literal["sequence", "frame"] = "frame",
        color_aug=False,
        box_crop_policy: Literal["none", "center", "fg_mask"] = "fg_mask",
        box_crop_mask_thr: float = 0.4,
        box_crop_context: float = 0.3,
        box_crop_square: bool = True,
        max_points: int | None = None,
        apply_adjust_upright: bool = False,
        **kwargs,
    ):
        """
            
        Args:
            data_path: The root folder of the dataset; all paths in frame / sequence
                annotations are defined w.r.t. this root. Has to be set if any of the
                load_* flabs below is true.
            return_images: Enable loading the frame RGB data.
            return_depths: Enable loading the frame depth maps.
            load_depth_masks: Enable loading the frame depth map masks denoting the
                depth values used for evaluation (the points consistent across views).
            return_masks: Enable loading frame foreground masks.
            return_pointclouds: Enable loading sequence-level point clouds.
            max_points: Cap on the number of loaded points in the point cloud;
                if reached, they are randomly sampled without replacement.
            mask_images: Whether to mask the images with the loaded foreground masks;
                0 value is used for background.
            mask_depths: Whether to mask the depth maps with the loaded foreground
                masks; 0 value is used for background.
            image_size: The height and width of the returned images, masks, and depth maps;
                aspect ratio is preserved during cropping/resizing.
            box_crop: Enable cropping of the image around the bounding box inferred
                from the foreground region of the loaded segmentation mask; masks
                and depth maps are cropped accordingly; cameras are corrected.
            box_crop_mask_thr: The threshold used to separate pixels into foreground
                and background based on the foreground_probability mask; if no value
                is greater than this threshold, the loader lowers it and repeats.
            box_crop_context: The amount of additional padding added to each
                dimension of the cropping bounding box, relative to box size.
        """
        self.target_image_size = image_size
        # Paths
        self.root_dir = data_path
        assert os.path.exists(self.root_dir), f"DATASET ROOT DIRECTORY NOT FOUND: {self.root_dir}"
        self.category_name = category_name
        category_path = os.path.join(self.root_dir, self.category_name)
        assert os.path.exists(category_path), f"DATASET CATEGORY DIRECTORY NOT FOUND: {category_path}"
        self.subset_name = subset_name
        self.split = split
        # Return values
        self.load_depth = return_depth
        self.load_fg_mask = return_mask
        self.load_pointcloud = return_pointcloud
        self.return_sequence = return_policy == "sequence"
        # Cropping parameters
        self.image_height, self.image_width = image_size if image_size is not None else (None, None)
        self.box_crop = box_crop_policy != "none"
        self.box_crop_policy = box_crop_policy
        if box_crop_policy == "fg_mask":
            self.load_fg_mask = True
        self.box_crop_mask_thr = box_crop_mask_thr
        self.box_crop_context = box_crop_context
        self.box_crop_square = box_crop_square
        # Augmentations
        self.color_aug = color_aug
        self.max_points = max_points
        self.T_adjust_upright = self._compute_upright_adjustment()
        self.apply_T_adjust_upright = apply_adjust_upright
        
        #        # Sort frames to have them grouped by sequence, ordered by timestamp
        # # pyre-ignore[16]
        # self.frame_annots = sorted(
        #     self.frame_annots,
        #     key=lambda f: (
        #         f["frame_annotation"].sequence_name,
        #         f["frame_annotation"].frame_timestamp or 0,
        #     ),
        # )
        
        # Load data split
        split_path = os.path.join(self.root_dir, self.category_name, 'set_lists', f'set_lists_{self.subset_name}.json')
        logger.info(f"Loading Co3D {self.split} split from {split_path}.")
        assert os.path.exists(split_path), f"DATASET SPLIT PATH NOT FOUND: {split_path}"
        with open(split_path, 'r') as file:
            data = json.load(file)
        # Map the images in the loaded split to their sequence.
        split_seq_frame_map = defaultdict(list)
        for sequence_name, frame_number, image_path in data[self.split]:
            split_seq_frame_map[sequence_name].append((frame_number, image_path))
        
        # Sort the data into datapoints that can be indexed.
        self._split_datapoints = []     # [[(seq_name, frame_num, im_path), (...)], [...], ...]
        match return_policy:
            case "frame":
                # One datapoint = One frame
                for sn in split_seq_frame_map:
                    for fn, ip in split_seq_frame_map[sn]:
                        self._split_datapoints.append([(sn, fn, ip)])
            case "sequence":
                # One datapoint = All training views of one sequence
                for sn in split_seq_frame_map:
                    sequence = []
                    for fn, ip in split_seq_frame_map[sn]:
                        sequence.append((sn, fn, ip))
                    self._split_datapoints.append(sequence)
        
        # Load per-frame annotations
        frame_annotations_path = os.path.join(self.root_dir, self.category_name, 'frame_annotations.jgz')
        logger.info(f"Loading Co3D frames from {frame_annotations_path}.")
        assert os.path.exists(frame_annotations_path), f"FRAME ANNOTATIONS PATH NOT FOUND: {frame_annotations_path}"
        frame_annotations = load_dataclass_jgzip(frame_annotations_path, List[FrameAnnotation])
        self.frame_annotations: Dict[str, Dict[str, FrameAnnotation]] = defaultdict(lambda: defaultdict(FrameAnnotation))
        for fa in frame_annotations:
            self.frame_annotations[fa.sequence_name][fa.frame_number] = fa
        
        # Load per-sequence annotations
        sequence_annotations_path = os.path.join(self.root_dir, self.category_name, 'sequence_annotations.jgz')
        logger.info(f"Loading Co3D sequences from {sequence_annotations_path}.")
        assert os.path.exists(sequence_annotations_path), f"SEQUENCE ANNOTATIONS PATH NOT FOUND: {sequence_annotations_path}"
        sequence_annotations = load_dataclass_jgzip(sequence_annotations_path, List[SequenceAnnotation])
        self.sequence_annotations: Dict[str, SequenceAnnotation] = {}
        for sa in sequence_annotations:
            self.sequence_annotations[sa.sequence_name] = sa

    def __len__(self) -> int:
        return len(self._split_datapoints)
    
    def __str__(self) -> str:
        return f"CO3DDataset_{self.category_name}_{self.subset_name}_{self.split}"
    
    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("Index out of range")

        _start_time = time.time()
        images, fg_masks, depths, poses, projs, cam_incl_adjusts, sequence_names, frame_numbers, ts = [], [], [], [], [], [], [], [], []
        
        # Data for the given index
        for sequence_name, frame_number, _ in self._split_datapoints[index]:
            try:        
                frame_annotation = self.frame_annotations[sequence_name][frame_number]
            except KeyError:
                raise ValueError(f"Frame annotation not found for sequence {sequence_name} with frame number {frame_number}.")
            try:        
                sequence_annotation = self.sequence_annotations[sequence_name]
            except KeyError:
                raise ValueError(f"Sequence annotation not found for sequence {sequence_name}.")
            # Load all necessary data
            bbox_xyxy, resize_scale, padding_mask = None, 1.0, None
            if self.load_fg_mask or self.box_crop:
                fg_mask, bbox_xyxy, resize_scale, padding_mask = self._load_fg_mask_with_bbox(frame_annotation)
                fg_masks.append(fg_mask)
                
            image = self._load_image(os.path.join(self.root_dir, frame_annotation.image.path), bbox_xyxy)
            # Map image to [-1, 1].
            image = image * 2. - 1.
            images.append(image)

            if self.load_depth:
                depth = self._load_depth(os.path.join(self.root_dir, frame_annotation.depth.path), frame_annotation.depth.scale_adjustment, bbox_xyxy)  
                depths.append(depth)
            
                
            pose, proj = self._extract_camera_params(frame_annotation, bbox_xyxy, resize_scale=resize_scale)
            poses.append(pose)
            projs.append(proj)
            # This transformation matrix can be used to align the Camera-Y-axis with the global Y-axis.
            # NOTE: Never worked (correctly?) for me
            cam_incl_adjust = self._compute_upright_adjustment(from_vec=pose[:3, 3])
            cam_incl_adjusts.append(cam_incl_adjust.T)
            
            sequence_names.append(sequence_name)
            frame_numbers.append(frame_number)
            ts.append(frame_annotation.frame_timestamp)
        
        xyz, rgb = [], []
        if self.load_pointcloud:
            xyz_, rgb_ = self._load_pointcloud(os.path.join(self.root_dir, sequence_annotation.point_cloud.path))
            
            if self.apply_T_adjust_upright:
                xyz_ = torch.cat([xyz_, torch.ones_like(xyz_[..., :1])], dim=-1)
                xyz_ = (self.T_adjust_upright @ xyz_.T).T
                xyz_ = xyz_[..., :3]
            
            xyz.append(xyz_)
            rgb.append(rgb_)
        
        # TODO when loading a full sequence, optionally normalize the scene scale
        
        data = {
            "imgs": images,            # list of [3, H, W]
            "projs": projs,            # list of [3, 3]
            "poses": poses,            # list of [3, 3]
            "depths": depths,          # list of [1, H, W]
            "pointcloud_xyz": xyz,
            "pointcloud_rgb": rgb,
            "sequence_names": sequence_names,
            "frame_numbers": frame_numbers,
            "hw_unnorm": torch.tensor([self.image_height, self.image_width]),
            "cam_incl_adjusts": cam_incl_adjusts,
            "ts": ts,
            "t__get_item__": torch.tensor([time.time() - _start_time]),
            "index": torch.tensor([index])
        }

        return data

    def _load_image(self, path: str, bbox_xyxy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns RGB image tensor in [0, 1]."""
        assert os.path.exists(path), f"IMAGE NOT FOUND: {path}"
        with Image.open(path) as pil_im:
            image = np.array(pil_im.convert("RGB"))
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image)
        if bbox_xyxy is not None and self.box_crop:
            image = self._crop_to_bbox(image, bbox_xyxy, path)
        image, minscale, _ = self._resize(image)
        return image 

    def _load_depth(self, path, scale_adjustment: float = 1.0, bbox_xyxy: Optional[torch.Tensor] = None):
        """Reads 16-bit depth from PNG and returns the cropped and resized depth map."""
        assert os.path.exists(path), f"DEPTH IMAGE NOT FOUND: {path}"
        with Image.open(path) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
            depth = torch.from_numpy(depth).unsqueeze(0) * scale_adjustment
            depth[torch.isinf(depth)] = 0.0
        if bbox_xyxy is not None and self.box_crop:
            depth = self._crop_to_bbox(depth, bbox_xyxy, path)
        depth, _, _ = self._resize(depth)
        return depth
    
    
    def _load_pointcloud(self, path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The PLY file expected to contain a single element called 'vertex' with the following properties:
            ('x', 'float'), 
            ('y', 'float'), 
            ('z', 'float'), 
            ('red', 'uchar'), 
            ('green', 'uchar'), 
            ('blue', 'uchar'))

        Corresponds to this line in the official implementation:
        https://github.com/ayushtewari/DFM/blob/50c6e20db124147f37ba44b256000de6ce524270/data_io/co3d/json_index_dataset.py#L1089
        """
        # Uses memory mapping for faster loading by default.
        ptc = PlyData.read(path)

        try:
            el = ptc["vertex"]
        except KeyError as e:
            raise KeyError("Expecting a single element with name 'vertex' in the PLY file.") from e
        
        data_memmap = el.data
        if self.max_points is not None and el.count > self.max_points:
            data_memmap = np.random.choice(data_memmap, size=self.max_points, replace=False)
        
        try:
            x = torch.from_numpy(data_memmap["x"].copy())
            y = torch.from_numpy(data_memmap["y"].copy())
            z = torch.from_numpy(data_memmap["z"].copy())
            r = torch.from_numpy(data_memmap["red"].copy())
            g = torch.from_numpy(data_memmap["green"].copy())
            b = torch.from_numpy(data_memmap["blue"].copy())
        except KeyError as e:
            raise KeyError(f"Expecting the element properties ['x', 'y', 'z', 'red', 'green', 'blue']") from e
        
        xyz = torch.stack([x, y, z], dim=1)
        rgb = torch.stack([r, g, b], dim=1)
            
        return xyz, rgb
        
    def _load_fg_mask_with_bbox(self, frame: FrameAnnotation, decrease_quant: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Returns the foreground mask and parameters for rescaling and cropping the image."""
        path = os.path.join(self.root_dir, frame.mask.path)
        assert os.path.exists(path), f"FOREGROUND MASK NOT FOUND: {path}"
        with Image.open(path) as pil_im:
            fg_mask = np.array(pil_im)
            fg_mask = fg_mask.astype(np.float32) / 255.0
            fg_mask = torch.from_numpy(fg_mask).unsqueeze(0)
        
        height, width = frame.image.size
        
        match self.box_crop_policy:
            case "none":
                bbox_xyxy = None
            case "center":
                # Square center crop
                center_point = torch.tensor([width // 2, height // 2])
                half_edge_length = center_point.min()
                bbox_xyxy = torch.tensor([
                    center_point[0] - half_edge_length,
                    center_point[1] - half_edge_length,
                    center_point[0] + half_edge_length,
                    center_point[1] + half_edge_length,
                ])
            case "fg_mask":
                # Square crop around the foreground mask with optional padding
                # https://github.com/ayushtewari/DFM/blob/50c6e20db124147f37ba44b256000de6ce524270/data_io/co3d/json_index_dataset.py#L991
                thr = copy(self.box_crop_mask_thr)
                masks_for_box = torch.zeros_like(fg_mask)
                while masks_for_box.sum() <= 1.0:
                    masks_for_box = (fg_mask > thr).float()
                    thr -= decrease_quant
                if thr <= 0.0:
                    warnings.warn(f"Empty masks_for_bbox (thr={thr}) => using full image.")
                    # TODO fall back to center crop
                def _get_1d_bounds(arr) -> Tuple[int, int]:
                    nz = np.flatnonzero(arr)
                    return nz[0], nz[-1] + 1

                x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
                y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))
                cx, cy = (x0+x1)/2, (y0+y1)/2
                w, h = x1 - x0, y1 - y0
                
                w_new, h_new = w, h
                if self.box_crop_context > 0.:
                    w_new = w + w * self.box_crop_context
                    h_new = h + h * self.box_crop_context
                    
                w_new = max(0, min(width, w_new))
                h_new = max(0, min(height, h_new))
                
                if self.box_crop_square:
                    w_new = h_new = min(w_new, h_new)
                
                x0_new = max(0, x0 + w - w_new)
                y0_new = max(0, y0 + h - h_new)
                # x0 = max(0, x1 - w_new)
                # y0 = max(0, y1 - h_new) 
                
                # x0 = max(0, cx - w//2)
                # y0 = max(0, cy - h//2) 
                # y0_new = y0 - (h_new - h)//2
                
                bbox_xyxy = torch.tensor([
                    # x0, y0, x0 + w, y0 + h
                    x0_new, y0_new, x0_new + w_new, y0_new + h_new
                ])
                    
                if (bbox_xyxy[2:] <= 1.).any():
                    raise ValueError(
                        f"Squashed image {frame.image.path}! The bounding box contains no pixels."
                    )
            case _:
                raise ValueError(f"Bounding box cropping policy {self.box_crop_policy} is not available.")
        
        # Clamp to max image bounds + round + crop foreground mask
        if bbox_xyxy is not None:
            bbox_xyxy[[0, 2]] = bbox_xyxy[[0, 2]].clamp(0, width)
            bbox_xyxy[[1, 3]] = bbox_xyxy[[1, 3]].clamp(0, height)
            bbox_xyxy = bbox_xyxy.round().long()
            fg_mask = self._crop_to_bbox(fg_mask, bbox_xyxy, frame.mask.path)
        
        fg_mask, resize_scale, padding_mask = self._resize(fg_mask, mode="nearest")
            
        return fg_mask, bbox_xyxy, resize_scale, padding_mask
    
    def _extract_camera_params(self, frame: FrameAnnotation, bbox_xyxy: Optional[torch.Tensor] = None, resize_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the camera pose and normalized intrinsics."""
        H, W = frame.image.size
        match frame.viewpoint.intrinsics_format.lower():
            case "ndc_isotropic":
                # The intrinsics provided in CO3Dv2 follow the isotropic-ndc convention from Pytorch3D.
                # https://github.com/facebookresearch/co3d/issues/4#issuecomment-1952224331
                # https://github.com/facebookresearch/co3d/blob/ff54df4dfcf5ac3d653d653b5a32b8e5653c70a9/co3d/dataset/data_types.py#L72
                sx = sy = min(H, W) / 2.
            case "ndc_norm_image_bounds":
                # Was used in CO3D
                sx = W / 2.
                sy = H / 2.
            case _:
                ValueError(f"Unknown intrinsics format: {frame.viewpoint.intrinsics_format}")
        
        fx, fy = frame.viewpoint.focal_length
        cx, cy = frame.viewpoint.principal_point
        # Convert NDC-space intrinsics to screen-space intrinsics.
        # Using the conversion equation from https://pytorch3d.org/docs/cameras
        fx_px = fx * sx
        fy_px = fy * sy
        cx_px = W/2. - cx * sx
        cy_px = H/2. - cy * sy
        # Now we have e.g. fx=1050 fy=1050 cx=185 cy=343
        
        if bbox_xyxy is not None and self.box_crop:
            # Move the principial point is the image is cropped.
            cx_px -= bbox_xyxy[0]
            cy_px -= bbox_xyxy[1]
        
        # Convert screen-space coordinates [0, H or W] to "normalized image coordinates" [-1, 1].
        intrinsics = torch.eye(3)
        # intrinsics[0, 0] = fx_px / W * 2. * 1/resize_scale
        # intrinsics[1, 1] = fy_px / H * 2. * 1/resize_scale
        # intrinsics[0, 2] = cx_px / W * 2. * 1/resize_scale - 1.
        # intrinsics[1, 2] = cy_px / H * 2. * 1/resize_scale - 1.
        intrinsics[0, 0] = fx_px / self.image_width * 2. * resize_scale
        intrinsics[1, 1] = fy_px / self.image_height * 2. * resize_scale
        intrinsics[0, 2] = cx_px / self.image_width * 2. * resize_scale - 1.
        intrinsics[1, 2] = cy_px / self.image_height * 2. * resize_scale - 1.
        
        # CO3Dv2 specifies the world-to-camera transformation [R|t] (PyTorch3D convention).
        # The pose is the inverse transformation.
        # https://github.com/facebookresearch/co3d/issues/77#issuecomment-1592701804
        R_c2w = torch.tensor(frame.viewpoint.R).inverse()
        t_c2w = - R_c2w @ torch.tensor(frame.viewpoint.T)
        pose = torch.eye(4)
        pose[:3, :3] = R_c2w
        pose[:3, 3:] = t_c2w[:, None]
        
        if self.apply_T_adjust_upright:
            # Transforming the camera with the direct matrix would move it as if it were part of the scene,
            # but we want the opposite effect (how the camera sees the scene).
            pose = self.T_adjust_upright.T @ pose
            
        # In PyTorch3D's coordinate system: "X = left; Y = up; Z = away".
        # Flip the directions of X and Y axes, such that "X = right; Y = down; Z = away".
        # Still a right-hand-coordinate-frame.
        pose = torch.diag(torch.tensor([-1. ,-1. ,1., 1.])) @ pose 
        
        return pose, intrinsics
    
    @staticmethod
    def _compute_upright_adjustment(to_vec=(0.0, -1.0, 0.0), from_vec=(-0.0396, -0.8306, -0.5554)) -> torch.Tensor:
        """
        Computes the transformation matrix to adjust scene to the desired 'upright' orientation based on the provided vectors.
        Explained here: https://github.com/facebookresearch/co3d/issues/28#issuecomment-1066245264
        Adjustment calculation following: https://github.com/facebookresearch/co3d/issues/64#issuecomment-1484885045
        
        NOTE: In the CO3D paper, they render fly-around-videos by fitting a plane to the training camera poses.
        https://github.com/facebookresearch/pytorch3d/blob/4ae25bfce7eb42042a34585acc3df81cf4be7d85/pytorch3d/implicitron/models/visualization/render_flyaround.py#L49
        https://github.com/facebookresearch/co3d/issues/69
        """
        to_vec = torch.FloatTensor(to_vec)
        from_vec = torch.FloatTensor(from_vec)

        # Calculate the rotation axis using Rodrigues' formula
        rot_axis = torch.cross(from_vec, to_vec)
        angle = torch.arccos(torch.dot(from_vec, to_vec) / (torch.norm(from_vec) * torch.norm(to_vec)))
        R_adjust = rodrigues_rotation_matrix(rot_axis, angle)

        T_adjust = torch.eye(4)
        T_adjust[:3, :3] = R_adjust
        
        return T_adjust
    
    @staticmethod
    def _crop_to_bbox(tensor: torch.Tensor, bbox: torch.Tensor, path: str = "") -> torch.Tensor:
        """Bounding box should be [x0, y0, x1, y1]."""
        tensor = tensor[..., bbox[1]:bbox[3], bbox[0]:bbox[2]]
        assert all(c > 0 for c in tensor.shape), f"Squashed image {path}"
        return tensor
    
    def _resize(self, tensor: torch.Tensor, mode="bilinear") -> Tuple[torch.Tensor, float, torch.Tensor]:
        """Resizes input tensor to the return shape and adds zero-padding to all sides if needed."""
        C, H, W = tensor.shape
        
        if self.image_height is None or self.image_width is None:
            # Skip the resizing
            return tensor, 1.0, torch.ones_like(tensor[:1])
        
        minscale = min(self.image_height / H, self.image_width / W)
        scaled_tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            scale_factor=minscale,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
            recompute_scale_factor=True,
        )[0]    # [c, h, w]
        
        pad_x = (self.image_width - scaled_tensor.size(2)) // 2
        pad_y = (self.image_height - scaled_tensor.size(1)) // 2
        ret = torch.zeros(C, self.image_height, self.image_width)
        ret[:, pad_y:pad_y+scaled_tensor.size(1), pad_x:pad_x+scaled_tensor.size(2)] = scaled_tensor
        
        valid_mask = torch.zeros(1, self.image_height, self.image_width)
        valid_mask[:, pad_y:pad_y+scaled_tensor.size(1), pad_x:pad_x+scaled_tensor.size(2)] = 1.0
        return ret, minscale, valid_mask

    def get_available_subset_names(
        self, dataset_root: str, category: str, path_manager: Optional[PathManager] = None,
    ) -> List[str]:
        """
        Get the available subset names for a given category folder inside a root dataset
        folder `dataset_root`.
        """
        category_dir = os.path.join(dataset_root, category)
        category_dir_exists = (
            (path_manager is not None) and path_manager.isdir(category_dir)
        ) or os.path.isdir(category_dir)
        if not category_dir_exists:
            raise ValueError(
                f"Looking for dataset files in {category_dir}. "
                + "Please specify a correct dataset_root folder."
            )

        set_list_dir = os.path.join(category_dir, "set_lists")
        set_list_jsons = (os.listdir if path_manager is None else path_manager.ls)(
            set_list_dir
        )

        return [
            json_file.replace("set_lists_", "").replace(".json", "")
            for json_file in set_list_jsons
        ]

    @classmethod
    def from_conf(cls, cfg: DataConfigCO3D, **kwargs):
        return cls(
            **asdict_lowercase_keys_override(cfg),
            **kwargs,
        )