from typing import Union
import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class Crop(nn.Module):
    def __init__(self, height: int, width: int, lefts: torch.Tensor, tops: torch.Tensor, *args, **kwargs) -> None:
        """
        lefts: [B] or [B, N]
        tops: [B] or [B, N]
        """
        super().__init__(*args, **kwargs)
        
        assert (tops >= 0).all() and (lefts >= 0).all()
        assert len(tops.shape) == len(lefts.shape) and len(tops.shape) in [1, 2]
        
        self.t = tops.long()
        self.b = (tops + height).long()
        self.l = lefts.long()
        self.r = (lefts + width).long()
        
    def forward(self, imgs):
        """imgs: [B, C, H, W] or [B, N, C, H, W]"""
        if len(imgs.shape) == 5:
            B, N, _, _, _ = imgs.shape
            imgs_ = imgs.clone().view(B*N, *imgs.shape[2:])
        elif len(imgs.shape) == 4:
            imgs_ = imgs.clone()
        else:
            raise ValueError()
        
        # The slicing syntax in PyTorch expects scalar integers or slice objects for indexing, 
        # not tensors with more than one element
        crops = []
        for i in range(len(imgs_)):
            t, b = self.t.flatten()[i].item(), self.b.flatten()[i].item()
            l, r = self.l.flatten()[i].item(), self.r.flatten()[i].item()
            crop = imgs_[i, :, t:b, l:r]
            crops.append(crop.unsqueeze(0))

        # Concatenate the list of cropped images along the batch dimension
        crops = torch.cat(crops, dim=0)
        if len(imgs.shape) == 5:
            return crops.view(B, N, *crops.shape[-3:])
        return crops
        # return crops.view(*imgs.shape[:-2], *crops.shape[-2:])