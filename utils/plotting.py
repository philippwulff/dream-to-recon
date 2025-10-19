from typing import List, Optional, Literal
import numpy as np
import torch
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.constants import DEFINITIONS
from dotdict import dotdict

from configs.structured_configs.data_config import VisVolume


cmap_spectral = mpl.colormaps["Spectral"]
cmap_magma = mpl.colormaps["magma"]
cmap_jet = mpl.colormaps["jet"]

PAGE_WIDTH_INCHES = 5.84
HALF_PAGE_WIDTH_INCHES = PAGE_WIDTH_INCHES // 2

def set_thesis_rcparams(dpi=None):
    SMALL_SIZE = 7
    TEXT_SIZE = 4
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 10
    
    # plt.rc('text.latex', preamble=r"\usepackage{lmodern}")
    # plt.rc('text.latex', unicode=True)
    
    # plt.rc('text', usetex=True)
    # plt.style.use('ggplot')
    # This is the default 'Times New Romant' font in the ICCV template.
    plt.rc('font', size=TEXT_SIZE, family="serif")
    plt.rc('mathtext', fontset="dejavuserif")
    plt.rc('axes', titlesize=TEXT_SIZE, labelsize=SMALL_SIZE, titlepad=2, labelpad=2)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE, title_fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE, dpi=500 if not dpi else dpi)


def set_spines(ax: plt.Axes, c="black", lw=1, ls="-", visible=True):
    """Helper for setting attributes for all spines of an axis."""
    for _ in ['bottom', 'top', 'left', 'right']:
        ax.spines[_].set(color=c, linewidth=lw, linestyle=ls, visible=visible)


def draw_bbox(im, size):
    b, c, h, w = im.shape
    h2, w2 = (h-size)//2, (w-size)//2
    marker = np.tile(np.array([[1.],[0.],[0.]]), (1,size))
    marker = torch.FloatTensor(marker)
    im[:, :, h2, w2:w2+size] = marker
    im[:, :, h2+size, w2:w2+size] = marker
    im[:, :, h2:h2+size, w2] = marker
    im[:, :, h2:h2+size, w2+size] = marker
    return im


def plot_image_grid(images, rows, cols, directions=None, imsize=(2, 2), title=None, show=True):
    fig, axs = plt.subplots(rows, cols, gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True, figsize=(rows * imsize[0], cols * imsize[1]))
    for i, image in enumerate(images):
        axs[i % rows][i // rows].axis("off")
        if directions is not None:
            axs[i % rows][i // rows].arrow(32, 32, directions[i][0] * 16, directions[i][1] * 16, color='red', length_includes_head=True, head_width=2., head_length=1.)
        axs[i % rows][i // rows].imshow(image, aspect='auto')
    plt.subplots_adjust(hspace=0, wspace=0)
    if title is not None:
        fig.suptitle(title, fontsize=12)
    if show:
        plt.show()
    return fig


def show_save(save_path, show=True, save=False):
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)


def color_tensor(tensor: torch.Tensor, cmap=None, norm=False):
    if norm:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    if cmap is None:
        cmap = cmap_magma
    elif isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    tensor = torch.tensor(cmap(tensor.cpu().numpy()), device=tensor.device)[..., :3]
    return tensor


def color_occlusion_masked_img(
    img: torch.Tensor, 
    mask: torch.Tensor, 
    c_occl: torch.Tensor | str | None = "red",
    c_inv: torch.Tensor | str | None = "blue",
    ) -> np.ndarray:
    """
    :param img: array of shape [B, C, H, W]
    :param masks: array of shape [B, 1, H, W]
    """
    
    img_ = img.clone().permute(0, 2, 3, 1)  # [B, H, W, C]
    is_occl = (mask == DEFINITIONS.IS_OCCLUDED).squeeze(1)  # [B, H, W]   
    is_inv = (mask == DEFINITIONS.IS_INVALID).squeeze(1)
    
    if c_occl is not None and isinstance(c_occl, torch.Tensor):
        c_occl = c_occl.to(device=img.device, dtype=img.dtype)
    elif c_occl == "random":
        c_occl = torch.rand_like(img_)[is_occl]
    elif c_occl == "zeros":
        c_occl = torch.zeros_like(img_)[is_occl]
    elif c_occl == "red":
        c_occl = torch.tensor([1.0, 0., 0.], device=img.device, dtype=img.dtype)
        
    if c_inv is not None and isinstance(c_inv, torch.Tensor):
        c_inv = c_inv.to(device=img.device, dtype=img.dtype)
    elif c_inv == "random":
        c_inv = torch.rand_like(img_)[is_inv]
    elif c_inv == "zeros":
        c_inv = torch.zeros_like(img_)[is_inv]
    elif c_inv == "blue":
        c_inv = torch.tensor([0., 0., 1.0], device=img.device, dtype=img.dtype)
        
    if c_occl is not None:
        img_[is_occl] = c_occl
    if c_inv is not None:
        img_[is_inv] = c_inv    
    # All other pixels keep their color.
    
    return img_.permute(0, 3, 1, 2)


OUT_RES = dotdict(
    # X_RANGE = (-9, 9),
    # Z_RANGE = (21, 3),
    
    X_RANGE = (-15, 15),
    Y_RANGE = (.0, .75),
    Z_RANGE = (30, 0),
    
    # X_RANGE = (-3, 3),
    # Y_RANGE = (-1, 1.),
    # Z_RANGE = (5, 0),
    
    P_RES_ZX = (256, 256),
    P_RES_Y = 64
)


def get_pts(x_range, y_range, z_range, x_res, y_res, z_res, cam_incl_adjust=None, device=None) -> torch.Tensor:
    """Outputs grid points with shape [y_res, z_res, x_res, 3]."""
    if not device and cam_incl_adjust is not None:
        device = cam_incl_adjust.device
    x = torch.linspace(x_range[0], x_range[1], x_res, device=device).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res, device=device).view(1, z_res, 1).expand(y_res, -1, x_res)
    y = torch.linspace(y_range[0], y_range[1], y_res, device=device).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
    if cam_incl_adjust is not None:
        xyz = xyz.view(-1, 3)
        xyz_h = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)
        xyz_h = (cam_incl_adjust.squeeze() @ xyz_h.mT).mT
        xyz = xyz_h[:, :3].view(y_res, z_res, x_res, 3)

    return xyz#.permute(2, 0, 1, 3)


def render_profile(
        net, 
        xyz_range_res, 
        cam_incl_adjust, 
        thresh: float = 8., 
        mode: Literal["density", "color", "mask", "uncert"] = "density", 
        mask_channel: int = -1, 
        color_profile: str = "magma"
    ) -> torch.Tensor:
    """"Outputs [B, H, W, 3] rgb map."""
    if isinstance(xyz_range_res, VisVolume):
        x_range, y_range, z_range = xyz_range_res.X_RANGE, xyz_range_res.Y_RANGE, xyz_range_res.Z_RANGE
        x_res, y_res, z_res = xyz_range_res.X_RES, xyz_range_res.Y_RES, xyz_range_res.Z_RES
    else:    
        x_range, y_range, z_range, x_res, y_res, z_res = xyz_range_res
    
    with torch.no_grad():
        if hasattr(net, "drgbf"):
            # Pseudo volume
            net_batch_size = net.drgbf.shape[0]
            device = net.drgbf.device
        elif hasattr(net, "grid_c_imgs"):
            # Nerual networks
            net_batch_size = net.grid_c_imgs.shape[0]
            device = net.grid_c_imgs.device
        else:
            raise NotImplementedError()
        
        q_pts = get_pts(x_range, y_range, z_range, x_res, y_res, z_res, cam_incl_adjust=cam_incl_adjust, device=device)
        q_pts = q_pts.view(1, -1, 3)

        batch_size = 50000
        if q_pts.shape[1] > batch_size:
            sigmas, invalid, rgbf, state_dicts = [], [], [], []
            l = q_pts.shape[1]
            for i in range(math.ceil(l / batch_size)):
                f = i * batch_size
                t = min((i + 1) * batch_size, l)
                q_pts_ = q_pts[:, f:t, :]
                rgbf_, invalid_, sigmas_, state_dict_ = net(q_pts_.expand(net_batch_size, -1, -1))
                rgbf.append(rgbf_)
                sigmas.append(sigmas_)
                invalid.append(invalid_)
                state_dicts.append(state_dict_)
            rgbf = torch.cat(rgbf, dim=1)
            sigmas = torch.cat(sigmas, dim=1)
            invalid = torch.cat(invalid, dim=1)
            state_dict = {k: torch.cat([sd[k] for sd in state_dicts], dim=1) for k in state_dicts[0]}
        else:
            rgbf, invalid, sigmas, state_dict = net(q_pts.expand(net_batch_size, -1, -1))
        
        invalid = invalid > 0.5 if invalid.dtype != torch.bool else invalid
        # "densities"
        sigmas = sigmas.reshape(net_batch_size, y_res, z_res, x_res)
        rgbf = rgbf.reshape(net_batch_size, y_res, z_res, x_res, rgbf.size(-1))
        # Vertical cumulative sum
        sigmas_sum = torch.cumsum(sigmas, dim=1)
        # Areas visible from BEV are smaller than some arbitrary threshold
        profile_sigmas = (sigmas_sum <= thresh).float().mean(dim=1)
        # Areas outside of the frustum as seen from BEV
        profile_invalid = invalid.reshape(net_batch_size, y_res, z_res, x_res).all(dim=1).float()
        # profile_invalid = invalid.reshape(net_batch_size, out_res.P_RES_Y, *out_res.P_RES_ZX).any(dim=1).float()
        
        match mode:
            case "density":
                # Visualizes density from BEV
                profile = color_tensor(profile_sigmas, cmap=color_profile, norm=True).float()     # [B, H, W, 3] values in [0, 1]
            case "color":
                # Visualizes density with color from BEV
                profile_rgb = rgbf[..., :3].float().mean(dim=1)
                profile = profile_sigmas * profile_rgb
            case "mask":
                # Visualizes density with a mask channel from BEV
                # mask = 1 - rgbf[..., -1]   # In my case "1=empty" and "0=marked"
                mask = rgbf[..., mask_channel]   # In my case "1=empty" and "0=marked"
                assert mask.min() >= 0., "Plot assumes positive values."
                # profile_mask = (mask / mask.max().clamp_min(1.)).mean(dim=1)
                mask_sum = torch.cumsum(sigmas * mask, dim=1)
                profile_mask = (mask_sum > 1.).float().mean(dim=1)
                # profile_mask = profile_sigmas.unsqueeze(-1) * profile_mask.unsqueeze(-1) * torch.tensor([0., 1., 0.], device=device)   # Green
                profile = color_tensor(profile_sigmas, cmap=color_profile, norm=True).float()
                # profile[:, :, :, 1] /= profile[:, :, :, 1].max()
                # profile[profile_mask.any(dim=-1)] = profile_mask[profile_mask.any(dim=-1)]
                # is_masked_and_dense = profile_mask < 0
                profile[profile_mask > 0] = profile_mask[profile_mask > 0].unsqueeze(-1) * torch.tensor([0., 1., 0.], device=device)   # Green
                # return profile / profile.max().clamp_min(1.)
            case "uncert":
                uncert = state_dict["uncertainties"]
                uncert = uncert.reshape(net_batch_size, y_res, z_res, x_res)
                uncert_sum = torch.cumsum(uncert, dim=1)
                profile_uncert = (uncert_sum > 100).float().mean(dim=1)
                profile = color_tensor(profile_uncert, cmap=color_profile, norm=True).float()
            case _:
                raise ValueError(f"Mode {mode} is not available.")

        profile[profile_invalid > 0] = torch.tensor([1., 1., 1.], device=device)   # White
        
        profile = torch.flip(profile, dims=(-3,))
            
        return profile
        

def plot_frustums(ax, poses, intrinsics, frustum_length=1., linewidths=None, linecolors=None, **kwargs):
    """Plots 2D frustums as seen from BEV onto the given plt.Axes."""
    if linewidths is None:
        linewidths = [1.] * len(poses)
    if linecolors is None:
        linecolors = ["red"] * len(poses)
        
    for i, (pose, intrinsic) in enumerate(zip(poses, intrinsics)):
        # Calculate the camera position by inverting the pose
        cam_pos = pose[:3, 3]
        # Corners of the image in image space
        corners_image = torch.tensor([
            [-1, -1, 1],  # Top-left
            # [1, -1, depth],  # Top-right
            [1., 1., 1],  # Bottom-right
            # [1., -1., depth],  # Bottom-left
        ], dtype=intrinsics.dtype) * frustum_length
        # Convert image corners to world space
        corners_world = (torch.linalg.inv(intrinsic) @ corners_image.T).T
        corners_world = (pose[:3, :3] @ corners_world.T).T + cam_pos[None, :]
        # Plot lines from camera position to corners in world space
        xz = [0, 2]
        lines = [
            (cam_pos[xz], corners_world[0, xz]),
            (corners_world[0, xz], corners_world[1, xz]),
            (corners_world[1, xz], cam_pos[xz]),
        ]
        for l in lines:
            ax.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], '-', color=linecolors[i], alpha=0.3, lw=linewidths[i])


def plot_profile(ax: plt.Axes, profile: torch.Tensor, vis_volume: VisVolume, poses: Optional[torch.Tensor] = None, projs: Optional[torch.Tensor] = None, flip: bool = False, **kwargs):
    """Plotting helper funtion that draws the given profiles with frustums."""
    set_thesis_rcparams()
    
    if flip:
        profile = torch.flip(profile, dims=(-3,))
    
    x0, x1 = vis_volume.X_RANGE
    z0, z1 = vis_volume.Z_RANGE
    ax.imshow(
        profile, 
        origin="lower",             # bottom-left is (0, 0)
        extent=[x0, x1, z0, z1],    # (left, right, bottom, top)
        interpolation="none",
    )
    ax.set(xlabel="X [m]", ylabel="Z [m]")
    
    if poses is not None and projs is not None:
        plot_frustums(ax, poses, projs, **kwargs)
        
        
def format_conditioning_imgs(conditioning_imgs: torch.Tensor) -> torch.Tensor:
    """Lays out different channels along the image width."""
    expand_shape = [1] * len(conditioning_imgs.shape)
    expand_shape[-3] = 3
    conditioning_image_parts = [conditioning_imgs[..., :3, :, :]]
    for c in range(3, conditioning_imgs.shape[-3]):
        conditioning_image_parts.append(
            conditioning_imgs[..., c:c+1, :, :].repeat(expand_shape)   # Single channel to RGB
        )
    formatted_imgs = torch.concat(conditioning_image_parts, dim=-1)
    return formatted_imgs