import torch
import torch.amp
import torch.nn.functional as F

# Used for avoiding zero-divisions
EPS = 1e-3


def normalize_calib(K: torch.Tensor, img_sizes: torch.Tensor) -> torch.Tensor:
    """Normalize the calibration matrices for fisheye cameras based on the image size

    Args:
        calib (torch.Tensor): B, n_views, 3, 3
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views 7
    """

    K[..., :2, :] = K[..., :2, :] / img_sizes.unsqueeze(-1) * 2.0
    K[..., :2, 2] = K[..., :2, 2] - 1.0

    return K


def unnormalize_calib(K: torch.Tensor, img_sizes: torch.Tensor) -> torch.Tensor:
    """Unnormalize the calibration matrices for fisheye cameras based on the image size

    Args:
        calib (torch.Tensor): B, n_views, 3, 3
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views 7
    """

    K[..., :2, 2] = K[..., :2, 2] + 1.0
    K[..., :2, :] = K[..., :2, :] * img_sizes.unsqueeze(-1) / 2.0

    return K


def pts_into_camera_or_world(pts: torch.Tensor, poses_w2c_or_c2w: torch.Tensor) -> torch.Tensor:
    """Project points from world coordinates into camera coordinate

    Args:
        pts (torch.Tensor): B, n_pts, 3
        poses_w2c (torch.Tensor): B, n_view, 4, 4

    Returns:
        torch.Tensor: B, n_views, n_pts, 3
    """

    # Add a singleton dimension to the input point cloud to match grid_f_poses_w2c shape
    pts = pts.unsqueeze(1)  # [B, 1, n_pts, 3]
    ones = torch.ones_like(
        pts[..., :1]
    )  ## Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
    pts = torch.cat(
        (pts, ones), dim=-1
    )  ## Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
    return (poses_w2c_or_c2w[:, :, :3, :]) @ pts.permute(0, 1, 3, 2)


def unproject_from_image(uv: torch.Tensor, Ks: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    """Unproject points from image coordinates into camera coordinates

    Args:
        uv (torch.Tensor): [B, n_views, n_pts, 2]. The pixel coordinates.
        Ks (torch.Tensor): [B, n_views, 3, 3]
        z (torch.Tensor, optional): [B, n_views, n_pts, 1]. The depth values in Z.

    Returns:
        torch.Tensor: B, n_views, n_pts, 3
    """

    # if z is None:
    pts = torch.cat(
        (uv, torch.ones_like(uv[..., :1])), dim=-1
    )  ## Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
    # else:
    #     pts = torch.cat((uv, z), dim=-1)

    Ks_inv = torch.inverse(Ks.float()).to(Ks)
    pts = Ks_inv.matmul(pts.transpose(-1, -2)).transpose(-1, -2)
    pts = pts * z
    return pts


def project_to_image(
    pts: torch.Tensor, Ks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project pts in camera coordinates into image coordinates.

    Args:
        pts (torch.Tensor): [B, n_views, n_pts, 3]
        Ks (torch.Tensor): [B, n_views, 3, 3]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (B, n_views, n_pts, 2), (B, n_views, n_pts, 1)
    """
    pts = (Ks @ pts).permute(
        0, 1, 3, 2
    )  ## Apply the intrinsic camera parameters to the projected points to get pixel coordinates
    xy = pts[
        :, :, :, :2
    ]  ## Extract the x,y coordinates and depth value from the projected points
    z_ = pts[:, :, :, 2:3]

    xy = xy / z_.clamp_min(EPS)

    return xy, z_


def outside_frustum(
    xy: torch.Tensor,
    z: torch.Tensor,
    limits_x: tuple[float, float] | tuple[int, int] = (-1.0, 1.0),
    limits_y: tuple[float, float] | tuple[int, int] = (-1.0, 1.0),
    limit_z: float = EPS,
) -> torch.Tensor:
    """_summary_

    Args:
        xy (torch.Tensor): _description_
        z (torch.Tensor): _description_
        limits_x (tuple[float, float] | tuple[int, int], optional): _description_. Defaults to (-1.0, 1.0).
        limits_y (tuple[float, float] | tuple[int, int], optional): _description_. Defaults to (-1.0, 1.0).
        limit_z (float, optional): _description_. Defaults to EPS.

    Returns:
        torch.Tensor: _description_
    """
    return (
        (z <= limit_z)
        | (xy[..., :1] < limits_x[0])
        | (xy[..., :1] > limits_x[1])
        | (xy[..., 1:2] < limits_y[0])
        | (xy[..., 1:2] > limits_y[1])
    )


def outside_frustum_diff(
    xy: torch.Tensor,
    z: torch.Tensor,
    limits_x: tuple[float, float] | tuple[int, int] = (-1.0, 1.0),
    limits_y: tuple[float, float] | tuple[int, int] = (-1.0, 1.0),
    limit_z: float = EPS,
    transition_sharpness: float = 1.,
) -> torch.Tensor:
    """Differentiable version of `outside_frustum`."""
    outside_probs = torch.stack([
        F.sigmoid((limit_z - z) * transition_sharpness),
        F.sigmoid((limits_x[0] - xy[..., :1]) * transition_sharpness),
        F.sigmoid((xy[..., :1] - limits_x[1]) * transition_sharpness),
        F.sigmoid((limits_y[0] - xy[..., 1:2]) * transition_sharpness),
        F.sigmoid((xy[..., 1:2] - limits_y[1]) * transition_sharpness)
    ], dim=-1)
    return outside_probs.max(dim=-1)[0]
    

def unproj_map(width, height, f, c=None, device="cpu", dtype=torch.float32, norm_dir=True, xy_offset=None):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = torch.tensor([[0.0, 0.0]], device=device)
    elif isinstance(c, float):
        c = torch.tensor([[c, c]], device=device)
    elif len(c.shape) == 0:
        c = c[None, None].expand(1, 2)
    elif len(c.shape) == 1:
        c = c.unsqueeze(-1).expand(1, 2)

    if isinstance(f, float):
        f = torch.tensor([[f, f]], device=device)
    elif len(f.shape) == 0:
        f = f[None, None].expand(1, 2)
    elif len(f.shape) == 1:
        f = f.unsqueeze(-1).expand(1, 2)
    n = f.shape[0]
    
    # x = torch.linspace(-1, 1, width, dtype=dtype, device=device).view(1, 1, width).expand(n, height, width)
    # y = torch.linspace(-1, 1, height, dtype=dtype, device=device).view(1, height, 1).expand(n, height, width)
    # xy = torch.stack((x, y), dim=-1)
    # xy = (xy - c.view(n, 1, 1, 2)) / f.view(n, 1, 1, 2)
    # z = torch.ones_like(x).unsqueeze(-1)
    # unproj = torch.cat((xy, z), dim=-1)
    
    pixel_width = 2 / width
    pixel_height = 2 / height

    x = torch.linspace(-1 + .5 * pixel_width, 1 - .5 * pixel_width, width, dtype=dtype, device=device).view(1, 1, width).expand(n, height, width)
    y = torch.linspace(-1 + .5 * pixel_height, 1 - .5 * pixel_height, height, dtype=dtype, device=device).view(1, height, 1).expand(n, height, width)

    if xy_offset is not None:
        x = x + xy_offset[0] * pixel_width
        y = y + xy_offset[1] * pixel_height

    xy_img = torch.stack((x, y), dim=-1)
    xy = (xy_img - c.view(n, 1, 1, 2)) / f.view(n, 1, 1, 2)
    z = torch.ones_like(x).unsqueeze(-1)
    unproj = torch.cat((xy, z), dim=-1)

    if norm_dir:
        unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj




# def unproj_grid(width, height, depth, z_near, z_far, K, inv_z = False, device="cpu", dtype=torch.float32):
#     """
#     Get camera unprojection grid for given image size and grid depth.
#     """
#     b = K.shape[0]

#     pixel_width = 2 / width
#     pixel_height = 2 / height

#     x = torch.linspace(-1 + .5 * pixel_width, 1 - .5 * pixel_width, width, dtype=dtype, device=device).view(1, 1, width, 1).expand(b, height, width, depth)
#     y = torch.linspace(-1 + .5 * pixel_height, 1 - .5 * pixel_height, height, dtype=dtype, device=device).view(1, height, 1, 1).expand(b, height, width, depth)

#     # encoding_mode is hardcoded to "z" here
    
#     if inv_z:
#         z = torch.linspace(1/z_near, 1/z_far, depth, dtype=dtype, device=device).view(1, 1, 1, depth).expand(b, height, width, depth)
#         z = (1 / z.clamp_min(EPS)).to(dtype)
#         # linear_space = torch.linspace(1 / end, 1 / start, steps)
#         # inv_space = 1 / linear_space
#         # TODO check if this is correct
#         # z = torch.div(1, z.clamp_min(EPS))
#         # z = z / z.max()
#         # z = z * (z_far - z_near) + z_near
#     else:
#         z = torch.linspace(0, 1, depth, dtype=dtype, device=device).view(1, 1, 1, depth).expand(b, height, width, depth)
#         z = z * (z_far - z_near) + z_near

#     xy1 = torch.stack((x, y, torch.ones_like(x)), dim=-1)
#     # xy = (xy_img - c.view(n, 1, 1, 2)) / f.view(n, 1, 1, 2)
#     # Cast K to FP32 for matrix multiplication
#     K_inv = torch.inverse(K.float()).to(xy1)
#     xy1_unproj = (K_inv @ xy1.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
#     xyz = xy1_unproj * z.unsqueeze(-1)

#     return xyz  # [B, H, W, D, 3]


# if __name__ == "__main__":
#     # Test the `unproj_map` function
#     # import matplotlib.pyplot as plt

#     f = 1.0
#     c = 0.0
#     width = 24
#     height = 12
#     depth = 64
#     K = torch.tensor([[f, 0, 0], [0, f, 0], [0, 0, 1]]).unsqueeze(0)
#     xyz = unproj_grid(width, height, depth, 3.0, 40, K, inv_z=True)

#     # xzy


#     pass