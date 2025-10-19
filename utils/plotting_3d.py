import io 
from PIL import Image
from typing import Optional

import torch
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData, PlyElement

from utils.plotting import cmap_magma


def plotly_fig2array(fig: go.Figure):
    """Converts a Plotly fig to an RGBA-array."""
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def make_3d_fig(
    title: str = "",
    up: tuple = (0, -1, 0),
    eye: tuple = (-1, -1, -1),
    center: tuple = (0, 0, 0),
    showgrid=True,
    transparent_plot_bg=True,
    transparent_paper_bg=True,
    width=500,
    height=500,
    aspectratio: dict | None = None,
    ) -> go.Figure:
    """Prepares the figure with common styling."""
    
    fig = go.Figure()

    aspect = {
        # can be 'data', 'cube', 'auto', 'manual'
        "aspectmode": "manual",
        "aspectratio": aspectratio,
    } if aspectratio else {
        "aspectmode": "data"
    }
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X [m]', showgrid=showgrid, zeroline=showgrid, visible=showgrid),
            yaxis=dict(title='Y [m]', showgrid=showgrid, zeroline=showgrid, visible=showgrid),
            zaxis=dict(title='Z [m]', showgrid=showgrid, zeroline=showgrid, visible=showgrid),
            **aspect
            # TODO set this such that the scene does not change scale in a video
        ),
        scene_camera=dict(
            up=dict(x=up[0], y=up[1], z=up[2]),  # Scene is oriented such that this vector points up
            center=dict(x=center[0], y=center[1], z=center[2]),
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),  # Init camera "behind" world frame
        ),
        width=width, 
        height=height,
        paper_bgcolor="rgba(0,0,0,0)" if transparent_paper_bg else "rgba(255,255,255,1)",
        margin=dict(r=0, l=0, b=0, t=0)
    )
    if transparent_plot_bg:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    
    return fig


def dpi2scale(width_default_px, width_in_mm=None, width_in_inches=None, dpi=1000):
    if width_in_mm is None:
        width_in_mm = width_in_inches * 25.4
    scale = (width_in_mm / 25.4) / (width_default_px / dpi)
    return scale


def draw_camera_with_frustum(
    fig: go.Figure, 
    camera_pose, 
    camera_intrinsics, 
    camera_id: Optional[str] = None,
    image: Optional[np.array] = None, 
    frustum_depth=1.0,
    frustum_line_width=2.0,
    frame_color="grey",
    frame_line_width=3.0,
    showlegend=False,
):
    """
    Draw the camera coordinate system and frustum in a 3D plot using Plotly, and optionally display an RGB image on the frustum plane.

    :param fig: Plotly figure object to which the camera and frustum will be added.
    :param camera_pose: Camera pose in homogeneous coordinates (4x4 matrix). Represents the camera-to-world transformation.
    :param camera_intrinsics: Camera intrinsics in homogeneous coordinates (3x3 matrix).
    :param image: Optional RGB image array to be displayed on the plane of the frustum. Shape: [H, W, 3] in range [0, 1].
    :param scale: Scaling factor for the size of the coordinate axes and frustum.
    """
    
    legend_group_name = "Camera" + f" {camera_id}" if camera_id else ""
    
    # Extracting camera position and rotation from camera pose
    camera_pos = camera_pose[:3, 3]
    R_cam2world = camera_pose[:3, :3]

    # Draw the camera coordinate frame in world coordinates.
    axes_cam = np.eye(3) * frustum_depth
    axes_world = R_cam2world @ axes_cam + camera_pos[:, None]

    for i, color, name in zip(range(3), ['red', 'green', 'blue'], ["X", "Y", "Z"]):
        fig.add_trace(go.Scatter3d(
            x=[camera_pos[0], axes_world[0, i]], 
            y=[camera_pos[1], axes_world[1, i]], 
            z=[camera_pos[2], axes_world[2, i]], 
            mode='lines', 
            line=dict(color=color, width=frame_line_width),
            hoverinfo="none",
            showlegend=showlegend, 
            legendgroup=legend_group_name,
            name=name
        ))
        
    # Homogeneous normalized image coordinates
    image_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])  # top left, top right, bottom right, bottom left
    image_corners = np.concatenate((image_corners, np.ones((4,1))*frustum_depth), axis=1)
    
    # Unproject and transform to world coordinates
    transformed_corners = (np.linalg.inv(camera_intrinsics) @ image_corners.T).T
    transformed_corners = (R_cam2world @ transformed_corners.T).T + camera_pos[None, :]
    
    # Adding frustum lines
    for i in range(4):
        next_i = (i + 1) % 4
        # Lines between frustum corners
        fig.add_trace(go.Scatter3d(
            x=[transformed_corners[i, 0], transformed_corners[next_i, 0]], 
            y=[transformed_corners[i, 1], transformed_corners[next_i, 1]], 
            z=[transformed_corners[i, 2], transformed_corners[next_i, 2]], 
            mode='lines', 
            line=dict(color=frame_color, width=frustum_line_width),
            showlegend=False,
            legendgroup=legend_group_name,
            hoverinfo="none"
        ))

        # Lines from camera center to frustum corners
        fig.add_trace(go.Scatter3d(
            x=[camera_pos[0], transformed_corners[i, 0]], 
            y=[camera_pos[1], transformed_corners[i, 1]], 
            z=[camera_pos[2], transformed_corners[i, 2]], 
            mode='lines', 
            line=dict(color='gray', width=2),
            showlegend=False,
            legendgroup=legend_group_name,
            hoverinfo="none"
        ))

    # Adding the image to the frustum plane if provided
    if image is not None:
        surface = create_3d_image_surface(image_array=image, corners=transformed_corners, legendgroup=legend_group_name)
        fig.add_trace(surface)

    return fig


def create_3d_image_surface(image_array: np.array, corners: np.array, legendgroup: Optional[str] = None) -> go.Surface:
    """
    Map an image onto an arbitrary quadrilateral in 3D space using all color channels.

    :param image_array: Numpy array of shape [H, W, 3] with range [0, 1].
    :param corners: List of four 3D corner points [(x1, y1, z1), ..., (x4, y4, z4)]. 
                    Order should be [top left, top right, bottom right, bottom left].
    :return: Plotly go.Surface object.
    """
    if not ((0 <= image_array) & (image_array <= 1)).all():
        image_array = np.clip(image_array / 255, 0, 1)
    
    # Generate grid for image dimensions.
    height, width = image_array.shape[:2]
    u = np.linspace(0, 1, width)
    v = np.linspace(0, 1, height)
    u_grid, v_grid = np.meshgrid(u, v)

    # Bilinear interpolation to map the image onto the arbitrary quadrilateral.
    x = (1 - u_grid) * (1 - v_grid) * corners[0][0] + \
        u_grid * (1 - v_grid) * corners[1][0] + \
        u_grid * v_grid * corners[2][0] + \
        (1 - u_grid) * v_grid * corners[3][0]

    y = (1 - u_grid) * (1 - v_grid) * corners[0][1] + \
        u_grid * (1 - v_grid) * corners[1][1] + \
        u_grid * v_grid * corners[2][1] + \
        (1 - u_grid) * v_grid * corners[3][1]

    z = (1 - u_grid) * (1 - v_grid) * corners[0][2] + \
        u_grid * (1 - v_grid) * corners[1][2] + \
        u_grid * v_grid * corners[2][2] + \
        (1 - u_grid) * v_grid * corners[3][2]

    # Extract RGB channels for the surface color.
    rgb = np.zeros(image_array.shape, dtype=np.float32)
    for i in range(3):
        rgb[:, :, i] = (1 - u_grid) * (1 - v_grid) * image_array[:, :, i] + \
                       u_grid * (1 - v_grid) * image_array[:, :, i] + \
                       u_grid * v_grid * image_array[:, :, i] + \
                       (1 - u_grid) * v_grid * image_array[:, :, i]

    # Create the surface color.
    # Reference: https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot
    rgb = Image.fromarray(np.uint8(rgb*255)).convert('P', palette='WEB', dither=None)
    dum_img = Image.fromarray(np.ones((3,3,3), dtype='uint8')).convert('P', palette='WEB')
    idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    
    return go.Surface(
        x=x, y=y, z=z, 
        surfacecolor=rgb, 
        colorscale=colorscale,
        cmin=0, 
        cmax=255,
        showscale=False,
        showlegend=False,
        legendgroup=legendgroup,
        hoverinfo="none",
    )
    
    
def draw_camera_ray(fig: go.Figure, ray_origins: np.array, ray_dirs: np.array, ray_samples: np.array) -> go.Figure:
    
    ray_lines = go.Scatter3d(
        x=dates, y=y, z=z,
        line=dict(
            color='darkblue',
            width=2
        )
    )
    
    ray_samples = go.Scatter3d(
        x=dates, y=y, z=z,
        marker=dict(
            size=4,
            color=z,
            colorscale='Viridis',
        )
    )
    
    fig.add_trace(ray_lines)
    fig.add_trace(ray_samples)
    
    return fig
    
    
def draw_point_cloud(
    fig: go.Figure, 
    xyz: np.ndarray, 
    rgb: np.ndarray | None = None,
    colorscale: Optional[str] = None,
    marker_size: float = 1.0,
) -> go.Figure:
    """Scatters pointcloud in 3D."""
    color = None
    if rgb is not None:
        color = rgb.astype(np.int64)
        color = [f"rgb({r}, {g}, {b})" for r, g, b in color]
        
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=color,
            colorscale=colorscale if colorscale else "plotly3",   # "viridis"
        ),
        hoverinfo="all",
        showlegend=True,
    ))
    
    return fig


def draw_density_field(
    fig: go.Figure,
    grid: np.ndarray,
    values: np.ndarray,
) -> go.Figure:
    """Draws a Plotly volume into the figure.
    
    Args:
        fig (go.Figure)
        grid (np.ndarray): Uniform and axis-aligned grid. Shape [x, y, z, 3].
        values (np.ndarray): Grid density values. Shape [x, y, z].
    """
    
    assert len(grid.shape) == 4
    assert len(values.shape) == 3
    x = grid[:, :, :, 0].flatten()
    y = grid[:, :, :, 1].flatten()
    z = grid[:, :, :, 2].flatten()
    vals = values.flatten()
    
    fig.add_trace(go.Volume(
        x=x,
        y=y,
        z=z,
        value=vals,
        isomin=0.1,
        isomax=np.round(np.max(values)*0.8, decimals=1),
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
    ))
    
    return fig


def build_voxels(ijks, x_res, y_res, z_res, xyz, colors, faces_t, is_occupied_padded, keep_all=False):
    device = ijks.device
    ids_offset = torch.tensor(
        [[1, 1, 0], [1, 0, 0],
         [0, 0, 0], [0, 1, 0],
         [1, 1, 1], [1, 0, 1],
         [0, 0, 1], [0, 1, 1]],
        dtype=torch.int32,
        device=device)

    # Compute global indices of voxel vertices
    ids = ijks.view(-1, 1, 3) + ids_offset.view(1, -1, 3)
    ids_flat = ids[..., 0] * y_res * z_res + ids[..., 1] * z_res + ids[..., 2]
    verts = xyz[:, ids_flat.reshape(-1)]

    # Determine visibility by checking if any vertex in the face is exposed (not surrounded on all sides)
    # Expand index bounds to handle padded occupancy grid
    check_offsets = torch.tensor([
        [-1, 0, 0], [1, 0, 0],
        [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1]
    ], dtype=torch.int32, device=device)

    # # Only keep faces where at least one vertex is exposed
    # mask = torch.stack([is_exposed(ijk) for ijk in ids.reshape(-1, 3)]).reshape(-1, 8)
    # mask_faces = mask[:, faces_t.flatten()].reshape(-1, 4).any(dim=1)

    # Calculate all neighbor indices in one go
    neighbor_indices = ids.unsqueeze(2) + check_offsets.unsqueeze(0).unsqueeze(0)
    # neighbor_indices = torch.clamp(neighbor_indices, 0, torch.tensor([x_res+1, y_res+1, z_res+1], device=device) - 1)
    neighbor_indices = torch.clamp(neighbor_indices, torch.tensor([0, 0, 0], device=device), torch.tensor([x_res+1, y_res+1, z_res+1], device=device) - 1)
    # Determine the occupancy of each neighbor
    is_occupied_neighbors = is_occupied_padded[neighbor_indices[..., 0], neighbor_indices[..., 1], neighbor_indices[..., 2]]
    is_exposed = is_occupied_neighbors.sum(dim=2) != 6  # A vertex is exposed if not all neighbors are occupied
    # Only keep faces where at least one vertex is exposed
    mask_faces = is_exposed.view(-1, 8)[:, faces_t.flatten()].view(-1, 4).any(dim=1)

    # Apply mask to faces and colors
    faces_off = torch.arange(0, ijks.shape[0] * 8, 8, device=device).view(-1, 1, 1) + faces_t.view(-1, 6, 4)
    # colors = y_to_color[ijks[:, 1], :].view(-1, 1, 3).expand(-1, 6, -1)
    colors = colors[ijks[:, 0], ijks[:, 1], ijks[:, 2], :].view(-1, 1, 3).expand(-1, 6, -1)
    
    if keep_all:
        mask_faces = torch.ones_like(mask_faces).bool()
    
    return verts.cpu().numpy().T, faces_off.reshape(-1, 4)[mask_faces].cpu().numpy(), colors.reshape(-1, 3)[mask_faces].cpu().numpy()

# def occupancy_grid_to_ply(is_occupied: torch.Tensor, xyz: torch.Tensor, color_grad_dir: tuple = (0., 1., 0.), keep_all=False):
#     device = is_occupied.device
    
#     x_res, y_res, z_res, _ = xyz.shape
    
#     color_grad_dir = torch.FloatTensor(color_grad_dir, device=device)
#     color_grad_dir = color_grad_dir / color_grad_dir.norm()
#     # Project the voxel centers onto the vector
#     centers = xyz[:-1, :-1, :-1] + torch.tensor([0.5 / x_res, 0.5 / y_res, 0.5 / z_res], device=device)
#     projections = torch.matmul(centers, color_grad_dir)
#     # Generate a colormap based on the projections.
#     projection_steps = (projections - projections.min()) / (projections.max() - projections.min())
#     # color_indices = (projection_steps * (cmap.N - 1)).long()
#     # colors = torch.tensor(cmap.colors[color_indices], device=device)[:, :3] * 255
#     # colors = colors.to(torch.uint8)
#     colors = (torch.tensor(list(map(cmap_magma, projection_steps.view(-1))), device=device)[:, :3] * 255).to(torch.uint8).view(x_res-1, y_res-1, z_res-1, 3)
    
#     faces = [[0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [4, 5, 6, 7]]
#     faces_t = torch.tensor(faces, device=device)
    
#     # Pad occupancy grid to handle boundary conditions
#     padded_occupancy = torch.zeros((x_res+2-1, y_res+2-1, z_res+2-1), dtype=torch.uint8, device=device)
#     padded_occupancy[1:-1, 1:-1, 1:-1] = is_occupied
    
#     # y_steps = (1 - (torch.linspace(0, 1 - 1/y_res, y_res) + 1 / (2 * y_res))).tolist()
#     # cmap = plt.cm.get_cmap("magma")
#     # y_to_color = (torch.tensor(list(map(cmap, y_steps)), device=device)[:, :3] * 255).to(torch.uint8)
    
#     verts, faces, colors = build_voxels(is_occupied.nonzero(), x_res, y_res, z_res, xyz.reshape(-1, 3).T, colors, faces_t, padded_occupancy, keep_all=keep_all)

#     verts = list(map(tuple, verts))
#     verts_data = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

#     face_data = np.array(faces, dtype='i4')
#     color_data = np.array(colors, dtype='u1')
#     ply_faces = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,)),  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

#     ply_faces['vertex_indices'] = face_data
#     ply_faces["red"] = color_data[:, 0]
#     ply_faces["green"] = color_data[:, 1]
#     ply_faces["blue"] = color_data[:, 2]

#     verts_el = PlyElement.describe(verts_data, "vertex")
#     faces_el = PlyElement.describe(ply_faces, "face")
    
#     return PlyData([verts_el, faces_el])


def occupancy_grid_to_ply(is_occupied: torch.Tensor, xyz: torch.Tensor, color_grad_dir: tuple = (0., 1., 0.), keep_all=False):
    import numpy.lib.recfunctions as rfn

    device = is_occupied.device
    x_res, y_res, z_res, _ = xyz.shape
    
    color_grad_dir = torch.FloatTensor(color_grad_dir).to(device)
    color_grad_dir = color_grad_dir / color_grad_dir.norm()
    
    centers = xyz[:-1, :-1, :-1] + torch.tensor([0.5 / x_res, 0.5 / y_res, 0.5 / z_res], device=device)
    projections = torch.matmul(centers, color_grad_dir)
    projection_steps = (projections - projections.min()) / (projections.max() - projections.min())
    colors = (torch.tensor(list(map(cmap_magma, projection_steps.view(-1))), device=device)[:, :3] * 255).to(torch.uint8).view(x_res-1, y_res-1, z_res-1, 3)

    faces = [[0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [4, 5, 6, 7]]
    faces_t = torch.tensor(faces, device=device)

    padded_occupancy = torch.zeros((x_res+2-1, y_res+2-1, z_res+2-1), dtype=torch.uint8, device=device)
    padded_occupancy[1:-1, 1:-1, 1:-1] = is_occupied

    verts, faces, face_colors = build_voxels(
        is_occupied.nonzero(), x_res, y_res, z_res,
        xyz.reshape(-1, 3).T, colors, faces_t,
        padded_occupancy, keep_all=keep_all
    )

    # Map face colors to vertex colors (averaging over faces per vertex)
    # verts = verts.T  # shape: (N, 3)
    color_accum = np.zeros((verts.shape[0], 3), dtype=np.float64)
    counts = np.zeros((verts.shape[0],), dtype=np.int32)

    for face, color in zip(faces, face_colors):
        for vi in face:
            color_accum[vi] += color
            counts[vi] += 1

    counts[counts == 0] = 1  # Prevent division by zero
    vertex_colors = (color_accum / counts[:, None]).astype(np.uint8)

    # Create structured arrays
    verts_data = np.zeros(verts.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    verts_data['x'] = verts[:, 0]
    verts_data['y'] = verts[:, 1]
    verts_data['z'] = verts[:, 2]
    verts_data['red'] = vertex_colors[:, 0]
    verts_data['green'] = vertex_colors[:, 1]
    verts_data['blue'] = vertex_colors[:, 2]

    face_data = np.array(faces, dtype='i4')
    ply_faces = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,))])
    ply_faces['vertex_indices'] = face_data
    
    verts_el = PlyElement.describe(verts_data, "vertex")
    faces_el = PlyElement.describe(ply_faces, "face")
    
    return PlyData([verts_el, faces_el])



# def occupancy_grid_to_ply(is_occupied: torch.Tensor, xyz: torch.Tensor, x_res, y_res, z_res):
#     # Following
#     # https://github.com/ruili3/Know-Your-Neighbors/blob/main/scripts/gen_kitti360_voxel.py
#     device = is_occupied.device
    
#     faces = [[0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [4, 5, 6, 7]]
#     faces_t = torch.tensor(faces, device=device)

#     y_steps = (1 - (torch.linspace(0, 1 - 1/y_res, y_res) + 1 / (2 * y_res))).tolist()
#     cmap = plt.cm.get_cmap("magma")
#     y_to_color = (torch.tensor(list(map(cmap, y_steps)), device=device)[:, :3] * 255).to(torch.uint8)
    
#     verts, faces, colors = build_voxels(is_occupied.nonzero(), x_res, y_res, z_res, xyz.squeeze(0).T, y_to_color, faces_t)

#     verts = list(map(tuple, verts))
#     verts_data = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

#     face_data = np.array(faces, dtype='i4')
#     color_data = np.array(colors, dtype='u1')
#     ply_faces = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,)),  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

#     ply_faces['vertex_indices'] = face_data
#     ply_faces["red"] = color_data[:, 0]
#     ply_faces["green"] = color_data[:, 1]
#     ply_faces["blue"] = color_data[:, 2]

#     verts_el = PlyElement.describe(verts_data, "vertex")
#     faces_el = PlyElement.describe(ply_faces, "face")
    
#     return PlyData([verts_el, faces_el])


# def draw_ply_mesh(
#     fig: go.Figure,
#     ply: PlyData,
# ) -> go.Figure:
#     """Draws a Plotly volume into the figure.
    
#     Args:
#         fig (go.Figure)
#         grid (np.ndarray): Uniform and axis-aligned grid. Shape [x, y, z, 3].
#         values (np.ndarray): Grid density values. Shape [x, y, z].
#     """
#     vertex = ply['vertex']
#     x, y, z = vertex['x'], vertex['y'], vertex['z']
    
#     face = ply['face']
#     faces = face.data['vertex_indices']
#     faces = np.stack(faces)
    
#     # r = face.data['red']
#     # g = face.data['green']
#     # b = face.data['blue']
    
#     triangles = []
#     facecolors = []
#     for quad, r, g, b in zip(faces, face.data['red'], face.data['green'], face.data['blue']):
#         triangles.append([quad[0], quad[1], quad[2]])
#         triangles.append([quad[0], quad[2], quad[3]])
#         facecolors.extend([f"rgb({r}, {g}, {b})"] * 2)
#     triangles = np.array(triangles)
    
    
#     fig.add_trace(go.Mesh3d(
#         x=x, 
#         y=y, 
#         z=z,
#         i=triangles[:, 0], 
#         j=triangles[:, 1], 
#         k=triangles[:, 2],
#         color='lightblue', 
#         opacity=1.0,
#         # facecolor=[f"rgb({r}, {g}, {b})" for r, g, b in zip(r, g, b)],
#         facecolor=facecolors,
#     ))
    
#     return fig


def draw_ply_mesh(fig: go.Figure, ply: PlyData) -> go.Figure:
    vertex = ply['vertex']
    x, y, z = vertex['x'], vertex['y'], vertex['z']
    
    face = ply['face']
    faces = np.stack(face.data['vertex_indices'])
    triangles = []
    for quad in faces:
        triangles.append([quad[0], quad[1], quad[2]])
        triangles.append([quad[0], quad[2], quad[3]])
    triangles = np.array(triangles)

    # Use vertex color
    color = [f"rgb({r},{g},{b})" for r, g, b in zip(vertex['red'], vertex['green'], vertex['blue'])]

    fig.add_trace(go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        vertexcolor=color,
        opacity=1.0,
    ))

    return fig



