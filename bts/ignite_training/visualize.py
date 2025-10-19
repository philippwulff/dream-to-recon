import math
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from utils.plotting import color_tensor
from torchvision.utils import make_grid
from vcm.utils.eval_utils import format_conditioning_imgs
from utils.utils import invert_depth


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):

    data = engine.state.output["output"]
    writer = logger.writer
    
    z_near = data["z_near"]
    z_far = data["z_far"]
    idxs = data["idxs"].squeeze().tolist()
    idxs = idxs if isinstance(idxs, list) else [idxs]

    images = torch.stack(data["imgs"], dim=1).detach()
    # Only display a subset of the batch and novel views
    take_n = min(images.shape[0], 10)

    profiles = data["profiles"].detach()
    images = images[:take_n, :].float() * .5 + .5
    images = images.permute(0, 2, 3, 1, 4).reshape(take_n, 3, images.shape[-2], -1)   # stack horizontally
    profiles = profiles[:take_n].float()
    profiles = profiles.permute(0, 3, 1, 2)
    if "profiles_pseudo" in data:
        profiles_pseudo = data["profiles_pseudo"].detach().permute(0, 3, 1, 2)
        profiles_pseudo = profiles_pseudo[:take_n].float()
        # profiles = torch.concat([profiles, profiles_pseudo], dim=-1)
        for i, idx in enumerate(idxs):
            writer.add_image(f"{tag}/IN_{idx}/profile_pseudo", profiles_pseudo[i].cpu(), global_step=step)

    has_gt = "images_synthetic_cond" in data
    if has_gt:    
        depths_gt_in = data["depths_in"].detach()
        depths_gt_in = depths_gt_in[:take_n, :].float()
        depths_gt_in_to_plot = depths_gt_in.permute(0, 2, 3, 1, 4).reshape(take_n, 1, depths_gt_in.shape[-2], -1)   # stack horizontally
        depths_gt_in = depths_gt_in[:, 0]
        # Synthetic GT
        depths_gt = data["depths_synthetic_gt"].detach()
        depths_reproj = data["depths_synthetic_reproj"].detach()
        # depths_gt = depths_gt.view(n*nv, 1, h, w)
        images_gt = data["images_synthetic_gt"].detach()#.view(n*nv, 3, h, w)
        # images_gt = images_gt * 0.5 + 0.5
        images_cond = data["images_synthetic_cond"].detach()#.view(n*nv, 3, h, w)

    has_renders = "fine" in data
    if has_gt:
        _, nv, _, h, w = depths_gt.shape 
    elif has_renders:
        _, nv, h, w, _, _ = data["fine"][0]["rgb"].shape         # FIXME
    else:
        nv, _, h, w = data["imgs"][0].shape
    take_nv = min(nv, 6)

    if has_renders:
        recon_imgs = data["fine"][0]["rgb"].detach()
        recon_imgs_in = data["fine_in"][0]["rgb"].detach()
        recon_depths = data["fine"][0]["depth"].detach()   # [n, nv, H, W]
        recon_depths_var = data["fine"][0]["depth_var"].detach()   # [n, nv, H, W]
        recon_depths_in = data["fine_in"][0]["depth"].detach()   # [n, nv, H, W]
        recon_depths_var_in = data["coarse_in"][0]["depth_var"].detach()

        recon_depths_in = recon_depths_in[:take_n, 0].float()
        recon_depths_var_in = recon_depths_var_in[:take_n, 0].float()
        recon_imgs_in = recon_imgs_in[:take_n].view(-1, 1, *recon_imgs_in.shape[2:4]).float()

        if has_gt:
            recon_err_in = (depths_gt_in.squeeze(-3) - recon_depths_in).abs()
            recon_err_in = color_tensor((recon_err_in / recon_err_in.max()).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2)
            
            recon_err_rel_in = (depths_gt_in.squeeze(-3) - recon_depths_in).abs() 
            recon_err_rel_in = color_tensor((recon_err_rel_in / depths_gt_in.squeeze(-3)).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2)
        
        recon_depths_in = color_tensor(invert_depth(recon_depths_in, z_near, z_far).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2)
        recon_depths_var_in = color_tensor((recon_depths_var_in/recon_depths_var_in.max()).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2)
    
        invalids_in = {}
        for k, v in data["coarse_in"][0].items():
            if k.startswith("invalid"):
                invalids_in[k] = color_tensor(v[:take_n, 0].detach().float().mean(-1), cmap="plasma").permute(0, 3, 1, 2)
        invalids = {}
        for k, v in data["coarse"][0].items():
            if k.startswith("invalid"):
                invalids[k] = [color_tensor(_.detach().float().mean(-1), cmap="plasma").permute(0, 3, 1, 2) for _ in v[:take_n, :take_nv]]
    
    # depths_gt_in = color_tensor((depths_gt_in / depths_gt_in.max()).clamp(0, 1).view(-1, *depths_gt_in.shape[-2:]), cmap="plasma").permute(0, 3, 1, 2)
    if has_gt:
        depths_gt_in_to_plot = color_tensor(invert_depth(depths_gt_in_to_plot, z_near, z_far).clamp(0, 1).view(-1, *depths_gt_in_to_plot.shape[-2:]), cmap="plasma").permute(0, 3, 1, 2)
    
    for i, idx in enumerate(idxs):
        writer.add_image(f"{tag}/IN_{idx}/profile", profiles[i].cpu(), global_step=step)
        writer.add_image(f"{tag}/IN_{idx}/input_imgs", images[i].cpu(), global_step=step)
        if has_gt:
            writer.add_image(f"{tag}/IN_{idx}/gt_depth", depths_gt_in_to_plot[i].cpu(), global_step=step)
        if has_renders:
            writer.add_image(f"{tag}/IN_{idx}/recon_imgs", recon_imgs_in[i].cpu(), global_step=step)
            writer.add_image(f"{tag}/IN_{idx}/recon_depth", recon_depths_in[i].cpu(), global_step=step)
            writer.add_image(f"{tag}/IN_{idx}/recon_depth_var", recon_depths_var_in[i].cpu(), global_step=step)
            if has_gt:
                writer.add_image(f"{tag}/IN_{idx}/recon_err", recon_err_in[i].cpu(), global_step=step)
                writer.add_image(f"{tag}/IN_{idx}/recon_err_rel", recon_err_rel_in[i].cpu(), global_step=step)
            for k, v in invalids_in.items():
                writer.add_image(f"{tag}/IN_{idx}/{k}", v[i].cpu(), global_step=step)

    # --- Visualize the novel GT view(s) ---

    if has_gt:
        depths_gt = depths_gt[:take_n, :take_nv].float()
        images_gt = images_gt[:take_n, :take_nv].float()
        images_cond = images_cond[:take_n, :take_nv].float()
        depths_reproj = depths_reproj[:take_n, :take_nv].float()

    if has_renders:
        recon_imgs = recon_imgs[:take_n, :take_nv].float()
        recon_depths = recon_depths[:take_n, :take_nv].float()
        recon_depths_var = recon_depths_var[:take_n, :take_nv].float()

        if has_gt:
            recon_err = (depths_gt.squeeze(-3) - recon_depths).abs()#.mean(dim=1)
            recon_err = [color_tensor(im, cmap="plasma").permute(0, 3, 1, 2) for im in (recon_err / recon_err.max()).clamp(0, 1)]
            
            recon_err_rel = ((depths_gt.squeeze(-3) - recon_depths) / depths_gt.squeeze(-3)).abs()
            recon_err_rel = [color_tensor(im, cmap="plasma").permute(0, 3, 1, 2) for im in recon_err_rel.clamp(0, 1)]
            
        recon_imgs = [_.permute(0, 3, 1, 2) for _ in recon_imgs.mean(dim=-2)]
        recon_depths = [color_tensor(d.squeeze(1).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2) for d in invert_depth(recon_depths, z_near, z_far)]
        recon_depths_var = [color_tensor(d.squeeze(1).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2) for d in invert_depth(recon_depths_var, z_near, z_far)]
    
    # Synthetic GT
    if has_gt:
        gt_reproj_err = (depths_gt.clamp(z_near, z_far).squeeze(-3) - depths_reproj.clamp(z_near, z_far).squeeze(-3)).abs()
        gt_reproj_err /= gt_reproj_err.max()
        gt_reproj_err = [color_tensor(im, cmap="plasma").permute(0, 3, 1, 2) for im in gt_reproj_err.clamp(0, 1)]
        depths_gt = [color_tensor(_.squeeze(1).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2) for _ in invert_depth(depths_gt, z_near, z_far)]
        depths_reproj = [color_tensor(_.squeeze(1).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2) for _ in invert_depth(depths_reproj, z_near, z_far)]
        images_cond = format_conditioning_imgs(images_cond)
        synthetic_gt = [_ for _ in torch.concat([images_cond, images_gt, torch.stack(depths_gt), torch.stack(depths_reproj), torch.stack(gt_reproj_err)], dim=-1)]     # list [nv, 3, h, w*3]

    if has_renders:
        if "alphas" in data["coarse"][0]:
            alphas = data["coarse"][0]["alphas"].detach()
            alphas = alphas[:take_n, :take_nv].float()
            alphas += 1e-5
            alpha_sum = alphas.mean(dim=-1).clamp(-1)
            alpha_sum = [color_tensor(_, cmap="plasma").permute(0, 3, 1, 2) for _ in alpha_sum]
        
            # TODO verify this
            depth_profile = alphas[:, :, [h//4, h//2, 3*h//4], :, :].permute(0, 1, 2, 4, 3).reshape(take_n, take_nv, -1, w)
            depth_profile = depth_profile.clamp_min(0) / depth_profile.max()
            depth_profile = [color_tensor(_, cmap="plasma").permute(0, 3, 1, 2) for _ in depth_profile]

            ray_density = alphas / alphas.sum(dim=-1, keepdim=True)
            ray_entropy = -(ray_density * torch.log(ray_density)).sum(-1) / (math.log2(alphas.shape[-1]))
            ray_entropy = [color_tensor(_, cmap="plasma").permute(0, 3, 1, 2) for _ in ray_entropy]
        
    # Write images
    nrow = int(take_nv ** .5)

    # writer.add_image(f"{tag}/profiles_pseudo", make_grid(profiles_pseudo, nrow=nrow_in), global_step=step)
    for i, idx in enumerate(idxs):
        if has_gt:
            writer.add_image(f"{tag}/NV_{idx}/synthetic_gt", make_grid(synthetic_gt[i].cpu(), nrow=1), global_step=step)
        if has_renders:
            writer.add_image(f"{tag}/NV_{idx}/recon_imgs", make_grid(recon_imgs[i].cpu(), nrow=nrow), global_step=step)
            writer.add_image(f"{tag}/NV_{idx}/recon_depth", make_grid(recon_depths[i].cpu(), nrow=nrow), global_step=step)
            writer.add_image(f"{tag}/NV_{idx}/recon_depth_var", make_grid(recon_depths_var[i].cpu(), nrow=nrow), global_step=step)
            if has_gt:
                writer.add_image(f"{tag}/NV_{idx}/recon_err", make_grid(recon_err[i].cpu(), nrow=nrow), global_step=step)
                writer.add_image(f"{tag}/NV_{idx}/recon_err_rel", make_grid(recon_err_rel[i].cpu(), nrow=nrow), global_step=step)
            for k, v in invalids.items():
                writer.add_image(f"{tag}/NV_{idx}/{k}", make_grid(v[i].cpu(), nrow=nrow), global_step=step)
            if "alphas" in data["coarse"][0]:
                writer.add_image(f"{tag}/NV_{idx}/depth_profile", make_grid(depth_profile[i].cpu(), nrow=nrow), global_step=step)
                writer.add_image(f"{tag}/NV_{idx}/ray_entropy", make_grid(ray_entropy[i].cpu(), nrow=nrow), global_step=step)
                writer.add_image(f"{tag}/NV_{idx}/alpha_sum", make_grid(alpha_sum[i].cpu(), nrow=nrow), global_step=step)
