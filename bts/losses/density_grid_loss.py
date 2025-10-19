import torch
import torch.nn as nn
from bts.losses.base_loss import BaseLoss
from torch import profiler
import torch.nn.functional as F


class DensityGridLoss(BaseLoss):
    def __init__(
        self, 
        weight: float = 1.0,
        criterion: str = "l2",
        bce_occ_threshold: float = 0.5,
        lambda_bce: float = 0.0,
        weight_occl_and_empty: float = 1.0,
        target_density_max: float = 10.0,
        eps: float = 1.0,
        align: bool = False,
        align_shift: bool = False,
        **kwargs,
        ) -> None:
        super().__init__()
        self.lambda_loss = weight
        self.lambda_bce = lambda_bce
        self.weight_occl_and_empty = weight_occl_and_empty
        self.bce_occ_threshold = bce_occ_threshold
        self.criterion = criterion
        self.target_density_max = target_density_max
        self.eps = eps
        self.align = align
        self.align_shift = align_shift
        match criterion:
            case "l2":
                self.crit = torch.nn.MSELoss(reduction="none")
            case "l1":
                self.crit = torch.nn.L1Loss(reduction="none")
            case "smooth_l1":
                self.crit = torch.nn.SmoothL1Loss(reduction="none")
            case "bce":
                self.crit = nn.BCELoss(reduction="none")
            case _:
                raise ValueError(f"Criterion {criterion} not available.")
        
    def get_loss_metric_names(self):
        names = ["density_loss_total", f"density_loss_{self.criterion}", ]
        return names

    def __call__(self, data: dict, **kwargs):
        loss_dict = {}
        with profiler.record_function("loss_computation"):
            pred = data["sampled_densities_pred"]
            target = data["sampled_densities_gt"]
            uncertainty_pred = data.get("sampled_uncertainties_pred", None)
            invalid = data.get("sampled_densities_invalid_comb", None)
            occl_and_empty = data.get("sampled_densities_occluded_and_empty", None)

            batch_size, _, x_res, y_res, z_res = pred.shape

            # from utils.plotting_3d import make_3d_fig, draw_point_cloud, draw_camera_with_frustum, draw_ply_mesh, occupancy_grid_to_ply
            # import torch.nn.functional as F
            # fig = make_3d_fig()
            # n_pts = 500_000
            # xyz = data["xyz"][0].reshape(-1, 3)
            # indices = torch.randperm(len(xyz))[:n_pts]
            # xyz = xyz[indices]
            # rgb = torch.zeros_like(xyz)
            # rgb[:, 0] = 1 - invalid[0].reshape(-1)[indices]
            # rgb[:, 2] = invalid[0].reshape(-1)[indices]
            # draw_point_cloud(fig, xyz.cpu().numpy(), rgb.cpu().numpy() * 255)
            # draw_camera_with_frustum(fig, torch.eye(4).numpy(), data["projs"][0][0].float().cpu().numpy())
            # fig.write_html("xyz.html")

            if self.align:
                with torch.no_grad():  # Backprop through this is very memory-expensive and not necessary.
                    # Solve for X in A * X = B. This aligns the target to the prediction.
                    A = target.view(batch_size, -1, 1)
                    if self.align_shift:
                        A = torch.concat([A, torch.ones_like(A)], dim=-1)       # [B, n_pts, 2]
                    B = pred.view(batch_size, -1, 1)
                    X = torch.linalg.lstsq(A.float(), B.float()).solution.to(pred.dtype)  # [B, 2, 1]
                    scales = torch.nan_to_num(X[:, 0, 0], nan=1.0).mean()
                    loss_dict["density_loss_scale"] = scales.item()
                    if self.align_shift:
                        shifts = torch.nan_to_num(X[:, 1, 0], nan=0.0).mean()
                        loss_dict["density_loss_shift"] = shifts.item()
                    else:
                        shifts = torch.tensor(0.0, device=scales.device)
                target = target * scales + shifts

            if self.criterion == "bce":
                pred = F.sigmoid(pred)
                target = (target / self.target_density_max).clamp(0, 1)

            loss = self.crit(pred, target)
            loss_dict[f"density_loss_{self.criterion}"] = loss.mean().item()

            if uncertainty_pred is not None:
                loss = loss * (2 ** .5) / (uncertainty_pred + self.eps) + torch.log(uncertainty_pred + self.eps)
                loss_dict["density_loss_uncertainty_mean"] = uncertainty_pred.mean().item()
                loss_dict["density_loss_uncertainty_max"] = uncertainty_pred.max().item()
            
            if occl_and_empty is not None and self.weight_occl_and_empty != 0:
                loss[occl_and_empty] *= self.weight_occl_and_empty
            if invalid is not None:
                loss *= 1 - invalid.to(loss)

            loss = loss.mean() * self.lambda_loss
            loss_dict[f"density_loss_total"] = loss.item()

        return loss, loss_dict
