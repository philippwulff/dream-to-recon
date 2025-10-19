import math

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import profiler

from bts.models.layers.layers import ssim
from bts.losses.invalid_policy import InvalidPolicy
from bts.losses.base_loss import BaseLoss
from collections import defaultdict
from bts.losses.utils import edge_aware_smoothness


class DepthReconstructionLoss(BaseLoss):
    def __init__(
        self, 
        criterion: str = "l2",
        invalid_policy: str = "none",
        lambda_in: float = 0.5,
        lambda_coarse: float = 1.0,
        lambda_fine: float = 1.0,
        lambda_entropy: float = 0.0,
        lambda_depth_reg: float = 0.0,
        lambda_alpha_reg: float = 0.0,
        lambda_surfaceness_reg: float = 0.0,
        lambda_edge_aware_smoothness: float = 0.0,
        lambda_depth_smoothness: float = 0.0,
        median_thresholding: bool = False,
        alpha_reg_reduction: str = "ray",
        alpha_reg_fraction: float = 1/8,
        use_automasking: bool = False,
        lambda_var: float = 0.,
        **kwargs,
        ) -> None:
        super().__init__()
        
        self.is_gnll = False
        match criterion:
            case "l2":
                self.coarse_crit = torch.nn.MSELoss(reduction="none")
                self.fine_crit = torch.nn.MSELoss(reduction="none")
            case "l1":
                self.coarse_crit = torch.nn.L1Loss(reduction="none")
                self.fine_crit = torch.nn.L1Loss(reduction="none")
            case "gnll":
                self.is_gnll = True
                self.coarse_crit = nn.GaussianNLLLoss(eps=1, reduction="none")
                self.fine_crit = nn.GaussianNLLLoss(eps=1, reduction="none")
            case _:
                raise ValueError(f"Criterion {criterion} not available.")
            
        self.lambda_coarse = lambda_coarse
        self.lambda_fine = lambda_fine
        self.lambda_in = lambda_in
        assert 0. <= lambda_in <= 1., "INVALID RANGE"
        self.lambda_var = lambda_var
        assert 0. <= lambda_var, "INVALID RANGE"
        self.lambda_entropy = lambda_entropy
        self.lambda_depth_reg = lambda_depth_reg
        self.lambda_surfaceness_reg = lambda_surfaceness_reg
        self.lambda_edge_aware_smoothness = lambda_edge_aware_smoothness
        self.lambda_depth_smoothness = lambda_depth_smoothness

        self.use_automasking = use_automasking
        self.median_thresholding = median_thresholding

        self.lambda_alpha_reg = lambda_alpha_reg
        self.alpha_reg_reduction = alpha_reg_reduction
        assert alpha_reg_reduction in ("ray", "slice"), f"Unknown reduction for alpha regularization: {self.alpha_reg_reduction}"
        self.alpha_reg_fraction = alpha_reg_fraction
   
        self.invalid_policy = invalid_policy
        assert self.invalid_policy in ["strict", "weight_guided", "weight_guided_diverse", None, "none"]
        self.ignore_invalid = self.invalid_policy is not None and self.invalid_policy != "none"
        self.invalid_policy_ = InvalidPolicy(self.invalid_policy)

    def get_loss_metric_names(self):
        return ["loss_rgb_coarse", "loss_rgb_fine", "loss_rgb_coarse_in", "loss_rgb_fine_in", "loss_rgb_total"]

    def __call__(self, data, **kwargs):
        with profiler.record_function("loss_computation"):
            n_scales = len(data["coarse"])
            b = len(data["coarse"][0]["depth"])

            z_near, z_far = data["z_near"], data["z_far"]

            rgb_gt = data["rgb_gt"].float()
            rgb_gt_in = data["rgb_gt_in"].float()
            
            def make_kwargs(key):
                kwargs = {"invalids": data[key][0]["invalid"]}
                if "weights" in data[key][0]:
                    kwargs["weights": data[key][0]["weights"]]
                if "rgb_samps" in data[key][0]:
                    kwargs["rgb_samps": data[key][0]["rgb_samps"]]
                return kwargs
            
            invalid_coarse = self.invalid_policy_(**make_kwargs("coarse"))
            invalid_coarse_in = self.invalid_policy_(**make_kwargs("coarse_in"))
            invalid_fine = self.invalid_policy_(**make_kwargs("fine"))
            invalid_fine_in = self.invalid_policy_(**make_kwargs("fine_in"))
            
            set_outside_range_invalid = lambda invalid, gt: torch.logical_or(invalid, torch.logical_or(z_near > gt, gt > z_far))
            invalid_coarse = set_outside_range_invalid(invalid_coarse, rgb_gt)
            invalid_fine = set_outside_range_invalid(invalid_fine, rgb_gt)
            invalid_coarse_in = set_outside_range_invalid(invalid_coarse_in, rgb_gt_in)
            invalid_fine_in = set_outside_range_invalid(invalid_fine_in, rgb_gt_in)
            
            render_types = {
                "coarse": [(1 - self.lambda_in) * self.lambda_coarse, invalid_coarse.float(), rgb_gt], 
                "fine": [(1 - self.lambda_in) * self.lambda_fine, invalid_fine.float(), rgb_gt], 
                "coarse_in": [self.lambda_in * self.lambda_coarse, invalid_coarse_in.float(), rgb_gt_in], 
                "fine_in": [self.lambda_in * self.lambda_fine, invalid_fine_in.float(), rgb_gt_in],
            }
            
            loss_dict = defaultdict(float)
            loss_total = torch.tensor(0., device=invalid_fine.device)

            for scale in range(n_scales):
                for render_type, (loss_lambda, loss_invalid, loss_gt) in render_types.items():
                    render_data = data[render_type][scale]
                    rgb = render_data["rgb"].float()

                    if self.use_automasking:
                        thresh_gt = loss_gt[..., -1:]
                        # rgb_coarse = rgb_coarse[..., :-1]
                        # rgb_fine = rgb_fine[..., :-1]
                        # TODO this is inplace... maybe bad when looping over it
                        loss_gt = loss_gt[..., :-1]

                    # b, pc, h, w, nv, c = rgb_coarse.shape

                    loss = self.coarse_crit(rgb.unsqueeze(-1), loss_gt)
                    
                    # Take minimum across all reconstructed views
                    loss = loss.amin(-2)

                    if self.use_automasking:
                        loss = torch.min(loss, thresh_gt)

                    if self.ignore_invalid:
                        loss = loss * (1 - loss_invalid.to(loss.dtype))

                    if self.median_thresholding:
                        threshold = torch.median(loss.view(b, -1), dim=-1)[0].view(-1, 1, 1, 1, 1)
                        loss = loss[loss <= threshold]

                    loss = loss.mean()
                    loss *= loss_lambda
                    loss_dict[f"loss_rgb_{render_type}"] += loss.item()
                    loss_total += loss

            loss_dict = {k: v / n_scales for k, v in loss_dict.items()}
            loss_total = loss_total / n_scales

        loss_dict["loss_rgb_total"] = loss_total.item()
        
        return loss_total, loss_dict
