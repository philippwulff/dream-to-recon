from typing import Tuple
import torch


def compute_occlusion_center_error(occluded: torch.Tensor, all_occluded: torch.Tensor, invalid: torch.Tensor, mode="center"):
    """
    Computes the masked MSE error between the two inputs with shape [B, N, 1, H, W].
    The mask is 0 if a pixel is invalid and weights the image center higher than the edges.
    """
    _, _, _, height, width = occluded.shape
    device = occluded.device
    dtype = occluded.dtype
    
    y, x = torch.meshgrid(torch.linspace(-1., 1., height, device=device, dtype=dtype), torch.linspace(-1., 1., width, device=device, dtype=dtype), indexing='ij')
    match mode:
        case "center":
            normal = torch.distributions.MultivariateNormal(torch.tensor([0, 0], device=device), torch.tensor([[1/4, 0], [0, 1/4]], device=device))
        case "center_bottom":
            normal = torch.distributions.MultivariateNormal(torch.tensor([1, 0], device=device), torch.tensor([[1/2, 0], [0, 1/4]], device=device))
    yx = torch.stack([y, x], dim=-1)  
    # Evaluate the PDF at each point  
    g = torch.exp(normal.log_prob(yx)).to(dtype)
    g /= g.detach().max().clamp_min(0.01)
    all_occluded = torch.ones_like(occluded)
    # loss_center = torch.pow((occluded - all_occluded) * (1-invalid.to(occluded)) * g, 2)
    # return loss_center.mean(dim=[2, 3, 4])      # [b, nv]
    loss_center = torch.pow((occluded - all_occluded) * (1-invalid.to(occluded)), 2) * g
    return loss_center.sum(dim=[2, 3, 4]) / g.sum().clamp_min(1.)      # [b, nv]
    

def compute_occlusion_edges_error(occluded: torch.Tensor, all_not_occluded: torch.Tensor, invalid: torch.Tensor):
    """
    The mask is 0 if a pixel is invalid and weights the image edges higher than the edges.
    """
    _, _, _, height, width = occluded.shape
    device = occluded.device
    dtype = occluded.dtype
    
    normal = torch.distributions.Normal(loc=torch.tensor(0, dtype=dtype, device=device), scale=torch.tensor(1/2, dtype=dtype, device=device))
    x = torch.linspace(-1., 1., width, device=device, dtype=dtype)
    g = torch.exp(normal.log_prob(x))[None, :].repeat(height, 1)
    g /= g.detach().max().clamp_min(0.01)
    # loss_edges = torch.pow((occluded - all_not_occluded) * (1-invalid.to(occluded)) * (1 - g), 2)
    # return loss_edges.mean(dim=[2, 3, 4])
    loss_edges = torch.pow((occluded - all_not_occluded) * (1-invalid.to(occluded)), 2) * (1 - g)
    return loss_edges.sum(dim=[2, 3, 4]) / (1 - g).sum().clamp_min(1.)      # [b, nv]


def gather_bottom_k_error(error: torch.Tensor, gather_input: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples the best k instances according to the error from the given tensor.
    
    Args:
        error: [B, NV]
        gather_input: [B, NV, ...]
    """
    idxs_sorted = torch.argsort(error, dim=1, descending=False)
    idxs_sorted = idxs_sorted[:, :k]
    # [b, num_proposals, 4, 4] -> [b, nv, 4, 4]
    expand_shape = gather_input.shape[2:]
    idxs_sorted = idxs_sorted.view(error.size(0), k, *[1 for _ in expand_shape]).expand(-1, -1, *expand_shape)
    # idxs_sorted = idxs_sorted.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)
    gather_input = torch.gather(gather_input, 1, idxs_sorted)
    return gather_input, idxs_sorted


