import torch
from dataclasses import asdict
from typing import List, Union, Dict


def get_interval_sample(interval_start: torch.Tensor, interval_size: torch.Tensor, generator=None) -> torch.Tensor:
    rand = torch.rand(interval_start.shape, generator=generator, device=interval_start.device, dtype=interval_start.dtype)
    return (interval_start + rand * abs(interval_size))


def asdict_lowercase_keys_override(d, **kwargs):
    """Unpacks a dataclass into a dict of lower-case keys and their values."""
    d_asdict = d if isinstance(d, dict) else asdict(d)
    d_asdict = {k.lower(): v for k, v in d_asdict.items()}
    d_asdict.update(kwargs)
    return d_asdict


def check_nested_for_nan(d: Union[Dict, List], parents_keys: List[str] = []):
    
    for i, el in enumerate(d):
        v = el if isinstance(d, (list, tuple)) else d[el]
        k = str(i) if isinstance(d, (tuple, list)) else el

        if isinstance(v, torch.Tensor):
            k_str = "_".join(parents_keys + [k])
            if torch.isnan(v).any():
                print(f"Found NaN under key '{k_str}':", v)
            elif torch.isinf(v).any():
                print(f"Found inf under key '{k_str}':", v)
        elif isinstance(v, (dict, list, tuple)):
            check_nested_for_nan(v, parents_keys + [k])
            
            
def invert_depth(d, d_near, d_far):
    return (1 / d - 1 / d_far) / (1 / d_near - 1 / d_far)