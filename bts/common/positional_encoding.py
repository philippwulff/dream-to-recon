from typing import Callable
import torch
import numpy as np
import torch.autograd.profiler as profiler


# TODO: rethink encoding mode
def encoding_mode(
    encoding_mode: str, d_min: float, d_max: float, inv_z: bool, EPS: float
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def _z(xy: torch.Tensor, z: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        if inv_z:
            z = (torch.div(1, z.clamp_min(EPS)) - 1 / d_max) / (1 / d_min - 1 / d_max)
        else:
            z = (z - d_min) / (d_max - d_min)
        z = 2 * z - 1  # to [-1, 1]
        return torch.cat(
            (xy, z), dim=-1
        )  ## concatenates the normalized x, y, and z coordinates

    def _distance(xy: torch.Tensor, z: torch.Tensor, distance: torch.Tensor):
        if inv_z:
            distance = (torch.div(1, distance.clamp_min(EPS)) - 1 / d_max) / (
                1 / d_min - 1 / d_max
            )
        else:
            distance = (distance - d_min) / (d_max - d_min)
        distance = 2 * distance - 1
        return torch.cat(
            (xy, distance), dim=-1
        )  ## Apply the positional encoder to the concatenated xy and depth/distance coordinates (it enables the model to capture more complex spatial dependencies without a significant increase in model complexity or training data)

    match encoding_mode:
        case "z":
            return _z
        case "distance":
            return _distance
        case _:
            return _z


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get("num_freqs", 6),
            d_in,
            conf.get("freq_factor", np.pi),
            conf.get("include_input", True),
        )
