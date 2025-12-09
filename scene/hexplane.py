import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

#coor: N,2  or B,N,2? 
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]
    # coords (N,2)
    # grid (1,channels,dim1, dim2)
    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0) 
    if coords.dim() == 2:
        coords = coords.unsqueeze(0) 
    # coords (1,N,2)
    # grid_dim = 2
    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]  # number of points
    interp = grid_sampler(
        grid,  # [B, feature_dim, dim1, dim2]
        coords,  # [B, 1, n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')  # we should rewrite this function to make the whole hashable
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  
    interp = interp.squeeze()  # (n, feature_dim)
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]  # add batch to the first dimension
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs



class HexPlaneField(nn.Module):
    def __init__(
        self,
        bounds,
        planeconfig,
        multires,
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        # self.Q = nn.Parameter(torch.tensor(0.1, requires_grad=False))
        self.Q = 0.1
        self.quantize_HEX = False
        self.rotation_matrix = nn.Parameter(torch.eye(3), requires_grad=True)
        self.pca_mean = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.variance = nn.Parameter(torch.ones(3), requires_grad=True) 
        
        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)



    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)
        
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_max = aabb[0]
        aabb_min = aabb[1]
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x       
          
    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        # pts = normalize_aabb(pts, self.aabb) #
        pts = self.contract_to_unisphere(pts.clone().detach(), aabb=self.aabb, derivative=False)
        timestamps = 2*timestamps - 1 # add rescaling
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])   # if rays then flatten all samples along the ray 
        features = self.interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        return features
    
    def set_rotation_matrix(self, rotation_matrix, mean, variance):
        self.rotation_matrix = nn.Parameter(rotation_matrix, requires_grad=False)
        self.pca_mean = nn.Parameter(mean, requires_grad=False)
        self.variance = nn.Parameter(variance, requires_grad=False)

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        pts = pts - self.pca_mean  # shape: (N, 4)
        # apply rotation: [N, 4] @ [4, 4] → [N, 4]
        pts = torch.matmul(pts, self.rotation_matrix.detach())
        # divide by variance (standardization)
        pts = pts / self.variance.detach()
        features = self.get_density(pts, timestamps)

        return features


    def register_hooks(self):
        """Register hooks for each plane in `grids` to track gradients."""
        for scale_idx, scale in enumerate(self.grids):
            for plane_idx, plane in enumerate(scale):
                plane.register_hook(
                    lambda grad, scale_idx=scale_idx, plane_idx=plane_idx:
                    print(f"Gradient for plane {plane_idx} in scale {scale_idx}: {grad}")
                )

        
    def _quantize(self, inputs, mode):
        _step_size = self.Q
        # print(_step_size)
        if mode == "noise":
            noise = (torch.rand(inputs.size(), device=inputs.device) - 0.5) * _step_size
            # noise = 0
            return inputs + noise
        elif mode == "symbols":
            return RoundNoGradient.apply(inputs / _step_size) * _step_size

    # size of pts: [N, 4]
    def interpolate_ms_features(self, pts: torch.Tensor,
                                ms_grids: Collection[Iterable[nn.Module]],
                                grid_dimensions: int,
                                concat_features: bool,
                                num_levels: Optional[int],
                                ) -> torch.Tensor:
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]), grid_dimensions)
        )  # combination of hexplane (x,y) (x,z), etc
        if num_levels is None:
            num_levels = len(ms_grids)
        multi_scale_interp = [] if concat_features else 0.
        grid: nn.ParameterList
        
        # each grid is shaped as (1,channels, dim1, dim2), 1 is for batch
        for scale_id, grid in enumerate(ms_grids[:num_levels]):
            interp_space = 1.
            for ci, coo_comb in enumerate(coo_combs):
                # interpolate in plane
                feature_dim = grid[ci].shape[1]
                if self.quantize_HEX: 
                    grid_i = self._quantize(grid[ci], "noise")
                else:
                    grid_i = grid[ci]
                interp_out_plane = (
                    grid_sample_wrapper(grid_i, pts[..., coo_comb]) # according to the dimension combo, selects two coordinates from pts (last4 
                    .view(-1, feature_dim))
                # compute product over planes
                interp_space = interp_space * interp_out_plane

            # combine over scales
            if concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp
