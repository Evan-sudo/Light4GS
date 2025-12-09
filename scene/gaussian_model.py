#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness
from scene.entropy_model import *
from scene.checkerboard import *
from utils.general_utils import *
import torch.nn.functional as F
from icecream import ic
import matplotlib.pyplot as plt



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._deformation = deform_network(args)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.setup_functions()
        ch = [args.kplanes_config][0]['output_coordinate_dim']
        print(ch)
        self.checkerboard = CheckerboardAutoregressive(N=ch//8, M =ch).cuda()
        self.entropy_bottleneck = EntropyBottleneck(45)
        self.quantize_SH = False
        self.sh_Q = 0.002


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            # self.grid,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        deform_state,
        self._deformation_table,
        
        # self.grid,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        # breakpoint()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        # self.grid = self.grid.to("cuda")
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))  # SH maybe
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        

        l = [    
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': list(self.checkerboard.get_parameters()), 'lr': training_args.entropy_lr_init * self.spatial_lr_scale, "name": "checker"},
            # {'params': [self._deformation.deformation_net.grid.Q], 'lr': 0.0005, "name": "step"}, 
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},   
            {'params': list(self.entropy_bottleneck.get_parameters()), 'lr': training_args.entropy_lr_init * self.spatial_lr_scale, "name": "entropy"},      
        ]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # for group in self.optimizer.param_groups:
        #    for param in group['params']:
        #     print(f"Parameter: {param.shape}, Requires Grad: {param.requires_grad}")


        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.checker_scheduler_args = get_expon_lr_func(lr_init=training_args.entropy_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.entropy_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)  
        self.entropy_scheduler_args = get_expon_lr_func(lr_init=training_args.entropy_lr_init*10*self.spatial_lr_scale,
                                                    lr_final=training_args.entropy_lr_final*10*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps) 
        # self.q_scheduler_args = get_expon_lr_func(lr_init=0.0005,
        #                                             lr_final=0.00005,
        #                                             lr_delay_mult=training_args.deformation_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)  

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if  "entropy" in param_group["name"]:
                lr = self.entropy_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "checker" in param_group["name"]:
                lr = self.checker_scheduler_args(iteration)
                param_group['lr'] = lr
            if  "step" in param_group["name"]:
                lr = self.q_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
        
    def compute_deformation(self,time):
        
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        weight_dic_e = torch.load(os.path.join(path, "entropy_bottleneck.pth"), map_location="cuda")
        self.entropy_bottleneck.load_state_dict(weight_dic_e)
        weight_dic_c = torch.load(os.path.join(path, "checkerboard.pth"), map_location="cuda")
        self.checkerboard.load_state_dict(weight_dic_c)
        
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # print(self._deformation.deformation_net.grid.)

    def save_entropy(self, path):
        torch.save(self.entropy_bottleneck.state_dict(), os.path.join(path, "entropy_bottleneck.pth"))

    def save_checkerboard(self, path):
        torch.save(self.checkerboard.state_dict(), os.path.join(path, "checkerboard.pth"))

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def set_pca(self):
        x = self._xyz[:, 0].detach().cpu().numpy()
        y = self._xyz[:, 1].detach().cpu().numpy()
        z = self._xyz[:, 2].detach().cpu().numpy()
        xyz = np.vstack([x, y, z]).T

        dense_points = torch.tensor(xyz, dtype=torch.float32).cuda()

        mean = torch.mean(dense_points, dim=0)
        centered_points = dense_points - mean
        cov_matrix  = torch.cov(centered_points.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        rotated_points = torch.matmul(centered_points, sorted_eigenvectors)

        standardized_points = rotated_points / torch.sqrt(eigenvalues[sorted_indices])
        
        print(f"mean: {mean}")
        print(f"eigenvalues: {eigenvalues[sorted_indices]}")
        print(f"eigenvectors: {sorted_eigenvectors}")
 
        self._deformation.deformation_net.grid.set_rotation_matrix(sorted_eigenvectors, mean, torch.sqrt(eigenvalues[sorted_indices]))
        print("PCA finished!")


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            if group["name"] == "step":continue 
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:continue
            if group["name"] == "step":continue  # Skip updating for step_size
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        # "deformation": new_deformation
       }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._deformation = optimizable_tensors["deformation"]
        
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # breakpoint()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        

        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)

    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    def get_displayment(self,selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point<xyz_max 
        mask_b = final_point>xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]
    

        return final_point, mask_d    
    def add_point_by_mask(self, selected_pts_mask, perturb=0):
        selected_xyz = self._xyz[selected_pts_mask] 
        new_xyz, mask = self.get_displayment(selected_xyz, self.get_xyz.detach(),perturb)

        new_features_dc = self._features_dc[selected_pts_mask][mask]
        new_features_rest = self._features_rest[selected_pts_mask][mask]
        new_opacities = self._opacity[selected_pts_mask][mask]
        
        new_scaling = self._scaling[selected_pts_mask][mask]
        new_rotation = self._rotation[selected_pts_mask][mask]
        new_deformation_table = self._deformation_table[selected_pts_mask][mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)
        return selected_xyz, new_xyz

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

     #### function added from lightGS
    def prune_gaussians(self, percent, import_score: list):
        ic(import_score.shape)
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)
   

    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent)
    
    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    def _plane_regulation(self):
        #multi_res_grids = self._deformation.deformation_net.grid.get_grids()
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:  # grids are single-res hexplanes
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]  #sorting correct position of position planes
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _time_regulation(self):
        # multi_res_grids = self._deformation.deformation_net.grid.get_grids()
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        # multi_res_grids = self._deformation.deformation_net.grid.get_grids()
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()

    def attri_entropy(self, quantization_mode):
        _, bits = self.entropy_bottleneck(self._features_rest.unsqueeze(0), quantization_step = self.sh_Q, quantize_mode=quantization_mode)
        return bits 

    def compress_attri(self, file_path: str):
        """
        把 _features_rest 用 EB + torchac 编码成 .b 文件。

        返回：
            file_bits: 实际总 bits（含 header）
            x_q:       量化后的属性张量 [N, 15, 3]
        """
        assert self._features_rest is not None, "features_rest 还没初始化"

        x = self._features_rest.unsqueeze(0)  # [1, N, 15, 3]
        file_bits, x_q_4d = self.entropy_bottleneck.compress(
            x,
            Q=self.sh_Q,
            file_path=file_path,
        )
        # 去掉 batch 维度，回到 [N, 15, 3]
        x_q = x_q_4d.squeeze(0)

        # 你可以选择把量化后的结果存起来
        # self._features_rest = x_q

        return file_bits, x_q

    def decompress_attri(self, file_path: str, device=None):
        """
        从 .b 文件解码出属性 SH，并更新到 self._features_rest。

        返回：
            x_hat: 解码 + 反量化后的 SH，形状 [N, 15, 3]
        """
        if device is None:
            device = self._features_rest.device if self._features_rest is not None else "cuda"

        x_hat_4d = self.entropy_bottleneck.decompress(
            file_path=file_path,
            Q=self.sh_Q,
            device=device,
        )   # [1, N, 15, 3]

        x_hat = x_hat_4d.squeeze(0)  # [N, 15, 3]
        # # 更新到当前 Gaussian
        # self._features_rest = x_hat

        return x_hat
    
    
    @torch.no_grad()
    def compress_hexplanes(self, file_path, Q=0.1):
        """
        压缩当前场景的 multiscale hexplanes 到 file_path。

        输入:
          file_path: 存 .pth 的路径（跟你 SH 的 .b 类似，只是这里直接用 torch.save）
          Q:         量化步长，默认 0.1（要和训练 / bits 估计时用的 Q 一致）

        返回:
          file_bits: float，总共写入的 bit 数（和 checkerboard.entropy_compress 里一致）
        """
        device = next(self.checkerboard.parameters()).device

        # 1. 取当前 hexplanes（list[list[tensor]]）
        #    和你 calculate_hexplane_bits 时一致：
        #    gaussians._deformation.deformation_net.grid.grids
        hexplanes = self._deformation.deformation_net.grid.grids

        # 2. 搬到模型所在 device（no_grad，不影响梯度）
        hexplanes_dev = [
            [p.to(device) for p in scale]
            for scale in hexplanes
        ]

        # 3. 调 checkerboard 的 entropy_compress，真正写 bitstream
        packed = self.checkerboard.entropy_compress(
            hexplanes_dev, Q=Q, output_path=file_path
        )

        # 4. 取出总 bit 数
        file_bits = float(packed["total_bits"])
        bpp = float(packed["bpp"])
        return file_bits, bpp

    @torch.no_grad()
    def decompress_hexplanes(self, file_path, device=None):
        """
        从 file_path 读回 bitstream，并解码出所有 multiscale hexplanes。

        输入:
          file_path: compress_hexplanes 写出来的 .pth 文件
          device:    要把输出放到哪个 device，默认跟 checkerboard 一样

        返回:
          decoded_hexplanes: List[List[Tensor]]，
              结构和 self._deformation.deformation_net.grid.grids 一样：
              decoded_hexplanes[scale][plane] = [1, C, H, W]
        """
        if device is None:
            device = next(self.checkerboard.parameters()).device

        # checkerboard.entropy_decompress 里已经支持传路径或 packed dict
        decoded = self.checkerboard.entropy_decompress(
            file_path, device=device
        )
        return decoded
    
    
    def calculate_mem(self, file_bits_hex, file_bits_sh):
        """
        计算总存储：
          - deformation MLP 参数
          - 熵模型 / context model 参数
          - hexplanes bitstream 大小（真实压缩）
          - GS frame 大小：SH(ac) bitstream + 其他属性（按 float32 存）
        """

        # ===== 1) Deformation MLP 参数大小 =====
        deform_para = self._deformation.get_mlp_parameters()
        deform_MB = calculate_memory(deform_para)  # 这个就是你以前用的那个函数

        # ===== 2) 熵模型 / context model 参数大小 =====
        selected_modules = [
            self.checkerboard.h_a,
            self.checkerboard.h_s,
            self.checkerboard.entropy_bottleneck,
            self.checkerboard.context_prediction,
            self.checkerboard.scale_context,
            self.entropy_bottleneck,
            self.checkerboard.entropy_parameters,
        ]
        parae_list = []
        for module in selected_modules:
            for param in module.parameters():
                parae_list.append(param)
        entropy_MB = calculate_memory(parae_list)

        # ===== 3) Hexplanes bitstream 大小（真实值） =====
        hex_MB = file_bits_hex / (8 * 1024**2)

        # ===== 4) GS frame 大小：SH(ac) + 其他属性 =====
        N = self._xyz.shape[0]

        # 其他属性按 float32 存：xyz(3) + color(3) + opacity(1) + rotation(4) + scaling(3)
        # 对应你原来公式里的 32*(3+1+4+3+3) 那块
        other_bits_per_point = 32 * (3 + 1 + 4 + 3 + 3)
        other_bits_total = other_bits_per_point * N
        other_MB = other_bits_total / (8 * 1024**2)

        # SH 部分用真实 ac bitstream
        sh_MB = file_bits_sh / (8 * 1024**2)

        PLY_MB = sh_MB + other_MB

        print("Deformation size: ", deform_MB)
        print("Context size: ", entropy_MB)
        print("Hexplane size: ", hex_MB)
        print("GS frame size: ", PLY_MB)

        return deform_MB + entropy_MB + hex_MB + PLY_MB
    
    








