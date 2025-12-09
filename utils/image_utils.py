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

import torch.nn.functional as F
import lpips
import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(3,1)
        mask = torch.where(mask!=0,True,False)
        img1 = img1[mask]
        img2 = img2[mask]
        
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

    else:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
    if mask is not None:
        if torch.isinf(psnr).any():
            print(mse.mean(),psnr.mean())
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
            psnr = psnr[~torch.isinf(psnr)]
        
    return psnr

@torch.no_grad()
def get_ssim(img1, img2, mask=None, C1=0.01**2, C2=0.03**2):
    """
    计算 SSIM (Structural Similarity Index)。
    
    参数:
    - img1: 第一个图像 (Tensor)，形状为 (B, C, H, W)
    - img2: 第二个图像 (Tensor)，形状为 (B, C, H, W)
    - mask: 掩码 (可选, Tensor)，形状为 (B, H, W)

    返回:
    - SSIM 值 (Tensor)
    """
    if mask is not None:
        # 如果有 mask，应用 mask 并展平图像
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)
        mask = mask.flatten(1).repeat(3, 1)
        mask = torch.where(mask != 0, True, False)
        img1 = img1[mask]
        img2 = img2[mask]
    else:
        # 如果没有 mask，直接计算
        img1 = img1.view(img1.size(0), -1)
        img2 = img2.view(img2.size(0), -1)
    
    # 计算图像的均值
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    # 计算 SSIM
    ssim_value = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    # 如果存在无穷大或 NaN，处理掉它们
    if torch.isinf(ssim_value).any() or torch.isnan(ssim_value).any():
        ssim_value = ssim_value[~torch.isinf(ssim_value)]
        ssim_value = ssim_value[~torch.isnan(ssim_value)]
    
    return ssim_value



@torch.no_grad()
def lpips_metric(img1, img2, mask=None, model_type='alex'):
    """
    计算 LPIPS (Learned Perceptual Image Patch Similarity)。
    
    参数:
    - img1: 第一个图像 (Tensor)，形状为 (B, C, H, W)
    - img2: 第二个图像 (Tensor)，形状为 (B, C, H, W)
    - mask: 掩码 (可选, Tensor)，形状为 (B, H, W)
    - model_type: LPIPS 模型类型 ('alex', 'vgg' or 'squeeze')，默认 'alex'
    
    返回:
    - LPIPS 值 (Tensor)
    """
    # 初始化 LPIPS 模型
    loss_fn = lpips.LPIPS(net=model_type)

    # 获取输入的设备，确保 img1 和 img2 在相同设备上
    device = img1.device  # 假设 img1 和 img2 在相同的设备上，确保一致
    loss_fn = loss_fn.to(device)  # 将 LPIPS 模型也移动到相同的设备

    if mask is not None:
        # 如果有 mask，应用 mask 并展平图像
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)
        mask = mask.flatten(1).repeat(3, 1)
        mask = torch.where(mask != 0, True, False)
        img1 = img1[mask]
        img2 = img2[mask]

    else:
        # 如果没有 mask，直接计算
        img1 = img1.view(img1.size(0), -1)
        img2 = img2.view(img2.size(0), -1)

    # 确保 img1 和 img2 都在相同的设备上
    img1 = img1.to(device)
    img2 = img2.to(device)

    # 计算 LPIPS
    lpips_value = loss_fn(img1.unsqueeze(0), img2.unsqueeze(0))

    # 如果存在无穷大或 NaN，处理掉它们
    if torch.isinf(lpips_value).any() or torch.isnan(lpips_value).any():
        lpips_value = lpips_value[~torch.isinf(lpips_value)]
        lpips_value = lpips_value[~torch.isnan(lpips_value)]

    return lpips_value.mean()