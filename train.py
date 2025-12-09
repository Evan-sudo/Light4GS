# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import random
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_bits
from tqdm import tqdm
from utils.image_utils import psnr, get_ssim, lpips_metric
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
from scene.prune import *
from icecream import ic


def print_grad(grad):
    print(grad)

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ============================== 压缩 / 解码工具函数 ==============================

@torch.no_grad()
def debug_compress_gaussians(gaussians, iteration, Q_hex=0.1, bitstream_dir="./bitstream"):
    """
    只做“压缩 + 误差统计”，不改动 gaussians 里真正的参数。
    同时处理：
      - SH (_features_rest)
      - multiscale hexplanes (deformation grid)

    返回:
      {
        "file_path_sh":   str,
        "file_path_hex":  str,
        "bps_file_sh":    float,
        "bps_file_hex":   float,
      }
    """
    os.makedirs(bitstream_dir, exist_ok=True)

    # ---- SH 属性 ----
    file_path_sh = os.path.join(bitstream_dir, f"debug_attri_iter_{iteration}.b")
    file_bits_sh, x_q = gaussians.compress_attri(file_path_sh)   # x_q: [N, 15, 3]
    num_symbols_sh = x_q.numel()
    bps_file_sh = file_bits_sh / num_symbols_sh

    x_hat = gaussians.decompress_attri(file_path_sh, device=x_q.device)
    recon_mse_sh = torch.mean((x_hat - x_q) ** 2).item()

    print(
        f"[iter {iteration}] SH compress debug:\n"
        f"  file bits/sym = {bps_file_sh:.6f}, "
        f"file_bits = {file_bits_sh:.1f}, "
        f"recon MSE(decode vs quantized) = {recon_mse_sh:.3e}"
    )

    # ---- Hexplanes ----
    if hasattr(gaussians, "checkerboard"):
        gaussians.checkerboard.update()

    file_path_hex = os.path.join(bitstream_dir, f"debug_hexplanes_iter_{iteration}.pth")
    file_bits_hex, bps_file_hex = gaussians.compress_hexplanes(file_path_hex, Q=Q_hex)

    hexplanes = gaussians._deformation.deformation_net.grid.grids
    decoded_hex = gaussians.decompress_hexplanes(file_path_hex)

    all_mse = []
    for i, scale in enumerate(hexplanes):
        for j, plane in enumerate(scale):
            x = plane
            x_hat_plane = decoded_hex[i][j].to(x.device)
            all_mse.append(torch.mean((x_hat_plane - x) ** 2))
    global_mse_hex = torch.stack(all_mse).mean().item()

    print(
        f"[iter {iteration}] HEX compress debug:\n"
        f"  file bits/sym = {bps_file_hex:.6f}, "
        f"file_bits = {file_bits_hex:.1f}, "
        f"recon MSE(decode vs original latent) = {global_mse_hex:.3e}"
    )

    return {
        "file_path_sh": file_path_sh,
        "file_path_hex": file_path_hex,
        "bps_file_sh": float(bps_file_sh),
        "bps_file_hex": float(bps_file_hex),
        "bits_file_sh": file_bits_sh,
        "bits_file_hex": file_bits_hex,
    }


@torch.no_grad()
def final_decode_and_eval(
    gaussians,
    scene,
    renderFunc,
    pipe,
    background,
    stage,
    dataset_type,
    tb_writer,
    iteration,
    bit_paths,
    Q_hex=0.1,
):
    """
    在训练最后一轮:
      1) 用最后一次压缩得到的 bitstream 做 decompress
      2) 把 decode 后的 SH & hexplanes 写回 gaussians
      3) 把 quantize_SH / quantize_HEX 重新置 False
      4) 用“压缩-解压后”的模型跑一次 test（调用 training_report）
    """
    file_path_sh = bit_paths.get("file_path_sh", None)
    file_path_hex = bit_paths.get("file_path_hex", None)

    if file_path_sh is None or file_path_hex is None:
        print("[final_decode_and_eval] 没有可用的 bitstream 路径，跳过最终解码测试。")
        return

    device = gaussians._xyz.device

    print("\n[FINAL] Decode SH & hexplanes back to GaussianModel ...")

    # ---- 1) 解码 SH，并写回 features_rest ----
    x_hat = gaussians.decompress_attri(file_path_sh, device=device)  # [N, 15, 3]
    gaussians._features_rest = x_hat

    # ---- 2) 解码 hexplanes，并写回 deformation grid ----
    decoded_hex = gaussians.decompress_hexplanes(file_path_hex, device=device)
    grids = gaussians._deformation.deformation_net.grid.grids

    for i in range(len(grids)):
        for j in range(len(grids[i])):
            target = grids[i][j]           # nn.Parameter
            src = decoded_hex[i][j].to(target.device)
            if target.shape != src.shape:
                print(f"[WARN] scale {i}, plane {j} shape mismatch: {target.shape} vs {src.shape}")
            else:
                target.data.copy_(src)

    # ---- 3) 关闭量化开关（因为已经是 decode 后的离散值了，不要再二次量化）----
    if hasattr(gaussians, "quantize_SH"):
        gaussians.quantize_SH = False
    try:
        if hasattr(gaussians._deformation.deformation_net.grid, "quantize_HEX"):
            gaussians._deformation.deformation_net.grid.quantize_HEX = False
    except AttributeError:
        pass

    print("[FINAL] Start evaluation with decoded parameters ...")

    _ = training_report(
        tb_writer,
        iteration,
        torch.tensor(0.0, device=device),  # 占位 Ll1
        torch.tensor(0.0, device=device),  # 占位 total loss
        l1_loss,
        0.0,                               # elapsed time placeholder
        [iteration],                       # 强制这一轮 test
        scene,
        renderFunc,
        [pipe, background],
        stage,
        dataset_type,
    )


# ============================== 主训练逻辑 ==============================

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer, expname):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
            
    # initialize PCA directions
    gaussians.set_pca()  

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("data loading done")

    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=16,
                collate_fn=list
            )
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(
                viewpoint_stack,
                batch_size=batch_size,
                shuffle=True,
                num_workers=16,
                collate_fn=list
            )
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    # dynerf, zerostamp_init
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 

    count = 0

    ## start training
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    count += 1
                    viewpoint_index = (count) % len(video_cams)
                    if (count // (len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    
                    net_image = render(
                        custom_cam, gaussians, pipe, background,
                        scaling_modifer, stage=stage, cam_type=scene.dataset_type
                    )["render"]

                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(
                        viewpoint_stack,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=32,
                        collate_fn=list
                    )
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:  # dnerf or coarse dynerf
            idx = 0
            viewpoint_cams = []

            while idx < batch_size:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background,
                stage=stage, cam_type=scene.dataset_type
            )
            image, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)

        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

        loss = Ll1
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(
                hyper.time_smoothness_weight,
                hyper.l1_time_planes,
                hyper.plane_tv_weight
            )
            loss += tv_loss

            # == 这里完全对齐 “旧逻辑” 的码率 schedule ==
            if iteration % 3 == 0:
                est_stats = gaussians.checkerboard.calculate_hexplane_bits(
                    gaussians._deformation.deformation_net.grid.grids,
                    Q=gaussians._deformation.deformation_net.grid.Q
                )
                bpp = est_stats["bpp"]
                bpp_sh = gaussians.attri_entropy(quantization_mode="noise")

                # 只要开始算 bpp，就让 HEX 走量化路径
                gaussians._deformation.deformation_net.grid.quantize_HEX = True

                if iteration > 2000:
                    gaussians.quantize_SH = True
                    loss += args.rlambda * (bpp + bpp_sh)
                else:
                    loss += args.rlambda * bpp
            else:
                bpp = "Not measured!"
                bpp_sh = "Not measured!"

            # 周期性真实压缩 debug（不改模型参数）
            if iteration % 10000 == 0:
                _ = debug_compress_gaussians(
                    gaussians,
                    iteration,
                    Q_hex=gaussians._deformation.deformation_net.grid.Q,
                    bitstream_dir="./bitstream",
                )

        else:
            bpp = "Not measured!"
            bpp_sh = "Not measured!"

        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0 - ssim_loss)

        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = \
                viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "psnr": f"{psnr_:.2f}",
                    "point": f"{total_point}",
                    "BPP": f"{bpp}",
                    "BPP_sh": f"{bpp_sh}"},
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            psnr_test = training_report(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render,
                [pipe, background], stage, scene.dataset_type
            )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
                
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                    or (iteration < 60000 and iteration % 100 == 99):
                    render_training_image(
                        scene, gaussians,
                        [test_cams[iteration % len(test_cams)]],
                        render, pipe, background,
                        stage + "test", iteration, timer.get_elapsed_time(),
                        scene.dataset_type
                    )
                    render_training_image(
                        scene, gaussians,
                        [train_cams[iteration % len(train_cams)]],
                        render, pipe, background,
                        stage + "train", iteration, timer.get_elapsed_time(),
                        scene.dataset_type
                    )

            timer.start()

            # light gaussian pruning
            if stage == "fine":
                if iteration in args.prune_iterations:
                    ic("Before prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
                    i = args.prune_iterations.index(iteration)
                    gaussian_list, imp_list = prune_list(
                        gaussians, scene, pipe, background, stage
                    )

                    if args.prune_type == "important_score":
                        gaussians.prune_gaussians(
                            (args.prune_decay**i) * args.prune_percent, imp_list
                        )
                    elif args.prune_type == "v_important_score":
                        v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                        gaussians.prune_gaussians(
                            (args.prune_decay**i) * args.prune_percent, v_list
                        )
                    elif args.prune_type == "max_v_important_score":
                        v_list = imp_list * torch.max(gaussians.get_scaling, dim=1)[0]
                        gaussians.prune_gaussians(
                            (args.prune_decay**i) * args.prune_percent, v_list
                        )
                    elif args.prune_type == "count":
                        gaussians.prune_gaussians(
                            (args.prune_decay**i) * args.prune_percent, gaussian_list
                        )
                    elif args.prune_type == "opacity":
                        gaussians.prune_gaussians(
                            (args.prune_decay**i) * args.prune_percent,
                            gaussians.get_opacity.detach(),
                        )
                    else:
                        raise Exception("Unsupportive pruning method")

                    ic("After prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = \
                        opt.opacity_threshold_fine_init - iteration * \
                        (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / \
                        (opt.densify_until_iter)
                    densify_threshold = \
                        opt.densify_grad_threshold_fine_init - iteration * \
                        (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / \
                        (opt.densify_until_iter)

                if iteration > opt.densify_from_iter and \
                   iteration % opt.densification_interval == 0 and \
                   gaussians.get_xyz.shape[0] < 360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(
                        densify_threshold, opacity_threshold,
                        scene.cameras_extent, size_threshold,
                        5, 5, scene.model_path, iteration, stage
                    )
                
                if iteration > opt.pruning_from_iter and \
                   iteration % opt.pruning_interval == 0 and \
                   gaussians.get_xyz.shape[0] > 200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(
                        densify_threshold, opacity_threshold,
                        scene.cameras_extent, size_threshold
                    )
                    
                if iteration % opt.densification_interval == 0 and \
                   gaussians.get_xyz.shape[0] < 360000 and opt.add_point:
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)

                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth"
                )

            # =================== fine 阶段最后一轮：真实压缩-解压评估 ===================
            if stage == "fine" and iteration == final_iter:
                summary_file = "./log/summary_results.txt"
                with open(summary_file, "a") as f:
                    f.write(f"Experiment: {expname}\n")
                    f.write(f"PSNR(train-eval): {psnr_test:.2f} dB\n")

                bit_info = debug_compress_gaussians(
                    gaussians,
                    iteration,
                    Q_hex=gaussians._deformation.deformation_net.grid.Q,
                    bitstream_dir="./bitstream",
                )

                total_MB = gaussians.calculate_mem(
                    file_bits_hex=bit_info["bits_file_sh"],
                    file_bits_sh=bit_info["bits_file_hex"],
                )
                print(f"[FINAL] Total storage (MB): {total_MB:.2f}")

                final_decode_and_eval(
                    gaussians,
                    scene,
                    render,
                    pipe,
                    background,
                    stage,
                    scene.dataset_type,
                    tb_writer,
                    iteration,
                    bit_info,
                    Q_hex=gaussians._deformation.deformation_net.grid.Q,
                )


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(
        dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
        checkpoint_iterations, checkpoint, debug_from,
        gaussians, scene, "coarse", tb_writer, opt.coarse_iterations, timer, expname
    )
    scene_reconstruction(
        dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
        checkpoint_iterations, checkpoint, debug_from,
        gaussians, scene, "fine", tb_writer, opt.iterations, timer, expname
    )


def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    psnr_eval = 0.0
    ssim_eval = 0.0

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',  'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())]
                                          for idx in range(10, 5000, 299)]},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(10, 5000, 299)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, stage=stage,
                                   cam_type=dataset_type, *renderArgs)["render"],
                        0.0, 1.0
                    )
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage + "/" + config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                image[None], global_step=iteration
                            )
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage + "/" + config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                    gt_image[None], global_step=iteration
                                )
                    except:
                        pass

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()

                    ssim_val = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                    ssim_test += ssim_val.double()

                num = len(config['cameras'])
                psnr_test /= num
                l1_test /= num
                ssim_test /= num

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test
                ))

                if config['name'] == 'test':
                    psnr_eval = psnr_test
                    ssim_eval = ssim_test

                if tb_writer:
                    base = stage + "/" + config['name']
                    tb_writer.add_scalar(base + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(base + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(base + '/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram",
                                    scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points',
                                 scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(
                f'{stage}/deformation_rate',
                scene.gaussians._deformation_table.sum() /
                scene.gaussians.get_xyz.shape[0],
                iteration
            )
            tb_writer.add_histogram(
                f"{stage}/scene/motion_histogram",
                scene.gaussians._deformation_accum.mean(dim=-1) / 100,
                iteration,
                max_bins=500
            )

        torch.cuda.empty_cache()

    return psnr_eval


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[3000, 10000, 20000, 25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[14000, 20000, 25000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    # prune configuration
    parser.add_argument("--prune_iterations", nargs="+", type=int, default=[13000, 17000])
    parser.add_argument(
        "--prune_type", type=str, default="important_score"
    )
    parser.add_argument("--prune_percent", type=float, default=0.4)
    parser.add_argument("--prune_decay", type=float, default=1)

    parser.add_argument("--rlambda", type=float, default=0.001)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, args.expname)

    print("\n Training complete.")











