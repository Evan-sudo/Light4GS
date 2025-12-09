import os
import argparse
from glob import glob

import numpy as np
from PIL import Image
import cv2

from skimage.metrics import structural_similarity as ssim


def load_image(path):
    """读图，返回 [0,1] 的 float32 RGB numpy 数组 (H,W,3)。"""
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    return img


def get_sorted_image_paths(folder):
    """按文件名排序拿到所有图片路径。"""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(folder, e)))
    paths = sorted(paths)
    return paths


# ----------------- PSNR -----------------

def psnr_single(gt, pred, eps=1e-10):
    """单帧 PSNR，gt/pred 为 [0,1] float32 numpy。"""
    mse = np.mean((gt - pred) ** 2)
    if mse < eps:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_avg_psnr(imgs_gt, imgs_pred):
    assert len(imgs_gt) == len(imgs_pred)
    vals = []
    for g, p in zip(imgs_gt, imgs_pred):
        vals.append(psnr_single(g, p))
    # 如果有 inf，就直接平均时当成一个很大的数
    vals = np.array(vals, dtype=np.float64)
    mean_psnr = float(np.mean(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else float("inf")
    return mean_psnr, vals.tolist()


# ----------------- tOF -----------------

def compute_optical_flow_pair(img1, img2):
    """
    计算从 img1->img2 的光流，使用 Farneback（OpenCV）。
    img1/img2: [0,1] float32 RGB
    返回 flow: (H,W,2) float32
    """
    gray1 = cv2.cvtColor((img1 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((img2 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow.astype(np.float32)


def compute_tof(imgs_gt, imgs_pred):
    """
    Temporal Optical Flow loss:
    对于每个相邻帧对 t,t+1：
        flow_gt   = Flow(gt_t,   gt_{t+1})
        flow_pred = Flow(pred_t, pred_{t+1})
        L1 = mean(|flow_gt - flow_pred|)
    最后在时间上平均。
    """
    assert len(imgs_gt) == len(imgs_pred)
    assert len(imgs_gt) >= 2

    losses = []
    for i in range(len(imgs_gt) - 1):
        flow_gt = compute_optical_flow_pair(imgs_gt[i], imgs_gt[i + 1])
        flow_pred = compute_optical_flow_pair(imgs_pred[i], imgs_pred[i + 1])

        # 尺寸不一致时把 pred 的光流 resize 到 gt 上
        if flow_gt.shape != flow_pred.shape:
            H, W = flow_gt.shape[:2]
            flow_pred = cv2.resize(flow_pred, (W, H), interpolation=cv2.INTER_LINEAR)

        l1 = np.mean(np.abs(flow_gt - flow_pred))
        losses.append(l1)

    mean_tof = float(np.mean(losses))
    return mean_tof, losses


# ----------------- flicker SSIM -----------------

def compute_flicker_ssim(imgs_pred):
    """
    flicker SSIM:
    只看预测序列内部相邻帧的 SSIM，取平均。
    数值越接近 1 表示越不闪（时间上越稳定）。
    """
    assert len(imgs_pred) >= 2
    vals = []

    for i in range(len(imgs_pred) - 1):
        im1 = imgs_pred[i]
        im2 = imgs_pred[i + 1]

        # skimage 新版用 channel_axis=-1，老版用 multichannel=True
        try:
            val = ssim(im1, im2, channel_axis=-1, data_range=1.0)
        except TypeError:
            val = ssim(im1, im2, multichannel=True, data_range=1.0)

        vals.append(val)

    mean_flicker_ssim = float(np.mean(vals))
    return mean_flicker_ssim, vals


# ----------------- main -----------------

def main(root_dir):
    gt_dir = os.path.join(root_dir, "gt")
    pred_dir = os.path.join(root_dir, "renders")

    gt_paths = get_sorted_image_paths(gt_dir)
    pred_paths = get_sorted_image_paths(pred_dir)

    if len(gt_paths) == 0 or len(pred_paths) == 0:
        raise RuntimeError("gt/ 或 renders/ 里面没找到图片。")

    if len(gt_paths) != len(pred_paths):
        raise RuntimeError(f"gt 和 renders 帧数不一致: {len(gt_paths)} vs {len(pred_paths)}")

    print(f"Found {len(gt_paths)} frames.")

    # 简单检查一下文件名是否对应
    for gp, pp in zip(gt_paths, pred_paths):
        if os.path.basename(gp) != os.path.basename(pp):
            print("警告：gt 和 renders 的文件名并不完全对应：")
            print("  gt:", os.path.basename(gp))
            print("  rd:", os.path.basename(pp))
            break

    # 读图
    imgs_gt = [load_image(p) for p in gt_paths]
    imgs_pred = [load_image(p) for p in pred_paths]

    # 把 renders resize 到 gt 的分辨率
    H, W = imgs_gt[0].shape[:2]
    imgs_pred_r = []
    for img in imgs_pred:
        if img.shape[:2] != (H, W):
            img_resized = np.array(
                Image.fromarray((img * 255).astype(np.uint8)).resize((W, H))
            ).astype(np.float32) / 255.0
        else:
            img_resized = img
        imgs_pred_r.append(img_resized)

    # 1) 平均 PSNR
    mean_psnr, psnr_list = compute_avg_psnr(imgs_gt, imgs_pred_r)

    # 2) tOF
    mean_tof, tof_list = compute_tof(imgs_gt, imgs_pred_r)

    # 3) flicker SSIM（只看 renders）
    mean_flicker_ssim, flicker_list = compute_flicker_ssim(imgs_pred_r)

    print("\n===== Metrics =====")
    print(f"Average PSNR (frame-wise): {mean_psnr:.4f} dB")
    print(f"tOF (mean L1 on flow):     {mean_tof:.6f}")
    print(f"flicker SSIM (renders):    {mean_flicker_ssim:.6f}")

    # 如果你想看每一帧/每一对的细节，可以自行打印 psnr_list / tof_list / flicker_list
    # 比如：
    # for i, (p, f) in enumerate(zip(psnr_list, flicker_list + [None])):
    #     print(f"frame {i}: PSNR={p:.3f}, flickerSSIM_pair_with_next={f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute avg PSNR, tOF and flicker SSIM for gt vs renders."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="test/ours_30000",
        help="根目录，下面应该有 gt/ 和 renders/ 两个子目录。",
    )
    args = parser.parse_args()
    main(args.root)