import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models.google import JointAutoregressiveHierarchicalPriors
from scene.layers import CheckerboardMaskedConv2d     # masked conv
from scene.modules import Demultiplexer, Multiplexer  # checkerboard Space2Depth / Depth2Space


# ========================== 小工具 ==========================

def normalize_coordinates(H, W, device=None):
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
    return grid.unsqueeze(0)  # [1, H, W, 2]


def _strings_num_bits(strings):
    if strings is None:
        return 0
    if isinstance(strings, (bytes, bytearray)):
        return len(strings) * 8
    if isinstance(strings, (list, tuple)):
        return sum(_strings_num_bits(s) for s in strings)
    if torch.is_tensor(strings):
        return int(strings.numel()) * 8
    raise TypeError(f"Unsupported stream type: {type(strings)}")


# ========================== 核心模型（checkerboard 版） ==========================

class CheckerboardAutoregressive(JointAutoregressiveHierarchicalPriors):
    """
    多尺度 hexplane + hyperprior + checkerboard + context.

    plane index：
      0: XY  (space-only)
      1: XZ  (space-only)
      2: XT  (space-time, hyperprior)
      3: YZ  (space-only)
      4: YT  (space-time, hyperprior)
      5: ZT  (space-time, hyperprior)

    设计：
      - scale 0：
          2,4,5: hyperprior + checkerboard
          0,1,3: inter-plane context (从 XT/YT/ZT) + checkerboard
      - scale > 0：
          所有 plane：hyperprior + inter-scale context（来自低层同 index plane），**不再用 checkerboard**
    """

    def __init__(self, N, M, **kwargs):
        """
        N: hyperprior 通道数
        M: latent 通道数（= 每个平面 channel 数）
        """
        super().__init__(N, M, **kwargs)
        self.M = M

        # checkerboard：根据 anchor full map 做 context（只在低层用）
        self.context_prediction = CheckerboardMaskedConv2d(
            in_channels=M,
            out_channels=2 * M,
            kernel_size=5,
            padding=2,
            stride=1,
        )

        # cross-plane / cross-scale context -> 变成 (2M) 作为高斯参数基础
        # !!! 修改 1：in_channels 从 2M 改成 M，out_channels 还是 2M
        self.context_to_params = nn.Conv2d(
            in_channels=M,
            out_channels=2 * M,
            kernel_size=5,
            padding=2,
            stride=1,
        )

    def get_parameters(self):
        """
        你之前是直接把所有参数丢进去训练，这里保持一致。
        """
        parameter_list = []
        for name, param in self.named_parameters():
            parameter_list.append(param)
        return parameter_list

    # ------------------------------------------------------------------
    #  space-only 的 inter-plane context: 从 XT/YT/ZT 构造 XY/XZ/YZ 的上下文
    # ------------------------------------------------------------------
    def _build_spaceonly_context_maps(self, y_plane, ctx_src_planes):
        """
        y_plane      : [1,M,H,W]，目标平面（XY / XZ / YZ）
        ctx_src_planes: list[Tensor]，每个 [1,M,Ha,Wa]，为对应的 XT/YT/ZT plane。

        规则：沿着“时间轴”(最后一维 Wa) 做 mean，然后根据 Ha 与目标 H/W 的对应关系
        去 expand 成 [1,M,H,W]。
        """
        _, _, H, W = y_plane.shape
        ctx_maps = []
        for src in ctx_src_planes:
            _, _, Ha, Wa = src.shape
            # 沿最后一维（时间轴）做 mean -> [1,M,Ha]
            ctx_1d = src.mean(dim=-1)  # [1,M,Ha]

            if Ha == H:
                # Ha 对应 H：每一行单独值，沿 W 复制
                ctx = ctx_1d.unsqueeze(-1).expand(-1, -1, H, W)
            elif Ha == W:
                # Ha 对应 W：每一列单独值，沿 H 复制
                ctx = ctx_1d.unsqueeze(2).expand(-1, -1, H, W)
            else:
                # 极端情况：Ha 和 H/W 都不相等，用插值到 H 再沿 W 复制
                ctx_hw = ctx_1d.unsqueeze(-1)  # [1,M,Ha,1]
                ctx_hw = F.interpolate(
                    ctx_hw, size=(H, 1), mode="bilinear", align_corners=True
                ).squeeze(-1)  # [1,M,H]
                ctx = ctx_hw.unsqueeze(-1).expand(-1, -1, H, W)

            ctx_maps.append(ctx)
        return ctx_maps

    # ====================== 低层：hyperprior + checkerboard ======================

    def _forward_ckbd_two_pass(self, y, Q_val, params_base):
        """
        给定 base params（hyperprior 或 context_to_params 输出），
        按照 compress/decompress 的 checkerboard 流程来建模：

          - anchor: 只用 params_base（ctx=0）
          - non-anchor: 用 params_base + checkerboard context(anchor_full)

        注意：
          - 这里不再做 quantize/dequantize，calculate_hexplane_bits 在外面已经统一加过 noise。
          - y 是 (加噪后的) latent 原始域值。
        """
        device = y.device
        y_scaled = y / Q_val  # 熵模型在 /Q 空间

        N, C, H, W = y_scaled.shape

        # --- anchor 路：ctx=0 ---
        zero_ctx = torch.zeros(N, 2 * self.M, H, W, device=device)
        gauss_anchor_full = self.entropy_parameters(
            torch.cat([params_base, zero_ctx], dim=1)
        )  # [N,4M,H,W]
        scales_anchor_full, means_anchor_full = gauss_anchor_full.chunk(2, 1)

        # 分出 anchor / non-anchor 参数
        scales_anchor, _ = Demultiplexer(scales_anchor_full)
        means_anchor, _ = Demultiplexer(means_anchor_full)

        # 分出 anchor / non-anchor 的 latent
        y_anchor_scaled, y_nonanchor_scaled = Demultiplexer(y_scaled)

        # anchor likelihood
        _, lkl_anchor = self.gaussian_conditional(
            y_anchor_scaled, scales_anchor, means=means_anchor
        )

        # --- 用 anchor 构 context（模拟 decode 看到的 anchor full map） ---
        # 这里直接用 y_anchor_scaled（已经是 noisy 的 /Q 值），再乘回 Q
        anchor_full_scaled = Multiplexer(
            y_anchor_scaled, torch.zeros_like(y_anchor_scaled)
        )  # [N,M,H,W]
        ctx_params = self.context_prediction(anchor_full_scaled * Q_val)  # conv 在原始域

        gauss_full = self.entropy_parameters(
            torch.cat([params_base, ctx_params], dim=1)
        )  # [N,4M,H,W]
        scales_full, means_full = gauss_full.chunk(2, 1)

        # non-anchor 参数
        _, scales_nonanchor = Demultiplexer(scales_full)
        _, means_nonanchor = Demultiplexer(means_full)

        # non-anchor likelihood
        _, lkl_nonanchor = self.gaussian_conditional(
            y_nonanchor_scaled, scales_nonanchor, means=means_nonanchor
        )

        # 用 Multiplexer 把 anchor / non-anchor 的 likelihood 拼回 full map
        lkl_full = Multiplexer(lkl_anchor, lkl_nonanchor)  # [N,C,H,W]
        return lkl_full

    def forward_low_space_time(self, y, Q):
        """
        低 scale + space-time (XT / YT / ZT)：
          hyperprior + checkerboard

        步骤：
          z = h_a(y/Q)
          z_hat, z_lkl = entropy_bottleneck(z)
          params_base = h_s(z_hat)
          y_lkl = _forward_ckbd_two_pass(y, Q, params_base)
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)

        y_scaled = y / Q_val
        z = self.h_a(y_scaled)
        z_hat, z_lkl = self.entropy_bottleneck(z)
        params_base = self.h_s(z_hat)  # [N,2M,H,W]

        y_lkl = self._forward_ckbd_two_pass(y, Q_val, params_base)

        return {
            "likelihoods": {"y": y_lkl, "z": z_lkl},
        }

    def forward_low_space_only(self, y, Q, ctx_src_planes):
        """
        低 scale + space-only (XY / XZ / YZ)：
          inter-plane context (XT/YT/ ZT) + checkerboard

        ctx_src_planes: list[Tensor]，对应同 scale 的 XT/YT/ZT（已经加噪）
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)
        device = y.device

        # 生成和 compress 一致的 context maps（shape=[1,M,H,W]）
        context_planes = self._build_spaceonly_context_maps(
            y_plane=y, ctx_src_planes=ctx_src_planes
        )
        # !!! 修改 2：两张 plane 先做平均，得到 [N,M,H,W]，再喂给 context_to_params
        if len(context_planes) == 1:
            scale_context = context_planes[0]          # [N,M,H,W]
        else:
            scale_context = sum(context_planes) / len(context_planes)

        params_base = self.context_to_params(scale_context)  # [N,2M,H,W]

        z_lkl = torch.empty(0, device=device)
        y_lkl = self._forward_ckbd_two_pass(y, Q_val, params_base)
        return {
            "likelihoods": {"y": y_lkl, "z": z_lkl},
        }

    # ====================== 高层：hyperprior + inter-scale context（无 checkerboard） ======================

    def forward_high_scale_no_ckbd(self, y, Q, lower_plane):
        """
        高 scale（i > 0）：
          hyperprior + inter-scale context（上一层同 index plane 上采样）
          不再用 checkerboard。
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)
        device = y.device

        y_scaled = y / Q_val

        # hyperprior
        z = self.h_a(y_scaled)
        z_hat, z_lkl = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)  # [N,2M,H,W]

        # 上采样低层 context
        _, _, H_cur, W_cur = y.shape
        grid = normalize_coordinates(H_cur, W_cur, device=device)
        ctx_up = F.grid_sample(
            lower_plane, grid, mode="bilinear", align_corners=True
        )  # [N,M,H_cur,W_cur]

        # !!! 修改 3：inter-scale 直接用 M 通道的 ctx_up
        scale_context = ctx_up                      # [N,M,H,W]
        ctx_params = self.context_to_params(scale_context)   # [N,2M,H,W]

        # hyper + ctx -> 高斯参数
        gaussian_params = self.entropy_parameters(
            torch.cat([hyper_params, ctx_params], dim=1)
        )  # [N,4M,H,W]
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        _, y_lkl = self.gaussian_conditional(y_scaled, scales_hat, means=means_hat)

        return {
            "likelihoods": {"y": y_lkl, "z": z_lkl},
        }

    # ====================== 训练时 bits（统一 noise 版） ======================

    def calculate_hexplane_bits(self, hexplanes, Q):
        """
        hexplanes: List[List[Tensor]]，形状同 entropy_compress：
            hexplanes[scale][plane] = [1,M,H,W]
            plane index 含义：
              0: XY (space-only)
              1: XZ (space-only)
              2: XT (space-time, hyperprior)
              3: YZ (space-only)
              4: YT (space-time, hyperprior)
              5: ZT (space-time, hyperprior)
        Q: 标量步长

        返回:
        {
            "total_bits": Tensor (0-dim),
            "bpp": Tensor (0-dim)
        }

        逻辑：
          - 先对所有 plane 做一次 uniform noise quantization（模拟最终离散 latent）：
              y_scaled = y/Q
              y_hat_scaled = quantize(y_scaled, "noise")
              y_noisy = y_hat_scaled * Q
          - 然后在这些 noisy latent 上跑跟 compress 对应的流程：
              * scale 0:
                    2,4,5 -> forward_low_space_time (hyperprior + ckbd)
                    0,1,3 -> forward_low_space_only (inter-plane + ckbd)
              * scale >0:
                    所有 plane -> forward_high_scale_no_ckbd (hyperprior + inter-scale ctx)
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)
        device = next(self.parameters()).device

        # ---------- 1) 统一加 noise quant ----------
        noisy_hexplanes = []
        for scale in hexplanes:
            noisy_scale = []
            for y in scale:
                y = y.to(device)
                y_scaled = y / Q_val
                y_hat_scaled = self.gaussian_conditional.quantize(
                    y_scaled, mode="noise"
                )  # y_scaled + U(-0.5,0.5)
                y_noisy = y_hat_scaled * Q_val
                noisy_scale.append(y_noisy)
            noisy_hexplanes.append(noisy_scale)

        hexplanes = noisy_hexplanes
        num_scales = len(hexplanes)

        total_bits = torch.zeros((), device=device)
        total_symbols = 0
        eps = 1e-9

        space_only_indices = [0, 1, 3]
        space_time_indices = [2, 4, 5]

        def bits_from_lkl(lkl: torch.Tensor) -> torch.Tensor:
            if lkl.numel() == 0:
                return torch.zeros((), device=lkl.device)
            lkl = lkl.clamp(min=eps)
            return (-torch.log2(lkl)).sum()

        for i, scale in enumerate(hexplanes):
            if i == 0:
                # ---------- 低 scale ----------
                # 1) space-time：hyperprior + checkerboard
                for j in space_time_indices:
                    y = scale[j]
                    out = self.forward_low_space_time(y, Q_val)

                    lkl_y = out["likelihoods"]["y"]
                    lkl_z = out["likelihoods"]["z"]

                    bits_y = bits_from_lkl(lkl_y)
                    bits_z = bits_from_lkl(lkl_z)

                    total_bits = total_bits + bits_y + bits_z
                    total_symbols += lkl_y.numel() + lkl_z.numel()

                # 2) space-only：inter-plane context + checkerboard
                for j in space_only_indices:
                    y = scale[j]

                    if j == 0:      # XY <- XT, YT
                        ctx_ids = [2, 4]
                    elif j == 1:    # XZ <- XT, ZT
                        ctx_ids = [2, 5]
                    elif j == 3:    # YZ <- YT, ZT
                        ctx_ids = [4, 5]
                    else:
                        ctx_ids = []

                    ctx_src_planes = [scale[k] for k in ctx_ids]
                    out = self.forward_low_space_only(y, Q_val, ctx_src_planes)

                    lkl_y = out["likelihoods"]["y"]
                    lkl_z = out["likelihoods"]["z"]  # 空 tensor

                    bits_y = bits_from_lkl(lkl_y)
                    bits_z = bits_from_lkl(lkl_z)

                    total_bits = total_bits + bits_y + bits_z
                    total_symbols += lkl_y.numel() + lkl_z.numel()

            else:
                # ---------- 高 scale：hyperprior + inter-scale ctx（无 ckbd） ----------
                prev_scale = hexplanes[i - 1]

                for j, y in enumerate(scale):
                    lower = prev_scale[j]   # [1,M,H_low,W_low]
                    out = self.forward_high_scale_no_ckbd(y, Q_val, lower)

                    lkl_y = out["likelihoods"]["y"]
                    lkl_z = out["likelihoods"]["z"]

                    bits_y = bits_from_lkl(lkl_y)
                    bits_z = bits_from_lkl(lkl_z)

                    total_bits = total_bits + bits_y + bits_z
                    total_symbols += lkl_y.numel() + lkl_z.numel()

        bpp = total_bits / (float(total_symbols) + eps)

        return {
            "total_bits": total_bits,
            "bpp": bpp,
        }

    # ====================== checkerboard compress/decompress core ======================

    @torch.no_grad()
    def _checkerboard_compress_core(self, y, Q_val, params_base):
        """
        y           : [N, M, H, W] 原始 latent
        Q_val       : float, 量化步长
        params_base : [N, 2M, H, W]，来自 hyperprior 或 context_to_params

        返回 (anchor_strings, non_anchor_strings)
        """
        device = y.device
        y_scaled = y / Q_val

        N, C, H, W = y_scaled.shape

        # PASS 1: anchor (ctx = 0)
        zero_ctx = torch.zeros(N, 2 * self.M, H, W, device=device)
        gauss_anchor_full = self.entropy_parameters(
            torch.cat([params_base, zero_ctx], dim=1)
        )
        scales_anchor_full, means_anchor_full = gauss_anchor_full.chunk(2, 1)

        # dequantize 模拟解码端 y_hat_scaled（anchor 的均值/scale）
        y_hat_full_scaled = self.gaussian_conditional.quantize(
            y_scaled, mode="dequantize", means=means_anchor_full
        )

        # 真正要编码的 anchor / non-anchor
        y_anchor_scaled, y_nonanchor_scaled = Demultiplexer(y_scaled)

        # dequant 的 anchor 构造 full map，非 anchor = 0
        y_anchor_hat_scaled, _ = Demultiplexer(y_hat_full_scaled)
        anchor_full_scaled = Multiplexer(
            y_anchor_hat_scaled, torch.zeros_like(y_anchor_hat_scaled)
        )

        # PASS 2: 用 anchor full map 做 context
        ctx_params = self.context_prediction(anchor_full_scaled * Q_val)
        gauss_full = self.entropy_parameters(
            torch.cat([params_base, ctx_params], dim=1)
        )
        scales_full, means_full = gauss_full.chunk(2, 1)

        # anchor 用 PASS1 参数
        scales_anchor, _ = Demultiplexer(scales_anchor_full)
        means_anchor, _ = Demultiplexer(means_anchor_full)

        # non-anchor 用 PASS2 参数
        _, scales_nonanchor = Demultiplexer(scales_full)
        _, means_nonanchor = Demultiplexer(means_full)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        indexes_nonanchor = self.gaussian_conditional.build_indexes(scales_nonanchor)

        anchor_strings = self.gaussian_conditional.compress(
            y_anchor_scaled, indexes_anchor, means=means_anchor
        )
        non_anchor_strings = self.gaussian_conditional.compress(
            y_nonanchor_scaled, indexes_nonanchor, means=means_nonanchor
        )

        return anchor_strings, non_anchor_strings

    @torch.no_grad()
    def _checkerboard_decompress_core(self, strings_pair, Q_val, params_base, y_shape):
        """
        strings_pair : (anchor_strings, non_anchor_strings)
        Q_val        : float
        params_base  : [N, 2M, H, W]
        y_shape      : [N, M, H, W]

        返回: y_hat (原始域)
        """
        anchor_strings, non_anchor_strings = strings_pair
        device = params_base.device
        N, _, H, W = params_base.shape

        # PASS 1: anchor
        zero_ctx = torch.zeros(N, 2 * self.M, H, W, device=device)
        gauss_anchor_full = self.entropy_parameters(
            torch.cat([params_base, zero_ctx], dim=1)
        )
        scales_anchor_full, means_anchor_full = gauss_anchor_full.chunk(2, 1)

        scales_anchor, _ = Demultiplexer(scales_anchor_full)
        means_anchor, _ = Demultiplexer(means_anchor_full)

        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        y_anchor_scaled = self.gaussian_conditional.decompress(
            anchor_strings, indexes_anchor, means=means_anchor
        )

        anchor_full_scaled = Multiplexer(
            y_anchor_scaled, torch.zeros_like(y_anchor_scaled)
        )

        # PASS 2: non-anchor
        ctx_params = self.context_prediction(anchor_full_scaled * Q_val)
        gauss_full = self.entropy_parameters(
            torch.cat([params_base, ctx_params], dim=1)
        )
        scales_full, means_full = gauss_full.chunk(2, 1)

        _, scales_nonanchor = Demultiplexer(scales_full)
        _, means_nonanchor = Demultiplexer(means_full)

        indexes_nonanchor = self.gaussian_conditional.build_indexes(scales_nonanchor)
        y_nonanchor_scaled = self.gaussian_conditional.decompress(
            non_anchor_strings, indexes_nonanchor, means=means_nonanchor
        )

        y_hat_scaled = Multiplexer(y_anchor_scaled, y_nonanchor_scaled)
        y_hat = y_hat_scaled * Q_val
        y_hat = y_hat.view(*y_shape)
        return y_hat

    # -------------------- 低层 space-time: hyperprior + checkerboard --------------------

    @torch.no_grad()
    def compress_space_time(self, y, Q):
        """
        y: [1, M, H, W] 原始 latent
        Q: float / 0-dim tensor
        返回: {"strings":[anchor,non,z_strings], "shape_z":(Hz,Wz)}
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)
        y_scaled = y / Q_val

        z = self.h_a(y_scaled)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params_base = self.h_s(z_hat)

        anchor_strings, non_anchor_strings = self._checkerboard_compress_core(
            y, Q_val, params_base
        )
        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape_z": z.size()[-2:],
        }

    @torch.no_grad()
    def decompress_space_time(self, strings, shape_z, y_shape, Q):
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)
        anchor_strings, non_anchor_strings, z_strings = strings

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape_z)
        params_base = self.h_s(z_hat)

        y_hat = self._checkerboard_decompress_core(
            (anchor_strings, non_anchor_strings), Q_val, params_base, y_shape
        )
        return y_hat

    # -------------------- 低层 space-only: inter-plane context + checkerboard --------------------

    @torch.no_grad()
    def compress_space_only_low(self, y, Q, ctx_src_planes):
        """
        y: [1, M, H, W]
        Q: float
        ctx_src_planes: list[Tensor]，对应 scale0 的 XT/YT/ZT 的 decode 结果
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)

        # 先根据 ctx_src_planes 构造 XY/XZ/YZ 的 context maps
        context_planes = self._build_spaceonly_context_maps(
            y_plane=y, ctx_src_planes=ctx_src_planes
        )
        # !!! 修改 4：这里也做平均，保证和 forward / decompress 一致
        if len(context_planes) == 1:
            scale_context = context_planes[0]
        else:
            scale_context = sum(context_planes) / len(context_planes)

        params_base = self.context_to_params(scale_context)

        anchor_strings, non_anchor_strings = self._checkerboard_compress_core(
            y, Q_val, params_base
        )
        return {
            "strings": [anchor_strings, non_anchor_strings],
            "shape_z": None,
        }

    @torch.no_grad()
    def decompress_space_only_low(self, strings, y_shape, Q, ctx_src_planes):
        """
        strings: [anchor_strings, non_anchor_strings]
        ctx_src_planes: compress 时对应的 space-time 解码结果
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)

        dummy_y = torch.zeros(y_shape, device=ctx_src_planes[0].device)
        context_planes = self._build_spaceonly_context_maps(
            y_plane=dummy_y, ctx_src_planes=ctx_src_planes
        )
        # !!! 修改 5：同样用平均，保持对称性
        if len(context_planes) == 1:
            scale_context = context_planes[0]
        else:
            scale_context = sum(context_planes) / len(context_planes)

        params_base = self.context_to_params(scale_context)

        y_hat = self._checkerboard_decompress_core(
            strings, Q_val, params_base, y_shape
        )
        return y_hat

    # -------------------- 高层：hyperprior + inter-scale context（无 checkerboard） --------------------

    @torch.no_grad()
    def compress_high_scale_no_ckbd(self, y, Q, lower_hat):
        """
        y        : [1,M,H,W] 当前高层 plane
        lower_hat: [1,M,H_low,W_low] 下层对应 plane 的 decode 结果
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)
        device = y.device

        y_scaled = y / Q_val

        # hyperprior
        z = self.h_a(y_scaled)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)  # [1,2M,H,W]

        # inter-scale context
        _, _, H_cur, W_cur = y.shape
        grid = normalize_coordinates(H_cur, W_cur, device=device)
        ctx_up = F.grid_sample(
            lower_hat, grid, mode="bilinear", align_corners=True
        )  # [1,M,H_cur,W_cur]

        # !!! 修改 6：直接用 ctx_up 当 context
        scale_context = ctx_up                      # [1,M,H,W]
        ctx_params = self.context_to_params(scale_context)

        gaussian_params = self.entropy_parameters(
            torch.cat([hyper_params, ctx_params], dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(
            y_scaled, indexes, means=means_hat
        )
        return {"strings": [y_strings, z_strings], "shape_z": z.size()[-2:]}

    @torch.no_grad()
    def decompress_high_scale_no_ckbd(self, strings, shape_z, y_shape, Q, lower_hat):
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)
        device = lower_hat.device

        y_strings, z_strings = strings

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape_z)
        hyper_params = self.h_s(z_hat)  # [1,2M,H,W]

        _, _, H_cur, W_cur = y_shape
        grid = normalize_coordinates(H_cur, W_cur, device=device)
        ctx_up = F.grid_sample(
            lower_hat, grid, mode="bilinear", align_corners=True
        )

        scale_context = ctx_up                      # [1,M,H,W]
        ctx_params = self.context_to_params(scale_context)

        gaussian_params = self.entropy_parameters(
            torch.cat([hyper_params, ctx_params], dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat_scaled = self.gaussian_conditional.decompress(
            y_strings, indexes, means=means_hat
        )
        y_hat = y_hat_scaled * Q_val
        return y_hat.view(*y_shape)

    # ======================= 多尺度 hexplanes 压缩 / 解压 =======================

    @torch.no_grad()
    def entropy_compress(self, hexplanes, Q, output_path=None):
        """
        hexplanes: List[List[Tensor]]，hexplanes[scale][plane] = [1,M,H,W]
        Q: 标量步长

        逻辑：
          - scale 0：
              2,4,5: hyperprior + checkerboard
              0,1,3: inter-plane context (XT/YT/ZT) + checkerboard
          - scale >0：
              所有 plane: hyperprior + inter-scale ctx（无 checkerboard）
        """
        Q_val = float(Q.detach().cpu().item()) if torch.is_tensor(Q) else float(Q)
        device = next(self.parameters()).device

        hexplanes = [[p.to(device) for p in scale] for scale in hexplanes]
        num_scales = len(hexplanes)

        decoded = [[None for _ in scale] for scale in hexplanes]

        streams = []
        total_bits = 0
        total_symbols = 0

        space_only_indices = [0, 1, 3]
        space_time_indices = [2, 4, 5]

        for i, scale in enumerate(hexplanes):
            if i == 0:
                # ---------- 最低层：先压 space-time，再压 space-only ----------
                # 1) space-time: XT/YT/ZT
                for j in space_time_indices:
                    y = scale[j]

                    out = self.compress_space_time(y, Q_val)
                    strings = out["strings"]
                    shape_z = out["shape_z"]

                    anchor_strings, non_anchor_strings, z_strings = strings
                    bits_y = _strings_num_bits(anchor_strings) + _strings_num_bits(
                        non_anchor_strings
                    )
                    bits_z = _strings_num_bits(z_strings)
                    bits_plane = bits_y + bits_z

                    total_bits += bits_plane
                    total_symbols += y.numel()

                    streams.append(
                        {
                            "scale_index": i,
                            "plane_index": j,
                            "kind": "space_time_lowest",
                            "shape_y": list(y.shape),
                            "shape_z": shape_z,
                            "strings": strings,
                        }
                    )

                    y_hat = self.decompress_space_time(strings, shape_z, y.shape, Q_val)
                    decoded[i][j] = y_hat.detach()

                # 2) space-only: XY/XZ/YZ，用刚 decode 出来的 XT/YT/ZT 做 context
                for j in space_only_indices:
                    y = scale[j]

                    if j == 0:      # XY <- XT, YT
                        ctx_ids = [2, 4]
                    elif j == 1:    # XZ <- XT, ZT
                        ctx_ids = [2, 5]
                    elif j == 3:    # YZ <- YT, ZT
                        ctx_ids = [4, 5]
                    else:
                        ctx_ids = []

                    ctx_src = [decoded[i][k] for k in ctx_ids]
                    for t in ctx_src:
                        assert t is not None

                    out = self.compress_space_only_low(y, Q_val, ctx_src)
                    strings = out["strings"]

                    bits_plane = _strings_num_bits(strings[0]) + _strings_num_bits(
                        strings[1]
                    )
                    total_bits += bits_plane
                    total_symbols += y.numel()

                    streams.append(
                        {
                            "scale_index": i,
                            "plane_index": j,
                            "kind": "space_only_lowest",
                            "shape_y": list(y.shape),
                            "shape_z": None,
                            "strings": strings,
                        }
                    )

                    y_hat = self.decompress_space_only_low(
                        strings, y.shape, Q_val, ctx_src
                    )
                    decoded[i][j] = y_hat.detach()

            else:
                # ---------- 高 scale：hyperprior + inter-scale ctx（无 checkerboard） ----------
                prev_decoded = decoded[i - 1]
                for j, y in enumerate(scale):
                    lower_hat = prev_decoded[j]
                    assert lower_hat is not None

                    out = self.compress_high_scale_no_ckbd(y, Q_val, lower_hat)
                    strings = out["strings"]
                    shape_z = out["shape_z"]

                    y_strings, z_strings = strings
                    bits_y = _strings_num_bits(y_strings)
                    bits_z = _strings_num_bits(z_strings)
                    bits_plane = bits_y + bits_z

                    total_bits += bits_plane
                    total_symbols += y.numel()

                    streams.append(
                        {
                            "scale_index": i,
                            "plane_index": j,
                            "kind": "high_scale_nockbd",
                            "shape_y": list(y.shape),
                            "shape_z": shape_z,
                            "strings": strings,
                        }
                    )

                    y_hat = self.decompress_high_scale_no_ckbd(
                        strings, shape_z, y.shape, Q_val, lower_hat
                    )
                    decoded[i][j] = y_hat.detach()

        bpp = total_bits / float(total_symbols + 1e-9)
        packed = {
            "Q": Q_val,
            "streams": streams,
            "total_bits": float(total_bits),
            "bpp": float(bpp),
        }
        if output_path is not None:
            torch.save(packed, output_path)
        return packed

    @torch.no_grad()
    def entropy_decompress(self, packed_or_path, device=None):
        if isinstance(packed_or_path, str):
            loaded = torch.load(packed_or_path, map_location=device or "cpu")
        else:
            loaded = packed_or_path

        Q_val = float(loaded["Q"])
        streams = loaded["streams"]

        if device is None:
            device = next(self.parameters()).device

        max_scale = max(s["scale_index"] for s in streams)
        num_scales = max_scale + 1

        per_scale_planes = [0 for _ in range(num_scales)]
        for s in streams:
            i = int(s["scale_index"])
            j = int(s["plane_index"])
            per_scale_planes[i] = max(per_scale_planes[i], j + 1)

        decoded = [
            [None for _ in range(per_scale_planes[i])]
            for i in range(num_scales)
        ]

        for entry in streams:
            i = int(entry["scale_index"])
            j = int(entry["plane_index"])
            kind = entry["kind"]
            shape_y = entry["shape_y"]
            shape_z = entry["shape_z"]
            strings = entry["strings"]

            y_shape = shape_y

            if kind == "space_time_lowest":
                y_hat = self.decompress_space_time(strings, shape_z, y_shape, Q_val)
                decoded[i][j] = y_hat.detach().to(device)

            elif kind == "space_only_lowest":
                # 需要同 scale 的 space-time decode 做 context
                if j == 0:
                    ctx_ids = [2, 4]
                elif j == 1:
                    ctx_ids = [2, 5]
                elif j == 3:
                    ctx_ids = [4, 5]
                else:
                    ctx_ids = []

                ctx_src = [decoded[i][k] for k in ctx_ids]
                for t in ctx_src:
                    assert t is not None

                y_hat = self.decompress_space_only_low(
                    strings, y_shape, Q_val, ctx_src
                )
                decoded[i][j] = y_hat.detach().to(device)

            else:  # "high_scale_nockbd"
                lower_hat = decoded[i - 1][j]
                assert lower_hat is not None

                y_hat = self.decompress_high_scale_no_ckbd(
                    strings, shape_z, y_shape, Q_val, lower_hat
                )
                decoded[i][j] = y_hat.detach().to(device)

        return decoded


