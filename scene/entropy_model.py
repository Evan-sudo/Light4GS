import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchac  # pip install torchac


# =============== 工具函数 ===============

def get_bits(likelihood: torch.Tensor) -> torch.Tensor:
    """-log2(p) 求总 bit 数."""
    return -torch.sum(torch.log2(likelihood))


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class Low_bound(torch.autograd.Function):
    """你原来用在 likelihood 上的 lower bound."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x < 1e-9] = 0
        except RuntimeError:
            grad1 = g.clone()
        pass_through_if = np.logical_or(
            x.cpu().detach().numpy() >= 1e-9,
            g.cpu().detach().numpy() < 0.0
        )
        t = torch.Tensor(pass_through_if + 0.0).to(grad1.device)
        return grad1 * t


# =============== Fully-factorized EntropyBottleneck ===============

class EntropyBottleneck(nn.Module):

    def __init__(self, channels, init_scale=8, filters=(3, 3, 3)):
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._channels = channels

        # 参数初始化（完全照你原来的）
        filters_all = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        self._matrices = nn.ParameterList()
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()

        for i in range(len(self._filters) + 1):
            # matrix: [C, out, in]
            matrix = Parameter(torch.FloatTensor(channels, filters_all[i + 1], filters_all[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters_all[i + 1]))
            matrix.data.fill_(init_matrix)
            self._matrices.append(matrix)

            # bias: [C, out, 1]
            bias = Parameter(torch.FloatTensor(channels, filters_all[i + 1], 1))
            bias.data.uniform_(-0.5, 0.5)
            self._biases.append(bias)

            # factor: [C, out, 1]
            factor = Parameter(torch.FloatTensor(channels, filters_all[i + 1], 1))
            factor.data.fill_(0.0)
            self._factors.append(factor)

    # ---------- 原始 EB 的核心函数 ----------

    def _logits_cumulative(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [C, 1, N]
        输出:   [C, 1, N]
        """
        logits = inputs
        for i in range(len(self._filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i]).to(logits.device)
            logits = torch.matmul(matrix, logits)
            logits = logits + self._biases[i].to(logits.device)
            factor = torch.tanh(self._factors[i]).to(logits.device)
            logits = logits + factor * torch.tanh(logits)
        return logits

    def _quantize(self, inputs, Q, mode):
        if mode == "symbols":
            return RoundNoGradient.apply(inputs / Q) * Q
        elif mode == "noise":
            noise = (torch.rand_like(inputs) - 0.5) * Q
            return inputs + noise
        else:
            raise ValueError("Unknown quantize mode")

    def _likelihood(self, inputs, Q):
        """
        fully-factorized EB 的 likelihood 计算：
        inputs: [1, N, H, W]，展平为 [C, N] 再走 CDF 网络。
        """
        B = inputs.shape[1]                       # N
        C = inputs.shape[2] * inputs.shape[3]     # H*W
        assert C == self._channels
        flat = inputs.view(B, C).permute(1, 0).contiguous()  # [C, N]
        flat = flat.unsqueeze(1)                                # [C, 1, N]
        lower = self._logits_cumulative(flat - 0.5 * Q)
        upper = self._logits_cumulative(flat + 0.5 * Q)
        sign = -torch.sign(lower + upper).detach()
        lik = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        return lik.view(-1)

    def forward(self, inputs, quantization_step, quantize_mode="noise"):
        if quantize_mode is not None:
            outputs = self._quantize(inputs, quantization_step, quantize_mode)
        else:
            outputs = inputs
        lik = self._likelihood(outputs, quantization_step)
        lik = Low_bound.apply(lik)
        bits_per_symbol = get_bits(lik) / lik.shape[0]
        return outputs, bits_per_symbol
    
    def save_entropy(self, path):
        """
        Saves the state of the EntropyBottleneck to a specified path.
        
        Args:
        - path (str): The directory where the entropy parameters will be saved.
        """
        torch.save(self.state_dict(), os.path.join(path, "entropy_bottleneck.pth"))

    def get_parameters(self):
        """
        This method returns all parameters related to the 'context' or 'checkerboard' layers.
        """
        parameter_list = []
        for name, param in self.named_parameters():
            parameter_list.append(param)
        return parameter_list

    # ---------- 严格按 encoder_factorized 构造 CDF ----------

    def _build_factorized_cdf_range(self, k_min: int, k_max: int, Q, device):
        """
            x_int_round = round(x / Q) ∈ [k_min, k_max]

            samples = range(k_min, k_max+1)  # [R]
            samples = samples.unsqueeze(0).unsqueeze(0).repeat(C,1,1)  # [C,1,R]

            lower = _logits_cumulative((samples-0.5)*Q)
            upper = _logits_cumulative((samples+0.5)*Q)
            sign  = -sign(lower+upper).detach()
            pmf   = |sigmoid(sign*upper) - sigmoid(sign*lower)|   # [C,1,R]
            cdf   = cumsum(pmf, dim=-1)                           # [C,1,R]
            lower = cat([0, cdf], dim=-1)                         # [C,1,R+1]
            lower = clamp(lower, 0, 1)

        最后返回:
          cdf_ch: [C, Lp]，每行一个 channel 的 CDF，Lp = R+1
        """
        C = self._channels
        Q_val = float(Q)

        # samples: [R]
        samples = torch.arange(k_min, k_max + 1,
                               dtype=torch.float32,
                               device=device)              # [R]
        R = samples.numel()

        # [C,1,R]
        samples_3d = samples.view(1, 1, R).repeat(C, 1, 1)

        lower_logits = self._logits_cumulative((samples_3d - 0.5) * Q_val)  # [C,1,R]
        upper_logits = self._logits_cumulative((samples_3d + 0.5) * Q_val)  # [C,1,R]

        sign = -torch.sign(lower_logits + upper_logits).detach()
        pmf = torch.abs(
            torch.sigmoid(sign * upper_logits) - torch.sigmoid(sign * lower_logits)
        )  # [C,1,R]

        cdf = torch.cumsum(pmf, dim=-1)  # [C,1,R]

        lower = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)  # [C,1,R+1]
        lower = torch.clamp(lower, 0.0, 1.0)

        cdf_ch = lower.squeeze(1)  # [C, R+1]
        return cdf_ch

    # ---------- compress / decompress：用 torchac，逻辑 ≈ encoder_factorized ----------

    def compress(self, inputs, Q, file_path):
        """
        严格对齐 encoder_factorized 的做法，只是把 arithmetic_encode 换成 torchac.encode_float_cdf。

        inputs:   [1, N, H, W]，其中 H*W == self._channels
        Q:        量化步长 (float / tensor)
        file_path: 输出 .b 文件路径

        返回:
          file_bits: 实际文件 bits（含 header）
          x_q_4d   : 量化后的张量 [1,N,H,W]
        """
        assert inputs.dim() == 4 and inputs.shape[0] == 1
        device_x = inputs.device

        B, N_pos, H, W = inputs.shape   # B=1
        C = H * W
        assert C == self._channels, f"Flattened C={C}, but EB channels={self._channels}"

        # 1) reshape 成 [N, C]，完全对应原始 encoder_factorized 的 x.shape=[N,C]
        x2d = inputs.view(N_pos, C)  # [N, C]

        Q_val = float(Q)

        # 2) 整数符号 k = round(x/Q)
        with torch.no_grad():
            x_int_round = torch.round(x2d / Q_val)  # [N, C]

        max_value = x_int_round.max()
        min_value = x_int_round.min()
        k_min = int(min_value.item())
        k_max = int(max_value.item())

        # 3) 构造每个 channel 的 CDF （range: [k_min, k_max]）
        device_model = next(self.parameters()).device
        cdf_ch = self._build_factorized_cdf_range(
            k_min, k_max, Q_val, device=device_model
        )   # [C, Lp]
        Lp = cdf_ch.shape[1]

        # 4) 按原代码：lower = lower.permute(1,0,2).repeat(N,1,1) -> [N,C,Lp] -> view(-1,Lp)
        cdf_all = (
            cdf_ch.unsqueeze(0)              # [1,C,Lp]
                  .permute(1, 0, 2)          # [C,1,Lp]
                  .contiguous()
        )  # 其实 [C,1,Lp]，不过跟原文一步一步来

        # 原文：lower = lower.permute(1,0,2).contiguous().repeat(x.shape[0],1,1)
        # 这里我们更直白：先 [C,1,Lp] -> [1,C,Lp] 再 repeat N 次
        cdf_all = cdf_ch.unsqueeze(0).repeat(N_pos, 1, 1)  # [N,C,Lp]
        cdf_all = cdf_all.view(-1, Lp)                     # [N*C, Lp]
        cdf_all = cdf_all.to("cpu")

        # 5) 符号 index: (x_int_round - k_min) -> int16，再展平 [N*C]
        x_int_round_idx = (x_int_round - k_min).to(torch.int16)
        x_int_round_idx = x_int_round_idx.view(-1).to("cpu")    # [N*C]

        # 6) 用 torchac 进行算术编码
        bitstream = torchac.encode_float_cdf(cdf_all, x_int_round_idx)
        if isinstance(bitstream, (bytes, bytearray)):
            bs_bytes = bitstream
        else:
            bs_bytes = bitstream.numpy().tobytes()

        # 7) 写 header + bitstream
        header = np.array(
            [B, N_pos, H, W, C, k_min, k_max, len(bs_bytes)],
            dtype=np.int32
        )
        with open(file_path, "wb") as f:
            f.write(header.tobytes())
            f.write(bs_bytes)

        file_bits = (header.nbytes + len(bs_bytes)) * 8

        # 返回量化后的 4D 张量
        x_q_4d = (x_int_round * Q_val).view(1, N_pos, H, W).to(device_x)
        return file_bits, x_q_4d

    def decompress(self, file_path, Q, device="cuda"):
        """
        用与 compress 同一套 EB + Q，从 .b 文件中解码出 [1,N,H,W]。
        """
        # 1) 读 header + bitstream
        with open(file_path, "rb") as f:
            header_bytes = f.read(8 * 4)
            header = np.frombuffer(header_bytes, dtype=np.int32)
            B, N_pos, H, W, C, k_min, k_max, bs_len = header.tolist()
            bs_bytes = f.read(bs_len)

        assert C == self._channels

        Q_val = float(Q)
        device_model = next(self.parameters()).device

        # 2) 构造与 compress 完全一样的 CDF
        cdf_ch = self._build_factorized_cdf_range(
            k_min, k_max, Q_val, device=device_model
        )  # [C, Lp]
        Lp = cdf_ch.shape[1]

        cdf_all = cdf_ch.unsqueeze(0).repeat(N_pos, 1, 1)  # [N,C,Lp]
        cdf_all = cdf_all.view(-1, Lp).to("cpu")           # [N*C,Lp]

        # 3) 解码得到符号 index
        sym = torchac.decode_float_cdf(cdf_all, bs_bytes)  # [N*C] int16
        sym = sym.to(torch.int32)

        # 4) 映回整数 k，再乘 Q 得到实数
        k = sym + int(k_min)                               # [N*C]
        x_hat_2d = (k.to(torch.float32) * Q_val).view(N_pos, C)  # [N,C]
        x_hat = x_hat_2d.view(B, N_pos, H, W).to(device)
        return x_hat


# # =============== 简单测试：训练rate vs 实际rate ===============

# def test_entropy_bottleneck():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     channels = 45
#     H, W = 15, 3
#     N_pos = 5000   # 可以调大一些
#     Q = 0.3      # 你可以改不同的量化步长再试

#     model = EntropyBottleneck(channels=channels).to(device)
#     model.eval()

#     # 随便造一组数据
#     x = torch.randn(1, N_pos, H, W, device=device) * 6

#     # ---------- (1) 训练时的理论 rate（用 _likelihood + Low_bound） ----------
#     with torch.no_grad():
#         _, bits_per_symbol_train = model(x, Q, quantize_mode="symbols")
#     print("==== Training-time theoretical bits ====")
#     print(f"bits per symbol (from _likelihood) = {float(bits_per_symbol_train):.6f}")

#     # ---------- (2) 实际编码写 .b 文件 ----------
#     file_path = "./test_eb_factorized_torchac.b"
#     if os.path.exists(file_path):
#         os.remove(file_path)

#     file_bits, x_q = model.compress(x, Q, file_path)
#     bits_per_symbol_file = file_bits / x.numel()
#     print("\n==== Actual bits from .b file ====")
#     print(f"total bits = {file_bits}  |  bits per symbol = {bits_per_symbol_file:.6f}")

#     # ---------- (3) 解码 & 重建误差 ----------
#     x_hat = model.decompress(file_path, Q, device=device)
#     mse_recon = torch.mean((x_hat - x_q) ** 2).item()
#     mse_quant = torch.mean((x_q - x) ** 2).item()

#     print("\n==== Reconstruction quality ====")
#     print(f"MSE( decode vs quantized x ) = {mse_recon:.6e}")
#     print(f"MSE( quantized x vs original x ) = {mse_quant:.6e}")

#     print("\n==== Summary ====")
#     print(f"Train bits / sym : {float(bits_per_symbol_train):.6f}")
#     print(f"File bits / sym  : {bits_per_symbol_file:.6f}")


# if __name__ == "__main__":
#     test_entropy_bottleneck()