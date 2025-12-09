from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
import torch
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import importlib.util
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
import seaborn as sns

def load_config_from_py_file(filepath):
    """
    Load the content of a Python configuration file and return it as a dictionary.
    """
    spec = importlib.util.spec_from_file_location("config", filepath)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def spatial_hash(x: torch.Tensor, y: torch.Tensor, plane_id: int, grid_shape: torch.Size) -> torch.Tensor:
    """
    Improved hash function for 2D coordinates (x, y) with a plane identifier.

    Args:
    - x: Tensor of x coordinates.
    - y: Tensor of y coordinates.
    - plane_id: Integer representing the plane identifier (e.g., 0 for x-y, 1 for x-z, etc.).
    - grid_shape: The shape of the hash table.
    
    Returns:
    - Hashed values for each (x, y) coordinate, modded by the grid size.
    """
    PRIMES = (1, 2654435761, 805459861, 3674653429)  # Prime numbers for hashing

    # Hash x, y coordinates along with the plane_id
    hashed = (x * PRIMES[1]) ^ (y * PRIMES[2]) ^ (plane_id * PRIMES[3])

    # Modulo operation to keep the hashed values within the grid's dimensions
    return hashed % grid_shape  # return the hash value


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        hash_table: torch.Tensor):
    """
    Initialize grid parameters using the hash function instead of random initialization.
    
    Args:
    - grid_nd: Number of dimensions to combine (e.g., 2D, 3D).
    - in_dim: Input dimension (should match the length of `reso`).
    - out_dim: Output feature dimension.
    - reso: List of resolutions for each dimension.
    - hash_table: A pre-initialized hash table with shape [hash_table_size, out_dim].
    
    Returns:
    - A list of tensors initialized using values from the hash table.
    """
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    
    # 获取 4D 空间中所有可能的 2D 平面组合
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    
    grid_coefs = []  # 存储所有的网格参数
    
    # 遍历每一个坐标组合
    for ci, coo_comb in enumerate(coo_combs):
        # 生成对应组合坐标的分辨率
        combo_reso = [reso[cc] for cc in coo_comb[::-1]]
        
        # 创建 2D 坐标网格
        x = torch.arange(combo_reso[0], dtype=torch.long)
        y = torch.arange(combo_reso[1], dtype=torch.long)
        coords = torch.meshgrid(x, y, indexing='ij')
        
        # 将坐标展开为1D，便于传递给哈希函数
        x_flat, y_flat = coords[0].flatten(), coords[1].flatten()

        # 使用哈希函数计算哈希索引
        hash_indices = spatial_hash(x_flat, y_flat, plane_id=ci, grid_shape=hash_table.shape[0])

        # 根据哈希索引从哈希表中获取值
        grid_values = hash_table[hash_indices]

        # 重塑为网格形状，并添加 batch 维度
        grid_values = grid_values.view([1, out_dim] + combo_reso)
        
        # 将结果存储到 list 中
        grid_coefs.append(grid_values)

    return grid_coefs

def get_grids(config, multiscale_res_multipliers,hash_table):
    grids = []
    for res in multiscale_res_multipliers:
        config["resolution"] = [
            r * res for r in config["resolution"][:4]
        ] 
        gp = init_grid_param(
            grid_nd=config["grid_dimensions"],
            in_dim=config["input_coordinate_dim"],
            out_dim=config["output_coordinate_dim"],
            reso=config["resolution"],
            hash_table = hash_table,
        )
        grids.append(gp)
    return grids




path = "output/dnerf/bouncingballs/point_cloud/iteration_20000/deformation.pth"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
config_path = "/home/gaussian4d/workspace/4DGaussians/arguments/dnerf/bouncingballs.py"
config = load_config_from_py_file(config_path)
grid_config = config.ModelHiddenParams['kplanes_config']
multiscale_res_multipliers = [1,2]


data = torch.load(path)
hash_table = data['deformation_net.grid.hash_table']

grids = get_grids(grid_config, multiscale_res_multipliers, hash_table)
for grid in grids:
    for i in grid:
        print(i.shape)

results_dir = "results"
low_res_dir = os.path.join(results_dir, "low_res")
high_res_dir = os.path.join(results_dir, "high_res")
os.makedirs(low_res_dir, exist_ok=True)
os.makedirs(high_res_dir, exist_ok=True)

# 定义可视化函数
def visualize_tensor(tensor, save_path):
    # 假设 batch size 为 1，移除 batch 维度
    tensor = tensor.squeeze(0)

    # 确保有 32 个通道
    assert tensor.size(0) == 32, f"Tensor channel size is {tensor.size(0)}, expected 32."

    # 设置画布大小
    plt.figure(figsize=(15, 15))

    # 创建 8x4 的 GridSpec 网格（因为32个通道）
    gs = gridspec.GridSpec(8, 4)

    for i in range(tensor.size(0)):
        ax = plt.subplot(gs[i])
        
        # 显示图像，使用 twilight 颜色映射
        ax.imshow(tensor[i].cpu(), cmap='twilight')
        ax.axis('off')
    
    # 调整子图布局
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
    plt.savefig(save_path)
    plt.close()

# 遍历 grids 列表并可视化每个 tensor
for grid_id1, grid in enumerate(grids):
    for grid_id2, tensor in enumerate(grid):
        # 获取 tensor 的高和宽
        height, width = tensor.shape[-2], tensor.shape[-1]
        
        # 确定保存路径
        if height == width:
            # 保存到 low_res 文件夹中
            save_path = os.path.join(low_res_dir, f"grid_{grid_id1}_{grid_id2}.png")
        else:
            # 保存到 high_res 文件夹中
            save_path = os.path.join(high_res_dir, f"grid_{grid_id1}_{grid_id2}.png")
        
        # 可视化并保存
        visualize_tensor(tensor.cpu(), save_path)

# 打印每个 grid 的最大值和最小值
for grid_id1, grid in enumerate(grids):
    for grid_id2, tensor in enumerate(grid):
        max_value = tensor.max().item()
        min_value = tensor.min().item()
        print(f"Maximum value in grid_{grid_id1}_{grid_id2}: {max_value}")
        print(f"Minimum value in grid_{grid_id1}_{grid_id2}: {min_value}")

# 计算值的分布
all_values = torch.cat([tensor.flatten() for grid in grids for tensor in grid])

# 将 tensor 转换为 numpy 以便使用 seaborn 绘图
all_values_np = all_values.cpu().numpy()

# 绘制值的概率分布
plt.figure(figsize=(10, 6))
sns.histplot(all_values_np, bins=100, kde=True)
plt.title('Value Distribution in Grids')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('./results/dis.png')
plt.show()