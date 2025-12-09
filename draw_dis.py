import torch
import matplotlib.pyplot as plt

# 加载模型路径
#path = "output/dynerf/coffee_martini0/point_cloud/iteration_14000/deformation2.pth"
path = "output/dnerf/jumpingjacks/point_cloud/iteration_20000/deformation.pth"

# 加载数据
data = torch.load(path)
print(data.keys())

grid_value = {}
for grid_id1 in range(2):
    for grid_id2 in range(6):
        key = f'deformation_net.grid.grids.{grid_id1}.{grid_id2}'
        grid_value[f"grid{grid_id1}{grid_id2}"] = data[key]
        print(f"grid{grid_id1}{grid_id2}", data[key].shape)

# Initialize lists to store the tensor values for each group
group1_values = []
group2_values = []

# Iterate over the grid_value dictionary and separate values based on the key
for key, tensor in grid_value.items():
    grid_id2 = key.split('grid')[1][1]  # Extract the second digit (grid_id2)
    if grid_id2 in ['0', '1', '3']:  # Group 1: 0.x, 1.x where x is 0, 1, 3
        group1_values.extend(tensor.cpu().numpy().flatten())
    elif grid_id2 in ['2', '4', '5']:  # Group 2: 0.x, 1.x where x is 2, 4, 5
        group2_values.extend(tensor.cpu().numpy().flatten())

# Plot the histograms
plt.figure(figsize=(10, 6))

plt.hist(group1_values, bins=50, alpha=0.5, label='Spatial', color='blue')
plt.hist(group2_values, bins=50, alpha=0.5, label='Temporal', color='red')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Tensor Values for jumpingjacks')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./results/dis.png')
