_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
    #  'output_coordinate_dim': 16,  # modified by evan
     'resolution': [64, 64, 64, 40]
    }
)