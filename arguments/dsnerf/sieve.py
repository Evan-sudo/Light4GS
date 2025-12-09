_base_="default.py"
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [32, 32, 32, 150]
    },
    render_process=True,
    no_dshs=False,
    defor_depth = 3,
)
OptimizationParams = dict(
)