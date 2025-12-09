import os, sys
if __name__ == "__main__":
    filedir = os.path.dirname(os.path.abspath(__file__))
    rootdir = os.path.dirname(filedir)
    sys.path.append(rootdir)
    
from scene.checkerboard import CheckerboardAutogressive
import torch

if __name__ == "__main__":
    # Create an instance of the CheckerboardAutogressive class
    checkerboard = CheckerboardAutogressive()
    # Print the parameters of the CheckerboardAutogressive class
    # print(checkerboard.get_parameters())
    y = torch.randn(1, 32, 24, 64)
    scale_max = True
    scale_context = None
    ret = checkerboard.forward(y, scale_max, scale_context)
    # print(ret)
    