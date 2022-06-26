import os
import torch
import sys
sys.path.append('../envs/base/')
from depth_backbone import *
from save_dagger_recurrent import get_load_path

depth_backbone = DepthOnlyFCBackbone58x87(-1, 32, -1, output_activation="tanh")
# depth_backbone = DepthOnlyViTBackbone58x87(-1, 32, -1, output_activation="tanh")

load_path = get_load_path(sys.argv[1])
print("Loading from ", load_path)

depth_backbone.load_state_dict(torch.load(load_path, map_location="cuda:0"))
depth_backbone.eval()
traced_depth_backbone = torch.jit.trace(depth_backbone, (torch.tensor([]), torch.zeros(1, 1, 58, 87), torch.tensor([])))

save_path = os.path.join(sys.argv[1], "traced_depth.pt")
traced_depth_backbone.save(save_path)
print("Saved traced module at ", save_path)

