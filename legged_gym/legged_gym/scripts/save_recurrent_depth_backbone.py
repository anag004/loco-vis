import os
import torch
import sys
sys.path.append('../envs/base/')
from depth_backbone import *
import legged_robot_config as cfg
from save_dagger_recurrent import get_load_path

env_cfg = cfg.LeggedRobotCfg
depth_backbone = RecurrentDepthBackbone(env_cfg)
print(depth_backbone)
real_world_obs_dim = env_cfg.env.num_observations - env_cfg.env.n_scan - env_cfg.env.n_priv - 3

load_path = get_load_path(sys.argv[1])
print("Loading from ", load_path)

depth_backbone.load_state_dict(torch.load(load_path, map_location="cuda:0"))
depth_backbone.eval()
traced_depth_backbone_rnn = torch.jit.trace(depth_backbone.rnn, (torch.zeros(1, 1, 32 + real_world_obs_dim), torch.zeros(1, 1, 512)))

save_path = os.path.join(sys.argv[1], "traced_depth_rnn.pt")
traced_depth_backbone_rnn.save(save_path)
print("Saved traced module at ", save_path)

if hasattr(depth_backbone, "output"):
    save_path = os.path.join(sys.argv[1], "traced_depth_output.pt")
    traced_depth_output = torch.jit.trace(depth_backbone.output, torch.zeros(1, 512), env_cfg.depth.depth_latent_size)
    traced_depth_output.save(save_path)
    print("Saved traced module at ", save_path)

