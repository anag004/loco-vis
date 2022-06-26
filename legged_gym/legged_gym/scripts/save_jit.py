import torch
from rsl_rl.modules import ActorCritic
import sys
import os

dim = int(sys.argv[2])
cfg = {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0, 'use_actor_tanh': True}

ac = ActorCritic(dim, dim, 12, **cfg)

ac.load_state_dict(torch.load(os.path.join(sys.argv[1], "model_1500.pt"))["model_state_dict"])

traced_module = torch.jit.script(ac.actor.cpu())
traced_module.save("/home/anag/policy_raisim.pt")
