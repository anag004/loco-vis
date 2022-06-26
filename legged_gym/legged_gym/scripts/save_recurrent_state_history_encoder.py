import os
import torch
import torch.nn as nn
import sys
sys.path.append('../envs/base/')
from depth_backbone import *
import legged_robot_config as cfg
from save_dagger_recurrent import get_load_path

HIDDEN_SIZE = 64

class RecurrentStateHistoryEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden_states_list = []
        self.hidden_states = None
        self.is_recurrent = True

    def forward(self, obs):
        _, self.hidden_states = self.rnn(obs[:, None, :], self.hidden_states)
        self.hidden_states_list.append(self.hidden_states)

        result = self.hidden_states.squeeze(0)
        result = self.output_layer(result)

        return result

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()
        self.hidden_states_list = [self.hidden_states]

env_cfg = cfg.LeggedRobotCfg
real_world_obs_dim = env_cfg.env.num_observations - env_cfg.env.n_scan - env_cfg.env.n_priv - 3
state_history_encoder = RecurrentStateHistoryEncoder(real_world_obs_dim, 8, HIDDEN_SIZE)

load_path = get_load_path(sys.argv[1])
print("Loading from ", load_path)

state_history_encoder.load_state_dict(torch.load(load_path, map_location="cuda:0"))
state_history_encoder.eval()
traced_state_history_encoder_rnn = torch.jit.trace(state_history_encoder.rnn, (torch.zeros(1, 1, real_world_obs_dim), torch.zeros(1, 1, HIDDEN_SIZE)))
traced_state_history_encoder_output = torch.jit.trace(state_history_encoder.output_layer, torch.zeros(1, HIDDEN_SIZE))

print(state_history_encoder.rnn)

save_path = os.path.join(sys.argv[1], "traced_state_history_encoder_rnn.pt")
traced_state_history_encoder_rnn.save(save_path)
print("Saved traced module at ", save_path)

print(state_history_encoder.output_layer)

save_path = os.path.join(sys.argv[1], "traced_state_history_encoder_output.pt")
traced_state_history_encoder_output.save(save_path)
print("Saved traced module at ", save_path)



