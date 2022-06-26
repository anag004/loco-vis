# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU


class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_total_obs,
                        num_critic_obs,
                        priv_obs_loc,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        use_actor_tanh=False,
                        use_info_encoder_tanh=False,
                        use_scandots_compression_tanh=False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRMA, self).__init__()

        self.num_actor_total_obs = num_actor_total_obs
        self.kwargs = kwargs

        self.priv_obs_start, self.priv_obs_end = priv_obs_loc
        self.num_actor_priv_obs = self.priv_obs_end - self.priv_obs_start
        
        activation = get_activation(activation)
        
        mlp_input_dim_a = num_actor_total_obs
        mlp_input_dim_c = num_critic_obs

        if kwargs["use_depth_backbone"]:
            raise Exception("use_depth_backbone no longer supported")

        if "scandots_compression" in kwargs:
            self.construct_scandots_compression_nn(
                kwargs["scandots_compression"],
                use_scandots_compression_tanh=use_scandots_compression_tanh
            )

            mlp_input_dim_a = mlp_input_dim_a - self.scandots_compression[0] + self.scandots_compression[-1]
            mlp_input_dim_c = mlp_input_dim_c

            self.priv_obs_start = self.priv_obs_start
            self.priv_obs_end = self.priv_obs_end
        else:
            self.scandots_compression_nn = None

        # Encoder
        encoder_dim = 8
        self.info_encoder =  nn.Sequential(*[
                                nn.Linear(self.num_actor_priv_obs, 256), activation,
                                nn.Linear(256, 128), activation,
                                nn.Linear(128, encoder_dim), 
                                nn.Tanh() if use_info_encoder_tanh else activation
                            ]) 
        
        # Policy
        self.actor_input_dim = mlp_input_dim_a - self.num_actor_priv_obs + encoder_dim
        actor_layers = []
        self.actor_input_dim = mlp_input_dim_a - self.num_actor_priv_obs + encoder_dim
        actor_layers.append(nn.Linear(mlp_input_dim_a - self.num_actor_priv_obs + encoder_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)

        if use_actor_tanh:
             actor_layers.append(nn.Tanh())

        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.info_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def construct_scandots_compression_nn(self, scandots_compression, use_scandots_compression_tanh=False):
        self.scandots_compression = scandots_compression
        self.scandots_compression_layers = []

        for i in range(len(scandots_compression) - 1):
            self.scandots_compression_layers.append(nn.Linear(scandots_compression[i], 
                                                              scandots_compression[i+1]))
            self.scandots_compression_layers.append(nn.ReLU())

        if use_scandots_compression_tanh:
            del self.scandots_compression_layers[-1]
            self.scandots_compression_layers.append(nn.Tanh())

        self.scandots_compression_nn = nn.Sequential(*self.scandots_compression_layers)
        print(self.scandots_compression_nn)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, priv_latent=None, **kwargs):        
        if self.scandots_compression_nn is None:
            if priv_latent is None:
                priv_latent = self.info_encoder(observations[:, self.priv_obs_start:self.priv_obs_end])
            
            actor_input = torch.cat([
                observations[:, :self.priv_obs_start], 
                priv_latent,
                observations[:, self.priv_obs_end:]
            ], 1)
        else:
            compressed_depth = self.scandots_compression_nn(observations[:, :self.scandots_compression[0]])

            if priv_latent is None:
                priv_latent = self.info_encoder(observations[:, self.priv_obs_start:self.priv_obs_end])
            
            actor_input = torch.cat([
                observations[:, self.scandots_compression[0]:self.priv_obs_start], 
                priv_latent,
                compressed_depth,
                observations[:, self.priv_obs_end:]
            ], 1)
        
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def compute_scandots_compression(self, obs):
        visual_obs = self.scandots_compression_nn(obs[:, :self.scandots_compression[0]])
        return visual_obs

    def act_depth_inference(self, observations, priv_latent=None, compressed_depth=None, **kwargs):
        if priv_latent is None:
            priv_latent = self.info_encoder(observations[:, self.priv_obs_start:self.priv_obs_end])
        
        actor_input = torch.cat([
            observations[:, self.scandots_compression[0]:self.priv_obs_start], 
            priv_latent,
            compressed_depth,
        ], 1)

        actions_mean = self.actor(actor_input)
        return actions_mean

    def act_inference(self, observations, priv_latent=None, scandots_latent=None, **kwargs):
        if self.scandots_compression_nn is None:
            if priv_latent is None:
                priv_latent = self.info_encoder(observations[:, self.priv_obs_start:self.priv_obs_end])
            
            actor_input = torch.cat([
                observations[:, :self.priv_obs_start], 
                priv_latent,
                observations[:, self.priv_obs_end:]
            ], 1)
        else:
            if scandots_latent is None:
                compressed_depth = self.scandots_compression_nn(observations[:, :self.scandots_compression[0]])
            else:
                compressed_depth = scandots_latent

            if priv_latent is None:
                priv_latent = self.info_encoder(observations[:, self.priv_obs_start:self.priv_obs_end])
            
            actor_input = torch.cat([
                observations[:, self.scandots_compression[0]:self.priv_obs_start], 
                priv_latent,
                compressed_depth,
                observations[:, self.priv_obs_end:]
            ], 1)

        actions_mean = self.actor(actor_input)
        return actions_mean

    def act_dagger(self, observations, priv_latent, return_scandots_latent=False):
        if self.scandots_compression_nn is not None:
            compressed_depth = self.scandots_compression_nn(observations[:, :self.scandots_compression[0]])
            actor_input = torch.cat([
                observations[:, self.scandots_compression[0]:self.priv_obs_start],  # assuming lin/ang vel is known
                priv_latent,
                compressed_depth,
                observations[:, self.priv_obs_end:]
            ], 1)
        else:
            actor_input = torch.cat([
                observations[:, :self.priv_obs_start],  # assuming lin/ang vel is known
                priv_latent,
                observations[:, self.priv_obs_end:]
            ], 1)
            

        actions_mean = self.actor(actor_input)

        if not return_scandots_latent:
            return actions_mean
        else:
            return actions_mean, compressed_depth


    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
        
        
class DepthBackbone(nn.Module):
    def __init__(self, backbone_type):
        super().__init__()

        self.backbone_type = backbone_type

        if self.backbone_type == "deepgait":
            # From DeepGait paper
            self.depth_backbone_deepgait = nn.Sequential(
                # [32, 32, 1]
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16, 
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
                # [32, 32, 16]
                nn.MaxPool2d(
                    kernel_size=2, 
                    stride=2
                ),
                nn.ReLU(),
                # [16, 16, 16]
                nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=4,
                    padding=2,
                    stride=1
                ),
                # [17, 17, 32]
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                ),
                nn.ReLU(),
                # [8, 8, 32]
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    stride=1
                ),
                nn.ReLU()
                # [8, 8, 32]
            )

            self.fc = nn.Sequential(
                nn.Linear(8 * 8 * 32, 128),
                nn.ReLU()
            )
        elif self.backbone_type == "fc":
            self.fc = nn.Linear(32 * 32, 128)
        elif self.backbone_type == "deepgait_coordconv":
            # From DeepGait paper with coordconv
            self.depth_backbone_deepgait = nn.Sequential(
                # [32, 32, 3]
                nn.Conv2d(
                    in_channels=3,
                    out_channels=16, 
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
                # [32, 32, 16]
                nn.MaxPool2d(
                    kernel_size=2, 
                    stride=2
                ),
                nn.ReLU(),
                # [16, 16, 16]
                nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=4,
                    padding=2,
                    stride=1
                ),
                # [17, 17, 32]
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                ),
                nn.ReLU(),
                # [8, 8, 32]
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    stride=1
                ),
                nn.ReLU()
                # [8, 8, 32]
            )

            self.fc = nn.Sequential(
                nn.Linear(8 * 8 * 32, 128),
                nn.ReLU()
            )
        elif backbone_type == "jumping_from_pixels":
            self.depth_backbone_jumping_from_pixels = nn.Sequential(
                # [1, 120, 160]
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
                # [16, 116, 156],
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                # [16, 58, 78]
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
                # [32, 54, 74]
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                # [32, 27, 37]
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU()
                # [64, 35, 35]
            )

            self.fc = nn.Linear(64 * 35 * 25, 128)
        elif backbone_type == "transformer":
            self.image_compression = nn.Sequential(
                nn.Linear(32 * 32, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 67)
            )

            self.pos_encoding = PositionalEmbeddings()

            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=100, nhead=4, batch_first=True, dim_feedforward=512),
                num_layers=3
            )
        elif backbone_type == "mlp":
            self.image_compression = nn.Sequential(
                nn.Linear(87 * 58 * 15 + 32 * 15, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 100)
            )
        elif backbone_type == "mlp_noprop":
            self.image_compression = nn.Sequential(
                nn.Linear(87 * 58 * 15, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 100)
            )
        elif backbone_type == "async_transformer":
            self.image_compression = nn.Sequential(
                # [1, 58, 87]
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                # [32, 54, 83]
                nn.MaxPool2d(kernel_size=2, stride=2),
                # [32, 27, 41]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                # [32, 25, 39]
                nn.Linear(64 * 25 * 39, 128),
            )

            self.prop_compression = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU()
            )

            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=148, nhead=4, batch_first=True, dim_feedforward=512),
                num_layers=3
            )
        elif backbone_type == "mlp_hierarchical":
            self.image_compression = nn.Sequential(
                # [1, 58, 87]
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                # [32, 54, 83]
                nn.MaxPool2d(kernel_size=2, stride=2),
                # [32, 27, 41]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                # [32, 25, 39]
                nn.Linear(64 * 25 * 39, 128),
            )

            self.mlp = nn.Sequential(
                nn.Linear(128 * 15 + 32 * 15, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 32)
            )
        elif backbone_type == "mlp_hierarchical_dualcam":
            self.image_compression = nn.Sequential(
                # [1, 58, 174]
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                # [32, 54, 170]
                nn.MaxPool2d(kernel_size=2, stride=2),
                # [32, 27, 85]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                # [32, 25, 83]
                nn.Linear(64 * 25 * 83, 128),
            )

            self.mlp = nn.Sequential(
                nn.Linear(128 * 5 + 32 * 5, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 100)
            )
        elif backbone_type == "mlp_hierarchical_nframes10":
            self.image_compression = nn.Sequential(
                # [1, 58, 87]
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                # [32, 54, 83]
                nn.MaxPool2d(kernel_size=2, stride=2),
                # [32, 27, 41]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                # [32, 25, 39]
                nn.Linear(64 * 25 * 39, 128),
            )

            self.mlp = nn.Sequential(
                nn.Linear(128 * 10 + 32 * 10, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 100)
            )
        elif backbone_type == "mlp_hierarchical_noprop":
            self.image_compression = nn.Sequential(
                # [1, 58, 87]
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
                # [32, 54, 83]
                nn.MaxPool2d(kernel_size=2, stride=2),
                # [32, 27, 41]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                # [32, 25, 39]
                nn.Linear(64 * 25 * 39, 128),
            )

            self.mlp = nn.Sequential(
                nn.Linear(128 * 15, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 100)
            )
        elif backbone_type == "mlp_scandots":
            self.scandots_compression = nn.Sequential(
                nn.Linear(28 * 12, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )

            self.mlp = nn.Sequential(
                nn.Linear(128 * 15 + 32 * 15, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 100)
            )

    def forward(self, x):
        if self.backbone_type == "deepgait":
            # x has shape (batch_size, 10, 32, 32)
            batch_size = x.shape[0]
            x = x.reshape((batch_size, 1, 32, 32))
            x = self.depth_backbone_deepgait(x)
            x = x.reshape((batch_size, 8 * 8 * 32))
            x = self.fc(x)
            x = x.reshape((batch_size, 128))
        elif self.backbone_type == "fc":
            batch_size = x.shape[0]
            x = x.reshape((batch_size, 32 * 32))
            x = self.fc(x)
            x = x.reshape((batch_size, 128))
        elif self.backbone_type == "deepgait_coordconv":
            # x has shape (batch_size, 10, 32, 32)
            batch_size = x.shape[0]
            x = x.reshape((batch_size, 1, 32, 32))
            
            # Concatenate coordinate information
            i_coord = torch.arange(32).repeat(32, 1).to(x.device)
            i_coord_repeated = i_coord.repeat(batch_size, 1, 1).reshape((batch_size, 1, 32, 32))
            j_coord = i_coord.T
            j_coord_repeated = j_coord.repeat(batch_size, 1, 1).reshape((batch_size, 1, 32, 32))
            x = torch.cat((x, i_coord_repeated, j_coord_repeated), dim=1)

            x = self.depth_backbone_deepgait(x)
            x = x.reshape((batch_size, 8 * 8 * 32))
            x = self.fc(x)
            x = x.reshape((batch_size, 128))
        elif self.backbone_type == "jumping_from_pixels":
            # x has shape (batch_size, N, 120, 160)
            batch_size = x.shape[0]

            x = x.reshape((batch_size, -1, 120, 160))
            num_frames = x.shape[1]
            x = x.reshape((batch_size * num_frames, 1, 120, 160))
            x = self.depth_backbone_jumping_from_pixels(x)
            x = x.reshape((batch_size * num_frames, 64 * 35 * 25))
            x = self.fc(x)
            x = x.reshape((batch_size, 128 * num_frames))
        elif self.backbone_type == "transformer":   
            batch_size = x.shape[0]
            prop_history = x[:, :33 * 15].reshape((batch_size, 15, 33))
            depth_frames = x[:, 33 * 15:].reshape((batch_size, 15, 1024))
            depth_frames = self.image_compression(depth_frames) # [B, 15, 67]

            transformer_input = torch.cat((prop_history, depth_frames), dim=-1) # [B, 15, 100]
            transformer_input = transformer_input + self.pos_encoding(transformer_input)

            encoded_output = self.transformer(transformer_input) # [B, 15, 100]

            x = encoded_output[:, 0, :]
        elif self.backbone_type == "mlp":
            x = self.image_compression(x)
        elif self.backbone_type == "mlp_noprop":
            x = self.image_compression(x)
        elif self.backbone_type == "async_transformer":
            batch_size = x.shape[0]
            prop_history = x[:, :52 * 15].reshape((batch_size * 15, 32 + 20))
            depth_frames = x[:, 52 * 15:].reshape((batch_size * 15, 58 * 87 + 20))
            depth_frames, depth_frames_time_embedding = depth_frames[:, :58 * 87], depth_frames[:, 58 * 87:]
            prop_history, prop_history_time_embedding = prop_history[:, :32], prop_history[:, 32:]
            
            depth_frames = depth_frames.reshape((-1, 1,  58, 87))
            depth_frames = self.image_compression(depth_frames)
            depth_frames = torch.cat((depth_frames, depth_frames_time_embedding), dim=-1)
            depth_frames = depth_frames.reshape((batch_size, 15, -1))

            prop_history = self.prop_compression(prop_history)
            prop_history = torch.cat((prop_history, prop_history_time_embedding), dim=-1)
            prop_history = prop_history.reshape((batch_size, 15, -1))

            transformer_input = torch.cat((prop_history, depth_frames), dim=1) # [B, 30, 148]

            encoded_output = self.transformer(transformer_input) # [B, 15, 100]

            x = encoded_output[:, 0, :]
        elif self.backbone_type == "mlp_hierarchical":
            batch_size = x.shape[0]
            prop_history = x[:, :32 * 15].reshape((batch_size, 15 * 32))
            depth_frames = x[:, 32 * 15:].reshape((batch_size * 15, 1, 58, 87))
            depth_frames = self.image_compression(depth_frames).reshape((batch_size, 15 * 128))

            x = self.mlp(torch.cat((prop_history, depth_frames), dim=-1))
        elif self.backbone_type == "mlp_hierarchical_dualcam":
            batch_size = x.shape[0]
            prop_history = x[:, :32 * 5].reshape((batch_size, 5 * 32))
            depth_frames = x[:, 32 * 5:].reshape((batch_size * 5, 1, 58, 87 * 2))
            depth_frames = self.image_compression(depth_frames).reshape((batch_size, 5 * 128))

            x = self.mlp(torch.cat((prop_history, depth_frames), dim=-1))
        elif self.backbone_type == "mlp_hierarchical_nframes10":
            batch_size = x.shape[0]
            prop_history = x[:, :32 * 10].reshape((batch_size, 10 * 32))
            depth_frames = x[:, 32 * 10:].reshape((batch_size * 10, 1, 58, 87))
            depth_frames = self.image_compression(depth_frames).reshape((batch_size, 10 * 128))

            x = self.mlp(torch.cat((prop_history, depth_frames), dim=-1))
        elif self.backbone_type == "mlp_hierarchical_noprop":
            batch_size = x.shape[0]
            depth_frames = x.reshape((batch_size * 15, 1, 58, 87))
            depth_frames = self.image_compression(depth_frames).reshape((batch_size, 15 * 128))

            x = self.mlp(depth_frames)
        elif self.backbone_type == "mlp_scandots":
            batch_size = x.shape[0]
            prop_history = x[:, :32 * 15].reshape((batch_size, 15 * 32))
            depth_frames = x[:, 32 * 15:].reshape((batch_size * 15, 28 * 12))
            depth_frames = self.scandots_compression(depth_frames).reshape((batch_size, 15 * 128))

            x = self.mlp(torch.cat((prop_history, depth_frames), dim=-1))

        return x

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='tanh',
                        init_noise_std=1.0,
                        use_actor_tanh=True,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.input_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.kwargs = kwargs

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        if kwargs["use_depth_backbone"]:
            raise Exception("use_depth_backbone not supported in ActorCritic")

        if "scandots_compression" in kwargs:
            raise Exception("scandots_compression not supported in ActorCritic")
            
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)

        if use_actor_tanh:
            actor_layers.append(nn.Tanh())

        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

        return mean

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_grad(self, observations, **kwargs):
        return self.actor(observations)
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, **kwargs):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
class PositionalEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        
    def get_embeddings(self, fn, shape, embedding_dim, device):
        result = torch.zeros(shape)
        
        freqs = 2 * torch.arange(shape[-1]).to(device) / embedding_dim
        freqs = 10000 ** freqs
        freqs = torch.arange(shape[1]).to(device)[:, None] / freqs[None, :]
        
        result = freqs.repeat(shape[0], 1, 1)
        
        return fn(result)
        
    def forward(self, embeddings):
        """
            embeddings -- (batch_size, sentence_length, embedding_dim)
        """
        
        batch_size, sentence_length, embedding_dim = embeddings.shape        
        result = torch.zeros_like(embeddings)
            
        result[:, :, ::2] = self.get_embeddings(torch.sin, (batch_size, sentence_length, result[:, :, ::2].shape[-1]), embedding_dim, device=embeddings.device)
        result[:, :, 1::2] = self.get_embeddings(torch.cos, (batch_size, sentence_length, result[:, :, 1::2].shape[-1]), embedding_dim, device=embeddings.device)
        
        return result
