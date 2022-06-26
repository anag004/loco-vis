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

from copy import deepcopy
from telnetlib import SUPPRESS_LOCAL_ECHO
from turtle import forward, hideturtle
import numpy as np
import warnings
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, ActorCriticRMA, get_activation, DepthBackbone
from rsl_rl.utils import unpad_trajectories

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x):
        return x
class ActorCriticRMARecurrent(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_total_obs,
                        num_critic_obs,
                        priv_obs_loc,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        use_scandots_compression_tanh=False,
                        rnn_hidden_state_noise=None,
                        **kwargs):
        
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)
        
        super_kwargs = deepcopy(kwargs)
        
        if "scandots_compression" in super_kwargs:
            del super_kwargs["scandots_compression"]

        if "use_depth_backbone" in super_kwargs and super_kwargs["use_depth_backbone"]:
            raise Exception("depth_backbone no longer supported inside ActorCriticRMARecurrent")

        self.rnn_hidden_size = rnn_hidden_size
        self.construct_rnn_mask(kwargs)

        super().__init__(num_actor_obs=self.output_rnn_dim + len(self.rnn_mask_indices),
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         **super_kwargs)

        self.construct_rnn_compression(kwargs)

        self.num_actor_total_obs = num_actor_total_obs
        self.kwargs = kwargs

        self.priv_obs_start, self.priv_obs_end = priv_obs_loc
        self.num_actor_priv_obs = self.priv_obs_end - self.priv_obs_start
        
        activation = get_activation(activation)
        
        self.mlp_input_dim_a = num_actor_total_obs
        self.mlp_input_dim_c = num_critic_obs

        if "scandots_compression" in kwargs:
            self.construct_scandots_compression_nn(
                kwargs["scandots_compression"],
                use_scandots_compression_tanh=use_scandots_compression_tanh
            )
        else:
            self.scandots_compression_nn = None

        # Encoder
        if "disable_info_encoder" in kwargs and kwargs["disable_info_encoder"]:
            warnings.warn("Disabling info_encoder", category=UserWarning)
            encoder_dim = 17
            self.info_encoder = nn.Identity()
        else:
            encoder_dim = 8 if not "info_encoder_dim" in kwargs else kwargs["info_encoder_dim"]
            self.info_encoder =  nn.Sequential(*[
                                    nn.Linear(self.num_actor_priv_obs, 256), activation,
                                    nn.Linear(256, 128), activation,
                                    nn.Linear(128, encoder_dim), activation,
                                ]) 

        self.actor_input_dim = self.mlp_input_dim_a - self.num_actor_priv_obs + encoder_dim + len(self.rnn_mask_indices)
        self.memory_a = Memory(self.mlp_input_dim_a - self.num_actor_priv_obs + encoder_dim - len(self.rnn_mask_indices), type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size, noise_scales=rnn_hidden_state_noise)
        self.memory_c = Memory(self.mlp_input_dim_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size, noise_scales=rnn_hidden_state_noise)

        print(f"Encoder MLP: {self.info_encoder}")
        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")
        print(f"RNN compression: {self.rnn_compression}")

    def construct_rnn_mask(self, kwargs):
        if "rnn_mask" in kwargs and len(kwargs["rnn_mask"]) != 0:
            self.rnn_mask_indices = []
            self.rnn_unmasked_indices = []

            for i, x in enumerate(kwargs["rnn_mask"]):
                if x == 1:
                    self.rnn_mask_indices.append(i)
                
                if x == 0:
                    self.rnn_unmasked_indices.append(i)
        else:
            self.rnn_mask_indices = []
            self.rnn_unmasked_indices = None

        if "rnn_compression" in kwargs:
            if self.rnn_unmasked_indices is None:
                warnings.warn("rnn must have masked indices to use rnn_compression layer")
                self.output_rnn_dim = self.rnn_hidden_size
            else:
                self.output_rnn_dim = kwargs["rnn_compression"]
        else:
            self.output_rnn_dim = self.rnn_hidden_size

    def construct_rnn_compression(self, kwargs):
        if "rnn_compression" in kwargs:
            self.rnn_compression = nn.Sequential(
                nn.Linear(self.rnn_hidden_size, kwargs["rnn_compression"]),
                nn.ReLU()
            )
        else:
            self.rnn_compression = None

    def construct_scandots_compression_nn(self, scandots_compression, use_scandots_compression_tanh=False):
        if scandots_compression is None:
            self.scandots_compression_nn = None
            return

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

        self.mlp_input_dim_a = self.mlp_input_dim_a - self.scandots_compression[0] + self.scandots_compression[-1]
        self.mlp_input_dim_c = self.mlp_input_dim_c

        self.priv_obs_start = self.priv_obs_start
        self.priv_obs_end = self.priv_obs_end

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def reset_with_grad(self, dones=None):
        self.memory_a.reset_with_grad(dones)
        self.memory_c.reset_with_grad(dones)

    def act(self, observations, masks=None, hidden_states=None, priv_latent=None):
        obs_shape = observations.shape
        observations = observations.reshape((-1, obs_shape[-1]))

        if self.scandots_compression_nn is None:
            if priv_latent is None:
                priv_latent = self.info_encoder(observations[:, self.priv_obs_start:self.priv_obs_end])
            
            actor_input = torch.cat([
                observations[:, :self.priv_obs_start], 
                priv_latent,
                observations[:, self.priv_obs_end:]
            ], 1)  
        else:
            scandots_latent = self.scandots_compression_nn(observations[:, :self.scandots_compression[0]])

            if priv_latent is None:
                priv_latent = self.info_encoder(observations[:, self.priv_obs_start:self.priv_obs_end])
            
            actor_input = torch.cat([
                scandots_latent,
                observations[:, self.scandots_compression[0]:self.priv_obs_start], 
                priv_latent,
                observations[:, self.priv_obs_end:]
            ], 1)

        if self.rnn_unmasked_indices is not  None:
            actor_input_masked = actor_input[:, self.rnn_unmasked_indices]
            actor_input_masked = actor_input_masked.reshape((*obs_shape[:-1], -1))
            missing_obs = actor_input[:, self.rnn_mask_indices]
            missing_obs = missing_obs.reshape((*obs_shape[:-1], -1))
            input_a = self.memory_a(actor_input_masked, masks, hidden_states)

            if self.rnn_compression is not None:
                input_a = self.rnn_compression(input_a)

            if masks is not None:
                missing_obs = unpad_trajectories(missing_obs, masks)
            input_a = torch.cat((
                input_a.squeeze(0), 
                missing_obs
            ), dim=-1)
        else:
            actor_input = actor_input.reshape((*obs_shape[:-1], -1))
            input_a = self.memory_a(actor_input, masks, hidden_states).squeeze(0)

        return super().act(input_a)

    def act_inference(self, observations, priv_latent=None, scandots_latent=None, override_rnn_output=None, **kwargs):
        obs_shape = observations.shape
        observations = observations.reshape((-1, obs_shape[-1]))

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
                scandots_latent = self.scandots_compression_nn(observations[:, :self.scandots_compression[0]])

            if priv_latent is None:
                priv_latent = self.info_encoder(observations[:, self.priv_obs_start:self.priv_obs_end])
            
            actor_input = torch.cat([
                scandots_latent,
                observations[:, self.scandots_compression[0]:self.priv_obs_start], 
                priv_latent,
                observations[:, self.priv_obs_end:]
            ], 1)

        if self.rnn_unmasked_indices is not  None:
            actor_input_masked = actor_input[:, self.rnn_unmasked_indices]
            actor_input_masked = actor_input_masked.reshape((*obs_shape[:-1], -1))
            missing_obs = actor_input[:, self.rnn_mask_indices]
            missing_obs = missing_obs.reshape((*obs_shape[:-1], -1))

            if override_rnn_output is not None:
                input_a = override_rnn_output
            else:
                input_a = self.memory_a(actor_input_masked, **kwargs)

                if self.rnn_compression is not None:
                    input_a = self.rnn_compression(input_a)

            input_a = torch.cat((
                input_a.squeeze(0), 
                missing_obs
            ), dim=-1)
        else:
            actor_input = actor_input.reshape((*obs_shape[:-1], -1))
            
            if override_rnn_output is not None:
                input_a = override_rnn_output
            else:
                input_a = self.memory_a(actor_input,**kwargs).squeeze(0)            

        return super().act_inference(input_a)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def act_dagger(self, observations, priv_latent, return_scandots_latent=False):
        raise NotImplemented

class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         **kwargs)

        activation = get_activation(activation)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations, **kwargs):
        input_a = self.memory_a(observations, **kwargs)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class LayerNormGRU(nn.GRU):
    def __init__(self, input_size=None, hidden_size=None, num_layers=None):
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        self.ln = nn.LayerNorm((hidden_size))
    
    def forward(self, input, hidden_states):
        out, hidden_states = super().forward(input, hidden_states)

        if hidden_states is not None:
            hidden_states = self.ln(hidden_states)

        return out, hidden_states
        
class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256, noise_scales=None):
        super().__init__()
        # RNN
        type = type.lower()

        if type == "gru":
            rnn_cls = nn.GRU
        elif type == "ln_gru":
            rnn_cls = LayerNormGRU
        elif type == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise NotImplemented

        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
        self.noise_scales = noise_scales
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")

            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        if self.hidden_states is None:
            return

        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0  

    def reset_with_grad(self, dones=None):
        if self.hidden_states is None:
            return 

        if dones is None:
            self.hidden_states = self.hidden_states * 0
        else:
            device = self.hidden_states.device

            if device.index != 0:
                # For some reason running this on GPU 1 leads to cuda error 
                # Hacky solution - do the buggy op on cpu instead
                self.hidden_states = self.hidden_states.cpu()
                dones = dones.cpu()
                self.hidden_states[:, dones, :] = self.hidden_states[:, dones, :] * 0
                self.hidden_states = self.hidden_states.to(device)
            else:
                self.hidden_states[:, dones, :] = self.hidden_states[:, dones, :] * 0