# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from configparser import Interpolation
from turtle import width
from xml.dom import HierarchyRequestErr
from cv2 import decomposeProjectionMatrix, magnitude
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import skimage
import numpy as np
import os
from tqdm import tqdm
import math
import random
import scipy.ndimage
import matplotlib.pyplot as plt
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
import warnings
import torch
from torch import Tensor, normal
from typing import Tuple, Dict
from collections import deque
import matplotlib.pyplot as plt
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from scipy.spatial.transform import Rotation as R
from torchaudio import get_audio_backend
from .legged_robot_config import LeggedRobotCfg
from math import sin, cos
import cv2
import pytorch3d.transforms as tx
import matplotlib.pyplot as plt
import torchvision
import wandb
from copy import deepcopy
from .scandots_noise_samplers import *
from .pid_controller import PIDController 

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def quat_from_axis_angle(axis, angle):
    w = torch.cos(angle / 2)
    xyz = - axis * torch.sin(angle / 2)

    return torch.cat((xyz, w), dim=-1)

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        self.cfg = cfg
        self.global_counter = 0
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = os.getenv("DEBUG_VIZ") is not None
        self.init_done = False
        self.new_height = False # True when a new height value is observed
        self.new_depth = False 
        self.init_config_curriculum()
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self.construct_edge_masks()
        self.visual_blackout_mask = torch.zeros(self.num_envs).bool()
        self.init_done = True
        self.depth_ctr = 0
        self.global_counter = 0
        self.total_env_steps_counter = 0
        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        # Old code for backward compatibility use curriculum
        if hasattr(self.cfg.env, "exec_lag_curriculum"):
            print("Using exec lag curriculum")
            
            if self.cfg.env.test_time:
                print("Using exec_lag = ", self.cfg.env.exec_lag_curriculum["max"])
                self.cfg.env.exec_lag = self.cfg.env.exec_lag_curriculum["max"]
            else:
                self.cfg.env.exec_lag = self.cfg.env.exec_lag_curriculum["initial"]

        self.initial_root_states = self.root_states.clone()
        self.nonzero_vel_steps = torch.zeros(self.num_envs, device=self.device)
        self.last_force_sensor_tensor = torch.zeros((self.num_envs, 24), device=self.device)

    def init_config_curriculum(self):
        if not hasattr(self.cfg.env, "test_time"):
            self.cfg.env.test_time = False

        if not hasattr(self.cfg.env, "curriculum"):
            return
        
        self.curriculum_indices = {}

        for x in self.cfg.env.curriculum:
            if self.cfg.env.test_time:
                idx = -1
            else:
                idx = 0
            
            value = x["values"][idx]
            exec("self.cfg.{} = {}".format(x["field"], value))
            self.curriculum_indices[x["field"]] = idx

            if wandb.run is not None:
                wandb.log({x["field"]: value})

    def double_penalty_coeff(self):
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            if name in self.doubling_terms:
                dfac = self.doubling_factors[self.doubling_terms.index(name)]
                self.reward_scales[name] *= dfac
        
        if "termination" in self.doubling_terms:
            dfac = self.doubling_factors[self.doubling_terms.index("termination")]
            self.reward_scales["termination"] *= dfac

        if "termination" in self.doubling_terms:
            dfac = self.doubling_factors[self.doubling_terms.index("termination")]
            self.reward_scales["termination"] *= dfac

    def zero_actions(self, actions):
        """
            Set actions to zero if ep_len is smaller than some interval.
            This is done in cases where robot needs to capture a depth frame for calibration
        """

        if not hasattr(self.cfg.env, "zero_action_duration"):
            return actions

        num_zero_action_steps = int(self.cfg.env.zero_action_duration / (self.cfg.control.decimation * self.cfg.sim.dt))
        mask = self.episode_length_buf <= num_zero_action_steps
        actions[mask] = 0

        return actions

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        actions = self.raisim2ig(actions)

        if hasattr(self.cfg.noise.noise_scales, "action"):
            actions += torch.randn_like(actions) * self.cfg.noise.noise_scales.action

        self.global_counter += 1
        self.total_env_steps_counter += 1
        decay_period = int(self.decay_period)
        if (self.total_env_steps_counter % (24 * decay_period) == 0 and self.total_env_steps_counter < (24 * (4*decay_period))): 
            self.double_penalty_coeff()
        
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.action_history_buffer[:, :-1] = self.action_history_buffer[:, 1:].clone()
        self.action_history_buffer[:, -1] = actions.clone()
        self.actions = actions
        
        # Get lagged actions if required
        if hasattr(self.cfg.env, "exec_lag"):
            exec_lag = self.cfg.env.exec_lag
        elif hasattr(self.cfg.env, "exec_lag_range"):
            exec_lag = random.randint(*self.cfg.env.exec_lag_range)
        else:
            exec_lag = 0

        laggy_actions = self.action_history_buffer[torch.arange(self.num_envs).to(self.device), -1 - exec_lag].clone()
        self.actions = laggy_actions

        # step physics and render each frame
        self.render()

        if hasattr(self.cfg.control, "decimation_range"):
            decimation = random.randint(*self.cfg.control.decimation_range)
        else:
            decimation = self.cfg.control.decimation

        for _ in range(decimation): 
            laggy_actions = self.zero_actions(laggy_actions) 
            self.torques = self._compute_torques(laggy_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def update_prop_history_buffer(self):
        if self.prop_history_buf is None:
            return

        if self.global_counter % self.height_updat != 0:
            return

        self.prop_history_buf[:, :-1] = self.prop_history_buf[:, 1:].clone()

        base_vels_scaled = torch.cat((
            self.get_noisy_measurement(self.base_lin_vel * self.obs_scales.lin_vel, self.cfg.noise.noise_scales.lin_vel),
            self.get_noisy_measurement(self.base_ang_vel  * self.obs_scales.ang_vel, self.cfg.noise.noise_scales.ang_vel),
        ), dim=-1)

        if hasattr(self.cfg.env, "mask_base_ang_vel") and self.cfg.env.mask_base_ang_vel:
            base_vels_scaled[:, 3:] *= 0

        if hasattr(self.cfg.env, "mask_base_lin_vel") and self.cfg.env.mask_base_lin_vel:
            base_vels_scaled[:, :3] *= 0

        self.prop_history_buf[:, -1] = torch.cat((
            base_vels_scaled,
            self.get_orientation_observations(),
            self.ig2raisim(self.get_noisy_measurement((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, self.cfg.noise.noise_scales.dof_pos)),
            self.ig2raisim(self.get_noisy_measurement(self.dof_vel * self.obs_scales.dof_vel, self.cfg.noise.noise_scales.dof_vel))
        ), dim=-1)

    def calibrate_depth(self, env_id, depth_image):
        if self.calibration_depth_buffer is None:
            return 

        if self.is_calibrated[env_id]:
            return

        if self.episode_length_buf[env_id] < self.calibrate_steps:
            return

        self.calibration_depth_buffer[env_id][0] = depth_image.clone()
        self.is_calibrated[env_id] = True

    def process_depth_image(self, depth_image, env_id):
        depth_image = self.add_depth_artifacts(depth_image)

        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image = torch.clip(depth_image, -self.cfg.depth.clip, self.cfg.depth.clip)
        self.save_depth_image(depth_image, env_id)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)

        return depth_image

    def update_depth_buffer(self):
        if not self.cfg.env.use_camera:
            return

        if hasattr(self, "depth_update_interval") and self.global_counter % self.depth_update_interval != 0:
            return

        if self.global_counter % self.height_update_interval != 0:
            return

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.refresh_patch_masks()

        for i in range(self.num_envs):
            depth_image_concat = None

            for cam_handle in self.cam_handles[i]:
                depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                    self.envs[i], 
                                                                    cam_handle,
                                                                    gymapi.IMAGE_DEPTH)
                
                depth_image = gymtorch.wrap_tensor(depth_image_)

                if self.visual_blackout_mask[i]:
                    depth_image[:] = 0

                depth_image = self.process_depth_image(depth_image, i)

                if ((depth_image > -1).sum() == 0):
                    warnings.warn("empty depth image receved", UserWarning)

                if depth_image_concat is None:
                    depth_image_concat = depth_image
                else:
                    depth_image_concat = torch.cat((depth_image_concat, depth_image), dim=-1)

            if i == self.lookat_id: 
                self.depth_image_concat_displayed = depth_image_concat.clone()

            self._depth_buffer[i, :-1] = self._depth_buffer[i, 1:].clone()    
            self._depth_buffer[i, -1] = depth_image_concat.clone()
            self.calibrate_depth(i, depth_image_concat)

        self.gym.end_access_image_tensors(self.sim)

        if hasattr(self, "depth_update_interval"):
            self.set_depth_update_interval()
            self.depth_flush_step = self.global_counter + random.randint(*self.depth_update_latency)

    def set_depth_update_interval(self):
        depth_update_dt = random.uniform(*self.cfg.env.depth_update_dt)
        self.depth_update_interval = int(depth_update_dt / self.cfg.control.decimation / self.cfg.sim.dt)
        
        if self.depth_update_interval == 0:
            self.depth_update_interval = 1

    def crop_depth_image(self, depth_image):
        if hasattr(self.cfg.depth, "crop_left"):
            depth_image = depth_image[:, self.cfg.depth.crop_left:]

        if hasattr(self.cfg.depth, "crop_right"):
            depth_image = depth_image[:, :-self.cfg.depth.crop_right]

        if hasattr(self.cfg.depth, "crop_top"):
            depth_image = depth_image[self.cfg.depth.crop_top:, :]

        if hasattr(self.cfg.depth, "crop_bottom"):
            depth_image = depth_image[:-self.cfg.depth.crop_bottom, :]

        return depth_image

    def save_depth_image(self, depth_image, env_id):
        if not hasattr(self.cfg.depth, "save_images") or not self.cfg.depth.save_images:
            return

        if self.lookat_id != env_id:
            return

        depth_image = -1 * depth_image.squeeze().cpu()
        plt.imsave("depth_images/depth_image_{}.png".format(self.global_counter), depth_image)

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1

        if self.cfg.depth.normalize_mean is not None:
            depth_image = depth_image - abs(self.cfg.depth.normalize_mean) # The abs is for backward compatibility

        if self.cfg.depth.normalize_std is not None:
            depth_image = depth_image / self.cfg.depth.normalize_std

        return depth_image

    def refresh_patch_masks(self):
        if self.cfg.depth.patch_selection_prob == 0:
            return

        self.patch_masks = {}

        def round_and_clip(x, i, shape):
            x = int(x)

            if x < 0:
                x = 0
                return 0

            if x > shape[i]:
                x = shape[i] - 1

            return x

        for patch_type in self.cfg.depth.patch_types:
            patch_mask = torch.zeros((self.cfg.depth.original[1], self.cfg.depth.original[0])).bool()
            mask = torch.rand(patch_mask.shape) < self.cfg.depth.patch_selection_prob
            patch_indices = mask.nonzero()
            
            for x1, x2 in patch_indices:
                patch_dim1 = np.random.uniform(*self.cfg.depth.patch_dim_interval[0])
                patch_dim2 = np.random.uniform(*self.cfg.depth.patch_dim_interval[1])

                patch_mask[
                    round_and_clip(x1 - patch_dim1 / 2, 0, patch_mask.shape) : round_and_clip(x1 + patch_dim1 / 2, 0, patch_mask.shape),
                    round_and_clip(x2 - patch_dim2 / 2, 1, patch_mask.shape) : round_and_clip(x2 + patch_dim2 / 2, 1, patch_mask.shape)
                ] = True

            self.patch_masks[patch_type] = patch_mask

    def add_depth_artifacts(self, depth_image):
        depth_image = depth_image.clone()

        if self.cfg.depth.patch_selection_prob != 0:
            for patch_type in self.cfg.depth.patch_types:
                if patch_type == "white":
                    patch_value = -1
                elif patch_type == "black":
                    patch_value = 0
                else:
                    raise NotImplemented("This patch value is not implemented")

                patch_mask = self.patch_masks[patch_type]
                depth_image[patch_mask] = patch_value
        
        if self.cfg.depth.noise_scale > 0:
            depth_image += torch.randn_like(depth_image).to(self.device) * self.cfg.depth.noise_scale 

        depth_image = -depth_image.cpu().numpy()

        if self.cfg.depth.salt_pepper_noise:
            depth_image = skimage.util.random_noise(depth_image, mode="s&p")

        if hasattr(self.cfg.depth, "noise_samplers"):
            for sampler in self.cfg.depth.noise_samplers:
                depth_image = sampler.add_noise(depth_image)

        depth_image = -torch.from_numpy(depth_image).to(self.device)

        return depth_image

    def update_visual_blackout_mask(self):
        if self.visual_blackout_mean_life is None:
            return

        off_probability = self.dt / self.visual_blackout_mean_life
        mask = torch.rand(self.num_envs, device=self.device) < off_probability

        self.visual_blackout_mask[mask] = True

        on_probability = self.dt / self.visual_blackout_mean_life_on
        mask = torch.rand(self.num_envs, device=self.device) < on_probability

        self.visual_blackout_mask[mask] = False

    def get_history_observations(self):
        return self.obs_history_buf
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.measured_heights = self._get_heights()
        self.measured_clean_heights = self._get_clean_heights()
        self.measured_ray_distances = self._get_ray_distances()
        self.update_depth_buffer()
        self.update_visual_blackout_mask()
        self.update_prop_history_buffer()
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.cfg.env.history_nsteps is not None:
            self.obs_history_buf = torch.cat([
                self.obs_history_buf[:, 1:],
                self.obs_buf[:, None, self.proprio_obs_start:self.proprio_obs_end]], dim=1)

        self.last_force_sensor_tensor = self.force_sensor_tensor.clone()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    
    def _reset_depth_buffer(self, env_ids):
        if self.cfg.env.use_camera:
            self.depth_buffer[env_ids] = 0
            self._depth_buffer[env_ids] = 0

        if self.calibration_depth_buffer is not None:
            self.calibration_depth_buffer[:] = 0
            self.is_calibrated[:] = False

    def check_feet_in_gap(self):
        if hasattr(self.cfg.env, "check_feet_in_gap") and not self.cfg.env.check_feet_in_gap:
            return torch.zeros(self.num_envs).bool().cuda()

        depth_threshold = -0.5
        feet_xy_pos = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 0:2]
        feet_z_pos = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 2]

        terrain_ht_under_feet = self.get_heights_at_abs_points(feet_xy_pos) - self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, 0, 2][:, None]
        feet_in_gap = (feet_z_pos < self.cfg.asset.feet_in_gap_cutoff) * (terrain_ht_under_feet <= depth_threshold)
        return torch.max(feet_in_gap, dim = 1)[0]

    def update_nonzero_vel_steps(self):
        zero_vel_mask = (self.commands[:, 0] <= 1e-3) & (self.base_lin_vel[:, 0] >= 0.01)
        self.nonzero_vel_steps[zero_vel_mask] += 1
        self.nonzero_vel_steps[~zero_vel_mask][:] = 0

    def check_excess_foot_force(self):
        if not hasattr(self.cfg.asset, "max_foot_force"):
            self.cfg.asset.max_foot_force = float('inf')

        foot_force_vertical = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, 2]
        excess_foot_force = foot_force_vertical > self.cfg.asset.max_foot_force
        excess_foot_force = excess_foot_force.sum(dim=-1)
        excess_foot_force = excess_foot_force.bool() 
        excess_foot_force = excess_foot_force & (self.episode_length_buf > 100) # Check after falling

        return excess_foot_force

    def check_excess_foot_impact(self):
        if not hasattr(self.cfg.rewards, "foot_impact_hard_limit"):
            reset_buf = torch.zeros(self.num_envs, device=self.device).bool()
            return reset_buf

        last_force = self.last_force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, :3]
        curr_force = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, :3]
        delta_force = torch.square(torch.norm(curr_force - last_force, dim=-1))
        total_impact = delta_force.sum(dim=-1)

        reset_buf = (total_impact >= self.cfg.rewards.foot_impact_hard_limit) 
        reset_buf &= (self.episode_length_buf >= 100)
        
        return reset_buf

    def check_max_yaw_nonflat(self):
        reset_buf = torch.zeros(self.num_envs, device=self.device).bool() 

        if not hasattr(self.cfg.asset, "max_yaw_nonflat"):
            return reset_buf

        r, p, y = euler_from_quaternion(self.base_quat) 
        r_init, p_init, y_init = euler_from_quaternion(self.initial_root_states[:, 3:7])

        flat_yaw_cutoff = (torch.abs(y - y_init) > self.cfg.asset.max_yaw_nonflat)
        nonflat_terrain_mask = self.env_terrain_types != 7
        reset_buf = (flat_yaw_cutoff & nonflat_terrain_mask) # Terminate only on non-flat ground if yaw exceeds certain value

        return reset_buf

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        r, p, y = euler_from_quaternion(self.base_quat) 
        r_init, p_init, y_init = euler_from_quaternion(self.initial_root_states[:, 3:7])

        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        roll_threshold = 0.6 if not hasattr(self.cfg.asset, "max_roll") else self.cfg.asset.max_roll
        yvel_threshold = self.cfg.asset.max_yvel if hasattr(self.cfg.asset, "max_yvel") else float('inf')
        yshift_threshold = self.cfg.asset.max_yshift if hasattr(self.cfg.asset, "max_yshift") else float('inf')
        yaw_threshold = self.cfg.asset.max_yaw if hasattr(self.cfg.asset, "max_yaw") else 0.17453
        heading_error_threshold = self.cfg.asset.max_heading_error if hasattr(self.cfg.asset, "max_heading_error") else float('inf')

        # Make sure the robot is making some forward progres if command is large enough
        forward_vel_threshold = torch.ones(self.num_envs).to(self.device) * 0.5
        forward_vel_threshold = torch.minimum(forward_vel_threshold, self.commands[:, 0] - 0.2)
        forward_vel_threshold = torch.maximum(forward_vel_threshold, forward_vel_threshold * 0)

        roll_cutoff = torch.abs(r) > roll_threshold
        pitch_threshold = 0.8 if not hasattr(self.cfg.asset, "max_pitch") else self.cfg.asset.max_pitch
        pitch_cutoff = torch.abs(p) > pitch_threshold
        yvel_cutoff = torch.abs(self.base_lin_vel[:, 1]) > yvel_threshold
        yshift_cutoff = torch.abs(self.root_states[:, 1] - self.initial_root_states[:, 1]) > yshift_threshold

        xvel_cutoff = torch.zeros(self.num_envs).to(self.device).bool()

        if self.terminate_xvel:
            xvel_cutoff = self.base_lin_vel[:, 0] < forward_vel_threshold
            xvel_cutoff = xvel_cutoff & (self.episode_length_buf > 200)

        if hasattr(self.cfg.asset, "min_height"):
            ht_cutoff = base_height < self.cfg.asset.min_height
        
        yaw_cutoff = torch.abs(y - y_init) > yaw_threshold # More than 10 degrees yaw

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_error = torch.abs(self.commands[:, 3] - heading)
        heading_error_cutoff = (heading_error > heading_error_threshold)

        self.update_nonzero_vel_steps()
        nonzero_vel_steps_cutoff = self.nonzero_vel_steps > self.nonzero_vel_steps_threshold
        self.reset_buf |= nonzero_vel_steps_cutoff

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= yaw_cutoff
        self.reset_buf |= yvel_cutoff
        self.reset_buf |= yshift_cutoff
        self.reset_buf |= xvel_cutoff
        self.reset_buf |= heading_error_cutoff
        self.reset_buf |= self.check_max_yaw_nonflat()
        self.reset_buf |= self.check_feet_in_gap()
        self.reset_buf |= self.check_excess_foot_force()
        self.reset_buf |= self.check_feet_on_edge()
        self.reset_buf |= self.check_excess_foot_impact()

        if hasattr(self.cfg.asset, "min_height"):
            self.reset_buf |= ht_cutoff

    def get_distance_to_edge(self):
        result = torch.ones((self.num_envs, 4), device=self.device) * self.cfg.env.feet_distance_soft_limit

        edge_radius = int(self.cfg.env.feet_distance_soft_limit / self.cfg.terrain.horizontal_scale)
        contact_mask = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, 2] > 1
        contact_mask = contact_mask[:, [1, 0, 3, 2]] # Change ordering to match feet_indices ordering
        feet_xy_pos_world = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 0:2]
        feet_xy_pos = feet_xy_pos_world + self.terrain.cfg.border_size
        feet_xy_pos = (feet_xy_pos / self.terrain.cfg.horizontal_scale).long()

        E, num_points, C = feet_xy_pos.shape
        assert(E == self.num_envs)
        assert(C == 2)

        px = feet_xy_pos[:, :, 0].view(-1)
        py = feet_xy_pos[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        self.feet_closest_points = torch.ones((self.num_envs, 4, 2), device=self.device).long() * -1

        for i in range(-edge_radius-1, edge_radius + 2):
            for dir_x, dir_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                qx = px + i * dir_x
                qy = py + i * dir_y
                qx = torch.clip(qx, 0, self.height_samples.shape[0]-2)
                qy = torch.clip(qy, 0, self.height_samples.shape[1]-2)
                qx_world = qx * self.cfg.terrain.horizontal_scale - self.terrain.cfg.border_size
                qy_world = qy * self.cfg.terrain.horizontal_scale - self.terrain.cfg.border_size
                q_world = torch.stack((qx_world, qy_world), dim=-1).reshape((self.num_envs, 4, 2))

                distance_to_foot = torch.norm(feet_xy_pos_world - q_world, dim=-1).reshape((self.num_envs, 4))
                is_edge_point = self.edge_mask[qx, qy].reshape((self.num_envs, 4))

                update_mask = (result > distance_to_foot) & is_edge_point
                self.feet_closest_points[update_mask] = torch.stack((
                    qx.reshape((self.num_envs, 4))[update_mask],
                    qy.reshape((self.num_envs, 4))[update_mask],
                ), dim=-1)

                result[update_mask] = distance_to_foot[update_mask]

        result[~contact_mask] = self.cfg.env.feet_distance_soft_limit
        self.min_edge_distance = torch.minimum(self.min_edge_distance, result.min(dim=-1)[0])

        return result

    def check_feet_on_edge(self):
        reset_buf = torch.zeros((self.num_envs, 4), device=self.device).bool()

        if not self.cfg.env.check_feet_on_edge:
            return reset_buf.sum(dim=-1).bool()

        assert(hasattr(self.cfg.env, "edge_distance_cutoff") or hasattr(self.cfg.env, "adaptive_edge_distance_cutoff")) 
        
        if hasattr(self.cfg.env, "edge_distance_cutoff"):
            edge_distance_cutoff = torch.ones(self.num_envs, device=self.device) * self.cfg.env.edge_distance_cutoff
        elif hasattr(self.cfg.env, "adaptive_edge_distance_cutoff"):
            edge_distance_cutoff =  torch.ones(self.num_envs, device=self.device) * -1

            for i in range(len(self.cfg.env.adaptive_edge_distance_cutoff)):
                if i == 0:
                    low_threshold = 0.0
                else:
                    low_threshold = self.cfg.env.adaptive_edge_distance_cutoff[i-1][0]
                
                high_threshold = self.cfg.env.adaptive_edge_distance_cutoff[i][0]
                mask = (self.env_terrain_difficulty >= low_threshold) & (self.env_terrain_difficulty <= high_threshold)

                edge_distance_cutoff[mask] = self.cfg.env.adaptive_edge_distance_cutoff[i][1]
        else:
            assert(False)


        reset_buf = self.get_distance_to_edge() <= edge_distance_cutoff[:, None]

        return reset_buf.sum(dim=-1).bool()

    def _resample_gains(self, env_ids):
        n = env_ids.shape[0]

        for i in range(self.num_dofs):
            for dof_name in self.cfg.control.stiffness.keys():
                self.p_gains[env_ids, i] = self.cfg.control.stiffness[dof_name]
                self.d_gains[env_ids, i] = self.cfg.control.damping[dof_name]

                if hasattr(self.cfg.control, "stiffness_delta"):
                    stiffness_delta = self.cfg.control.stiffness_delta
                    self.p_gains[env_ids, i] += torch.rand(n).to(self.device) * 2 * stiffness_delta - stiffness_delta

                if hasattr(self.cfg.control, "damping_delta"):
                    damping_delta = self.cfg.control.damping_delta
                    self.d_gains[env_ids, i] += torch.rand(n).to(self.device) * 2 * damping_delta - damping_delta
            
    def _reset_commands(self, env_ids):
        if self.keyboard_control_heading or self.keyboard_control_angvel:
            self.commands[env_ids, 3] = 0
            self.commands[env_ids, 2] = 0

        if self.keyboard_control_xvel:
            self.commands[env_ids, 0] = self.default_keyboard_vel

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._reset_depth_buffer(env_ids)
        self._resample_commands(env_ids)
        self._resample_gains(env_ids)
        self._reset_commands(env_ids)
        self.scandots_noise_sampler.set_num_envs(self.num_envs)
        self.scandots_noise_sampler.resample_noise(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.action_history_buffer[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.nonzero_vel_steps[env_ids] = 0
        
        if self.cfg.env.history_nsteps is not None:
            self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.max_foot_impact[env_ids] = 0
        self.min_edge_distance[env_ids] = self.cfg.env.feet_distance_soft_limit
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """

        self.rew_buf[:] = 0.
        flat_mask = self.env_terrain_types == 7

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]

            if "_flat" in name:
                rew = rew * flat_mask
            
            if "_nonflat" in name:
                rew = rew * (~flat_mask)

            assert(rew.shape[0] == self.num_envs)
            self.rew_buf += rew
            self.named_rew_buf[name] = rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def get_orientation_observations(self):
        body_angles = self.get_body_orientation()

        if hasattr(self.cfg.env, "use_rpy_trig_values") and self.cfg.env.use_rpy_trig_values:
            return torch.cat((
                body_angles, 
                torch.sin(body_angles),
                torch.cos(body_angles)
            ), dim=-1)
        else:
            return body_angles

    def get_body_orientation(self, return_yaw=False):
        # base_quat_real_first = torch.cat((
        #     self.base_quat[:, -1][:, None],
        #     self.base_quat[:, :-1]
        # ), dim=-1)
        # body_angles = tx.quaternion_to_axis_angle(base_quat_real_first)

        body_angles = euler_from_quaternion(self.base_quat)
        body_angles = torch.stack(body_angles, dim=-1)

        if not return_yaw:
            return body_angles[:, :-1] # roll = rot_x, pitch = rot_y
        else:
            return body_angles
    
    def compute_delta_pose(self, p1, p2):
        delta_pos = p1[:, :3] - p2[:, :3]
        delta_quat = quat_mul(p1[:, 3:7], quat_conjugate(p2[:, 3:7]))
        return torch.cat((delta_pos, delta_quat), dim=-1)
    def ig2raisim(self, vec):
        if self.cfg.env.reorder_dofs:
            # Need to reorder DOFS to match what the a1 hardware gives -(FR (hip, thigh, calf), FL, RR, RL)
            if not hasattr(self, "ig_2_raisim_reordering_idx"):
                # self.dof_names = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
                self.ig_2_raisim_reordering_idx = []
                dof_order_a1_robot = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]

                for name in dof_order_a1_robot:
                    self.ig_2_raisim_reordering_idx.append(self.dof_names.index(name))

            vec = vec[:, self.ig_2_raisim_reordering_idx]

        return vec

    def raisim2ig(self, vec):
        if self.cfg.env.reorder_dofs:
            if not hasattr(self, "raisim2ig_reordering_idx"):
                self.raisim2ig_reordering_idx = []

                dof_order_a1_robot = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]

                for name in self.dof_names:
                    self.raisim2ig_reordering_idx.append(dof_order_a1_robot.index(name))

            vec = vec[:, self.raisim2ig_reordering_idx]
        
        return vec

    def set_height_update_interval(self):
        if hasattr(self.cfg.env, "height_update_dt"):
            height_update_dt = random.uniform(*self.cfg.env.height_update_dt)
            self.height_update_interval = int(height_update_dt / self.cfg.control.decimation / self.cfg.sim.dt)
            
            if self.height_update_interval == 0:
                self.height_update_interval  = 1
        else:
            self.height_update_interval = 1

    def add_timing_info(self, obs):
        if self.cfg.env.scandots_timing_dim == 0:
            return obs

        timing_info = self.get_time_encoding(self.cfg.env.scandots_timing_dim, 
                                             self.global_counter - self.scandots_update_step,
                                             self.num_envs)

        obs = torch.cat((timing_info, obs), dim=-1)
        return obs

    def process_ray_observations(self, obs):
        if not self.cfg.terrain.measure_rays:
            raise Exception("process_ray_observations called even though rays not being measured")

        if not hasattr(self, "_ray_distances"):
            num_scandots = self.cfg.env.n_scan
            self.scandots_flush_step = self.global_counter
            num_scandots -= self.cfg.env.scandots_timing_dim
            self._ray_distances = torch.zeros(self.num_envs, num_scandots).to(self.device)
        elif (self.global_counter % self.height_update_interval) == 0:
            self._ray_distances = torch.clip(self.measured_ray_distances, -1, 1) * self.obs_scales.height_measurements
            self._ray_distances = self.get_noisy_measurement(self._ray_distances, self.cfg.noise.noise_scales.height_measurements)

            self.set_height_update_interval()
            self.scandots_flush_step = self.global_counter + random.randint(*self.height_update_latency)

        if self.global_counter == self.scandots_flush_step:
            self.ray_distances = self._ray_distances.clone()
            self.scandots_update_step = self.global_counter
            self.new_height = True
        else:
            self.new_height = False

        obs = self.add_timing_info(obs)
        obs = torch.cat((self.ray_distances, obs), dim=-1)

        return obs

    def quantize(self, arr, quantization):
        if quantization is None:
            return arr

        arr = arr / quantization
        arr = arr.int()
        arr = arr.float() * quantization

        return arr

    def blackout_scandots(self, heights):
        if self.cfg.env.visual_blackout_mode == "noise":
            heights[self.visual_blackout_mask] = torch.randn_like(heights[self.visual_blackout_mask]) # Add large noise to visual blackout
        elif self.cfg.env.visual_blackout_mode == "zeros":
            heights[self.visual_blackout_mask] = 0.25 - self.heights_offset
        elif self.cfg.env.visual_blackout_mode == "no_offset_zeros":
            heights[self.visual_blackout_mask] = 0

        return heights

    def clip_measured_heights(self, measured_heights):
        if not hasattr(self.cfg.noise, "clip_ranges"):
            return self.root_states[:, 2].unsqueeze(1) - measured_heights

        heights = self.root_states[:, 2].unsqueeze(1) - measured_heights

        for low, high, val in self.cfg.noise.clip_ranges:
            mask = (heights >= low) & (heights <= high)
            heights[mask] = val

        return heights

    def process_scandots_observations(self, obs):
        if not self.cfg.terrain.measure_heights:
            raise Exception("process_scandots_observations called even though heights not being measured")
            
        if not hasattr(self, "_heights"):
            num_scandots = self.cfg.env.n_scan
            self.scandots_flush_step = self.global_counter
            num_scandots -= self.cfg.env.scandots_timing_dim
            self._heights = torch.zeros(self.num_envs, num_scandots).to(self.device)
            self._clean_heights = torch.zeros(self.num_envs, num_scandots).to(self.device)
            self.heights = torch.zeros_like(self._heights)
            self.clean_heights = torch.zeros_like(self._clean_heights)
        elif (self.global_counter % self.height_update_interval) == 0: 
            clipped_heights = self.clip_measured_heights(self.measured_heights)
            self._heights = torch.clip(clipped_heights - self.heights_offset, -1, 1.) * self.obs_scales.height_measurements

            clipped_clean_heights = self.clip_measured_heights(self.measured_clean_heights)
            self._clean_heights = torch.clip(clipped_clean_heights - self.heights_offset, -1, 1.) * self.obs_scales.height_measurements

            self._heights = self.scandots_noise_sampler.add_vertical_noise(self._heights)
            self._heights = self.quantize(self._heights, self.cfg.noise.height_quantization)
            self.pose_at_last_depth = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, 0, :7]
            
            self.set_height_update_interval()
            self.scandots_flush_step = self.global_counter + random.randint(*self.height_update_latency)
        
        if self.global_counter == self.scandots_flush_step:
            self.heights = self._heights.clone() 
            self.clean_heights = self._clean_heights.clone()
            self.heights = self.blackout_scandots(self.heights)
            self.scandots_update_step = self.global_counter
            self.new_height = True
        else:
            self.new_height = False

        obs = self.add_timing_info(obs)
        obs = torch.cat((self.heights, obs), dim=-1)

        return obs

    def compute_observations(self):
        """ 
        Computes observations
        """
        previous_actions = self.action_history_buffer[:, -1]

        current_obs = torch.cat((  
                                    self.get_noisy_measurement(self.base_lin_vel * self.obs_scales.lin_vel, self.cfg.noise.noise_scales.lin_vel),
                                    self.get_noisy_measurement(self.base_ang_vel  * self.obs_scales.ang_vel, self.cfg.noise.noise_scales.ang_vel),
                                    self.get_foot_contacts(),
                                    self.get_orientation_observations(),
                                    self.commands[:, :3] * self.commands_scale,
                                    self.ig2raisim(self.get_noisy_measurement((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, self.cfg.noise.noise_scales.dof_pos)),
                                    self.ig2raisim(self.get_noisy_measurement(self.dof_vel * self.obs_scales.dof_vel, self.cfg.noise.noise_scales.dof_vel)),
                                    self.ig2raisim(previous_actions),
                                ),dim=-1)

        current_obs = torch.cat((
            current_obs,
            self.mass_params,
            self.friction_coeffs_tensor.squeeze(-1),
            self.motor_strength - 1
        ), dim=-1)

        self.compute_terrain_height()

        if self.cfg.env.include_terrain_height:
            current_obs = torch.cat((current_obs, self.env_terrain_heights.unsqueeze(1)), dim=-1)

        if self.cfg.terrain.measure_heights:
            current_obs = self.process_scandots_observations(current_obs)
        
        if self.cfg.terrain.measure_rays:
            current_obs = self.process_ray_observations(current_obs)

        if self.cfg.env.history_nsteps is not None and self.cfg.env.add_history_to_obs:
            current_obs = torch.cat((
                current_obs, 
                self.obs_history_buf.reshape((self.num_envs, -1))
            ), dim=-1)

        if self.cfg.env.use_camera:
            if hasattr(self, "depth_flush_step") :
                if self.global_counter == self.depth_flush_step:
                    self.depth_buffer = self._depth_buffer.clone()
                    self.new_depth = True
                else:
                    self.new_depth = False
            elif self.global_counter == self.scandots_flush_step:
                self.depth_buffer = self._depth_buffer.clone()

        if hasattr(self.cfg.env, "obs_mask"):
            current_obs *= (1 - torch.tensor(self.cfg.env.obs_mask).to(self.device)[None, :])

        self.obs_buf = current_obs.clone()

    def get_time_encoding(self, dim, nstep, n):
        assert(dim % 2 == 0)

        result = torch.zeros((n, dim)).to(self.device)
        powers_of_two = (1 / 2) ** torch.arange(dim / 2).to(self.device)
        omega = math.pi / (self.cfg.sim.dt * self.cfg.control.decimation)

        t = nstep * self.cfg.sim.dt * self.cfg.control.decimation
        result[:, ::2] = torch.sin(omega * powers_of_two * t)
        result[:, 1::2] = torch.cos(omega * powers_of_two * t)

        return result

    def get_noisy_measurement(self, x, scale):
        if self.cfg.noise.add_noise:
            if hasattr(self.cfg.noise, "noise_type") and self.cfg.noise.noise_type == "uniform":
                x = x + (2 * torch.rand_like(x) - 1) * scale * self.cfg.noise.noise_level
            else:
                x = x + torch.randn_like(x) * scale * self.cfg.noise.noise_level

        return x

    def construct_edge_masks(self):
        edge_threshold = int(self.cfg.env.edge_threshold / self.cfg.terrain.vertical_scale)
        dim0, dim1 = self.height_samples.shape
        zero_vec0 = torch.zeros(dim1, device=self.device)
        zero_vec1 = torch.zeros(dim0, device=self.device)
        self.edge_mask = torch.zeros_like(self.height_samples).bool()

        # Perturb to the right
        perturbed = torch.cat((zero_vec0[None, :], self.height_samples[:-1]), dim=0)
        self.edge_mask |= (self.height_samples - perturbed >= edge_threshold)

        # Perturb to the left
        perturbed = torch.cat((self.height_samples[1:], zero_vec0[None, :]), dim=0)
        self.edge_mask |= (self.height_samples - perturbed >= edge_threshold)

        # Perturb to the right
        perturbed = torch.cat((zero_vec1[:, None], self.height_samples[:, :-1]), dim=-1)
        self.edge_mask |= (self.height_samples - perturbed >= edge_threshold)

        perturbed = torch.cat((self.height_samples[:, 1:], zero_vec1[:, None]), dim=-1)
        self.edge_mask |= (self.height_samples - perturbed >= edge_threshold)

    def compute_terrain_height(self):
        if hasattr(self, "env_terrain_heights") and self.global_counter % 50 != 0:
            # Only update every 1s for efficiency
            return

        self.env_terrain_heights = torch.zeros(self.num_envs).to(self.device)
        root_coords = self.root_states[:, :3]
        pairwise_dist = torch.cdist(root_coords.unsqueeze(0), self.terrain_origins.reshape((1, -1, 3))).squeeze(0) # [num_envs, num_rows * num_cols]
        _, closest_origin_indices = torch.min(pairwise_dist, dim=-1)
        closest_origin_indices_x = torch.div(closest_origin_indices, self.cfg.terrain.num_cols, rounding_mode="floor")
        closest_origin_indices_y = closest_origin_indices % self.cfg.terrain.num_cols
        self.env_terrain_heights = self.terrain_heights[closest_origin_indices_x, closest_origin_indices_y]

        border_mask = (root_coords[:, 0] < 0) 
        border_mask |= (root_coords[:, 1] < 0)
        border_mask |= (root_coords[:, 0] > self.cfg.terrain.terrain_length * self.cfg.terrain.num_rows)
        border_mask |= (root_coords[:, 1] > self.cfg.terrain.terrain_width * self.cfg.terrain.num_cols)
        self.env_terrain_heights[border_mask] = 0

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            self.terrain_indices = torch.from_numpy(self.terrain.env_terrain_types).to(self.device)
            self.terrain_difficulty = torch.from_numpy(self.terrain.env_terrain_difficulty).to(self.device)

        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            # print(len(props))
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        # else:
        #     pass
        #     if env_id==0:
        #         # prepare friction randomization
        #         friction_range = self.cfg.domain_rand.friction_range
        #         num_buckets = 64
        #         bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
        #         self.friction_coeffs = torch.zeros(self.num_envs, 1, device='cpu')
        #     self.friction_coeffs[env_id, 0] = props[0].friction
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass

        # No need to use tensors as only called upon env creation
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])
        return props, mass_params

    def _push_feet(self):
        random_indices = torch.randint(low=0, high=4, size=(self.num_envs,), device=self.device)
        chosen_feet_indices = self.feet_indices[random_indices]
        env_indices = torch.arange(self.num_envs, device=self.device)
        
        force_positions = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, :, :3].clone()
        num_bodies = force_positions.shape[1]

        forces = torch.zeros((self.num_envs, num_bodies, 3), device=self.device, dtype=torch.float)
        uniform_random = torch.rand((self.num_envs, 3), device=self.device)
        forces[env_indices, chosen_feet_indices] = (2 * uniform_random  - 1) * self.cfg.domain_rand.feet_force_magnitude

        self.gym.apply_rigid_body_force_at_pos_tensors(
            self.sim, 
            gymtorch.unwrap_tensor(forces), 
            gymtorch.unwrap_tensor(force_positions),
            gymapi.ENV_SPACE
        )

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            error = wrap_to_pi(self.commands[:, 3] - heading)

            self.commands[:, 2] = self.cfg.commands.heading_command_controller.compute_command(error)

            # self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -3.14, 3.14)
        
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        if self.cfg.domain_rand.push_feet and (self.common_step_counter % self.cfg.domain_rand.push_feet_interval == 0):
            self._push_feet()

        if self.cfg.commands.yvel_command:
            yshift = self.root_states[:, 1] - self.initial_root_states[:, 1]
            yvel = torch.clip(-yshift, -1, 1)
            self.commands[:, 1] = yvel

        self.update_config_curriculum()

        # Code for backward compatibility use curriculum array to define curriculum
        if hasattr(self.cfg.env, "exec_lag_curriculum"):
            if self.global_counter != 0 and \
               self.global_counter % self.cfg.env.exec_lag_curriculum["increment_tsteps"] == 0 and \
               self.cfg.env.exec_lag < self.cfg.env.exec_lag_curriculum["max"]:
               
                print("Incrementing exec_lag")
                self.cfg.env.exec_lag += 1

    def update_config_curriculum(self):
        if self.cfg.env.test_time:
            return

        if not hasattr(self.cfg.env, "curriculum"):
            return

        for x in self.cfg.env.curriculum:
            if self.global_counter % x["increment_interval"] != 0:
                continue

            current_idx = self.curriculum_indices[x["field"]]
            if current_idx == len(x["values"]) - 1:
                continue

            self.curriculum_indices[x["field"]] += 1
            value = x["values"][current_idx + 1]
            
            exec("self.cfg.{} = {}".format(x["field"], value))
            
            if wandb.run is not None:
                wandb.log({x["field"]: value})

    def _resample_commands_nav_loco(self, env_ids):
        # Sample settings (0) - curve following (1) - turning in place
        sampling_modes = torch.randint(low=0, high=2, size=(len(env_ids),), device=self.device)

        if not self.keyboard_control_xvel:
            in_place_commands = torch_rand_float(0, 0.0, (len(env_ids), 1), device=self.device).squeeze(1)
            curve_follow_commands = torch_rand_float(0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 0] = torch.where(sampling_modes == 0, in_place_commands, curve_follow_commands)
        
        if not self.keyboard_control_angvel: 
            assert(not self.cfg.commands.heading_command)
            assert(not self.keyboard_control_heading)

            in_place_commands = torch_rand_float(-0.6, 0.6, (len(env_ids), 1), device=self.device).squeeze(1)
            curve_follow_commands = torch_rand_float(-0.4, 0.4, (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] = torch.where(sampling_modes == 0, in_place_commands, curve_follow_commands)

    def _resample_commands_v1(self, env_ids):
        """
            On terrain fixed linear velocity of 0.35 and heading commands in [-10, 10 degrees]
        """

        assert(self.cfg.commands.heading_command)

        terrain_lin_vel = torch.ones(len(env_ids), device=self.device) * 0.35
        terrain_heading = torch_rand_float(-0.175, 0.175, (len(env_ids), 1), device=self.device).squeeze(1) 

        # There are two sampling modes, in place and curve-following
        sampling_modes = torch.randint(low=0, high=2, size=(len(env_ids),), device=self.device)
        curve_following_mask = (sampling_modes == 0)
        in_place_mask = (sampling_modes == 1)
        
        curve_following_lin_vel = torch_rand_float(0.2, 0.7, (len(env_ids), 1), device=self.device).squeeze(1) 
        curve_following_heading = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1) 

        in_place_lin_vel = torch.zeros(len(env_ids), device=self.device)
        in_place_heading = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device).squeeze(1) 

        terrain_mask = (self.env_terrain_types[env_ids] != 7)
        flat_mask = (self.env_terrain_types[env_ids] == 7)

        self.commands[env_ids[terrain_mask], 0] = terrain_lin_vel[terrain_mask]
        self.commands[env_ids[terrain_mask], 3] = terrain_heading[terrain_mask]

        self.commands[env_ids[flat_mask & curve_following_mask], 0] = curve_following_lin_vel[flat_mask & curve_following_mask]
        self.commands[env_ids[flat_mask & curve_following_mask], 3] = curve_following_heading[flat_mask & curve_following_mask]

        self.commands[env_ids[flat_mask & in_place_mask], 0] = in_place_lin_vel[flat_mask & in_place_mask]
        self.commands[env_ids[flat_mask & in_place_mask], 3] = in_place_heading[flat_mask & in_place_mask]

        # Sanity check
        assert(not torch.any(curve_following_mask & in_place_mask))
        assert(torch.all(curve_following_mask | in_place_mask)) 

        assert(not torch.any(terrain_mask & flat_mask))
        assert(torch.all(flat_mask | terrain_mask))

    def _resample_commands_v2(self, env_ids):
        """
            Add a complete stop setting on flat
        """

        assert(self.cfg.commands.heading_command)
        assert(self.cfg.asset.max_yaw == float('inf'))

        terrain_lin_vel = torch.ones(len(env_ids), device=self.device) * 0.35
        terrain_heading = torch_rand_float(-0.175, 0.175, (len(env_ids), 1), device=self.device).squeeze(1) 

        # There are two sampling modes, in place and curve-following
        # sampling_modes = torch.randint(low=0, high=2, size=(len(env_ids),), device=self.device)
        if len(env_ids) > 0:
            sampling_modes = torch.multinomial(torch.tensor([0.33, 0.33, 0.33]), num_samples=len(env_ids), replacement=True).to(self.device)
        else:
            sampling_modes = torch.tensor([], device=self.device)

        curve_following_mask = (sampling_modes == 0)
        in_place_mask = (sampling_modes == 1)
        complete_stop_mask = (sampling_modes == 2)
        
        curve_following_lin_vel = torch_rand_float(0.2, 0.7, (len(env_ids), 1), device=self.device).squeeze(1) 
        curve_following_heading = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1) 

        in_place_lin_vel = torch.zeros(len(env_ids), device=self.device)
        in_place_heading = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device).squeeze(1) 

        complete_stop_lin_vel = torch.zeros(len(env_ids), device=self.device)
        complete_stop_heading = torch.zeros(len(env_ids), device=self.device)

        terrain_mask = (self.env_terrain_types[env_ids] != 7) & (self.env_terrain_types[env_ids] != 19)
        flat_mask = (self.env_terrain_types[env_ids] == 7) | (self.env_terrain_types[env_ids] == 19)

        self.commands[env_ids[terrain_mask], 0] = terrain_lin_vel[terrain_mask]
        self.commands[env_ids[terrain_mask], 3] = terrain_heading[terrain_mask]

        self.commands[env_ids[flat_mask & curve_following_mask], 0] = curve_following_lin_vel[flat_mask & curve_following_mask]
        self.commands[env_ids[flat_mask & curve_following_mask], 3] = curve_following_heading[flat_mask & curve_following_mask]

        self.commands[env_ids[flat_mask & in_place_mask], 0] = in_place_lin_vel[flat_mask & in_place_mask]
        self.commands[env_ids[flat_mask & in_place_mask], 3] = in_place_heading[flat_mask & in_place_mask]

        self.commands[env_ids[flat_mask & complete_stop_mask], 0] = complete_stop_lin_vel[flat_mask & complete_stop_mask]
        self.commands[env_ids[flat_mask & complete_stop_mask], 3] = complete_stop_heading[flat_mask & complete_stop_mask]

        # Sanity check
        assert(not torch.any(curve_following_mask & in_place_mask))
        assert(torch.all(curve_following_mask | in_place_mask | complete_stop_mask))

        assert(not torch.any(terrain_mask & flat_mask))
        assert(torch.all(flat_mask | terrain_mask))

        # Clip commands if needed
        if self.cfg.commands.clip_lin_vel_x is not None:
            self.commands[:, 0] = torch.clip(self.commands[:, 0], min=self.cfg.commands.clip_lin_vel_x[0], max=self.cfg.commands.clip_lin_vel_x[1])

        if self.cfg.commands.clip_heading is not None:
            self.commands[:, 3] = torch.clip(self.commands[:, 3], min=self.cfg.commands.clip_heading[0], max=self.cfg.commands.clip_heading[1])            

    def _resample_commands_v3(self, env_ids, **kwargs):
        """
            Add a complete stop setting on flat
        """

        assert(self.cfg.commands.heading_command)
        assert(self.cfg.asset.max_yaw == float('inf'))

        terrain_lin_vel = torch_rand_float(*kwargs["terrain_lin_vel"], (len(env_ids), 1), device=self.device).squeeze(1) 
        terrain_heading = torch_rand_float(*kwargs["terrain_heading"], (len(env_ids), 1), device=self.device).squeeze(1) 

        # There are two sampling modes, in place and curve-following
        # sampling_modes = torch.randint(low=0, high=2, size=(len(env_ids),), device=self.device)
        if len(env_ids) > 0:
            sampling_modes = torch.multinomial(torch.tensor([0.33, 0.33, 0.33]), num_samples=len(env_ids), replacement=True).to(self.device)
        else:
            sampling_modes = torch.tensor([], device=self.device)

        curve_following_mask = (sampling_modes == 0)
        in_place_mask = (sampling_modes == 1)
        complete_stop_mask = (sampling_modes == 2)
        
        curve_following_lin_vel = torch_rand_float(*kwargs["curve_following_lin_vel"], (len(env_ids), 1), device=self.device).squeeze(1) 
        curve_following_heading = torch_rand_float(*kwargs["curve_following_heading"], (len(env_ids), 1), device=self.device).squeeze(1) 

        in_place_lin_vel = torch_rand_float(*kwargs["in_place_lin_vel"], (len(env_ids), 1), device=self.device).squeeze(1) 
        in_place_heading = torch_rand_float(*kwargs["in_place_heading"], (len(env_ids), 1), device=self.device).squeeze(1) 

        complete_stop_lin_vel = torch.zeros(len(env_ids), device=self.device)
        complete_stop_heading = torch.zeros(len(env_ids), device=self.device)

        terrain_mask = (self.env_terrain_types[env_ids] != 7)
        flat_mask = (self.env_terrain_types[env_ids] == 7)

        self.commands[env_ids[terrain_mask], 0] = terrain_lin_vel[terrain_mask]
        self.commands[env_ids[terrain_mask], 3] = terrain_heading[terrain_mask]

        self.commands[env_ids[flat_mask & curve_following_mask], 0] = curve_following_lin_vel[flat_mask & curve_following_mask]
        self.commands[env_ids[flat_mask & curve_following_mask], 3] = curve_following_heading[flat_mask & curve_following_mask]

        self.commands[env_ids[flat_mask & in_place_mask], 0] = in_place_lin_vel[flat_mask & in_place_mask]
        self.commands[env_ids[flat_mask & in_place_mask], 3] = in_place_heading[flat_mask & in_place_mask]

        self.commands[env_ids[flat_mask & complete_stop_mask], 0] = complete_stop_lin_vel[flat_mask & complete_stop_mask]
        self.commands[env_ids[flat_mask & complete_stop_mask], 3] = complete_stop_heading[flat_mask & complete_stop_mask]

        # Sanity check
        assert(not torch.any(curve_following_mask & in_place_mask))
        assert(torch.all(curve_following_mask | in_place_mask | complete_stop_mask))

        assert(not torch.any(terrain_mask & flat_mask))
        assert(torch.all(flat_mask | terrain_mask))

    def _resample_commands_nav_loco_flat_only(self, env_ids):
        self._resample_commands_nav_loco(env_ids)

        default_command_mask = (self.env_terrain_types[env_ids] != 7).nonzero().squeeze()
        self.commands[env_ids[default_command_mask], 3] = 0
        self.commands[env_ids[default_command_mask], 2] = 0
        self.commands[env_ids[default_command_mask], 1] = 0
        self.commands[env_ids[default_command_mask], 0] = 0.35

    def _resample_commands_nav_loco_heading_flat_only(self, env_ids):
        sampling_modes = torch.randint(low=0, high=2, size=(len(env_ids),), device=self.device)

        if not self.keyboard_control_xvel:
            in_place_commands = torch_rand_float(0, 0.0, (len(env_ids), 1), device=self.device).squeeze(1)
            curve_follow_commands = torch_rand_float(0, 0.4, (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 0] = torch.where(sampling_modes == 0, in_place_commands, curve_follow_commands)
        
        if not self.keyboard_control_angvel: 
            assert(not self.keyboard_control_heading)
            assert(not self.keyboard_control_angvel)

            self.commands[env_ids, 3] = torch_rand_float(*self.cfg.commands.max_ranges.heading, (len(env_ids), 1), device=self.device).squeeze(1)

        default_command_mask = (self.env_terrain_types[env_ids] != 7).nonzero().squeeze()
        self.commands[env_ids[default_command_mask], 3] = 0 
        self.commands[env_ids[default_command_mask], 0] = 0.35

    def _resample_commands_nav_loco_clip_small_commands(self, env_ids):
        self._resample_commands_nav_loco(env_ids)
        self.commands[env_ids, :3] *= (torch.abs(self.commands[env_ids, :3]) > 0.1)

    def _resample_commands_flat_only(self, env_ids):
        if not self.keyboard_control_xvel:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if not self.cfg.commands.yvel_command:
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if not self.keyboard_control_heading: 
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # Set default commands on non-flat terrain
        default_command_mask = (self.env_terrain_types[env_ids] != 7).nonzero().squeeze()
        self.commands[env_ids[default_command_mask], 3] = 0
        self.commands[env_ids[default_command_mask], 1] = 0
        self.commands[env_ids[default_command_mask], 0] = 0.35

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _resample_commands_flat_only_binary(self, env_ids):
        if not self.keyboard_control_xvel:
            self.commands[env_ids, 0] = torch.tensor(self.command_ranges["lin_vel_x"])[torch.randint(low=0, high=2, size=(len(env_ids),))].to(self.device)
        
        if not self.cfg.commands.yvel_command:
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if not self.keyboard_control_heading: 
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # Set default commands on non-flat terrain
        default_command_mask = (self.env_terrain_types[env_ids] != 7).nonzero().squeeze()
        self.commands[env_ids[default_command_mask], 3] = 0
        self.commands[env_ids[default_command_mask], 1] = 0
        self.commands[env_ids[default_command_mask], 0] = 0.35

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        
        if self.cfg.commands.sampling_strategy == "standard":
            if not self.keyboard_control_xvel:
                self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            if not self.cfg.commands.yvel_command:
                self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            if not self.keyboard_control_heading and not self.keyboard_control_angvel: 
                if self.cfg.commands.heading_command:
                    self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        else:
            if hasattr(self.cfg.commands, "sampling_strategy_kwargs"):
                kwargs = self.cfg.commands.sampling_strategy_kwargs
            else:
                kwargs = {}

            exec("self._resample_commands_{}(env_ids, **kwargs)".format(self.cfg.commands.sampling_strategy))

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.p_gains*(self.motor_strength * actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            
            if self.cfg.env.randomize_start_positions: 
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center

            if self.cfg.init_state.randomize_init_yaw is not None:
                low, high = self.cfg.init_state.randomize_init_yaw
                random_angles = torch.rand(env_ids.shape[0], device=self.device) * (high - low) + low
                axis = torch.zeros((env_ids.shape[0], 3), device=self.device)
                axis[:, 2] = 1
                self.root_states[env_ids, 3:7] = quat_from_angle_axis(random_angles, axis)

        else: 
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.initial_root_states[env_ids] = self.root_states[env_ids].clone()

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:10] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_terrain_types[env_ids] = self.terrain_indices[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if "tracking_lin_vel_og" in self.episode_sums.keys():
            if torch.mean(self.episode_sums["tracking_lin_vel_og"][env_ids]) / self.max_episode_length > 0.7 * self.reward_scales["tracking_lin_vel_og"]:
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - self.cfg.commands.crclm_increment.lin_vel_x, -self.cfg.commands.max_ranges.lin_vel_x[0], 0.)
                self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + self.cfg.commands.crclm_increment.lin_vel_x, 0., self.cfg.commands.max_ranges.lin_vel_x[1])
            
                self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - self.cfg.commands.crclm_increment.lin_vel_y, -self.cfg.commands.max_ranges.lin_vel_y[0], 0.)
                self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + self.cfg.commands.crclm_increment.lin_vel_y, 0., self.cfg.commands.max_ranges.lin_vel_y[1])
        
        if "tracking_ang_vel_og" in self.episode_sums.keys():
            if torch.mean(self.episode_sums["tracking_ang_vel_og"][env_ids]) / self.max_episode_length > 0.7 * self.reward_scales["tracking_ang_vel_og"]:
                self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - self.cfg.commands.crclm_increment.ang_vel_yaw, -self.cfg.commands.max_ranges.ang_vel_yaw[0], 0.)
                self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + self.cfg.commands.crclm_increment.ang_vel_yaw, 0., self.cfg.commands.max_ranges.ang_vel_yaw[1])

                self.command_ranges["heading"][0] = np.clip(self.command_ranges["heading"][0] - self.cfg.commands.crclm_increment.heading, -self.cfg.commands.max_ranges.heading[0], 0.)
                self.command_ranges["heading"][1] = np.clip(self.command_ranges["heading"][1] + self.cfg.commands.crclm_increment.heading, 0., self.cfg.commands.max_ranges.heading[1])

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        force_sensor_tensor_ = self.gym.acquire_force_sensor_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
            
        # create some wrapper tensors for different slices
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor_)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor)
        self.pose_at_last_depth = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, 0, :7]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.env.history_nsteps is not None:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_nsteps, self.proprio_obs_end-self.proprio_obs_start, device=self.device, dtype=torch.float)
        
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.measured_clean_heights = 0
        self.max_foot_impact = torch.zeros(self.num_envs, device=self.device)
        self.min_edge_distance = torch.ones(self.num_envs, device=self.device) * self.cfg.env.feet_distance_soft_limit

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True

                    if hasattr(self.cfg.control, "stiffness_delta"):
                        stiffness_delta = self.cfg.control.stiffness_delta
                        self.p_gains[:, i] += np.random.uniform(-stiffness_delta, stiffness_delta)

                    if hasattr(self.cfg.control, "damping_delta"):
                        damping_delta = self.cfg.control.damping_delta
                        self.d_gains[:, i] += np.random.uniform(-damping_delta, damping_delta)
                        
            assert(found)

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if not hasattr(self.cfg.env, "camera_config"):
            self.num_cameras = 1
        else:
            self.num_cameras = len(self.cfg.env.camera_config)

        if self.cfg.env.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.num_depth_frames, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0] * self.num_cameras).to(self.device)
            
            self._depth_buffer = self.depth_buffer.clone()
        else:
            self.depth_buffer = None
            self._depth_buffer = None

        self.create_calibration_buffer()

        if hasattr(self.cfg.env, "use_prop_history_buf") and self.cfg.env.use_prop_history_buf > 0:
            self.prop_history_buf = torch.zeros(self.num_envs, self.cfg.depth.num_depth_frames, 32).to(self.device)
        else:
            self.prop_history_buf = None

        self.action_history_buffer = torch.zeros(self.num_envs, 10, 12).to(self.device)

        self.scandots_update_step = self.global_counter
        self.set_height_update_interval()

        if hasattr(self.cfg.env, "depth_update_dt"):
            self.set_depth_update_interval()
            self.depth_update_step = self.global_counter

    def create_calibration_buffer(self):
        if not hasattr(self.cfg.depth, "calibrate_duration"):
            self.calibration_depth_buffer = None
            self.is_calibrated = None
            return
        
        self.calibrate_steps = int(self.cfg.depth.calibrate_duration / (self.cfg.control.decimation * self.cfg.sim.dt))
        self.calibration_depth_buffer = torch.zeros(self.num_envs, 1, 
                                                    self.cfg.depth.resized[1],
                                                    self.num_cameras * self.cfg.depth.resized[0]).to(self.device)
        self.is_calibrated = torch.zeros(self.num_envs).bool().to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """

        self.named_rew_buf = {} # Store each reward separately for debugging

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                if not hasattr(self.cfg.rewards, "scale_by_time"):
                    self.cfg.rewards.scale_by_time = True
                
                if self.cfg.rewards.scale_by_time:
                    self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            self.named_rew_buf[name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            name = '_reward_' + name

            if "_flat" in name:
                name = name[:-5]

            if "_nonflat" in name:
                name = name[:-8]

            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.env.use_camera:
            if not hasattr(self.cfg.env, "camera_config"):
                # backward compatibility
                camera_props = gymapi.CameraProperties()
                camera_props.width = self.cfg.depth.original[0]
                camera_props.height = self.cfg.depth.original[1]
                camera_horizontal_fov = self.cfg.depth.horizontal_fov 

                if hasattr(self.cfg.depth, "horizontal_fov_delta"):
                    camera_horizontal_fov += np.random.uniform(-self.cfg.depth.horizontal_fov_delta, self.cfg.depth.horizontal_fov_delta)

                    camera_props.horizontal_fov = camera_horizontal_fov

                camera_props.enable_tensors = True
                
                camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.cam_handles[i].append(camera_handle)
                
                local_transform = gymapi.Transform()
                local_transform.p = gymapi.Vec3(0.50, 0, 0.2)
                local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(90))
                root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
                
                self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

                self.cam_params[i].append({
                    "position": local_transform.p,
                    "vertical_rot": np.radians(90),
                    "horizontal_rot": 0,
                    "horizontal_fov": camera_props.horizontal_fov,
                    "vertical_fov": camera_props.horizontal_fov * camera_props.height / camera_props.width
                })
            else:
                for config in self.cfg.env.camera_config:
                    camera_props = gymapi.CameraProperties()
                    camera_props.width = self.cfg.depth.original[0]
                    camera_props.height = self.cfg.depth.original[1]
                    camera_props.enable_tensors = True
                    
                    camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                    self.cam_handles[i].append(camera_handle)
                    
                    local_transform = gymapi.Transform()
                    
                    camera_position = deepcopy(config["position"])

                    if "position_delta" in config.keys():
                        for j, delta in enumerate(config["position_delta"]):
                            camera_position[j] = camera_position[j] + random.uniform(-delta, delta)

                    camera_angle = config["angle"]
                    
                    if "angle_delta" in config.keys():
                        camera_angle += np.random.uniform(-config["angle_delta"], config["angle_delta"])

                    if "h_angle" not in config.keys():
                        h_angle = 0

                    if "h_angle_delta" in config.keys():
                        h_angle += np.random.uniform(-config["h_angle_delta"], config["h_angle_delta"])

                    local_transform.p = gymapi.Vec3(*camera_position)
                    local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), np.radians(h_angle))
                    root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
                    
                    self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

                    self.cam_params[i].append({
                        "position": camera_position,
                        "vertical_rot": np.radians(camera_angle),
                        "horizontal_rot": np.radians(h_angle),
                        "horizontal_fov": np.radians(camera_props.horizontal_fov),
                        "vertical_fov": np.radians(camera_props.horizontal_fov * camera_props.height / camera_props.width)
                    })

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        # Add contact sensors to each foot -- this is the reliable way of getting foot contacts
        self.sensor_indices = []
        self.raisim_feet_indices = []

        for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            self.raisim_feet_indices.append(feet_idx)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
            self.sensor_indices.append(sensor_idx)

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = [[] for _ in range(self.num_envs)]
        self.cam_params = [[] for _ in range(self.num_envs)] # Camera parameters for viz
        self.mass_params = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            
            # attach camera to each env
            self.attach_camera(i, env_handle, anymal_handle)
            
            self.mass_params[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])  

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            if hasattr(self.cfg.terrain, "start_at_zero") and self.cfg.terrain.start_at_zero:
                self.terrain_levels = torch.zeros(self.num_envs, device=self.device).long()
            else:
                self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_heights = torch.from_numpy(self.terrain.env_terrain_height).to(self.device).float()
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.env_terrain_types = self.terrain_indices[self.terrain_levels, self.terrain_types]
            self.env_terrain_difficulty = self.terrain_difficulty[self.terrain_levels, self.terrain_types]
        else:
            raise NotImplemented

            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        
        if hasattr(self.cfg.rewards, "scales_flat"):
            for key, value in class_to_dict(self.cfg.rewards.scales_flat).items():
                key = key + "_flat"
                self.reward_scales[key] = value

        if hasattr(self.cfg.rewards, "scales_nonflat"):
            for key, value in class_to_dict(self.cfg.rewards.scales_nonflat).items():
                key = key + "_nonflat"
                self.reward_scales[key] = value
        
        self.doubling_terms = self.cfg.rewards.doubling_terms
        self.doubling_factors = self.cfg.rewards.doubling_factors

        if not hasattr(self.cfg.env, "add_history_to_obs"):
            self.cfg.env.add_history_to_obs = False

        if not hasattr(self.cfg.env, "history_nsteps"):
            self.cfg.env.history_nsteps = None

        if hasattr(self.cfg.rewards, "decay_period"):
            self.decay_period = self.cfg.rewards.decay_period
        else:
            self.decay_period = 10000
        if self.cfg.commands.curriculum:
            self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        else:
            self.command_ranges = class_to_dict(self.cfg.commands.max_ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
    
        if not hasattr(self.cfg.terrain, "measure_rays"):
            self.cfg.terrain.measure_rays = False

        if not hasattr(self.cfg.terrain, "normalize_rays"):
            self.cfg.terrain.normalize_rays = False

        if not hasattr(self.cfg.terrain, "normalize_rays_static"):
            self.cfg.terrain.normalize_rays_static = False

        if not hasattr(self.cfg.terrain, "scale_rays"):
            self.cfg.terrain.scale_rays = False

        if not hasattr(self.cfg.env, "randomize_start_positions"):
            self.cfg.env.randomize_start_positions = True

        if not hasattr(self.cfg.commands, "clip_lin_vel_x"):
            self.cfg.commands.clip_lin_vel_x = None

        if not hasattr(self.cfg.commands, "clip_heading"):
            self.cfg.commands.clip_heading = None

        if not hasattr(self.cfg.depth, "salt_pepper_noise"):
            self.cfg.depth.salt_pepper_noise = False
        
        if not hasattr(self.cfg.env, "feet_distance_soft_limit"):
            self.cfg.env.feet_distance_soft_limit = 0.1

        if hasattr(self.cfg.env, "height_update_latency"):
            self.height_update_latency = [0, 0]
            for i in range(2):
                self.height_update_latency[i] = int(self.cfg.env.height_update_latency[i] / self.cfg.control.decimation / self.cfg.sim.dt)
        else:
            self.height_update_latency = [0, 0]

        if not hasattr(self.cfg.env, "include_terrain_height"):
            self.cfg.env.include_terrain_height = False
        if hasattr(self.cfg.env, "depth_update_latency"):
            self.depth_update_latency = [0, 0]
            for i in range(2):
                self.depth_update_latency[i] = int(self.cfg.env.depth_update_latency[i] / self.cfg.control.decimation / self.cfg.sim.dt)            
        else:
            self.depth_update_latency = [0, 0]

        if not hasattr(self.cfg.env, "scandots_timing_dim"):
            self.cfg.env.scandots_timing_dim = 0

        if not hasattr(self.cfg.depth, "normalize"):
            self.cfg.depth.normalize = False
        
        if not hasattr(self.cfg.depth, "normalize_mean"):
            self.cfg.depth.normalize_mean = None
        
        if not hasattr(self.cfg.env, "check_feet_on_edge"):
            self.cfg.env.check_feet_on_edge = False
        
        if not hasattr(self.cfg.env, "edge_threshold"):
            self.cfg.env.edge_threshold = 0.16

        if not hasattr(self.cfg.env, "visual_blackout_mode"):
            self.cfg.env.visual_blackout_mode = "noise"

        if not hasattr(self.cfg.depth, "normalize_std"):
            self.cfg.depth.normalize_std = None

        if not hasattr(self.cfg.depth, "invert"):
            self.cfg.depth.invert = False

        if not hasattr(self.cfg.depth, "patch_selection_prob"):
            self.cfg.depth.patch_selection_prob = 0

        if not hasattr(self.cfg.depth, "patch_types"):
            self.cfg.depth.patch_types = ["white"]

        if not hasattr(self.cfg.commands, "heading_command_controller"):
            self.cfg.commands.heading_command_controller = PIDController(Kp=0.5, Kd=0, Ki=0)

        if not hasattr(self.cfg.depth, "noise_scale"):
            self.cfg.depth.noise_scale = 0

        if hasattr(self.cfg.noise, "scandots_noise_sampler"):
            self.scandots_noise_sampler = self.cfg.noise.scandots_noise_sampler
        elif hasattr(self.cfg.noise, "config"):
            self.scandots_noise_sampler = construct_scandots_noise_sampler(self.cfg.noise.config)
        else:
            self.scandots_noise_sampler = NoOpSampler() 

        if not hasattr(self.cfg.terrain, "heights_offset"):
            self.heights_offset = 0.5
        else:
            self.heights_offset = self.cfg.terrain.heights_offset

        if not hasattr(self.cfg.init_state, "randomize_init_yaw"):
            self.cfg.init_state.randomize_init_yaw = None

        if not hasattr(self.cfg.commands, "keyboard_control_heading"):
            self.keyboard_control_heading = False
        else:
            self.keyboard_control_heading = self.cfg.commands.keyboard_control_heading

        if not hasattr(self.cfg.commands, "keyboard_control_angvel"):
            self.keyboard_control_angvel = False
        else:
            self.keyboard_control_angvel = self.cfg.commands.keyboard_control_angvel

        if not hasattr(self.cfg.commands, "keyboard_control_xvel"):
            self.keyboard_control_xvel = False
        else:
            self.keyboard_control_xvel = self.cfg.commands.keyboard_control_xvel

        if not hasattr(self.cfg.commands, "default_keyboard_vel"):
            self.default_keyboard_vel = 0.35
        else:
            self.default_keyboard_vel = self.cfg.commands.default_keyboard_vel

        if not hasattr(self.cfg.env, "terminate_xvel"):
            self.terminate_xvel = False
        else:
            self.terminate_xvel = self.cfg.env.terminate_xvel
        
        if not hasattr(self.cfg.asset, "feet_in_gap_cutoff"):
            self.cfg.asset.feet_in_gap_cutoff = -0.05

        if not hasattr(self.cfg.commands, "sampling_strategy"):
            self.cfg.commands.sampling_strategy = "standard"

        if not hasattr(self.cfg.asset, "max_nonzero_vel_steps"):
            self.nonzero_vel_steps_threshold = float('inf')
        else:
            self.nonzero_vel_steps_threshold = self.cfg.asset.max_nonzero_vel_steps

        if not hasattr(self.cfg.noise, "height_quantization"):
            self.cfg.noise.height_quantization = None

        if not hasattr(self.cfg.commands, "yvel_command"):
            self.cfg.commands.yvel_command = False
        
        if not hasattr(self.cfg.domain_rand, "push_feet"):
            self.cfg.domain_rand.push_feet = False
        elif self.cfg.domain_rand.push_feet:
            self.cfg.domain_rand.push_feet_interval = np.ceil(self.cfg.domain_rand.push_feet_interval_s / self.dt)

        if not hasattr(self.cfg.env, "visual_blackout_mean_life"):
            self.visual_blackout_mean_life = None
        else:
            self.visual_blackout_mean_life = self.cfg.env.visual_blackout_mean_life

        if not hasattr(self.cfg.env, "visual_blackout_mean_life_on"):
            # Half life with which depth is re-enabled
            self.visual_blackout_mean_life_on = self.visual_blackout_mean_life
        else:
            self.visual_blackout_mean_life_on = self.cfg.env.visual_blackout_mean_life_on

        if not hasattr(self.cfg.rewards, "foot_impact_soft_limit"):
            self.cfg.rewards.foot_impact_soft_limit = 0 

        if not hasattr(self.cfg.depth, "save_images"):
            self.cfg.depth.save_images = False
        elif self.cfg.depth.save_images:
            os.system("rm -rf depth_images")
            os.system("mkdir depth_images")

    def rotate_by_angle(self, c1, c2, angle):
        c1 = c1 * cos(angle) - c2 * sin(angle)
        c2 = c1 * sin(angle) + c2 * cos(angle)

        return c1, c2

    def compute_ray_dir(self, h_angle, v_angle):
        vx = cos(h_angle) * cos(v_angle)
        vy = sin(h_angle)
        vz = -cos(h_angle) * sin(v_angle)

        return np.array([vx, vy, vz])

    def draw_rays(self):
        """ 
            Draw rays as specified in the config
        """

        if not self.cfg.terrain.measure_rays:
            return

        ray_origin = np.array(self.cfg.terrain.ray_origin)
        start_point_list = []
        ray_dir_list = []
        num_rays_per_env = len(self.cfg.terrain.measured_horizontal_angles) * len(self.cfg.terrain.measured_vertical_angles)

        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            ray_origin_transformed = quat_apply(self.base_quat[i], torch.from_numpy(ray_origin).float().to(self.device))
            ray_origin_transformed = ray_origin_transformed.cpu().numpy()
            start_point = ray_origin_transformed + base_pos

            for h_angle in self.cfg.terrain.measured_horizontal_angles:
                for v_angle in self.cfg.terrain.measured_vertical_angles:
                    ray_dir = self.compute_ray_dir(h_angle, v_angle)
                    ray_dir = quat_apply(self.base_quat[i], torch.from_numpy(ray_dir).float().to(self.device))
                    ray_dir = ray_dir.cpu().numpy()

                    start_point_list.append(start_point)
                    ray_dir_list.append(ray_dir)

                    end_point = start_point + ray_dir

                    self.gym.add_lines(self.viewer, 
                                       self.envs[i], 
                                        1, 
                                        np.stack([start_point, end_point]).astype(np.float32), 
                                        np.array([1, 1, 0], dtype=np.float32))

        # Draw sphere geoms at intersection points
        start_point_list = np.stack(start_point_list)
        start_point_list = torch.from_numpy(start_point_list).to(self.device)
        ray_dir_list = np.stack(ray_dir_list)
        ray_dir_list = torch.from_numpy(ray_dir_list).to(self.device)
        intersection_dist = self.get_ray_intersection(start_point_list, ray_dir_list, self.cfg.terrain.ray_tol)
        intersection_points = start_point_list + intersection_dist[:, None] * ray_dir_list

        for i in range(self.num_envs):
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))

            for pt in intersection_points[i * num_rays_per_env: (i + 1) * num_rays_per_env]:
                sphere_pose = gymapi.Transform(gymapi.Vec3(*pt), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)


    def draw_height_lines(self):
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return

        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.heights[i].cpu().numpy() / self.cfg.normalization.obs_scales.height_measurements
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()

            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = base_pos[2] - (heights[j] + self.heights_offset)
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def draw_calibration_sphere(self):
        """Draw a sphere on the robot if calibration frame has been captured"""

        if self.calibration_depth_buffer is None:
            return

        uncalibrated_sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 0))
        calibrated_sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 0))
        
        for i in range(self.num_envs):
            if self.is_calibrated[i]:
                sphere_geom = calibrated_sphere_geom
            else:
                sphere_geom = uncalibrated_sphere_geom

            sphere_xyz = (self.root_states[i, :3]).cpu().numpy()
            sphere_xyz[2] += 0.2
            sphere_pose = gymapi.Transform(gymapi.Vec3(*sphere_xyz), r=None)

            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def draw_edges(self):
        red_sphere_geom = gymutil.WireframeSphereGeometry(self.cfg.terrain.horizontal_scale / 2, 4, 4, None, color=(1, 0, 0))
        green_sphere_geom = gymutil.WireframeSphereGeometry(self.cfg.terrain.horizontal_scale / 2, 4, 4, None, color=(0, 1, 0))
        contact_mask = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, 2] > 1
        contact_mask = contact_mask[:, [1, 0, 3, 2]] # Change ordering to match feet_indices ordering

        # Draw closest points
        for i in range(self.num_envs):
            for j in range(4):
                if torch.all(self.feet_closest_points[i, j] >= 0):
                    x, y = self.feet_closest_points[i, j]
                    x = x.item()
                    y = y.item()
                    
                    sphere_xyz = np.array([
                        x * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size,
                        y * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size,
                        self.height_samples[x, y].cpu().item() * self.cfg.terrain.vertical_scale
                    ])
                    sphere_pose = gymapi.Transform(gymapi.Vec3(*sphere_xyz), r=None)

                    if contact_mask[i, j]:
                        gymutil.draw_lines(green_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                    else:
                        gymutil.draw_lines(red_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        # if hasattr(self, "edges_rendered"):
        #     return 

        
    
        # for i, j in tqdm(np.ndindex(self.edge_mask.shape), desc="Drawing edges", total=self.edge_mask.shape[0] * self.edge_mask.shape[1]):
        #     if self.edge_mask[i][j]:
        #         sphere_xyz = np.array([
        #             i * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size,
        #             j * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size,
        #             self.height_samples[i, j].cpu().item() * self.cfg.terrain.vertical_scale
        #         ])

        #         sphere_pose = gymapi.Transform(gymapi.Vec3(*sphere_xyz), r=None)
        #         gymutil.draw_lines(green_sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

        # self.edges_rendered = True

    def _draw_debug_vis(self):
        """ 
            Draws visualizations for debugging (slows down simulation a lot).
            Default behaviour: draws height measurement points and spheres on robots which have captured calibration frames
        """

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.draw_height_lines()
        # self.draw_calibration_sphere()
        self.draw_rays()
        # self.draw_cameras()
        # self.draw_edges()

    def draw_cameras(self):
        for i in range(self.num_envs):
            for cam in self.cam_params[i]:
                # Draw this camera
                camera_position = np.array(cam["position"])
                camera_v_angle = cam["vertical_rot"]
                camera_h_angle = cam["horizontal_rot"]
                horizontal_fov = cam["horizontal_fov"]
                vertical_fov = cam["vertical_fov"]


                base_pos = (self.root_states[i, :3]).cpu().numpy()
                cam_origin_transformed = quat_apply(self.base_quat[i], torch.from_numpy(camera_position).float().to(self.device))
                cam_origin_transformed = cam_origin_transformed.cpu().numpy()
                start_point = cam_origin_transformed + base_pos

                for v_dir in [-1, 1]:
                    for h_dir in [-1, 1]:
                        v_angle = camera_v_angle + v_dir * vertical_fov / 2
                        h_angle = camera_h_angle + h_dir * horizontal_fov / 2

                        ray_dir = self.compute_ray_dir(h_angle, v_angle)
                        ray_dir = quat_apply(self.base_quat[i], torch.from_numpy(ray_dir).float().to(self.device))
                        ray_dir = ray_dir.cpu().numpy()

                        end_point = start_point + ray_dir

                        self.gym.add_lines(self.viewer, 
                                           self.envs[i], 
                                           1, 
                                           np.stack([start_point, end_point]).astype(np.float32), 
                                           np.array([1, 1, 0], dtype=np.float32))



    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_foot_contacts(self):
        self.foot_contacts_from_sensor = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, 2] > 1

        if hasattr(self.cfg.env, "include_foot_contacts") and self.cfg.env.include_foot_contacts:
            return self.foot_contacts_from_sensor
        else:
            return torch.zeros(self.num_envs, 0).to(self.device)

    # def world2terrain_cooords(self, points):
    #     """
    #         points has shape (self.num_envs, K, 2)
    #     """

    #     E, num_points, C = points.shape

    #     assert(E == self.num_envs)
    #     assert(C == 2)

    #     points = points + self.terrain.cfg.border_size
    #     points = (points/self.terrain.cfg.horizontal_scale).long()

    #     px = points[:, :, 0].view(-1)
    #     py = points[:, :, 1].view(-1)
    #     px = torch.clip(px, 0, self.height_samples.shape[0]-2)
    #     py = torch.clip(py, 0, self.height_samples.shape[1]-2)

    #     points_transformed = torch.stack((
    #         px.reshape((self.num_envs, num_points)),
    #         py.reshape((self.num_envs, num_points)),
    #     ), dim =- 1)

    #     assert(points_transformed.shape == (self.num_envs, num_points, 2))

    #     return points   

    def get_heights_at_abs_points(self, points):
        points = points + self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        return heights

    def get_ray_intersection(self, points, dirs, tol):
        """
            Do binary search to find the t such that points + t * dirs intersects terrain
            `points` and `dirs` must be in global frame of reference
        """

        batch_size = points.shape[0]
        low = torch.zeros(batch_size).to(self.device)
        high = torch.ones(batch_size).to(self.device) * 10

        while torch.any(high - low >= tol):
            mid = (high + low) / 2
            coords = points + mid[:, None] * dirs
            heights = self.get_heights_at_abs_points(coords[None, :]).flatten()

            # Set low = mid if point coords is above hmap
            low = torch.where(coords[:, 2] > heights, mid, low)

            # Set high = mid if point is below hmap
            high = torch.where(coords[:, 2] <= heights, mid, high)

        return low

    def get_heights_at_points(self, points):
        points = quat_apply_yaw(self.base_quat.repeat(1, points.shape[1]), points) + (self.root_states[:, :3]).unsqueeze(1)

        # Add noise to the points if required
        if hasattr(self.cfg.noise.noise_scales, "height_measurements_horizontal"):
            points += torch.randn_like(points) * self.cfg.noise.noise_scales.height_measurements_horizontal

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights(self):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """

        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        
        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # Add noise to the points if required
        points = self.scandots_noise_sampler.add_horizontal_noise(points)
        # self.noisy_height_points = quat_apply_yaw_inverse(self.base_quat.repeat(1, self.num_height_points), points - self.root_states[:, :3].unsqueeze(1))
        # self.noisy_height

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_clean_heights(self):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """

        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        
        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # Add noise to the points if required
        # points = self.scandots_noise_sampler.add_horizontal_noise(points)
        # self.noisy_height_points = quat_apply_yaw_inverse(self.base_quat.repeat(1, self.num_height_points), points - self.root_states[:, :3].unsqueeze(1))
        # self.noisy_height

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def get_baseline_distances(self, ray_origin_transformed, ray_dirs_transformed):
        """
            Get the ray distances assuming the ground is flat 
        """

        return -(ray_origin_transformed[:, 2] / ray_dirs_transformed[:, 2])


    def _get_ray_distances(self):
        if not self.cfg.terrain.measure_rays:
            return

        num_rays_per_env = len(self.cfg.terrain.measured_horizontal_angles) * len(self.cfg.terrain.measured_vertical_angles)

        if not hasattr(self, "ray_origin"):
            self.ray_origin = torch.tensor(self.cfg.terrain.ray_origin).to(self.device)
            self.ray_origin = self.ray_origin.repeat(self.num_envs, 1) # (num_envs, 3)

            # Add systematic noise to ray_origin
            if hasattr(self.cfg.terrain, "ray_origin_delta"):
                ray_origin_delta = torch.tensor(self.cfg.terrain.ray_origin_delta).to(self.device) # (3)
                ray_origin_delta = (2 * torch.rand((self.num_envs, 3)) - 1) * ray_origin_delta[None, :] # (num_envs, 3)
                self.ray_origin += ray_origin_delta 

        if not hasattr(self, "ray_dirs"):
            self.ray_dirs = []

            for i in range(self.num_envs):
                if hasattr(self.cfg.terrain, "measured_angle_delta_vertical"):
                    delta = self.cfg.terrain.measured_angle_delta_vertical
                    v_angle_delta = np.random.uniform(-delta, delta)
                else:
                    v_angle_delta = 0
                
                if hasattr(self.cfg.terrain, "measured_angle_delta_horizontal"):
                    delta = self.cfg.terrain.measured_angle_delta_horizontal
                    h_angle_delta = np.random.uniform(-delta, delta)
                else:
                    h_angle_delta = 0

                for h_angle in self.cfg.terrain.measured_horizontal_angles:
                    for v_angle in self.cfg.terrain.measured_vertical_angles:
                        ray_dir = self.compute_ray_dir(h_angle + h_angle_delta, v_angle + v_angle_delta)
                        self.ray_dirs.append(ray_dir)

            self.ray_dirs = torch.from_numpy(np.stack(self.ray_dirs)).float().to(self.device) # (num_rays * num_envs, 3)

        base_pos = self.root_states[:, :3] # (num_envs, 3)
        ray_origin_transformed = quat_apply(self.base_quat, self.ray_origin)
        ray_origin_transformed = ray_origin_transformed + base_pos # (num_envs, 3)
        ray_origin_transformed = ray_origin_transformed.repeat_interleave(num_rays_per_env, dim=0) # (num_rays * num_envs, 3)

        ray_dirs_transformed = quat_apply(self.base_quat.repeat_interleave(num_rays_per_env, dim=0), self.ray_dirs) # (num_rays * num_envs, 3)
        intersection_dist = self.get_ray_intersection(ray_origin_transformed, ray_dirs_transformed, self.cfg.terrain.ray_tol)
        
        if self.cfg.terrain.normalize_rays:
            baseline_dist = self.get_baseline_distances(ray_origin_transformed, ray_dirs_transformed)
            intersection_dist = intersection_dist - baseline_dist # This is done to normalize observations

        if self.cfg.terrain.normalize_rays_static:
            if not hasattr(self, "static_baseline_dist"):
                # Normalize using baseline_dist with default parameters
                static_ray_origin = torch.zeros(num_rays_per_env * self.num_envs, 3).cuda()
                static_ray_origin[:, 2] = 0.28 + self.cfg.terrain.ray_origin[2]
                self.static_baseline_dist = self.get_baseline_distances(static_ray_origin, self.ray_dirs)
                
            intersection_dist = intersection_dist - self.static_baseline_dist

        if self.cfg.terrain.scale_rays:
            # Scale rays by z component of ray_dirs so that same distances correspond to same vertical displacement
            intersection_dist = intersection_dist * ray_dirs_transformed[:, 2]

        return intersection_dist.reshape((self.num_envs, num_rays_per_env))

    #------------ reward functions----------------

    # Energy loco reward functions
    def _reward_energy_loco_energy_clipped(self):
       return torch.clip(torch.sum(self.torques * self.dof_vel, dim = 1), min=0.0) 

    def _reward_feet_edge_distance(self):
        return torch.sum(self.cfg.env.feet_distance_soft_limit - self.get_distance_to_edge(), dim=-1) # If distance is large this is small

    def _reward_energy_loco_energy(self):
        return torch.sum(self.torques * self.dof_vel, dim = 1)

    def _reward_energy_loco_track_linvel_x(self):
        return torch.abs(self.base_lin_vel[:, 0] - self.commands[:, 0])

    def _reward_energy_loco_track_linvel_y(self):
        return self.base_lin_vel[:, 1] ** 2

    def _reward_foot_clearance(self):
        clearance_threshold = 0.1
        # Order is [FL, FR, RL, RR]
        feet_xy_pos = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 0:2]
        feet_z_pos = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 2]
        
        terrain_ht_under_feet = self.get_heights_at_abs_points(feet_xy_pos)
        
        self.foot_clearance = (feet_z_pos - terrain_ht_under_feet)
        self.foot_clearance = torch.clip(self.foot_clearance, max=clearance_threshold)

        return torch.sum(self.foot_clearance, dim=1)
    
    def _reward_energy_loco_track_angvel_yaw(self):
        return self.base_ang_vel[:, 2] ** 2

    def _reward_foot_impact(self):
        last_force = self.last_force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, :3]
        curr_force = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, :3]

        delta_force = torch.square(torch.norm(curr_force - last_force, dim=-1))
        total_impact = torch.sum(delta_force, dim=-1)

        reward = torch.clip(total_impact - self.cfg.rewards.foot_impact_soft_limit, min=0.0)
        reward = reward * (self.episode_length_buf >= 100)

        self.max_foot_impact = torch.maximum(self.max_foot_impact, total_impact * (self.episode_length_buf >= 100))
    
        return reward

    def _reward_foot_impact_linear(self):
        last_force = self.last_force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, :3]
        curr_force = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, :3]

        delta_force = torch.norm(curr_force - last_force, dim=-1)
        total_impact = torch.sum(delta_force, dim=-1)

        reward = torch.clip(total_impact - self.cfg.rewards.foot_impact_soft_limit, min=0.0)
        reward = reward * (self.episode_length_buf >= 100)

        self.max_foot_impact = torch.maximum(self.max_foot_impact, total_impact * (self.episode_length_buf >= 100))
    
        return reward

    def _reward_feet_in_gap(self):
        depth_threshold = -0.5
        feet_xy_pos = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 0:2]
        feet_z_pos = self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 2]
        
        terrain_ht_under_feet = self.get_heights_at_abs_points(feet_xy_pos) - self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, 0, 2][:, None]
        feet_in_gap_mask = (feet_z_pos < 0.05) * (terrain_ht_under_feet <= depth_threshold)
        reward = (-feet_z_pos) * feet_in_gap_mask

        return torch.sum(reward, dim=1)

    def _reward_energy_loco_alive(self):
        return self.commands[:, 0]

    def _reward_energy_loco(self):
        alpha_1 = 0.04
        alpha_2 = 20
        
        # Get reward from energy locomotion paper
        r_forward = 0
        r_forward -= alpha_2 * torch.abs(self.base_lin_vel[:, 0] - self.commands[:, 0])
        r_forward -= self.base_lin_vel[:, 1] ** 2
        r_forward -= self.base_ang_vel[:, 2] ** 2

        r_alive = 20 * self.commands[:, 0]

        r_energy = -torch.abs(torch.sum(self.torques * self.dof_vel, dim = 1))

        return r_forward + alpha_1 * r_energy + r_alive

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_lin_vel_y(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_yshift(self):
        return torch.abs(self.root_states[:, 1] - self.initial_root_states[:, 1])

    def _reward_hip_action_inward(self):
        # Penalize inward hip movement
        if not hasattr(self, "hip_indices"):
            self.hip_indices = []
            hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]

            for name in hip_names:
                self.hip_indices.append(self.dof_names.index(name))

        hip_action = self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]

        # Only positive values penalized since positive rot corresponds to inward
        hip_action[:, [0, 2]] = torch.clip(hip_action[:, [0, 2]], min=0.0) 
        # Only negative values penalized since negative rot corresponds to inward
        hip_action[:, [1, 3]] = torch.clip(hip_action[:, [1, 3]], max=0.0) 

        return torch.sum(torch.square(hip_action), dim=1)

    def _reward_hip_action(self):
        if not hasattr(self, "hip_indices"):
            self.hip_indices = []
            hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]

            for name in hip_names:
                self.hip_indices.append(self.dof_names.index(name))

        hip_action = self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]

        return torch.sum(torch.square(hip_action), dim=1)

    def _reward_alive_bonus(self):
        return torch.ones(self.num_envs, device=self.device)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_heading(self):
        # Penalize not heading in straight direction
        return torch.square(self.get_body_orientation(return_yaw=True)[:, -1])

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_sideways_penalty(self):
        return torch.norm(self.actions[:,[0,3,6,9]], dim=-1)

    def _reward_act_penalty(self):
        return torch.norm(self.actions, dim=-1)

    def _reward_act_penalty_complete_stop(self):
        reward = torch.norm(self.actions, dim=-1)
        complete_stop_mask = torch.abs(self.commands[:, 0]) < 0.05
        complete_stop_mask &= torch.abs(self.commands[:, 2]) < 0.1 
        reward[~complete_stop_mask] = 0

        return reward

    def _reward_act_penalty_complete_stop_l1(self):
        reward = torch.sum(torch.abs(self.actions), dim=-1)
        complete_stop_mask = torch.abs(self.commands[:, 0]) < 0.05
        complete_stop_mask &= torch.abs(self.commands[:, 2]) < 0.1 
        reward[~complete_stop_mask] = 0

        return reward

    def _reward_act_penalty_squared(self):
        return torch.square(torch.norm(self.actions, dim=-1))

    def _reward_act_penalty_zero_command(self):
        penalty = torch.norm(self.actions, dim=-1)
        penalty[self.commands[:, 0] >= 1e-3] = 0 # Don't penalize for nonzero vel

        return penalty

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_delta_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_work(self):
        # Penalize energy
        return torch.abs(torch.sum(self.torques * self.dof_vel, dim = 1))

    def _reward_work_actual(self):
        # Penalize energy
        return torch.sum(self.torques * self.dof_vel, dim = 1)

    def _reward_work_clipped(self):
        work = torch.sum(self.torques * self.dof_vel, dim = 1)
        work_clipped = torch.maximum(torch.zeros_like(work), work)

        return work_clipped
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel_square(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.minimum(self.commands[:, 0], self.base_lin_vel[:, 0]).clip(min=-0.1) ** 2

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        return torch.minimum(self.commands[:, 0], self.base_lin_vel[:, 0]).clip(min=-0.1)
    
    def _reward_tracking_lin_vel_y(self):
        return torch.abs(self.commands[:, 1] - self.base_lin_vel[:, 1]).clip(min=-0.1)

    def _reward_tracking_lin_vel_v2(self):
        # Tracking of linear velocity commands (xy axes)
        return torch.minimum(self.commands[:, 0], self.base_lin_vel[:, 0]).clip(min=-0.1) - self.commands[:, 0]

    def _reward_tracking_lin_vel_og(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_start_stop(self):
        # Tracking of linear velocity commands (xy axes) - if commands is less than a certain value just stop
        large_command_rew = torch.minimum(self.commands[:, 0], self.base_lin_vel[:, 0]).clip(min=-0.1)
        small_command_rew = self.cfg.commands.max_ranges.lin_vel_x[1] - torch.abs(self.base_lin_vel[:, 0])
        
        rewards = torch.where(self.commands[:, 0] < 0.05, small_command_rew, large_command_rew)

        return rewards

    def _reward_tracking_lin_vel_start_stop_v2(self):
        # Tracking of linear velocity commands (xy axes) - if commands is less than a certain value just stop
        large_command_rew = torch.minimum(self.commands[:, 0], self.base_lin_vel[:, 0]).clip(min=-0.1) - self.commands[:, 0]
        small_command_rew = -torch.abs(self.base_lin_vel[:, 0])
        
        rewards = torch.where(self.commands[:, 0] < 0.05, small_command_rew, large_command_rew)

        return rewards

    def _reward_tracking_lin_vel_start_stop_v3(self):
        # Tracking of linear velocity commands (xy axes) - if commands is less than a certain value just stop
        large_command_rew = torch.minimum(self.commands[:, 0], self.base_lin_vel[:, 0]).clip(min=-0.1) - self.commands[:, 0]
        small_command_rew = -torch.abs(self.base_lin_vel[:, 0]) * 5
        
        rewards = torch.where(self.commands[:, 0] < 0.05, small_command_rew, large_command_rew)

        return rewards
            
    def _reward_tracking_lin_vel_exact(self):
        lin_vel_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return lin_vel_error

    def _reward_tracking_lin_vel_exact_scaled(self):
        lin_vel_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return lin_vel_error / (self.commands[:, 0] + 1e-2)
    
    def _reward_tracking_ang_vel_og(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        return torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # Tracking of angular velocity commands (yaw) 
        # ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_drag(self):
        # Penalize if feet is in contact with ground and has nonzero velocity
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # contact_filt = torch.logical_or(contact, self.last_contacts) 
        # self.last_contacts = contact
        contact = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, 2] > 1

        feet_xy_vel = torch.abs(self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 7:9]).sum(dim=-1)
        dragging_vel = contact * feet_xy_vel
        return dragging_vel.sum(dim=-1) 

    def _reward_feet_drag_rma(self):
        # Penalize if feet is in contact with ground and has nonzero velocity
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # contact_filt = torch.logical_or(contact, self.last_contacts) 
        # self.last_contacts = contact
        contact = self.force_sensor_tensor.reshape((self.num_envs, 4, 6))[:, :, 2] > 1

        feet_xy_vel = torch.norm(self.rigid_body_states.reshape((self.num_envs, -1, 13))[:, self.feet_indices, 7:9], dim=-1)
        feet_xy_vel = torch.square(feet_xy_vel)
        dragging_vel = contact * feet_xy_vel
        dragging_vel = dragging_vel.sum(dim=-1) 

        # Don't penalize at the first second of episode to allow time to settle down
        dragging_vel = dragging_vel * (self.episode_length_buf > 100)

        return dragging_vel

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_jerk(self):
        # Penalize high jerk (causes damage to joints)
        if not hasattr(self, "last_contact_forces"):            
            result = torch.zeros(self.num_envs).to(self.device)
        else:
            result = torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :] - self.last_contact_forces, dim=-1), dim=-1)
        
        self.last_contact_forces = self.contact_forces[:, self.feet_indices, :].clone()

        return result

    def _reward_vel_matching_nav_loco(self):
        xvel_matching = torch.abs(self.base_lin_vel[:, 0] - self.commands[:, 0])
        ang_vel_matching = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])

        return ang_vel_matching + xvel_matching

    def _reward_vel_matching_nav_loco_v2(self): # This is a buggy reward when used in combination with heading_command = True
        

        # This one has the commands subtracted
        xvel_matching = torch.abs(self.base_lin_vel[:, 0] - self.commands[:, 0])
        ang_vel_matching = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])

        return torch.abs(self.commands[:, 0]) - ang_vel_matching - xvel_matching

    def _reward_tracking_lin_vel_exact_v2(self): 
        # This one has the commands subtracted
        xvel_matching = torch.abs(self.base_lin_vel[:, 0] - self.commands[:, 0])

        return 2 * torch.abs(self.commands[:, 0]) - xvel_matching

    def _reward_tracking_ang_vel_exact_v2(self): 
        # This one has the commands subtracted
        ang_vel_matching = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])

        return 3.14 - ang_vel_matching

    def _reward_lateral_movement_nav_loco(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_survival_bonus_nav_loco(self):
        v_x = torch.abs(self.base_lin_vel[:, 0])
        ang_vel_yaw = torch.abs(self.base_ang_vel[:, 2])

        return 10 + 20 * (v_x + ang_vel_yaw)

    def _reward_survival_bonus_nav_loco_v2(self):
        # This one does not include ang_vel_yaw in the bonus since that leads to jumping behavior 
        v_x = torch.abs(self.base_lin_vel[:, 0])
        ang_vel_yaw = torch.abs(self.base_ang_vel[:, 2])

        return 10 + 20 * (v_x)

    # Manip-loco reward functions
    def _reward_manip_loco_energy_square(self):
        energy = torch.sum(torch.square(self.torques[:, :12] * self.dof_vel[:, :12]), dim=1)

        return energy
    
    def _reward_manip_loco_survive(self):
        return torch.ones(self.num_envs, device=self.device)

    def _reward_manip_loco_tracking_lin_vel_x_l1(self):
        error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return - error + torch.abs(self.commands[:, 0])

    def _reward_manip_loco_tracking_ang_vel_yaw_exp(self):
        error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_manip_loco_hip_action_l2(self):
        action_l2 = torch.sum(self.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2

    
