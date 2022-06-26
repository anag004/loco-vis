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

# TODO mask out teacher observations when visualizing a student policy

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import statistics as stats
import torch
import cv2
from collections import deque
import statistics
import faulthandler
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn as nn
import seaborn as sns
from play_scandots_depth_dataset import get_num_scandots, get_latest_checkpoint
sns.set_theme()

def play(args):
    faulthandler.enable()
    log_pth = args.load_run
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    env_cfg.env.test_time = True

    if not args.use_train_env:
        # override some parameters for testing
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.noise.add_noise = True
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.push_feet = False
        env_cfg.domain_rand.push_interval_s = 2
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False
    
    error_history = []

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    if env_cfg.play.load_student_config:
        train_cfg.policy = train_cfg.student_policy
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    if args.teacher is not None:
        # load policy
        teacher_train_cfg = deepcopy(train_cfg)
        teacher_args = deepcopy(args)
        # load teacher policy
        teacher_train_cfg.policy = teacher_train_cfg.teacher_policy
        teacher_train_cfg.runner = deepcopy(train_cfg.teacher_runner)
        teacher_train_cfg.runner.resume = True
        teacher_train_cfg.runner.load_run = args.teacher
        teacher_args.teacher = None
        teacher_args.load_run = None
        override_num_obs = teacher_train_cfg.teacher_runner.override_num_obs

        ppo_runner, _ = task_registry.make_alg_runner(env=env, 
                                                      name=args.task, 
                                                      args=teacher_args, 
                                                      init_wandb=False,
                                                      train_cfg=teacher_train_cfg,
                                                      override_num_obs=override_num_obs)
        teacher_policy = ppo_runner.get_inference_policy(device=env.device)
        num_vis_obs = train_cfg.algorithm.num_vis_obs

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    rewbuffer = deque(maxlen=args.buffer_maxlen)
    lenbuffer = deque(maxlen=args.buffer_maxlen)
    num_terrain = len(env_cfg.terrain.terrain_proportions)
    lenbuffer_terrain = [deque(maxlen=args.buffer_maxlen) for _ in range(num_terrain)]
    rewbuffer_terrain = [deque(maxlen=args.buffer_maxlen) for _ in range(num_terrain)]
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    priv_dims = train_cfg.algorithm.priv_dims
    print("Masking out priv_dims: ", priv_dims)
    
    if env_cfg.play.num_error_points > 0:
        plt.ion()


    # plt.ion()
    value_buffer = deque(maxlen=5000)
    value_history = []

    # Load a second policy to switch between
    if args.load_run_aux is not None:
        actor_critic_aux = deepcopy(ppo_runner.alg.actor_critic)
        checkpoint_name = get_latest_checkpoint(args.load_run_aux)
        actor_critic_aux.load_state_dict(torch.load(checkpoint_name)["model_state_dict"])
        actor_critic_aux.eval()
        policy_aux = actor_critic_aux.act_inference

    for i in range(10*int(env.max_episode_length)):
        print("#################################################")
        if args.teacher is not None:
            if num_vis_obs != 0:
                teacher_actions = teacher_policy(obs[:, :-num_vis_obs].detach())
            else:
                teacher_actions = teacher_policy(obs.detach())

        if priv_dims != 0:
            obs[:, :priv_dims] = 0

        with torch.no_grad():
            actions = policy(obs.detach())
        
        if args.teacher is not None:
            error = torch.nn.functional.mse_loss(teacher_actions, actions)
            print("error: ", error.item())

            if env_cfg.play.num_error_points > 0:
                error_history.append(error.item())
                plt.plot(error_history)
                plt.xlim((len(error_history) - env_cfg.play.num_error_points, len(error_history)))
                plt.title("Error b/w teacher and student")
                plt.xlabel("Time steps")
                plt.ylabel("MSE error")
                plt.draw()
                plt.pause(0.01)


        obs, _, rews, dones, infos = env.step(actions.detach())
        ppo_runner.alg.actor_critic.reset_with_grad(dones=dones)
        # print(torch.mean(rews))
        # Log stuff
        cur_reward_sum += rews
        cur_episode_length += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        
        for i in range(num_terrain):
            len_filtered = cur_episode_length[new_ids][:, 0].cpu().numpy()
            len_filtered = len_filtered[env.env_terrain_types[new_ids][:, 0].cpu().numpy() == i]
            lenbuffer_terrain[i].extend(len_filtered.tolist()) 

            rew_filtered = cur_reward_sum[new_ids][:, 0].cpu().numpy()
            rew_filtered = rew_filtered[env.env_terrain_types[new_ids][:, 0].cpu().numpy() == i]
            rewbuffer_terrain[i].extend(rew_filtered.tolist())

        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0
        print("command: ", env.commands[env.lookat_id])
        print("terrain_type: ", env.env_terrain_types[env.lookat_id])
        print("terrain_difficulty: ", env.env_terrain_difficulty[env.lookat_id])

        if len(rewbuffer) > 0:
            avg_reward = statistics.mean(rewbuffer)
            print("avg_reward: ", avg_reward)

        if len(lenbuffer) > 0:
            avg_len = statistics.mean(lenbuffer)
            print("avg_len: ", avg_len)

        for i in range(num_terrain):
            if len(lenbuffer_terrain[i]) > 0:
                avg_len = statistics.mean(lenbuffer_terrain[i])
                print("avg_len_{}: ".format(i), avg_len)

            if len(rewbuffer_terrain[i]) > 0:
                avg_rew = statistics.mean(rewbuffer_terrain[i])
                print("avg_reward_{}: ".format(i), avg_rew)
        
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            pass
            # logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

        # Print some info
        print("lin_vel_x: {:.2}".format(env.base_lin_vel[env.lookat_id][0]))
        print("lin_vel_y: {:.2}".format(env.base_lin_vel[env.lookat_id][1]))
        print("lin_vel_z: {:.2}".format(env.base_lin_vel[env.lookat_id][2]))
        print("foot_contacts: ", env.foot_contacts_from_sensor[env.lookat_id])
        
        for rew_name, rew_values in env.named_rew_buf.items():
            print("rew_{}: {:.2} , {:.2}".format(rew_name, rew_values[env.lookat_id], rew_values[env.lookat_id] / env.reward_scales[rew_name]))
        
        if ppo_runner.alg.actor_critic.scandots_compression_nn is not None:
            print("scandots_latent: ", ppo_runner.alg.actor_critic.scandots_compression_nn(obs[env.lookat_id, :env_cfg.env.n_scan][None, :]))
        
        print("total_rew: {:.2}".format(env.rew_buf[env.lookat_id]))
        print("env_terrain_height: {:.2}".format(env.env_terrain_heights[env.lookat_id]))
        print("yshift: {:.2}".format(env.root_states[env.lookat_id, 1] - env.initial_root_states[env.lookat_id, 1]))
        print("max_foot_impact: ", env.max_foot_impact[env.lookat_id])
        print("min_edge_distance: ", env.min_edge_distance[env.lookat_id])

        forces = env.force_sensor_tensor.reshape((env.num_envs, 4, 6))[env.lookat_id, :, :3]

        for i, leg_name in enumerate(["FL", "FR", "RL", "RR"]):
            print("{}_force: ".format(leg_name), forces[i])

        print("#################################################")

        # Plot some value
        # value = avg_len
        # value_history.append(value)

        # plt.plot(value_history)
        # plt.xlim((len(value_history) - 50, len(value_history)))
        # plt.title("foot_force")
        # plt.xlabel("Time steps")
        # plt.draw()
        # plt.pause(0.00001)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
