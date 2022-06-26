# Reconstruct privileged information from the RMA latent

from copy import deepcopy
from cv2 import grabCut
from legged_gym import LEGGED_GYM_ROOT_DIR
import warnings
import os

use_wand = os.getenv("USE_WAND") is not None

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger, init_wandb, init_ml_runlog
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
from shutil import copyfile
import torch
import cv2
from collections import deque
import multiprocessing as mp
if use_wand:
    import wandb
import statistics
import uuid
from legged_gym.utils.helpers import *
from tqdm import tqdm
import datetime
from legged_gym.scripts.train_dagger import DaggerAgent
import ml_runlog
from collections import deque
from random import randint
from train_scandots_depth_dagger import *
import itertools
from copy import deepcopy
from play_scandots_depth_dataset import get_latest_checkpoint
import torch.nn.functional as F
from colorama import Fore, Style

def normalize(tensor):
    assert(len(tensor.shape) == 2)
    
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    
    return (tensor - mean) / std

def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.terrain.curriculum = False
    env_cfg.env.test_time = True

    log_pth = args.load_run
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Load the teacher policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg)
    collect_policy = ppo_runner.get_inference_policy()
    num_scandots = get_num_scandots(env_cfg)
    num_prop = env_cfg.env.num_observations - num_scandots

    criterion = nn.MSELoss()

    if args.reconstruct_from == "latent":
        input_dim = 8
    elif args.reconstruct_from == "hidden_state":
        input_dim = train_cfg.policy.rnn_hidden_size
    else:
        raise Exception("args.reconstruct_from should be either latent or hidden_state")

    latent_reconstruction = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, env_cfg.env.n_priv)
    )

    latent_reconstruction = latent_reconstruction.cuda()

    optimizer = torch.optim.Adam(latent_reconstruction.parameters())
    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    save_path = os.path.join(args.load_run, "latent_reconstruction_{}".format(timestamp))
    print("Saving checkpoints to ", save_path)

    if wandb.run.url is not None:
        ml_runlog.log_data(
            log_dir=save_path, 
            offset=-6
        )
    
    os.mkdir(save_path)

    obs = env.get_observations()
    pbar = tqdm(range(args.max_iterations))
    
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_loss = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    if 1 in env_cfg.env.obs_mask[env_cfg.env.priv_obs_loc[0]:env_cfg.env.priv_obs_loc[1]]:
        print(Fore.RED + "Warning: priv info is masked" + Style.RESET_ALL)

    for iter in pbar:
        predicted_priv_info_buffer = []
        priv_info_buffer = []

        for step in range(train_cfg.runner.num_steps_per_env):
            priv_info = obs[:, env_cfg.env.priv_obs_loc[0]:env_cfg.env.priv_obs_loc[1]].clone()

            if args.mask_priv:
                obs[:, env_cfg.env.priv_obs_loc[0]:env_cfg.env.priv_obs_loc[1]] = 0

            with torch.no_grad():
                teacher_actions = collect_policy(obs)

            priv_info_buffer.append(priv_info)

            if args.reconstruct_from == "latent":
                with torch.no_grad():
                    priv_latent = ppo_runner.alg.actor_critic.info_encoder(priv_info)
                predicted_priv_info = latent_reconstruction(priv_latent)
            elif args.reconstruct_from == "hidden_state":
                hidden_state = ppo_runner.alg.actor_critic.memory_a.hidden_states
                hidden_state = hidden_state.squeeze(0).clone()
                predicted_priv_info = latent_reconstruction(hidden_state)

            predicted_priv_info_buffer.append(predicted_priv_info)

            obs, _, rews, dones, infos = env.step(teacher_actions)

            cur_reward_sum += rews.detach()
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)

            # Log rewards and episode lengths
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0
            cur_loss[new_ids] = 0
            ppo_runner.alg.actor_critic.reset_with_grad(dones=dones)

            if len(rewbuffer) > 0:
                avg_reward = statistics.mean(rewbuffer)
                wandb.log({"avg_reward": avg_reward})

            if len(lenbuffer) > 0:
                avg_len = statistics.mean(lenbuffer)
                wandb.log({"avg_len": avg_len})

        # Compute and backprop the loss
        predicted_priv_info_buffer = torch.cat(predicted_priv_info_buffer, dim=0)  # [L, num_envs, 12]
        priv_info_buffer = torch.cat(priv_info_buffer, dim=0)

        if args.normalize:
            predicted_priv_info_buffer = normalize(predicted_priv_info_buffer)
            priv_info_buffer = normalize(priv_info_buffer)

        loss = criterion(predicted_priv_info_buffer, priv_info_buffer)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"Loss": loss.item()})
        pbar.set_description("Loss: {:.6}".format(loss.item()))

        avg_error = torch.abs(predicted_priv_info_buffer - priv_info_buffer).mean(dim=0)
        print("Avg Error: ", avg_error)
        wandb.log({"avg_error": avg_error})

        if iter != 0 and iter % train_cfg.runner.save_interval == 0:
            save_name = os.path.join(save_path, "model_{}.pt".format(iter))
            print("Saving latent_reconstruction ", save_name)
            torch.save(latent_reconstruction.state_dict(), save_name)


if __name__ == "__main__":
    if use_wand:
        init_wandb()

    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    main(args)
