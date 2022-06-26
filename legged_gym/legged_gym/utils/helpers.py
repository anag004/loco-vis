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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
import wandb
import git
import uuid
from isaacgym import gymutil
import ml_runlog
import datetime
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import sys

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
     if checkpoint==-1:
         models = [file for file in os.listdir(root) if model_name_include in file]
         models.sort(key=lambda m: '{0:0>15}'.format(m))
         model = models[-1]
     else:
         model = "model_{}.pt".format(checkpoint) 

     load_path = os.path.join(root, model)
     return load_path

def update_cfg_from_args(env_cfg, train_cfg, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if train_cfg is not None:
        if args.seed is not None:
            train_cfg.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            train_cfg.runner.max_iterations = args.max_iterations
        if args.resume:
            train_cfg.runner.resume = args.resume
        if args.experiment_name is not None:
            train_cfg.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            train_cfg.runner.run_name = args.run_name
        if args.load_run is not None:
            train_cfg.runner.load_run = args.load_run
        if args.checkpoint is not None:
            train_cfg.runner.checkpoint = args.checkpoint

    if args.override_env_config is not None and env_cfg is not None:
        exec(args.override_env_config)

    if args.override_train_config is not None and train_cfg is not None:
        exec(args.override_train_config)

    return env_cfg, train_cfg

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "a1", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},

        {"name": "--dataset_path", "type": str, "help": "Path to ScandotsDepthDataset"},
        {"name": "--teacher", "type": str, "help": "Name of the teacher policy to use when distilling"},
        {"name": "--exptid", "type": str, "help": "exptid"},
        {"name": "--depth_checkpoint", "type": str, "default": None},
        {"name": "--use_train_env", "action": "store_true", "default": False},
        {"name": "--mask_priv", "action": "store_true", "default": False},
        {"name": "--buffer_maxlen", "type": int,  "help": "Maximum length of buffer when playing", "default": 100},
        {"name": "--disable_depth", "help": "Zero out scandots_latent", "default": False, "action": "store_true"},
        {"name": "--load_run_aux", "default": None, "type": str},
        {"name": "--finetune_run", "default": None, "type": str},
        {"name": "--reconstruct_from", "default": "latent", "type": str},
        {"name": "--normalize", "action": "store_true", "default": False},
        {"name": "--override_env_config", "default": None, "type": str, "help": "Override config parameters, useful for launching batch jobs"},
        {"name": "--override_train_config", "default": None, "type": str, "help": "Override config parameters, useful for launching batch jobs"},
        {"name": "--baseline_name", "default": None, "type": str},
        {"name": "--baseline_folder", "default": None, "type": str},
        {"name": "--terrain", "default": None, "type": str},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"

    return args

def init_ml_runlog(log_dir):
    if wandb.run.url is not None:
        ml_runlog.init(
            "/home/anag/creds.json",
            "vision-loco"
        )

        ml_runlog.log_data(
            timestamp=datetime.datetime.now(),
            run_name=wandb.run.name,
            wandb_url=wandb.run.url,
            machine=os.uname().nodename,
            script_name=sys.argv[0],
            log_dir=log_dir,
            teacher="None",
            obs_dim="NA",
            obs_type="NA",
            commit_hash=get_current_commit_hash(),
            comments=os.getenv("SHEET_LOGGER")
        )


def get_current_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    branch = repo.active_branch.name

    return "{}@{}".format(branch, sha)

def get_git_diff_patch():
    repo = git.Repo(search_parent_directories=True)
    t = repo.head.commit.tree
    patch = repo.git.diff(t)

    return patch

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

def init_wandb():
    wandb.init(project="legged_gym", 
                sync_tensorboard=True, 
                dir="../../")

    exp_uuid = uuid.uuid1()
    os.system("mkdir /tmp/{}".format(exp_uuid))
    os.system("cp ../envs/a1/a1_config.py /tmp/{}/a1_config.py".format(exp_uuid))
    os.system("cp ../envs/go1/go1_config.py /tmp/{}/go1_config.py".format(exp_uuid))
    os.system("cp ../envs/base/legged_robot_config.py /tmp/{}/legged_robot_config.py".format(exp_uuid))
 
    diff_patch = get_git_diff_patch()
    with open("/tmp/{}/diff.patch".format(exp_uuid), 'w') as f:
        f.write(diff_patch)

    wandb.save("/tmp/{}/diff.patch".format(exp_uuid))
    wandb.save("/tmp/{}/a1_config.py".format(exp_uuid))
    wandb.save("/tmp/{}/go1_config.py".format(exp_uuid))
    wandb.save("/tmp/{}/legged_robot_config.py".format(exp_uuid))
    
