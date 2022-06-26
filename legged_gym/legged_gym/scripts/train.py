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

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import init_wandb
import numpy as np
import os
from datetime import datetime

use_wand = os.getenv("USE_WAND") is not None

from shutil import copyfile
import torch
if use_wand: import wandb
import uuid

def train(args):
    if not use_wand:
        log_pth = LEGGED_GYM_ROOT_DIR + "/logs/rough_a1/" + args.exptid
        try:
            os.makedirs(log_pth)
        except:
            pass
        copyfile(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", log_pth + "/legged_robot_config.py")
        copyfile(LEGGED_GYM_ENVS_DIR + "/a1/a1_config.py", log_pth + "/a1_config.py")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    if use_wand:
        if args.load_run is not None:
            log_root = args.load_run
        else:
            log_root = "default"

        ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_root, env=env, name=args.task, args=args)
    else:
        ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    # Log configs immediately
    if use_wand:
        # wandb.init(project="legged_gym", 
        #            sync_tensorboard=True, 
        #            dir="../../")

        # exp_uuid = uuid.uuid1()
        # print(exp_uuid)
        # os.system("mkdir /tmp/{}".format(exp_uuid))
        # os.system("cp ../envs/a1/a1_config.py /tmp/{}/a1_config.py".format(exp_uuid))
        # os.system("cp ../envs/base/legged_robot_config.py /tmp/{}/legged_robot_config.py".format(exp_uuid))
   
        # wandb.save("/tmp/{}/a1_config.py".format(exp_uuid))
        # wandb.save("/tmp/{}/legged_robot_config.py".format(exp_uuid))

        init_wandb()
    
    args = get_args()
    train(args)
