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

from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from .base_config import BaseConfig
import torch.nn as nn
class LeggedRobotCfg(BaseConfig):
    class play:
        load_student_config = False
        mask_priv_obs = False
        num_error_points = 50
    class env:
        num_envs = 4096
        # num_observations = 48 + 45
        # num_observations = 235 + 10 * 180 * 320
        # num_observations = 42 + 200
        # num_observations = 235 + 10 * 32 * 32
        n_scan = 187
        n_proprio_teacher = 6
        n_proprio_hist = 41
        n_proprio = n_proprio_teacher + n_proprio_hist
        n_priv = 5 + 12 
        reorder_dofs = True
        latent_update_freq = 0.1
        
        use_prop_history_buf = True
        depth_pos_encoding_dim = 10
        mask_base_lin_vel = True
        num_observations = n_scan + n_proprio + n_priv + 58 * 87 * 15 + 32 * 15 + 10
        priv_obs_loc = (n_scan + n_proprio, n_scan + n_proprio + n_priv)  #(187 + 47, 187 + 47 + 5 + 12)
        proprio_loc = (n_scan + n_proprio_teacher, n_scan + n_proprio_teacher + n_proprio_hist)  #(187 + 6, 187 + 6 + 41)  # history does not includ lin/ang vel
        
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        obs_type = "og"

        use_camera = True
        concatenate_depth = True
        history_encoding = True

        camera_config = [
            {
                "position": [0.30, 0, 0.12],
                "angle": 58,
                # "position_delta": [0.05, 0.05, 0.05],
                # "angle_delta": 0
            }
        ]
         

    class depth: 
        original = (212, 142)
        resized = (87, 58)
        horizontal_fov = 87
        # horizontal_fov_delta = 3
        dt = 0.2
        viz = True
        num_depth_frames = 15
        clip = 1
        scale = 1
        invert = True
        normalize_mean = -0.5
        normalize_std = 0.4


    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.05 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        # height = [0.02, 0.04]
        height = [0.00, 0.00]
        gap_size = [0.05, 0.1]
        platform_size = 4
        stepping_stone_distance = [0.06, 0.1]
        downsampled_scale = 0.05
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # measured_points_x = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
        # measured_points_y = [-0.4, -0.2, 0., 0.2, 0.4]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, 
        #                 rough slope, 
        #                 rough stairs up, 
        #                 rough stairs down, 
        #                 discrete, 
        #                 stepping stones
        #                 gaps, 
        #                 smooth flat]
        # terrain_proportions = [0.0, 0.3, 0.15, 0.15, 0.0, 0.3, 0.0, 0.1]
        #terrain_proportions = [0.0, 0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.1]
        #terrain_proportions = [0.0, 0.0, 0.15, 0.15, 0.5, 0.0, 0.2, 0.0]
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        
        # Easy ranges
        class ranges:
            lin_vel_x = [0.35, 0.35] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

        # Easy ranges
        class max_ranges:
            lin_vel_x = [0.35, 0.35] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

        # Full ranges
        # class max_ranges:
        #     lin_vel_x = [-1.0, 1.0] # min max [m/s]
        #     lin_vel_y = [-1.0, 1.0]   # min max [m/s]
        #     ang_vel_yaw = [-1, 1]    # min max [rad/s]
        #     heading = [-3.14, 3.14]
        class crclm_increment:
            lin_vel_x = 0.1 # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1    # min max [rad/s]
            heading = 0.5


    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.3, 1.25]
        randomize_base_mass = True
        added_mass_range = [-2., 6.]
        randomize_base_com = True
        added_com_range = [-0.15, 0.15]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.3

        randomize_motor = True
        motor_strength_range = [0.9, 1.1]
        
    class rewards:
        class scales:
            termination = -1.
            tracking_lin_vel = 5.0
            tracking_ang_vel = -0.5
            lin_vel_z = -0.01
            lin_vel_y = -0.02
            hip_action = -0.01
            ang_vel_xy = -0.02
            heading = -1e-4
            orientation = -1.
            torques = -0.0
            delta_torques = -1.0e-7#-1.0e-5
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  0.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            act_penalty = -0.1
            work = -0.002
            alive_bonus = 0.0

        doubling_terms = ['work', 'delta_torques', 'tracking_lin_vel', 'act_penalty']
        doubling_factors = [2.0, 2.0, 1.0, 0.5]

        only_positive_rewards = not  True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        quantize_height = not True
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.07
            height_measurements_horizontal = 0.00 # Horizontal error

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
 
    class distil:
        num_episodes = 10000
        num_epochs = 10000
        num_teacher_obs = 235 - 12 - 24 - 3
        logging_interval = 5
        save_interval = 1000  
        epoch_save_interval = 10
        batch_size = 256
        num_steps = 100
        num_training_iters = 10
        lr = 1e-3
        training_device = "cuda:0"
        max_buffer_length = 50000
        num_warmup_steps = 100
    class policy:
        use_actor_tanh = True
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        use_depth_backbone = True
        backbone_type = "mlp_hierarchical"
        num_input_vis_obs = 15 * 32 + 15 * 87 * 58
        num_output_vis_obs = 100
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
        scandots_compression = [187, 256, 128, 32]

    class teacher_policy:
        use_actor_tanh = True
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        use_depth_backbone = False
        backbone_type = "deepgait_coordconv"
        num_input_vis_obs = 10 * 32 * 32
        num_output_vis_obs = 1280
        override_num_obs = 251
        scandots_compression = [187, 256, 128, 32]
    class student_policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        use_depth_backbone = False
        backbone_type = "deepgait_coordconv"
        num_input_vis_obs = 10 * 32 * 32
        num_output_vis_obs = 1280
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        priv_dims = 190
        num_vis_obs = 32 * 15 + 15 * 58 * 87 + 10
        teacher_alpha = 0.0

    class teacher_runner:
        policy_class_name = 'ActorCriticRMA'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    class runner:
        policy_class_name = 'ActorCriticRMA'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 3000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
