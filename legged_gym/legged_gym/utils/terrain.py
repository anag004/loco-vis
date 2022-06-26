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
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import skimage
from pydelatin import Delatin
import math
from scipy.interpolate import BPoly
from scipy import ndimage

def random_uniform_terrain_raw(shape, horizontal_scale, vertical_scale, min_height, max_height, step=1, downsampled_scale=None,):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    
    width, length = shape

    if downsampled_scale is None:
        downsampled_scale = horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / vertical_scale)
    max_height = int(max_height / vertical_scale)
    step = int(step / vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(width * horizontal_scale / downsampled_scale), int(
        length * horizontal_scale / downsampled_scale)))

    x = np.linspace(0, width * horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, length * horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, width * horizontal_scale, width)
    y_upsampled = np.linspace(0, length * horizontal_scale, length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    return z_upsampled.astype(np.int16)
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.env_terrain_types = np.zeros((cfg.num_rows, cfg.num_cols))
        self.env_terrain_height = np.zeros((cfg.num_rows, cfg.num_cols))
        self.env_terrain_difficulty = np.zeros((cfg.num_rows, cfg.num_cols))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        
        if hasattr(self.cfg, "rough_borders") and self.cfg.rough_borders:
            # Add noise to borders of the height field
            borders = random_uniform_terrain_raw((self.tot_rows, self.tot_cols),
                                                  self.cfg.horizontal_scale,
                                                  self.cfg.vertical_scale,
                                                  min_height=-self.cfg.height[1],
                                                  max_height=self.cfg.height[1],
                                                  step=0.005,
                                                  downsampled_scale=self.cfg.downsampled_scale)
            
            self.height_field_raw[:self.border, :] = borders[:self.border, :]
            self.height_field_raw[-self.border:, :] = borders[-self.border:, :]
            self.height_field_raw[:, :self.border] = borders[:, :self.border]
            self.height_field_raw[:, -self.border:] = borders[:, -self.border:]

        if cfg.curriculum:  
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        if hasattr(self.cfg, "pepper_noise_amount"):
            mask = np.random.binomial(n=1, p=self.cfg.pepper_noise_amount, size=self.height_field_raw.shape) == 1
            self.height_field_raw[mask] -= 500 

        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            if hasattr(self.cfg, "delatin") and self.cfg.delatin:
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw,
                                                                                       self.cfg.horizontal_scale,
                                                                                       self.cfg.vertical_scale)
            else:
                self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                
            
            if hasattr(self.cfg, "remove_tall_triangles") and self.cfg.remove_tall_triangles is not None:
                if type(remove_tall_triangles) is bool:
                    prob = 1.0
                else:
                    prob = self.cfg.remove_tall_triangles

                self.vertices, self.triangles = remove_tall_triangles(self.vertices, self.triangles, threshold=0.4, prob=prob)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.05, 0.25, 0.5, 0.75, 0.95])
            terrain, terrain_type, env_origin_z = self.make_terrain(choice, difficulty, return_terrain_type=True)
            self.add_terrain_to_map(terrain, i, j, difficulty, terrain_type=terrain_type, env_origin_z=env_origin_z)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain, terrain_type, env_origin_z = self.make_terrain(choice, difficulty, return_terrain_type=True)
                self.add_terrain_to_map(terrain, i, j, difficulty, terrain_type=terrain_type, env_origin_z=env_origin_z)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j, -1)
    
    def add_roughness(self, terrain, terrain_type, difficulty=None):
        assert(not hasattr(self.cfg, "height_flat"))

        if difficulty is not None:
            height = self.cfg.height[0] + difficulty * (self.cfg.height[1] - self.cfg.height[0])
        else:
            height = random.uniform(*self.cfg.height)

        terrain.height = height

        if hasattr(self.cfg, "per_terrain_height") and terrain_type in self.cfg.per_terrain_height:
            if difficulty is not None:
                height_range = self.cfg.per_terrain_height[terrain_type]
                height = height_range[0] + difficulty * (height_range[1] - height_range[0])
            else:    
                height = random.uniform(*self.cfg.per_terrain_height[terrain_type])
        
        if hasattr(self.cfg, "use_fractal_terrain") and self.cfg.use_fractal_terrain:
            noise = generate_fractal_noise_2d(
                xSize=20,
                ySize=20,
                xSamples=self.width_per_env_pixels,
                ySamples=self.length_per_env_pixels,
                zScale=height/0.25
            ) / self.cfg.vertical_scale
            noise = noise.astype("int16")

            terrain.height_field_raw += noise
        else:
            terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def add_pillars(self, terrain, pillar_radius=0.3, pillar_dist=3):
        pillar_radius = int(pillar_radius / terrain.horizontal_scale)
        pillar_dist = int(pillar_dist / terrain.horizontal_scale)
        start_y = 0
        start_x = 0

        while start_x + 2 * pillar_radius <= terrain.length:
            shape = (2 * pillar_radius, 2 * pillar_radius)
            stop_x = start_x + 2 * pillar_radius
            start_y = np.random.randint(pillar_radius, pillar_radius + pillar_dist)
            stop_y = start_y + 2 * pillar_radius

            while start_y + 2 * pillar_radius <= terrain.width:
                stop_y = start_y + 2 * pillar_radius
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = generate_circle(shape, 0, height=500)
                start_y += 2 * pillar_radius + pillar_dist
            
            start_x += 2 * pillar_radius + pillar_dist

    def make_terrain(self, choice, difficulty, return_terrain_type=False):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        terrain.height = 0
        slope = difficulty * 0.4
        if hasattr(self.cfg, "slope_multiplier"):
            slope *= self.cfg.slope_multiplier

        if not hasattr(self.cfg, "discrete_obstacles_height_multiplier"):
            discrete_obstacles_height_multiplier = 1
        else:
            discrete_obstacles_height_multiplier = self.cfg.discrete_obstacles_height_multiplier
        
        step_height = 0.03 + 0.13 * difficulty
        discrete_obstacles_height = 0.02 + difficulty * 0.11 * discrete_obstacles_height_multiplier
        env_origin_z = None

        if hasattr(self.cfg, "max_obstacle_height"):
            discrete_obstacles_height = min(discrete_obstacles_height, self.cfg.max_obstacle_height)

        if hasattr(self.cfg, "gap_height_rand"):
            gap_height_rand = self.cfg.gap_height_rand
        else:
            gap_height_rand = 0
        
        if hasattr(self.cfg, "platform_height_rand"):
            platform_height_rand = self.cfg.platform_height_rand
        else:
            platform_height_rand = 0

        gap_size = 1. * difficulty
        periodic_gap_size = 0.2 * difficulty
        pit_depth = 1. * difficulty

        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1

        if not hasattr(self.cfg, "stepping_stones_height_multiplier"):
            stepping_stones_height_multiplier = 1
        else:
            stepping_stones_height_multiplier = self.cfg.stepping_stones_height_multiplier
        
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_type = 0
        elif choice < self.proportions[1]:
            if choice < self.proportions[1]/ 2:
                slope *= -1

            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_type = 1
            self.add_roughness(terrain, terrain_type)
        elif choice < self.proportions[3]:
            stair_direction = 1
            if choice<self.proportions[2]:
                stair_direction = -1
                step_height *= -1
            
            step_width = 0.27 + np.random.random() * 0.18

            # Override values from config if exist
            if hasattr(self.cfg, "step_dim"):
                # width, height
                width_range, height_range = self.cfg.step_dim
                step_width = random.uniform(*width_range)
                step_height = (height_range[0] + difficulty * (height_range[1] - height_range[0])) * stair_direction

            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=3.)
            terrain_type = 3
            self.add_roughness(terrain, terrain_type)
        elif choice < self.proportions[4]:
            num_rectangles = 100
            rectangle_min_size = 0.5 if not hasattr(self.cfg, "rectangle_min_size") else self.cfg.rectangle_min_size
            rectangle_max_size = 1.5 if not hasattr(self.cfg, "rectangle_max_size") else self.cfg.rectangle_max_size
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=2.5)
            add_platform(terrain, platform_size=1, platform_height_rand=platform_height_rand)
            
            terrain_type = 4
            self.add_roughness(terrain, terrain_type)
        elif choice < self.proportions[5]:
            stones_size = 1.5 - 1.2 * difficulty
            #terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stepping_stone_distance, max_height=0., platform_size=4., depth = -2000.)
            #stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            stepping_stones_terrain(terrain, stone_size=0.35-0.06*difficulty, stone_distance=0.05+0.06*difficulty, stone_distance_rand=0.06*difficulty, max_height=0.06*difficulty*stepping_stones_height_multiplier)
            add_platform(terrain, platform_size=2, platform_height_rand=platform_height_rand)
            add_borders(terrain, border_size=1)
            terrain_type = 5
            self.add_roughness(terrain, terrain_type)
        elif choice < self.proportions[6]:
            gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_terrain(terrain, gap_size=gap_size)
            add_platform(terrain, platform_size=3, platform_height_rand=platform_height_rand)
            terrain_type = 6
            self.add_roughness(terrain, terrain_type)
        elif choice < self.proportions[7]:
            terrain_type = 7
            self.add_roughness(terrain, terrain_type, difficulty=difficulty)
            add_platform(terrain, platform_size=3, platform_height_rand=platform_height_rand)
        elif choice < self.proportions[8]:
            gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            platform_size = random.uniform(self.cfg.platform_size[0], self.cfg.platform_size[1])
            periodic_gap_terrain(terrain, 
                                 gap_size=gap_size, 
                                 height_rand=gap_height_rand)
            add_platform(terrain, platform_size=platform_size, platform_height_rand=0.0)
            terrain_type = 8
            self.add_roughness(terrain, terrain_type)
        elif len(self.proportions) > 9 and choice < self.proportions[9]:
            gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            terrain_type = 9
        elif len(self.proportions) > 10 and choice < self.proportions[10]:
            gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            platform_size = random.uniform(self.cfg.platform_size[0], self.cfg.platform_size[1])
            ramp_width = random.uniform(self.cfg.ramp_width[0], self.cfg.ramp_width[1])
            periodic_gap_terrain(terrain, 
                                 gap_size=gap_size, 
                                 platform_size=platform_size,
                                 height_rand=gap_height_rand)
            centre_ramp(terrain, ramp_width)
            add_platform(terrain, platform_size=platform_size, platform_height_rand=0.0)
            terrain_type = 10
            self.add_roughness(terrain, terrain_type)
        elif len(self.proportions) > 11 and choice < self.proportions[11]:
            ramp_width = random.uniform(self.cfg.ramp_width[0], self.cfg.ramp_width[1])


            stepping_stones_terrain(terrain, 
                                    stone_size=0.45-0.03*difficulty, 
                                    stone_distance=0.02+0.03*difficulty,
                                    stone_distance_rand=0.03*difficulty, 
                                    max_height=0.06*difficulty*stepping_stones_height_multiplier, 
                                    randomized_shape=True)
            add_platform(terrain, platform_size=2, platform_height_rand=platform_height_rand)
            terrain_type = 11
            self.add_roughness(terrain, terrain_type)
        elif len(self.proportions) > 12 and choice < self.proportions[12]:
            ramp_width = random.uniform(self.cfg.ramp_width[0], self.cfg.ramp_width[1])
            stepping_stones_terrain(terrain, 
                                    stone_size=0.45-0.03*difficulty, 
                                    stone_distance=0.02+0.03*difficulty,
                                    stone_distance_rand=0.03*difficulty, 
                                    max_height=0.06*difficulty*stepping_stones_height_multiplier, 
                                    randomized_shape=True)
            add_platform(terrain, platform_size=2, platform_height_rand=platform_height_rand)
            centre_ramp(terrain, ramp_width)
            terrain_type = 12
            self.add_roughness(terrain, terrain_type)
        elif len(self.proportions) > 13 and choice < self.proportions[13]:
            ramp_width = random.uniform(self.cfg.ramp_width[0], self.cfg.ramp_width[1])
            stepping_stones_terrain(terrain, stone_size=0.35-0.06*difficulty, stone_distance=0.05+0.06*difficulty, stone_distance_rand=0.06*difficulty, max_height=0.06*difficulty*stepping_stones_height_multiplier)
            add_platform(terrain, platform_size=2, platform_height_rand=platform_height_rand)
            centre_ramp(terrain, ramp_width)
            terrain_type = 13
            self.add_roughness(terrain, terrain_type)
        elif len(self.proportions) > 14 and choice < self.proportions[14]:
            ramp_width = random.uniform(self.cfg.ramp_width[0], self.cfg.ramp_width[1])
            stepping_stones_terrain(terrain, stone_size=0.35-0.06*difficulty, stone_distance=0.01+0.06*difficulty, stone_distance_rand=0.06*difficulty, max_height=0.06*difficulty*stepping_stones_height_multiplier, circular_shape=True)
            add_platform(terrain, platform_size=2, platform_height_rand=platform_height_rand)
            centre_ramp(terrain, ramp_width)
            terrain_type = 14
            self.add_roughness(terrain, terrain_type)

        elif len(self.proportions) > 15 and choice < self.proportions[15]:
            if hasattr(self.cfg, "stools_diameter_range"):
                a = self.cfg.stools_diameter_range[1]
                b = self.cfg.stools_diameter_range[1] - self.cfg.stools_diameter_range[0] 
            else:
                a = 0.335
                b = 0.07
            
            if hasattr(self.cfg, "stools_distance_range_x"):
                stone_distance_x = random.uniform(*self.cfg.stools_distance_range_x)
            else:
                stone_distance_x = 0.01+0.06*difficulty

            if hasattr(self.cfg, "stools_distance_range_y"):
                stone_distance_y = random.uniform(*self.cfg.stools_distance_range_y)
            else:
                stone_distance_y = 0.01+0.06*difficulty

            stepping_stones_terrain_stools(terrain, 
                                            platform_size=2, 
                                            stone_size=a - b * difficulty, 
                                            stone_distance_x=stone_distance_x,
                                            stone_distance_y=stone_distance_y, 
                                            stone_distance_rand=0.1*difficulty, 
                                            max_height=0.03*difficulty*stepping_stones_height_multiplier)
            terrain_type = 15
            self.add_roughness(terrain, terrain_type)
        elif len(self.proportions) > 16 and choice < self.proportions[16]:
            gap_size = random.uniform(*self.cfg.barrier_width)
            platform_size = random.uniform(self.cfg.platform_size[0], self.cfg.platform_size[1])
            ramp_width = random.uniform(self.cfg.ramp_width[0], self.cfg.ramp_width[1])
            height = random.uniform(*self.cfg.barrier_height)

            periodic_gap_terrain(terrain, 
                                 gap_size=gap_size, 
                                 platform_size=platform_size,
                                 height=height,
                                 height_rand=gap_height_rand)
            add_platform(terrain, platform_size=2, platform_height_rand=platform_height_rand)
            terrain_type = 16
            self.add_roughness(terrain, terrain_type)
        elif len(self.proportions) > 17 and choice < self.proportions[17]:
            terrain_type = 17 # Hard rough terrain
            self.add_roughness(terrain, terrain_type)
        elif len(self.proportions) > 18 and choice < self.proportions[18]:
            terrain_type = 18
            pillar_dist = random.uniform(*self.cfg.pillar_dist)
            pillar_radius = random.uniform(*self.cfg.pillar_radius)

            self.add_pillars(terrain, pillar_dist=pillar_dist, pillar_radius=pillar_radius)
            self.add_roughness(terrain, terrain_type)
            add_platform(terrain, platform_size=2, platform_height_rand=platform_height_rand)
            env_origin_z = 0 # Have to override this since otherwise env_origin_z will be highest point of the terrain which is top of pillar
        elif len(self.proportions) > 19 and choice < self.proportions[19]:
            terrain_type = 19
            corridor_width = int(random.uniform(*self.cfg.corridor_width))
            corridor_terrain(terrain, corridor_width=corridor_width)
            env_origin_z = 0
        if return_terrain_type:
            return terrain, terrain_type, env_origin_z
        else:
            return terrain, env_origin_z

    def add_terrain_to_map(self, terrain, row, col, difficulty, terrain_type=None, env_origin_z=None):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        if hasattr(terrain, "spawn_offset"):
            env_origin_x = i * self.env_length + terrain.spawn_offset[0]
            env_origin_y = j * self.env_width + terrain.spawn_offset[1]
        else:
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width

        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        if env_origin_z is None:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

        if terrain_type is not None:
            self.env_terrain_types[i, j] = terrain_type
        
        self.env_terrain_difficulty[i, j] = difficulty
        self.env_terrain_height[i, j] = terrain.height

def add_platform(terrain, platform_size=1, platform_height_rand=0):
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_height_rand = int(platform_height_rand / terrain.vertical_scale)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2

    terrain.height_field_raw[x1:x2, y1:y2] = random.randint(-platform_height_rand, platform_height_rand)

def add_borders(terrain, border_size=1):
    border_size = int(border_size / terrain.horizontal_scale)

    terrain.height_field_raw[:border_size] = 0
    terrain.height_field_raw[-border_size:] = 0
    terrain.height_field_raw[:, :border_size] = 0
    terrain.height_field_raw[:, -border_size:] = 0

def generate_random_shape(shape, height_range, depth):
    result = np.zeros(shape)
    result[:, :] = np.random.choice(height_range)

    if 0 in shape:
        return result

    for i in range(result.shape[0]):
        start = random.randint(0, int(result.shape[1] * 0.1))
        end = random.randint(int(result.shape[1] * 0.9), result.shape[1])

        result[i, : start + 1] = depth
        result[i, end - 1:] = depth

    for i in range(result.shape[1]):
        start = random.randint(0, int(result.shape[0] * 0.1))
        end = random.randint(int(result.shape[0] * 0.9), result.shape[0])
        
        result[:start + 1, i] = depth
        result[end - 1:, i] = depth

    return result

def generate_circle(shape, depth, height_range=None, height=None):
    result = np.zeros(shape)
    
    if height_range is not None:
        result[:, :] = np.random.choice(height_range)
    if height is not None:
        result[:, :] = height
    
    a, b = result.shape
    a /= 2
    b /= 2

    if 0 in shape:
        return result

    for i in range(result.shape[0]):
        x = i - a
        y = b * math.sqrt(1 - x ** 2 / a ** 2)
        start = int(b - y)
        end = int(b + y)

        result[i, : start + 1] = depth
        result[i, end - 1:] = depth

    return result

def remove_tall_triangles(vertices, triangles, threshold, prob):
    for i, (a, b, c) in enumerate(triangles):
        z_ab = abs(vertices[a, 2] - vertices[b, 2])
        z_bc = abs(vertices[b, 2] - vertices[c, 2])
        z_ca = abs(vertices[c, 2] - vertices[a, 2])

        if max(z_ab, z_bc, z_ca) >= threshold and random.uniform(0, 1) <= prob:
            triangles[i] = (0, 0, 0)

    return vertices, triangles

def remove_low_triangles(vertices, triangles, threshold):
    for i, (a, b, c) in enumerate(triangles):
        if max(vertices[a, 2], vertices[b, 2], vertices[c, 2]) > threshold:
            continue

        triangles[i] = (0, 0, 0)

    return vertices, triangles

def corridor_terrain(terrain, corridor_width=30):
    dilate_iters = corridor_width
    max_height = 1.0
    n_interp_pts = 200
    spawn_ratio = 0.05
    h, w = terrain.height_field_raw.shape
    safe_h = int(h * 0.8)
    safe_w = int(w * 0.8)
    ctpt0 = (0, np.random.randint(0, w))
    ctpt1 = (np.random.randint(h-safe_h, safe_h), np.random.randint(w-safe_w, safe_w))
    ctpt2 = (np.random.randint(h-safe_h, safe_h), np.random.randint(w-safe_w, safe_w))
    ctpt3 = (h-1, np.random.randint(0, w))
    control_points = np.array([ctpt0, ctpt1, ctpt2, ctpt3])
    curve = BPoly(control_points[:, np.newaxis, :], [0, 1])
    x = np.linspace(0, 1, n_interp_pts)
    sampled_points = curve(x)

    new_terrain = np.zeros_like(terrain.height_field_raw)
    sampled_points_int = np.floor(sampled_points).astype(np.int)
    new_terrain[sampled_points_int[:, 0], sampled_points_int[:, 1]] = 1
    new_terrain = ndimage.binary_dilation(new_terrain, iterations=dilate_iters)

    terrain.height_field_raw[new_terrain==1] = 0
    terrain.height_field_raw[new_terrain==0] = int(max_height / terrain.vertical_scale)
    terrain.slope_vector = 0
    terrain.spawn_offset = sampled_points_int[int(spawn_ratio*n_interp_pts), :] * terrain.horizontal_scale 

def stepping_stones_terrain_stools(terrain, stone_size, stone_distance_x, stone_distance_y, max_height, stone_distance_rand=0.0, platform_size=1., depth=-0.5, platform_height_rand=0):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    platform_height_rand = int(platform_height_rand / terrain.vertical_scale)
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance_x = int(stone_distance_x / terrain.horizontal_scale)
    stone_distance_y = int(stone_distance_y / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    stone_distance_rand = int(stone_distance_rand / terrain.horizontal_scale)

    start_x = 0
    start_y = int(terrain.length / 2 - stone_distance_y / 2 - stone_size)
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)

    ramp_width = int(0.6 / terrain.horizontal_scale)
    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - ramp_width) // 2
    y2 = (terrain.length + ramp_width) // 2

    if terrain.length >= terrain.width:
        for i in range(2):
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = int(stone_distance_x / 2)

            # fill first hole
            stop_x = start_x + stone_size
            stone_height = int(np.random.normal() * max_height)
            terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = generate_circle(terrain.height_field_raw[start_x:stop_x, start_y:stop_y].shape, int(depth / terrain.vertical_scale), height=stone_height)

            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                create_stone = True
                
                if x1 <= start_x and start_x <= x2:
                    create_stone = False  
                elif x1 <= stop_x and stop_x <= x2:
                    create_stone = False

                    # Extend the ramp until this point
                    terrain.height_field_raw[start_x:x1, y1:y2] = 0

                if create_stone:
                    this_start_y = min(terrain.length, start_y + max(int(stone_distance_rand * np.random.normal()), 1))
                    this_stop_y = min(terrain.length, this_start_y + stone_size)
                    stone_height = int(np.random.normal() * max_height)
                    terrain.height_field_raw[start_x: stop_x, this_start_y: this_stop_y] = generate_circle(terrain.height_field_raw[start_x: stop_x, this_start_y: this_stop_y].shape, int(depth / terrain.vertical_scale), height=stone_height)
                    start_x += stone_size + max(1, stone_distance_x + int(stone_distance_rand * (0.2 + np.random.random())))
                else:

                    start_x = x2 + max(stone_distance_x + int(stone_distance_rand * (0.2 + np.random.random())), 1)

            start_y += stone_size + max(1, stone_distance_y)

    elif terrain.width > terrain.length:
        raise NotImplemented

    platform_size = int(2 / terrain.horizontal_scale)

    terrain.height_field_raw[x1:x2, y1:y2] = 0

    # Add a 25cm boundary to the terrain for safety
    boundary = int(0.25 / terrain.horizontal_scale)
    terrain.height_field_raw[:boundary, :] = 0
    terrain.height_field_raw[-boundary:, :] = 0
    terrain.height_field_raw[:, -boundary:] = 0
    terrain.height_field_raw[:, :boundary] = 0

    return terrain

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, stone_size_rand=0.0, stone_distance_rand=0.0, platform_size=1., depth=-0.5, randomized_shape=False, platform_height_rand=0, circular_shape=False):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    platform_height_rand = int(platform_height_rand / terrain.vertical_scale)
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)
    stone_size_rand = int(stone_size_rand / terrain.horizontal_scale)
    stone_distance_rand = int(stone_distance_rand / terrain.horizontal_scale)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)

            if randomized_shape:
                terrain.height_field_raw[0:stop_x, start_y:stop_y] = generate_random_shape(terrain.height_field_raw[0:stop_x, start_y:stop_y].shape, height_range, int(depth / terrain.vertical_scale))
            elif circular_shape:
                terrain.height_field_raw[0:stop_x, start_y:stop_y] = generate_circle(terrain.height_field_raw[0:stop_x, start_y:stop_y].shape, int(depth / terrain.vertical_scale), height_range=height_range)
            else:
                terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)

            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                this_start_y = min(terrain.length, start_y + np.random.randint(-stone_distance_rand, stone_distance_rand+1))
                this_stop_y = min(terrain.length, this_start_y + stone_size)

                if randomized_shape:
                    terrain.height_field_raw[start_x: stop_x, this_start_y: this_stop_y] = generate_random_shape(terrain.height_field_raw[start_x: stop_x, this_start_y: this_stop_y].shape, height_range, int(depth / terrain.vertical_scale))
                elif circular_shape:
                    terrain.height_field_raw[start_x: stop_x, this_start_y: this_stop_y] = generate_circle(terrain.height_field_raw[start_x: stop_x, this_start_y: this_stop_y].shape, int(depth / terrain.vertical_scale), height_range=height_range)
                else:
                    terrain.height_field_raw[start_x: stop_x, this_start_y: this_stop_y] = np.random.choice(height_range)

                start_x += stone_size + stone_distance + np.random.randint(-stone_distance_rand, stone_distance_rand+1)
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    return terrain

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, max_error=1e-3)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2] * vertical_scale
    return vertices, mesh.triangles

def gap_terrain(terrain, gap_size, platform_size=1., platform_height_rand=0.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    platform_height_rand = int(platform_height_rand / terrain.vertical_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

def periodic_gap_terrain(terrain, gap_size, platform_size=1, height_rand=0, height=-10):
    height = int(height / terrain.vertical_scale)
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_rand = int(height_rand / terrain.vertical_scale)

    terrain.height_field_raw[:] = 0
    x = 0 

    while x + gap_size + platform_size < terrain.length:
        terrain.height_field_raw[x:x+gap_size, :] = height
        terrain.height_field_raw[x+gap_size:x+gap_size+platform_size] += random.randint(-height_rand, height_rand)
        x += gap_size + platform_size
    

def centre_ramp(terrain, ramp_width, ramp_gap=0.5):
    ramp_width = int(ramp_width / terrain.horizontal_scale)
    ramp_gap = int(ramp_gap / terrain.horizontal_scale)

    y1 = (terrain.width - ramp_width) // 2
    y2 = (terrain.width + ramp_width) // 2

    terrain.height_field_raw[:, :y1] -= 100
    terrain.height_field_raw[:, y2:] -= 100

    # Add ramps to the left
    y1_ctr = y1 - ramp_width - ramp_gap
    y2_ctr = y2 - ramp_width - ramp_gap

    while (y1_ctr >= 0):
        terrain.height_field_raw[:, y1_ctr:y2_ctr] += 100 
        y1_ctr -= (ramp_width + ramp_gap)
        y2_ctr -= (ramp_width + ramp_gap)

    # Add ramps to the right
    y1_ctr = y1 + ramp_width + ramp_gap
    y2_ctr = y2 + ramp_width + ramp_gap

    while(y2_ctr < terrain.width):
        terrain.height_field_raw[:, y1_ctr:y2_ctr] += 100 
        y1_ctr += (ramp_width + ramp_gap)
        y2_ctr += (ramp_width + ramp_gap)

    terrain.height_field_raw[terrain.height_field_raw <= -100] = -100

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=0.01)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) * 0.5 + 0.5

def generate_fractal_noise_2d(xSize=20, ySize=20, xSamples=1600, ySamples=1600, \
    frequency=10, fractalOctaves=2, fractalLacunarity = 2.0, fractalGain=0.25, zScale = 0.23):
    xScale = frequency * xSize
    yScale = frequency * ySize
    amplitude = 1
    shape = (xSamples, ySamples)
    noise = np.zeros(shape)
    for _ in range(fractalOctaves):
        noise += amplitude * generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
        amplitude *= fractalGain
        xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

    return noise
