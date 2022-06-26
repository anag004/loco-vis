import numpy as np
import torch

class NoOpSampler:
    """
        Does not do anything
    """
    
    def __init__(self) -> None:
        pass

    def resample_noise(self, env_ids):
        return

    def add_horizontal_noise(self, points):
        return points

    def add_vertical_noise(self, heights):
        return heights

    def set_num_envs(self, num_envs):
        self.num_envs = num_envs

class PerTimeStepSampler(NoOpSampler):
    """
        Independently samples noise for each scandot per time step
    """

    def __init__(self, device="cuda", h_noise_scale=0.0, v_noise_scale=0.0) -> None:
        self.device = device
        self.h_noise_scale = h_noise_scale
        self.v_noise_scale = v_noise_scale

    def add_horizontal_noise(self, points):
        points = points + torch.randn_like(points) * self.h_noise_scale

        return points

    def add_vertical_noise(self, heights):
        heights = heights + torch.randn_like(heights) * self.v_noise_scale

        return heights

class PerTimeStepSystematicSampler(NoOpSampler):
    """
        Independently samples noise for all scandots per time step
    """

    def __init__(self, device="cuda", h_noise_scale=0.0, v_noise_scale=0.0) -> None:
        self.device = device
        self.h_noise_scale = h_noise_scale
        self.v_noise_scale = v_noise_scale

    def add_horizontal_noise(self, points):
        noise = torch.randn(self.num_envs, device=self.device)[:, None, None] * self.h_noise_scale
        points = points + noise

        return points

    def add_vertical_noise(self, heights):
        noise = torch.randn(self.num_envs, device=self.device)[:, None] * self.v_noise_scale
        heights = heights + noise

        return heights

class PerEpisodeSampler(NoOpSampler):
    def __init__(self, device="cuda", h_noise_scale=0.0, v_noise_scale=0.0) -> None:
        self.device = device
        self.h_noise_scale = h_noise_scale
        self.v_noise_scale = v_noise_scale
        
        self.ep_h_noise = None
        self.ep_v_noise = None
        self.num_envs = None

    def resample_noise(self, env_ids):
        if self.ep_h_noise is None:
            self.ep_h_noise = torch.zeros(self.num_envs, device=self.device)
            self.ep_v_noise = torch.zeros(self.num_envs, device=self.device)

        self.ep_h_noise[env_ids] = torch.randn(self.num_envs, device=self.device)[env_ids] * self.h_noise_scale
        self.ep_v_noise[env_ids] = torch.randn(self.num_envs, device=self.device)[env_ids] * self.v_noise_scale

    def add_horizontal_noise(self, points):
        points = points +  self.ep_h_noise[:, None, None]

        return points

    def add_vertical_noise(self, heights):
        heights = heights + self.ep_v_noise[:, None]

        return heights

class RandomlyChoosedSampler(NoOpSampler):
    """
        Randomly choose from one sampler each episode
    """

    def __init__(self, samplers, probabilities, device="cuda") -> None:
        super().__init__()

        self.device = device
        self.samplers = samplers
        self.probabilities = torch.tensor(probabilities, device=self.device)
        self.sampler_choice = None
        self.num_envs = None

    def set_num_envs(self, num_envs):
        self.num_envs = num_envs

        for sampler in self.samplers:
            sampler.set_num_envs(num_envs)
    
    def resample_noise(self, env_ids):
        if self.sampler_choice is None:
            self.sampler_choice = torch.multinomial(self.probabilities, num_samples=self.num_envs, replacement=True)
        
        self.sampler_choice[env_ids] = torch.multinomial(self.probabilities, num_samples=len(env_ids), replacement=True)

        for sampler in self.samplers:
            sampler.resample_noise(env_ids)

    def add_horizontal_noise(self, points):
        points_list = []
        
        for sampler in self.samplers:
            points_list.append(sampler.add_horizontal_noise(points.clone()))

        for i in range(self.probabilities.shape[0]):
            mask = self.sampler_choice == i
            points[mask] = points_list[i][mask] 

        return points

    def add_vertical_noise(self, points):
        points_list = []
        
        for sampler in self.samplers:
            points_list.append(sampler.add_vertical_noise(points.clone()))

        for i in range(self.probabilities.shape[0]):
            mask = self.sampler_choice == i
            points[mask] = points_list[i][mask] 
        
        return points

class OutlierVerticalNoise(NoOpSampler):
    def __init__(self, device="cuda", v_noise_scale=0.0, selection_prob=0.0) -> None:
        self.device = device
        self.selection_prob = selection_prob
        self.v_noise_scale = v_noise_scale
        self.num_envs = None
    
    def add_vertical_noise(self, heights):
        selection_mask = torch.rand_like(heights) < self.selection_prob
        noisy_heights = heights + torch.randn_like(heights) * self.v_noise_scale
        heights[selection_mask] = noisy_heights[selection_mask]

        return heights

# class Systematic

class SequentialSampler(NoOpSampler):
    """
        Use samplers in sucession
    """

    def __init__(self, *samplers) -> None:
        self.samplers = samplers

    def resample_noise(self, env_ids):
        for sampler in self.samplers:
            sampler.resample_noise(env_ids)

    def add_horizontal_noise(self, points):
        for sampler in self.samplers:
            points = sampler.add_horizontal_noise(points)

        return points

    def add_vertical_noise(self, heights):
        for sampler in self.samplers:
            heights = sampler.add_vertical_noise(heights)

        return heights

    def set_num_envs(self, num_envs):
        for sampler in self.samplers:
            sampler.set_num_envs(num_envs)

def construct_single_sampler(config):
    return SequentialSampler(*[
        PerTimeStepSampler(
            h_noise_scale=config["h_noise_iid"],
            v_noise_scale=config["v_noise_iid"]
        ),
        PerEpisodeSampler(
            h_noise_scale=config["h_noise_ep"],
            v_noise_scale=config["v_noise_ep"]
        ),
        OutlierVerticalNoise(
            v_noise_scale=config["v_noise_outlier"],
            selection_prob=config["outlier_prob"]
        )
    ])

def construct_scandots_noise_sampler(config):
    probabilities = []
    samplers = []

    for x in config:
        probabilities.append(x["probability"])
        samplers.append(construct_single_sampler(x))

    scandots_noise_sampler = RandomlyChoosedSampler(samplers, probabilities)

    return scandots_noise_sampler