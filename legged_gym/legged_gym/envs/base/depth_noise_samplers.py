import torch
import numpy as np
import skimage

class PepperNoise:
    """
        Replaces random values with zeros
    """

    def __init__(self, amount=0.05) -> None:
        self.amount = amount

    def add_noise(self, depth_image):
        depth_image = skimage.util.random_noise(depth_image, mode="pepper", amount=self.amount)

        return depth_image

class QuantizationNoise:
    def __init__(self, baseline=35130, sigma_s=1/2, sigma_d=1/6) -> None:
        self.baseline = baseline
        self.sigma_s = sigma_s
        self.sigma_d = sigma_d

    def add_noise(self, depth_image):
        x1 = np.arange(depth_image.shape[0]).repeat(depth_image.shape[1])
        x2 = np.tile(np.arange(depth_image.shape[1]), depth_image.shape[0])
        
        x1 = x1 + np.random.normal() * self.sigma_s
        x1 = x1.astype("int32")
        x1 = np.clip(x1, 0, depth_image.shape[0] - 1)
        
        x2 = x2 + np.random.normal() * self.sigma_s
        x2 = x2.astype("int32")
        x2 = np.clip(x2, 0, depth_image.shape[1] - 1)

        depth_image_perturbed = depth_image[x1, x2].reshape(depth_image.shape)

        denominator = self.baseline / depth_image_perturbed
        denominator += np.random.normal(size=denominator.shape) * self.sigma_d + 0.5
        denominator = denominator.astype("int32")

        return self.baseline / denominator

class PerlinNoise:
    """
        Adds perline noise everywhere
    """
    
    # raise NotImplemented
    pass
    # def __init__(self) -> None:
        

    # def interp(self, t):
    #     return 3 * t**2 - 2 * t ** 3

    # def perlin(self, width, height, scale=10, device=None):
    #     gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    #     xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    #     ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)


    #     wx = 1 - self.interp(xs)
    #     wy = 1 - self.interp(ys)

    #     dots = 0
    #     dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    #     dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    #     dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    #     dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))

    #     return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

    # def perlin_ms(self, octaves=[1, 1, 1, 1], width=2, height=2, device=None):
    #     scale = 2 ** len(octaves)
    #     out = 0
    #     for oct in octaves:
    #         p = self.perlin(width, height, scale, device)
    #         out += p * oct
    #         scale //= 2
    #         width *= 2
    #         height *= 2
    #     return out


