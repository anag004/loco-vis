from turtle import forward
import torch
import torch.nn as nn
import sys
import torchvision

sys.path.append("../../../mobilenetv3.pytorch/")

from mobilenetv3 import mobilenetv3_large, mobilenetv3_small

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, env_cfg) -> None:
        super().__init__()

        real_world_obs_dim = env_cfg.env.num_observations - env_cfg.env.n_scan - env_cfg.env.n_priv - 3

        if hasattr(env_cfg.depth, "depth_rnn_num_layers"):
            num_layers = env_cfg.depth.depth_rnn_num_layers
        else:
            num_layers = 1

        if hasattr(env_cfg.depth, "depth_rnn_hidden_dim"):
            hidden_dim = env_cfg.depth.depth_rnn_hidden_dim
        else:
            hidden_dim = 512

        self.rnn = nn.GRU(input_size=real_world_obs_dim + 32, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers)
        
        if hasattr(env_cfg.depth, "depth_latent_size"):
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, env_cfg.depth.depth_latent_size),
                nn.ReLU()
            )
            
        self.hidden_states_list = []
        self.hidden_states = None

    def forward(self, compressed_depth_image, proprioception):
        depth_latent = torch.cat((compressed_depth_image, proprioception), dim=-1)
        output, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        self.hidden_states_list.append(self.hidden_states)

        result = output[:, -1, :]

        if hasattr(self, "output"):
            result = self.output(result)

        return result 

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()
        self.hidden_states_list = [self.hidden_states]

class FCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation="relu", num_frames=1):
        super().__init__()

        self.num_frames = num_frames

        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_state_dim + prop_dim + 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.ReLU()

    def forward(self, prop, images, hidden_state):
        images_compressed = self.image_compression(images)
        
        latent = self.fc(torch.cat([
            images_compressed, 
            prop,
            hidden_state
        ], dim=-1))

        latent = self.output_activation(latent)

        return latent

class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation="relu", num_frames=1):
        super().__init__()

        self.num_frames = num_frames

        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            nn.ReLU(),
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.ReLU()

    def forward(self, prop, images, hidden_state):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent

class DepthOnlyFCBackbone80x93xN(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation="relu", num_frames=1):
        super().__init__()

        self.num_frames = num_frames

        self.image_compression = nn.Sequential(
            # [1, 80, 93]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            # [32, 76, 89]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 38, 44]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            # [64, 36, 42]
            nn.Flatten(),
            nn.Linear(self.num_frames * 64 * 36 * 42, 128),
            nn.ReLU(),
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.ReLU()

    def forward(self, prop, images, hidden_state):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent

class DepthOnlyFCBackbone80x93x2(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation="relu", num_frames=1):
        super().__init__()

        self.num_frames = num_frames

        self.image_compression = nn.Sequential(
            # [1, 80, 186]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            # [32, 76, 182]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 38, 91]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            # [64, 36, 89]
            nn.Flatten(),
            nn.Linear(64 * 36 * 89, 128),
            nn.ReLU(),
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.ReLU()

    def forward(self, prop, images, hidden_state):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent

class DepthOnlyViTBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation="relu", num_frames=1):
        super().__init__()

        self.num_frames = num_frames

        self.vit = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=4
        )

        self.resize_transform = torchvision.transforms.Resize((90, 60), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.unfold = nn.Unfold(kernel_size=15, stride=15)
        self.output_linear_layer = nn.Linear(128, scandots_output_dim)
        self.image_embedding = nn.Linear(225, 128)

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.ReLU()  
        
    def get_attention_mask(self, images):
        with torch.no_grad():
            attention_mask_list = []
            images_patchified = self.patchify_images(images) # [B, n_patches, patch_height, patch_width]
            x = self.image_embedding(images_patchified) # [B, n_patches, embedding_dim]

            for layer in self.vit.layers:
                _, attention_mask = layer.self_attn(x, x, x)
                attention_mask = attention_mask[:, 0, :].reshape((-1, 4, 6))
                attention_mask_list.append(attention_mask)
                x = layer(x)

        attention_mask_list = torch.stack(attention_mask_list, dim=1)
        return attention_mask_list

    def patchify_images(self, images):
        # images have dim [B, 58, 87] and need to be sliced into patches of size [15x15]
        images = self.resize_transform(images)
        images = self.unfold(images).permute(0, 2, 1)

        return images

    def forward(self, prop, images, hidden_state):
        images_patchified = self.patchify_images(images) # [B, n_patches, patch_height, patch_width]
        images_embedded = self.image_embedding(images_patchified) # [B, n_patches, embedding_dim]

        latent = self.vit(images_embedded)
        latent = latent[:, 0, :]
        latent = self.output_linear_layer(latent)
        latent = self.output_activation(latent)

        return latent

class MobileNetBackbone(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation="relu", num_frames=1, network_type="small"):
        super().__init__()

        if num_frames != 1:
            raise Exception("Only num_frames = 1 supported with MobileNet backbone")

        if network_type not in ["small", "large"]:
            raise Exception("Only small / large network_types allowed")

        self.num_frames = num_frames
        self.mobile_net, self.num_features = self.create_mobile_net(network_type)

        self.feature_transform = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.ReLU()

    def create_mobile_net(self, network_type):
        if network_type == "small":
            net = mobilenetv3_small()
            net.load_state_dict(torch.load('../../../mobilenetv3.pytorch/pretrained/mobilenetv3-small-55df8e1f.pth'))
            num_features = 96 * 2 * 3
        
        if network_type == "large":
            net = mobilenetv3_large()
            net.load_state_dict(torch.load('../../../mobilenetv3.pytorch/pretrained/mobilenetv3-large-1cd25616.pth'))
            num_features = 160 * 2 * 3
        
        return net, num_features

    def forward(self, prop, images, hidden_state):
        images = images.repeat(1, 3, 1, 1)
        features = self.mobile_net.features(images)
        features = self.feature_transform(features)
        features = self.output_activation(features)

        return features
