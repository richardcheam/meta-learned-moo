from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms 

from enum import Enum
class ImageSize(Enum):
    IMG64x64 = 12 * 12
    IMG32x32 = 4 * 4
    IMG36x36 = 6 * 6

# help function to compute the output size of the conv layers
def compute_conv_output_size(
    input_size: int,
    kernel_sizes: list[int],
    n_conv_layers: int,
    pool_kernel: int = 2,
) -> int:
    size = input_size
    for i in range(n_conv_layers):
        size = size - kernel_sizes[i] + 1  # conv
        size = size // pool_kernel          # maxpool
    return size


class LeNetHyper(nn.Module):
    """LeNet Hypernetwork"""

    def __init__(
        self,
        kernel_size: List[int],
        ray_hidden_dim=100,
        out_dim=[1, 1],
        target_hidden_dim=50,
        n_kernels=10,
        n_conv_layers=2,
        n_hidden=1,
        n_tasks=2,
        img_size:ImageSize=ImageSize.IMG64x64,
    ):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks

        assert len(kernel_size) == n_conv_layers, (
            "kernel_size is list with same dim as number of "
            "conv layers holding kernel size for each conv layer"
        )

        # Projection of the ray vector
        # output of size ray_hidden_dim
        self.ray_mlp = nn.Sequential(
            nn.Linear(n_tasks, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        # Weights of the first convolution layer
        self.conv_0_weights = nn.Linear(
            ray_hidden_dim, n_kernels * kernel_size[0] * kernel_size[0]
        )
        # Bias of the first convolution layer
        self.conv_0_bias = nn.Linear(ray_hidden_dim, n_kernels)

        # Run all the convolution layers
        for i in range(1, n_conv_layers):
            # previous number of kernels
            p = 2 ** (i - 1) * n_kernels
            # current number of kernels
            c = 2 ** i * n_kernels

            # Add the i-th convolution layer to the network
            setattr(
                self,
                f"conv_{i}_weights",
                nn.Linear(ray_hidden_dim, c * p * kernel_size[i] * kernel_size[i]),
            )
            setattr(self, f"conv_{i}_bias", nn.Linear(ray_hidden_dim, c))

        # output size of last conv layer of target network
        # depends on the size of the input images
        # latent = img_size.value (this was hardcoded before for dSprites)
        spatial = compute_conv_output_size(
            input_size=36, # MultiMNIST
            kernel_sizes=kernel_size,
            n_conv_layers=n_conv_layers,
        )
        latent = spatial * spatial
        self.hidden_0_weights = nn.Linear(
            ray_hidden_dim,
            target_hidden_dim * (2 ** (n_conv_layers - 1) * n_kernels) * latent
        )
        # before with dSprites:
        # self.hidden_0_weights = nn.Linear(
        #     ray_hidden_dim, target_hidden_dim * 2 ** i * n_kernels * latent # 
        # )
        self.hidden_0_bias = nn.Linear(ray_hidden_dim, target_hidden_dim)

        # for each task, we have to generate the weights and bias
        # of the target newtork's final layer
        for j in range(n_tasks):
            setattr(
                self,
                f"task_{j}_weights",
                nn.Linear(ray_hidden_dim, target_hidden_dim * out_dim[j]),
            )
            setattr(self, f"task_{j}_bias", nn.Linear(ray_hidden_dim, out_dim[j]))

    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if "task" not in n])

    def forward(self, ray):
        # calculate the features according to the given ray
        features = self.ray_mlp(ray)

        out_dict = {}
        layer_types = ["conv", "hidden", "task"]

        # for each layer of the target network,
        # compute the weights and bias
        for i in layer_types:
            if i == "conv":
                n_layers = self.n_conv_layers
            elif i == "hidden":
                n_layers = self.n_hidden
            elif i == "task":
                n_layers = self.n_tasks

            for j in range(n_layers):
                out_dict[f"{i}{j}.weights"] = getattr(self, f"{i}_{j}_weights")(
                    features
                )
                out_dict[f"{i}{j}.bias"] = getattr(self, f"{i}_{j}_bias")(
                    features
                ).flatten()

        return out_dict


class LeNetTarget(nn.Module):
    """LeNet target network"""

    def __init__(
        self,
        kernel_size:list[int],
        n_kernels:int=10,
        out_dim:list[int]=[1, 1],
        target_hidden_dim:int=50,
        n_conv_layers:int=2,
        n_tasks:int=2,
        img_size:int=ImageSize.IMG64x64,
    ):
        super().__init__()
        assert len(kernel_size) == n_conv_layers, (
            "kernel_size is list with same dim as number of "
            "conv layers holding kernel size for each conv layer"
        )
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.n_conv_layers = n_conv_layers
        self.n_tasks = n_tasks
        self.target_hidden_dim = target_hidden_dim
        self.img_size = img_size
        # self.resizer = transforms.Resize((32, 32)) 
        # self.padder = transforms.Pad(1)

    def forward(self, x, weights=None):
        # because the dataset is 64x64 images, if we want to
        # train on 32x32 instead, we have to resize the images
        # if (self.img_size != ImageSize.IMG64x64):
        #     x = self.resizer(x)
        #     x = self.padder(x)
        
        # first convolution layer
        x = F.conv2d(
            x,
            weight=weights["conv0.weights"].reshape(
                self.n_kernels, 1, self.kernel_size[0], self.kernel_size[0]
            ),
            bias=weights["conv0.bias"],
            stride=1,
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # following convolution layers
        for i in range(1, self.n_conv_layers):
            x = F.conv2d(
                x,
                weight=weights[f"conv{i}.weights"].reshape(
                    int(2 ** i * self.n_kernels),
                    int(2 ** (i - 1) * self.n_kernels),
                    self.kernel_size[i],
                    self.kernel_size[i],
                ),
                bias=weights[f"conv{i}.bias"],
                stride=1,
            )
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

        # flatten the output to make it suitable
        # as dense layer input
        x = torch.flatten(x, 1)

        # hidden layer
        x = F.linear(
            x,
            weight=weights["hidden0.weights"].reshape(
                self.target_hidden_dim, x.shape[-1]
            ),
            bias=weights["hidden0.bias"],
        )

        # final layers for each task
        logits = []
        for j in range(self.n_tasks):
            logits.append(
                F.linear(
                    x,
                    weight=weights[f"task{j}.weights"].reshape(
                        self.out_dim[j], self.target_hidden_dim
                    ),
                    bias=weights[f"task{j}.bias"],
                )
            )

        return logits
    

# hypernetwork for tabular data
class TabularHyperNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, ray_hidden_dim=128, n_tasks=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.ray_mlp = nn.Sequential(
            nn.Linear(n_tasks, ray_hidden_dim),
            nn.ReLU(),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(),
        )

        self.fc1_w = nn.Linear(ray_hidden_dim, hidden_dim * in_dim)
        self.fc1_b = nn.Linear(ray_hidden_dim, hidden_dim)

        self.fc2_w = nn.Linear(ray_hidden_dim, hidden_dim * hidden_dim)
        self.fc2_b = nn.Linear(ray_hidden_dim, hidden_dim)

        self.fc3_w = nn.Linear(ray_hidden_dim, hidden_dim)
        self.fc3_b = nn.Linear(ray_hidden_dim, 1)

    def forward(self, ray):
        z = self.ray_mlp(ray)

        return {
            "fc1.weight": self.fc1_w(z).view(self.hidden_dim, self.in_dim),
            "fc1.bias": self.fc1_b(z),
            "fc2.weight": self.fc2_w(z).view(self.hidden_dim, self.hidden_dim),
            "fc2.bias": self.fc2_b(z),
            "fc3.weight": self.fc3_w(z).view(1, self.hidden_dim),
            "fc3.bias": self.fc3_b(z),
        }


# stateless target network for tabular data, replicating M1 architecture used in LibMOON
class TabularTargetNet(nn.Module):
    def forward(self, x, w):
        x = F.linear(x, w["fc1.weight"], w["fc1.bias"])
        x = F.relu(x)
        x = F.linear(x, w["fc2.weight"], w["fc2.bias"])
        x = F.relu(x)
        x = F.linear(x, w["fc3.weight"], w["fc3.bias"])
        return x.squeeze(-1)

# hypernetwork for Electricity Demand (time series)
class TemporalTargetNet(nn.Module):
    """
    Target network for Electricity Demand (time series).
    Parameters are injected by name from the hypernetwork.
    """

    def __init__(self, seq_len=96, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward_impl(self, x):
        """
        Standard forward pass WITHOUT hypernet parameters.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)

    def forward(self, x, weights):
        """
        x: (B, T)
        weights: dict(name -> tensor) produced by hypernet
        """

        # Inject hypernet weights safely
        for name, param in self.named_parameters():
            param.data.copy_(weights[name])

        return self.forward_impl(x)
    

class TemporalHyperNet(nn.Module):
    """
    Hypernetwork for Electricity Demand (time series)
    Generates parameters for:
      fc1: seq_len -> hidden_dim
      fc2: hidden_dim -> 1
    """

    def __init__(self, seq_len=96, hidden_dim=128, ray_hidden_dim=128, n_tasks=2):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Ray encoder
        self.ray_mlp = nn.Sequential(
            nn.Linear(n_tasks, ray_hidden_dim),
            nn.ReLU(),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(),
        )

        # fc1 parameters
        self.fc1_w = nn.Linear(ray_hidden_dim, hidden_dim * seq_len)
        self.fc1_b = nn.Linear(ray_hidden_dim, hidden_dim)

        # fc2 parameters
        self.fc2_w = nn.Linear(ray_hidden_dim, hidden_dim)
        self.fc2_b = nn.Linear(ray_hidden_dim, 1)

    def forward(self, ray):
        z = self.ray_mlp(ray)

        return {
            "fc1.weight": self.fc1_w(z).view(self.hidden_dim, self.seq_len),
            "fc1.bias": self.fc1_b(z),
            "fc2.weight": self.fc2_w(z).view(1, self.hidden_dim),
            "fc2.bias": self.fc2_b(z),
        }

class TemporalTargetNet(nn.Module):
    """
    Target network for Electricity Demand (time series)
    """

    def __init__(self, seq_len=96, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, weights):
        x = torch.nn.functional.linear(
            x,
            weights["fc1.weight"],
            weights["fc1.bias"],
        )
        x = self.relu(x)

        x = torch.nn.functional.linear(
            x,
            weights["fc2.weight"],
            weights["fc2.bias"],
        )

        return x.squeeze(-1)