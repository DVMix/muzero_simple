import os
import typing
from hashlib import sha1
from typing import Dict, List

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .base_utils import num_blocks, num_filters, Action, device
except:
    from base_utils import num_blocks, num_filters, Action, device


# Nets
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, activation=nn.ReLU, batch_norm=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True
        )
        self.activation = activation()
        self.bn = batch_norm(out_channels)

    def forward(self, x, activation=True):
        h = self.conv(x)
        h = self.activation(h)
        h = self.bn(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU, batch_norm=nn.BatchNorm2d):
        super().__init__()
        self.conv_f = Conv(
            in_channels, in_channels, kernel_size, groups=in_channels, activation=activation, batch_norm=batch_norm
        )
        self.conv_dwc = Conv(
            in_channels, out_channels, kernel_size=1, groups=1, activation=activation, batch_norm=batch_norm
        )
        self.batch_norm_f = batch_norm(in_channels)

    def forward(self, x):
        # F.relu(x + (self.conv(x)))
        h = self.conv_f(x)
        h = h + x
        # h = self.batch_norm_f(h)
        h = self.conv_dwc(h)
        return h


class Representation(nn.Module):
    """
        Conversion from observation to inner abstract state
    """

    def __init__(self, input_shape, kernel_size=3, activation=nn.ReLU, batch_norm=nn.BatchNorm2d, avalanche=True):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        modules = []

        in_ch = self.input_shape[0]
        out_ch = in_ch * 2
        self.layer0 = Conv(in_ch, out_ch, kernel_size, activation=activation, batch_norm=batch_norm)
        for idx in range(num_blocks):
            if avalanche:
                if idx == 0:
                    in_ch = out_ch
                    out_ch = out_ch * 2
                elif idx < num_blocks - 1:
                    if idx < 4:
                        in_ch = out_ch
                        out_ch = in_ch * 2
                    else:
                        in_ch = out_ch
                        out_ch = int(in_ch / 2)
                else:
                    in_ch = out_ch
                    out_ch = self.input_shape[0]
            else:
                in_ch, out_ch = 2, 2
            modules.append(ResidualBlock(in_ch, out_ch, kernel_size, activation=activation, batch_norm=batch_norm))

        self.blocks = nn.ModuleList(modules)
        self.activation = activation()
        self.batch_norm = batch_norm(self.input_shape[0])

    def forward(self, x):
        h = self.layer0(x)
        for block in self.blocks:
            h = block(h)
        return h


class Prediction(nn.Module):
    """
        Policy and value prediction from inner abstract state
    """

    def __init__(self, in_channels, action_shape, kernel_size=3,
                 activation=nn.Tanh, batch_norm=nn.BatchNorm2d, avalanche=True):
        super().__init__()
        self.board_size = 42
        self.action_size = action_shape
        if isinstance(in_channels, tuple) or isinstance(in_channels, list):
            if len(in_channels) == 3:
                in_ch = in_channels[0]
            elif len(in_channels) == 4:
                in_ch = in_channels[1]
            else:
                raise ValueError('Prediction: in_channels wrong format')
        elif isinstance(in_channels, np.ndarray) or isinstance(in_channels, torch.Tensor):
            if len(in_channels.shape) == 3:
                in_ch = in_channels.shape[0]
            elif len(in_channels.shape) == 4:
                in_ch = in_channels.shape[1]
            else:
                raise ValueError('Prediction: in_channels wrong format')
        else:
            raise ValueError('Prediction: in_channels wrong format')
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.MaxPool2d(2)
        modules = []
        out_ch = in_ch * 2
        levels = np.floor(np.log2(min(in_channels[-2:]))).astype(int) - 1
        for idx in range(levels):
            module = nn.Sequential(
                Conv(in_ch, out_ch, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm),
                self.max_pooling
            )
            modules.append(module)
            in_ch = out_ch
            out_ch = in_ch * 2
        self.blocks = nn.ModuleList(modules)

        hidden_state = in_ch
        # num_point for Bezier curve strategy selection
        self.policy_importance = nn.Linear(hidden_state, self.action_size, bias=False)
        self.classification_levels = []

        for i in range(2, 2 + self.action_size):
            self.classification_levels.append(nn.Linear(hidden_state, i * 2, bias=False))
        self.value = nn.Linear(hidden_state, 1, bias=True)

    def forward(self, rp):
        # h_p = F.relu(self.conv_p1(rp))
        # h_p = self.conv_p2(h_p).view(-1, self.action_size)
        for block in self.blocks:
            rp = block(rp)
        n, c, _, _ = rp.shape
        rp = self.global_pooling(rp).view(n, c)
        policy = F.softmax(self.policy_importance(rp), dim=-1)  # .argmax(dim=1)
        point_coordinates = {
            idx: self.classification_levels[idx](rp).reshape(n, -1, 2)
            for idx in range(self.action_size)
        }
        value = torch.tanh(self.value(rp))
        return policy, value, point_coordinates


class Dynamics(nn.Module):
    """
        Abstruct state transition
    """

    def __init__(self, input_shape, rp_shape, kernel_size=3, activation=nn.ReLU,
                 batch_norm=nn.BatchNorm2d, avalanche=True):
        super().__init__()
        self.input_shape = input_shape
        self.rp_shape = rp_shape
        self.activation = activation
        self.batch_norm = batch_norm

        in_ch = self.input_shape[0]
        out_ch = in_ch * 2
        self.layer0 = Conv(in_ch, out_ch, kernel_size, activation=activation, batch_norm=batch_norm)
        modules = []

        for idx in range(num_blocks):
            if idx == 0:
                in_ch = out_ch
                out_ch = out_ch * 2
            elif idx < num_blocks - 1:
                if idx < 4:
                    in_ch = out_ch
                    out_ch = in_ch * 2
                else:
                    in_ch = out_ch
                    out_ch = int(in_ch / 2)
            else:
                in_ch = out_ch
                out_ch = self.input_shape[0]
            modules.append(ResidualBlock(in_ch, out_ch, kernel_size, activation=activation, batch_norm=batch_norm))
        self.blocks = nn.ModuleList(modules)

    def forward(self, state, action):
        h = torch.cat([state, action], dim=1)
        h = self.layer0(state)
        for block in self.blocks:
            h = block(h)

        return h


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float] or str
    coordinates: Dict[str, list]


def torch2numpy(h):
    return h.cpu().detach().numpy()


def load_image(path):
    return torch.load(path)


def save_image(h):
    hash_value = sha1(torch2numpy(h)).hexdigest()
    save_name = f'./states/{hash_value}.ptt'
    if not os.path.exists(save_name):
        torch.save(obj=h, f=save_name)
    return save_name


class Network(nn.Module):

    def __init__(self, input_shape: tuple or list = (1, 512, 512), action_space_size: int = 7):
        super().__init__()
        rp_shape = (num_filters, *input_shape[1:])
        self.steps = 0
        self.input_shape = input_shape
        self.action_space_size = action_space_size
        self.representation = Representation(input_shape).to(device)
        self.prediction = Prediction(input_shape, action_space_size).to(device)
        self.dynamics = Dynamics(input_shape, rp_shape).to(device)
        self.eval()

    def predict_initial_inference(self, x):
        assert x.ndim in (3, 4)
        orig_x = x
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        x = torch.Tensor(x).to(device)
        h = self.representation(x)

        policy, value, coordinates = self.prediction(h)

        if orig_x.ndim == 3:
            return h[0], policy[0], value[0], coordinates
        else:
            return h, policy, value, coordinates

    def predict_recurrent_inference(self, x, a):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0) if isinstance(x, numpy.ndarray) else x.unsqueeze(0)

        a = numpy.full(x.shape, a['action'] if isinstance(a, dict) else a)
        a = torch.Tensor(a).to(device)
        g = self.dynamics(x, a)
        policy, value, coordinates = self.prediction(g)
        return g[0], policy[0], value[0], coordinates

    def initial_inference(self, image, return_tensor=False) -> NetworkOutput:
        # representation + prediction function
        image = image.astype(numpy.float32)
        h, p, v, c = self.predict_initial_inference(image)
        if not return_tensor:
            h = save_image(h)
        return NetworkOutput(v, 0, p, h, c)

    def recurrent_inference(self, hidden_state, action, return_tensor=False) -> NetworkOutput:
        # dynamics + prediction function
        g, p, v, c = self.predict_recurrent_inference(hidden_state, action)
        if not return_tensor:
            g = save_image(g)
        return NetworkOutput(v, 0, p, g, c)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps


def main():
    state0 = torch.zeros(1, 1, 512, 512)
    net_repr = Representation(state0.shape[1:])
    net_pred = Prediction(1, 2)
    rp_shape = [1, *state0.shape[1:]]
    net_dyn = Dynamics(rp_shape, state0.shape[1:])

    state1 = net_repr(state0)
    print(state1.shape)
    point_coordinates, value = net_pred(state1)
    print(point_coordinates, value)
    latent_state = net_dyn(state1, point_coordinates)
    print(latent_state.shape)
    pass


if __name__ == '__main__':
    main()
