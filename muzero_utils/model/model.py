import typing
from typing import Dict, List

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..globals_ import num_filters, num_blocks, device
from ..muzero_components.action import Action


# Nets
class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))


class Representation(nn.Module):
    ''' Conversion from observation to inner abstract state '''

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        return h


class Prediction(nn.Module):
    '''
    Policy and value prediction from inner abstract state
    '''

    def __init__(self, action_shape):
        super().__init__()
        self.board_size = 42
        self.action_size = action_shape

        self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
        self.conv_p2 = Conv(4, 1, 1)

        self.conv_v = Conv(num_filters, 4, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

    def forward(self, rp):
        h_p = F.relu(self.conv_p1(rp))
        h_p = self.conv_p2(h_p).view(-1, self.action_size)

        h_v = F.relu(self.conv_v(rp))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

        # range of value is -1 ~ 1
        return F.softmax(h_p, dim=-1), torch.tanh(h_v)


class Dynamics(nn.Module):
    '''Abstruct state transition'''

    def __init__(self, rp_shape, act_shape):
        super().__init__()
        self.rp_shape = rp_shape
        self.layer0 = Conv(rp_shape[0] + act_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, rp, a):
        h = torch.cat([rp, a], dim=1)
        h = self.layer0(h)
        for block in self.blocks:
            h = block(h)
        return h


class Network(nn.Module):

    def __init__(self, action_space_size: int):
        super().__init__()
        self.steps = 0
        self.action_space_size = action_space_size
        input_shape = (2, 6, 7)
        rp_shape = (num_filters, *input_shape[1:])
        self.representation = Representation(input_shape).to(device)
        self.prediction = Prediction(action_space_size).to(device)
        self.dynamics = Dynamics(rp_shape, (2, 6, 7)).to(device)
        self.eval()

    def predict_initial_inference(self, x):
        assert x.ndim in (3, 4)
        assert x.shape == (2, 6, 7) or x.shape[1:] == (2, 6, 7)
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 2, 6, 7)

        x = torch.Tensor(x).to(device)
        h = self.representation(x)
        policy, value = self.prediction(h)

        if orig_x.ndim == 3:
            return h[0], policy[0], value[0]
        else:
            return h, policy, value

    def predict_recurrent_inference(self, x, a):

        if x.ndim == 3:
            x = x.reshape(1, 2, 6, 7)

        a = numpy.full((1, 2, 6, 7), a)

        g = self.dynamics(x, torch.Tensor(a).to(device))
        policy, value = self.prediction(g)

        return g[0], policy[0], value[0]

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        h, p, v = self.predict_initial_inference(image.astype(numpy.float32))
        return NetworkOutput(v, 0, p, h)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        g, p, v = self.predict_recurrent_inference(hidden_state, action)
        return NetworkOutput(v, 0, p, g)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps
