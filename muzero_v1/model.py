import typing
from typing import Dict, List

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .action import Action
from .globals_ import num_blocks, num_filters


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
    """
        Conversion from observation to inner abstract state
    """

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
    """
        Policy and value prediction from inner abstract state
    """

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
    """
        Abstract state transition
    """

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

    def __init__(self, action_space_size: int, device='cpu'):
        super().__init__()
        self.steps = 0
        self.device = device
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

        x = torch.Tensor(x).to(self.device)
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

        g = self.dynamics(x, torch.Tensor(a).to(self.device))
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


# ----------------------------------------------------------------------------------------------------------------------
# Part 2: Training
# ----------------------------------------------------------------------------------------------------------------------
def train_network(config, storage, replay_buffer, device='cpu'):
    network = Network(config.action_space_size, device)
    while True:
        optimizer = optim.SGD(
            network.parameters(),
            lr=0.01,
            weight_decay=config.lr_decay_rate,
            momentum=config.momentum
        )

        while not len(replay_buffer.buffer) > 0:
            pass

        for i in range(config.training_steps):
            print(i, config.training_steps)
            if i % config.checkpoint_interval == 0 and i > 0:
                storage.save_network(i, network)
                # Test against random agent
                # vs_random_once = vs_random(network)
                # print('network_vs_random = ', sorted(vs_random_once.items()), end='\n')
                # vs_older = latest_vs_older(storage.latest_network(), storage.old_network())
                # print('lastnet_vs_older = ', sorted(vs_older.items()), end='\n')
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            update_weights(batch, network, optimizer)
        storage.save_network(config.training_steps, network)


def update_weights(batch, network, optimizer, device='cpu'):
    network.train()

    p_loss, v_loss = 0, 0

    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_logits))

        for prediction, target in zip(predictions, targets):
            if (len(target[2]) > 0):
                _, value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target

                p_loss += torch.sum(-torch.Tensor(numpy.array(target_policy)).to(device) * torch.log(policy_logits))
                v_loss += torch.sum((torch.Tensor([target_value]).to(device) - value) ** 2)

    optimizer.zero_grad()
    total_loss = (p_loss + v_loss)
    total_loss.backward()
    optimizer.step()
    network.steps += 1
    print('p_loss %f v_loss %f' % (p_loss / len(batch), v_loss / len(batch)))


def make_uniform_network(config, device):
    # return Network(make_connect4_config().action_space_size).to(device)
    return Network(config.action_space_size).to(device)
