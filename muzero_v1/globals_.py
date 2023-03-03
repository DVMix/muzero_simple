import collections
import enum

import torch

num_filters = 2
num_blocks = 8
MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")
