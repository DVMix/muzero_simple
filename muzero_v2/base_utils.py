import collections
from typing import List, Optional
import enum

import torch


MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])
num_filters = 2
num_blocks = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")


class MinMaxStats(object):
    """
        A class that holds the min-max values of the tree.
    """

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Action(object):
    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other

    def __gt__(self, other):
        return self.index > other


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ActionHistory(object):
    """
        Simple history container used inside the search.
        Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [i for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        if len(self.history) % 2 == 0:
            return Player.white
        else:
            return Player.black
