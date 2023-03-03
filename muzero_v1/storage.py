import math
from typing import List

import numpy

from .action import Node, Action
from .globals_ import Player, Winner
from .min_max_stats import MinMaxStats
from .model import Network, NetworkOutput, make_uniform_network
from .utils import softmax_sample


class SharedStorage(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self._networks = {}

    def latest_network(self):
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(self.config, self.device)

    def old_network(self):
        if self._networks:
            return self._networks[min(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(self.config, self.device)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(
        node: Node,
        to_play: Player,
        actions: List[Action],
        network_output: NetworkOutput
):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


def add_exploration_noise(config, node):
    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def ucb_score(config, parent, child, min_max_stats) -> float:
    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    pb_c = math.log(
        (parent.visit_count + config.pb_c_base + 1) / config.pb_c_base
    ) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


def select_child(config, node, min_max_stats):
    # Select the child with the highest UCB score.
    _, action, child = max(
        (
            ucb_score(config, node, child, min_max_stats),
            action,
            child
        ) for action, child in node.children.items())
    return action, child


def backpropagate(search_path, value, to_play, discount, min_max_stats):
    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def run_mcts(config, root, action_history, network):
    # Core Monte Carlo Tree Search algorithm.
    # To decide on an action, we run N simulations, always starting at the root of
    # the search tree and traversing the tree according to the UCB formula until we
    # reach a leaf node.
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     history.last_action())
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(),
                      config.discount, min_max_stats)


def select_action(config, num_moves, node, network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
    _, action = softmax_sample(visit_counts, t)
    return action


# Battle against random agents
def vs_random(config, network, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            if turn:
                root = Node(0)
                current_observation = game.make_image(-1)
                expand_node(root, game.to_play(), game.legal_actions(),
                            network.initial_inference(current_observation))
                add_exploration_noise(config, root)
                run_mcts(config, root, game.action_history(), network)
                action = select_action(config, len(game.history), root, network)
            else:
                action = numpy.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn
        if ((game.environment.winner == Winner.white and first_turn)
                or (game.environment.winner == Winner.black and not first_turn)):
            r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
              or (game.environment.winner == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results


def random_vs_random(config, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            action = numpy.random.choice(game.legal_actions())
            game.apply(action)
            turn = not turn
        if ((game.environment.winner == Winner.white and first_turn)
                or (game.environment.winner == Winner.black and not first_turn)):
            r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
              or (game.environment.winner == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results


def latest_vs_older(config, last, old, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            if turn:
                root = Node(0)
                current_observation = game.make_image(-1)
                expand_node(root, game.to_play(), game.legal_actions(),
                            last.initial_inference(current_observation))
                add_exploration_noise(config, root)
                run_mcts(config, root, game.action_history(), last)
                action = select_action(config, len(game.history), root, last)
            else:
                root = Node(0)
                current_observation = game.make_image(-1)
                expand_node(root, game.to_play(), game.legal_actions(),
                            old.initial_inference(current_observation))
                add_exploration_noise(config, root)
                run_mcts(config, root, game.action_history(), old)
                action = select_action(config, len(game.history), root, old)
            game.apply(action)
            turn = not turn
        if ((game.environment.winner == Winner.white and first_turn)
                or (game.environment.winner == Winner.black and not first_turn)):
            r = 1
        elif ((game.environment.winner == Winner.black and first_turn)
              or (game.environment.winner == Winner.white and not first_turn)):
            r = -1
        results[r] = results.get(r, 0) + 1
    return results
