# Lint as: python3
"""Pseudocode description of the MuZero algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=assignment-from-no-return

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from typing import List, Optional

import numpy
import torch
import torch.optim as optim
import threading
from hashlib import sha1
import time

# custom imports
from muzero_v2.base_utils import Player, Winner, Action, ActionHistory, Node, device, KnownBounds, MinMaxStats
from muzero_v2.environment import Environment
from muzero_v2.softmaxes import softmax_sample, visit_softmax_temperature
from muzero_v2.net import Network, NetworkOutput


# ----------------------------------------------------------------------------------------------------------------------
# START: Helpers
# ----------------------------------------------------------------------------------------------------------------------
class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, configs):
        self.environment = Environment(configs)
        self.environment.reset()  # Game specific environment.
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = configs.action_space_size
        self.discount = configs.discount
        self.canvas_size = self.environment.target_image.shape

    def terminal(self) -> bool:
        # Game specific termination rules.
        return self.environment.done

    def legal_actions(self) -> List[Action]:
        # Game specific calculation of legal actions.
        return self.environment.legal_actions()

    def apply(self, action: Action):
        reward = self.environment.step(action)
        # reward = reward if self.environment.turn % 2 != 0 and reward == 1 else -reward
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(
            child['policy'].visit_count for child in root.children.values()
        )
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a]['policy'].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        # Game specific feature planes.
        o = self.environment.reset()

        for current_index in range(0, state_index):
            o.step(self.history[current_index])

        # black_ary, white_ary = o.black_and_white_plane()
        # state = [black_ary, white_ary] if o.player_turn() == Player.black else [white_ary, black_ary]
        # return numpy.array(state)
        return numpy.expand_dims(self.environment.current_image, axis=0) / 255

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i  # pytype: disable=unsupported-operands

            if current_index < len(self.root_values):
                targets.append(
                    (value, self.rewards[current_index], self.child_visits[current_index])
                )
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        return self.environment.player_turn

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)


class SharedStorage(object):

    def __init__(self, configs):
        self.configs = configs
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(
                action_space_size=self.configs.action_space_size,
                input_format=self.configs.input_format
            )

    def old_network(self) -> Network:
        if self._networks:
            return self._networks[min(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(
                action_space_size=self.configs.action_space_size,
                input_format=self.configs.input_format
            )

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


class MuZeroConfig(object):
    def __init__(
            self,
            action_space_size: int,
            input_format: tuple,
            max_moves: int,
            discount: float,
            dirichlet_alpha: float,
            num_simulations: int,
            batch_size: int,
            td_steps: int,
            num_actors: int,
            lr_init: float,
            lr_decay_steps: float,
            visit_softmax_temperature_fn,
            known_bounds: Optional[KnownBounds] = None
    ):
        # Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors
        self.input_format = input_format
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        # Training
        self.training_steps = int(100)  # 1e6
        self.checkpoint_interval = int(100)
        self.window_size = int(100)  # 1e6
        self.batch_size = batch_size
        self.num_unroll_steps = 4
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    def new_game(self, configs):
        return Game(configs=configs)


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [
            (
                g.make_image(i),
                g.history[i:i + num_unroll_steps],
                g.make_target(i, num_unroll_steps, td_steps, g.to_play())
            )
            for (g, i) in game_pos
        ]

    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return numpy.random.choice(self.buffer)

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return numpy.random.choice(len(game.history))


# ----------------------------------------------------------------------------------------------------------------------
# End: Helpers
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# START: Play modes
# ----------------------------------------------------------------------------------------------------------------------
def vs_random(network, n=100):
    # Battle against random agents
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game(config)
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


def random_vs_random(n=3):  # 100
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        game = config.new_game()
        r = 0
        while not game.terminal():
            action = numpy.random.choice(game.legal_actions())
            points = []
            for i in range(action):
                points.append(
                    [
                        numpy.random.choice(game.canvas_size[0]),
                        numpy.random.choice(game.canvas_size[1])
                    ]
                )
            game.apply(points)
        #     turn = not turn
        # if ((game.environment.winner == Winner.white and first_turn)
        #         or (game.environment.winner == Winner.black and not first_turn)):
        #     r = 1
        # elif ((game.environment.winner == Winner.black and first_turn)
        #       or (game.environment.winner == Winner.white and not first_turn)):
        #     r = -1
        # results[r] = results.get(r, 0) + 1
    return results


def latest_vs_older(last, old, n=100):
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


# ----------------------------------------------------------------------------------------------------------------------
# END: Play modes
# ----------------------------------------------------------------------------------------------------------------------
def make_board_game_config(
        action_space_size: int,
        input_format: tuple,
        max_moves: int,
        dirichlet_alpha: float,
        lr_init: float
) -> MuZeroConfig:
    return MuZeroConfig(
        action_space_size=action_space_size,
        input_format=input_format,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=5,  # 10
        batch_size=64,  # 64
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=1,  # 2
        lr_init=lr_init,
        lr_decay_steps=100e2,  # 400e3
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1))


def make_connect4_config(action_space_size, input_format) -> MuZeroConfig:
    # return make_board_game_config(action_space_size=7, max_moves=20, dirichlet_alpha=0.03, lr_init=0.01)
    return make_board_game_config(
        action_space_size=action_space_size,
        input_format=input_format,
        max_moves=20, dirichlet_alpha=0.03, lr_init=0.01
    )


def make_uniform_network(action_space_size, input_format):
    return Network(
        input_format,
        make_connect4_config(
            action_space_size=action_space_size,
            input_format=input_format
        ).action_space_size).to(device)


def launch_job(f, *args):
    f(*args)


def muzero(config: MuZeroConfig):
    # MuZero training is split into two independent parts: Network training and
    # self-play data generation.
    # These two parts only communicate by transferring the latest network checkpoint
    # from the training to the self-play, and the finished games from the self-play
    # to the training.
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)

    # # Start n concurrent actor threads
    # threads = list()
    # for _ in range(config.num_actors):
    #     t = threading.Thread(target=launch_job, args=(run_selfplay, config, storage, replay_buffer))
    #     threads.append(t)
    #
    # # Start all threads
    # for x in threads:
    #     x.start()

    run_selfplay(config, storage, replay_buffer)
    train_network(config, storage, replay_buffer)
    return storage.latest_network()


# ----------------------------------------------------------------------------------------------------------------------
# Part 1: Self-Play
# ----------------------------------------------------------------------------------------------------------------------
def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    pb_c = math.log(
        (parent.visit_count + config.pb_c_base + 1) / config.pb_c_base
    ) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    # Select the child with the highest UCB score.
    _, action, child, points = max(
        (
            ucb_score(config, node, child['policy'], min_max_stats), action, child['policy'], child['points']
        )
        for action, child in node.children.items()
    )
    return action, child, points


def expand_node(
        node: Node,
        to_play: Player,
        actions: List[Action],
        network_output: NetworkOutput
):
    # We expand a node using the value, reward and policy prediction obtained from
    # the neural network.
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a].item()) for a in actions}
    policy_sum = sum(policy.values())
    h, w = config.input_format[-2:]
    for action, p in policy.items():
        coord = network_output.coordinates[action].cpu().detach()
        coord[..., 0] *= w
        coord[..., 1] *= h
        node.children[action] = {
            'policy': Node(p / policy_sum),
            'action': action,
            'points': coord  # network_output.coordinates[action]
        }


def add_exploration_noise(config: MuZeroConfig, node: Node):
    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a]['policy'].prior = node.children[a]['policy'].prior * (1 - frac) + n * frac


def backpropagate(search_path: List[Node], value: float, to_play: Player, discount: float, min_max_stats: MinMaxStats):
    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def run_mcts(config: MuZeroConfig, root: Node, game: Game, network: Network):
    action_history =game.action_history()
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
            action, node, points = select_child(config, node, min_max_stats)
            # history.add_action(action)
            history.add_action({'action': action, 'points': points})
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(
            torch.load(parent.hidden_state),
            history.last_action()
        )
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(search_path, network_output.value, history.to_play(),
                      config.discount, min_max_stats)


def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: Network):
    visit_counts = [
        (child['policy'].visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
    _, action = softmax_sample(visit_counts, t)
    return action


def play_game(config: MuZeroConfig, network: Network) -> Game:
    # Each game is produced by starting at the initial board position, then
    # repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    # of the game is reached.
    game = config.new_game(config)
    # iter_ = 0
    while not game.terminal() and len(game.history) < config.max_moves:  # config.max_moves
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        # print(f'Iter #{iter_}')
        # iter_ += 1
        root = Node(0)
        to_play = game.to_play()
        actions = game.legal_actions()
        current_observation = game.make_image(-1)
        network_output = network.initial_inference(current_observation)
        expand_node(node=root, to_play=to_play, actions=actions, network_output=network_output)
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game, network)
        action = select_action(config, len(game.history), root, network)
        game.apply(root.children[action])  # ['points'] - action
        game.store_search_statistics(root['policy'] if isinstance(root, dict) else root)
    return game


def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    # Each self-play job is independent of all others; it takes the latest network
    # snapshot, produces a game and makes it available to the training job by
    # writing it to a shared replay buffer.
    # games_num = 0
    # games_limit = 5
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)
        if len(replay_buffer.buffer) < config.window_size:
            print(f'Current ReplayBuffer size = {len(replay_buffer.buffer)}')
        else:
            print('sleep for 10 seconds')
            time.sleep(10)
        # print(f'Games number = {games_num}')
        # if games_num == games_limit:
        #     break
        # games_num += 1


# ----------------------------------------------------------------------------------------------------------------------
# End Self-Play
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Part 2: Training
# ----------------------------------------------------------------------------------------------------------------------
def update_weights(batch, network, optimizer):
    network.train()
    p_loss, v_loss = 0, 0

    # value, rewards, child_visits = targets

    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state, coordinates = network.initial_inference(image, return_tensor=True)
        predictions = [(1.0, value, reward, policy_logits)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            result = network.recurrent_inference(hidden_state, action, return_tensor=True)
            value, reward, policy_logits, hidden_state, coordinates = result
            predictions.append((1.0 / len(actions), value, reward, policy_logits))

        for prediction, target in zip(predictions, targets):
            if len(target[2]) > 0:
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


def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    network = Network(input_shape=config.input_format, action_space_size=config.action_space_size)
    # network = Network(config.action_space_size).to(device)
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
            # if i % config.checkpoint_interval == 0 and i > 0:
            #     storage.save_network(i, network)
            #     # Test against random agent
            #     vs_random_once = vs_random(network)
            #     print('network_vs_random = ', sorted(vs_random_once.items()), end='\n')
            #     vs_older = latest_vs_older(storage.latest_network(), storage.old_network())
            #     print('lastnet_vs_older = ', sorted(vs_older.items()), end='\n')
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            update_weights(batch, network, optimizer)
        storage.save_network(config.training_steps, network)


# ----------------------------------------------------------------------------------------------------------------------
# End Training
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    config = make_connect4_config(action_space_size=2, input_format=(1, 32, 32))
    # print('random_vs_random = ', sorted(random_vs_random().items()), end='\n')
    network = muzero(config)
