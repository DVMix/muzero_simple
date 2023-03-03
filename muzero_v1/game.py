from typing import List

import numpy

from .action import Action, ActionHistory, Node
from .globals_ import Player
from .mode import add_exploration_noise, expand_node, run_mcts, select_action


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, environment, action_space_size: int, discount: float):
        self.environment = environment  # Game specific environment.
        self.environment.reset()
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        # Game specific termination rules.
        return self.environment.done

    def legal_actions(self) -> List[Action]:
        # Game specific calculation of legal actions.
        return self.environment.legal_actions()

    def apply(self, action: Action):
        reward = self.environment.step(action)
        reward = reward if self.environment.turn % 2 != 0 and reward == 1 else -reward
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        # Game specific feature planes.
        # o = Environment().reset()
        o = self.environment.reset()

        for current_index in range(0, state_index):
            o.step(self.history[current_index])

        black_ary, white_ary = o.black_and_white_plane()
        state = [black_ary, white_ary] if o.player_turn() == Player.black else [white_ary, black_ary]
        return numpy.array(state)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                    to_play: Player):
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
                targets.append((value, self.rewards[current_index],
                                self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        return self.environment.player_turn

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)


# ----------------------------------------------------------------------------------------------------------------------
# Part 1: Self-Play
# ----------------------------------------------------------------------------------------------------------------------
def run_selfplay(config, storage, replay_buffer):
    # Each self-play job is independent of all others; it takes the latest network
    # snapshot, produces a game and makes it available to the training job by
    # writing it to a shared replay buffer.
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)


def play_game(config, network):
    # Each game is produced by starting at the initial board position, then
    # repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    # of the game is reached.
    game = config.new_game()
    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(),
                    network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game
