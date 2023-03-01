import numpy
from typing import List
from .base import Player, Environment, Node, Winner
from .muzero_components.action import Action, ActionHistory
from .muzero_components.self_play import expand_node, add_exploration_noise, run_mcts, select_action


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment().reset()  # Game specific environment.
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
        o = Environment().reset()

        for current_index in range(0, state_index):
            o.step(self.history[current_index])

        black_ary, white_ary = o.black_and_white_plane()
        state = [black_ary, white_ary] if o.player_turn() == Player.black else [white_ary, black_ary]
        return numpy.array(state)

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
        game = Game(config.action_space_size, config.discount)  # game = config.new_game()
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
