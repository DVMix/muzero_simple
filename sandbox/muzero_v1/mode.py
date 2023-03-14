import numpy

from .action import Node
from .globals_ import Winner
from .utils import expand_node, add_exploration_noise, run_mcts, select_action


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
        if (
                (game.environment.winner == Winner.white and first_turn) or
                (game.environment.winner == Winner.black and not first_turn)
        ):
            r = 1
        elif (
                (game.environment.winner == Winner.black and first_turn) or
                (game.environment.winner == Winner.white and not first_turn)
        ):
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
