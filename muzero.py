from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import torch

# custom imports
from muzero_v1.config import MuZeroConfig, make_board_game_config
from muzero_v1.game import run_selfplay
from muzero_v1.model import train_network
from muzero_v1.replay_buffer import ReplayBuffer
from muzero_v1.storage import SharedStorage
from muzero_v1.utils import launch_job
from muzero_v1.mode import random_vs_random


def make_connect4_config(batch_size=64, num_actors=5) -> MuZeroConfig:
    return make_board_game_config(
        batch_size=batch_size,
        num_actors=num_actors,
        action_space_size=7,
        max_moves=20,
        dirichlet_alpha=0.03,
        lr_init=0.01
    )


def muzero(config_, device_):
    # MuZero training is split into two independent parts: Network training and
    # self-play data generation.
    # These two parts only communicate by transferring the latest network checkpoint
    # from the training to the self-play, and the finished games from the self-play
    # to the training.
    storage = SharedStorage(config_, device_)
    replay_buffer = ReplayBuffer(config_)

    # Start n concurrent actor threads
    threads = list()

    for _ in range(config_.num_actors):
        t = threading.Thread(target=launch_job, args=(run_selfplay, config_, storage, replay_buffer))
        threads.append(t)

    # Start all threads
    for x in threads:
        x.start()
    # run_selfplay(config_, storage, replay_buffer)
    train_network(config_, storage, replay_buffer, device=device_)
    return storage.latest_network()


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    config = make_connect4_config(batch_size=64, num_actors=1)
    vs_random_once = random_vs_random(config)
    print('random_vs_random = ', sorted(vs_random_once.items()), end='\n')
    network = muzero(config, device)
