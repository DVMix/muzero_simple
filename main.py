from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

# custom imports
from muzero_utils.model.network_training import train_network
from muzero_utils.muzero_components.config import MuZeroConfig
from muzero_utils.muzero_components.shared_storage import SharedStorage
from muzero_utils.muzero_components.replay_buffer import ReplayBuffer
from muzero_utils.muzero_components.self_play import run_selfplay
from muzero_utils.utils import launch_job, make_connect4_config
from muzero_utils.game import random_vs_random
# ----------------------------------------------------------------------------------------------------------------------
# |  MuZero training is split into two independent parts: Network training and                                         |
# |  self-play data generation.                                                                                        |
# |  These two parts only communicate by transferring the latest network checkpoint                                    |
# |  from the training to the self-play, and the finished games from the self-play                                     |
# |  to the training.                                                                                                  |
# ----------------------------------------------------------------------------------------------------------------------


def MuZero(config: MuZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    # Start n concurrent actor threads
    threads = list()
    for _ in range(config.num_actors):
        t = threading.Thread(
            target=launch_job,
            args=(run_selfplay, config, storage, replay_buffer)
        )
        threads.append(t)

    # Start all threads
    for x in threads:
        x.start()

    train_network(config, storage, replay_buffer)
    return storage.latest_network()


if __name__ == '__main__':
    config = make_connect4_config()
    vs_random_once = random_vs_random(config)
    print('random_vs_random = ', sorted(vs_random_once.items()), end='\n')
    network = MuZero(config)
