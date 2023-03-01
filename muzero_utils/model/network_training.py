import numpy
import torch
import torch.optim as optim
from ..globals_ import device
from ..game import vs_random, latest_vs_older
from .model import Network
from ..muzero_components.config import MuZeroConfig
from ..muzero_components.shared_storage import SharedStorage
from ..muzero_components.replay_buffer import ReplayBuffer

# ----------------------------------------------------------------------------------------------------------------------
# Part 2: Training
# ----------------------------------------------------------------------------------------------------------------------


def train_network(config: MuZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network(config.action_space_size).to(device)

    while True:

        optimizer = optim.SGD(network.parameters(), lr=0.01, weight_decay=config.lr_decay_rate,
                              momentum=config.momentum)

        while not len(replay_buffer.buffer) > 0:
            pass

        for i in range(config.training_steps):
            if i % config.checkpoint_interval == 0 and i > 0:
                storage.save_network(i, network)
                # Test against random agent
                vs_random_once = vs_random(network)
                print('network_vs_random = ', sorted(vs_random_once.items()), end='\n')
                vs_older = latest_vs_older(storage.latest_network(), storage.old_network())
                print('lastnet_vs_older = ', sorted(vs_older.items()), end='\n')
            batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
            update_weights(batch, network, optimizer)
        storage.save_network(config.training_steps, network)


def update_weights(batch, network, optimizer):
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
# ----------------------------------------------------------------------------------------------------------------------
# Part 2: End Training
# ----------------------------------------------------------------------------------------------------------------------
