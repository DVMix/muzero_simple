import numpy
from .globals_ import KnownBounds, device
from .muzero_components.config import MuZeroConfig
from .model.model import Network


# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    if temperature == 0:
        temperature = 1
    distribution = numpy.array(distribution) ** (1 / temperature)
    p_sum = distribution.sum()
    sample_temp = distribution / p_sum
    return 0, numpy.argmax(numpy.random.multinomial(1, sample_temp, 1))


def launch_job(f, *args):
    f(*args)


def make_uniform_network():
    return Network(make_connect4_config().action_space_size).to(device)


def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=action_space_size,
        max_moves=max_moves,
        discount=1.0,
        dirichlet_alpha=dirichlet_alpha,
        num_simulations=10,
        batch_size=64,
        td_steps=max_moves,  # Always use Monte Carlo return.
        num_actors=1,
        lr_init=lr_init,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1)
    )


def make_connect4_config() -> MuZeroConfig:
    return make_board_game_config(action_space_size=7, max_moves=20, dirichlet_alpha=0.03, lr_init=0.01)
