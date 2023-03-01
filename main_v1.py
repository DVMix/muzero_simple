from muzero_v1.sandbox import *


if __name__ == '__main__':
    config = make_connect4_config()
    vs_random_once = random_vs_random(config)
    print('random_vs_random = ', sorted(vs_random_once.items()), end='\n')
    network = muzero(config)
