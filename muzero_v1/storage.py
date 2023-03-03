from .model import Network, make_uniform_network


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
