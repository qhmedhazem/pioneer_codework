import torch.nn as nn


class LossBase(nn.Module):

    def __init__(self):
        super().__init__()
        self._logs = {}
        self._metrics = {}

    @property
    def logs(self):
        return self._logs

    @property
    def metrics(self):
        return self._metrics

    def add_metric(self, key, val):
        self._metrics[key] = val.detach()
