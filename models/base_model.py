import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self._logs = {}
        self._losses = {}

        self._input_keys = ["rgb"]

    def _forward_unimplemented(self, *args):
        pass

    @property
    def logs(self):
        return self._logs

    @property
    def losses(self):
        return self._losses

    def add_loss(self, key, val):
        self._losses[key] = val.detach()

    def forward(self, batch, return_logs=False, **kwargs):
        raise NotImplementedError(
            "Please implement forward function in your own subclass model."
        )
