import torch
from torch import nn
from typing import Any, Tuple, Union

MIN_SIGMA = -20
MAX_SIGMA = 2


class Net(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 0,
            hidden_layers: Tuple[int, ...] = (128,),
            activation: str = 'tanh',
            output_activation: Union[None, str] = None,
    ) -> None:
        super().__init__()
        assert len(hidden_layers) >= 1
        assert activation is not None
        activation = activation.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out = []

        net = []
        dim0 = input_dim
        for dim1 in hidden_layers:
            net.append(nn.Linear(dim0, dim1))
            if activation == 'tanh':
                net.append(nn.Tanh())
            elif activation == 'relu':
                net.append(nn.ReLU(inplace=True))
            dim0 = dim1
        if output_dim > 0:
            net.append(nn.Linear(dim0, output_dim))
            if output_activation is not None:
                output_activation = output_activation.lower()
                if output_activation == 'tanh':
                    net.append(nn.Tanh())
                elif output_activation == 'sigmod':
                    net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class GaussianNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 0,
            hidden_layers: Tuple[int, ...] = (128,),
    ) -> None:
        super(GaussianNet, self).__init__()
        self.net = Net(input_dim, 0, hidden_layers, output_activation='tanh')
        self.mu = nn.Linear(hidden_layers[-1], output_dim)
        self.sigma = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        net_out = self.net(s)
        mu = torch.tanh(self.mu(net_out))
        sigma = torch.clamp(self.sigma(net_out), min=MIN_SIGMA, max=MAX_SIGMA).exp()
        return (mu, sigma)
