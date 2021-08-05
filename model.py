from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal


class Actor(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds, n_hiddens=128):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens
        assert action_bounds[0] == -action_bounds[1], "computation of bounds of the mu should change!"
        self.action_bounds = action_bounds

        self.hidden = nn.Linear(self.n_states, self.n_hiddens)
        self.mu = nn.Linear(self.n_hiddens, self.n_actions)
        self.sigma = nn.Linear(self.n_hiddens, self.n_actions)

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity="relu")
        self.hidden.bias.data.zero_()
        nn.init.xavier_uniform_(self.mu.weight)
        self.mu.bias.data.zero_()
        nn.init.kaiming_normal_(self.sigma.weight, nonlinearity="relu")
        self.sigma.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.hidden(x))
        mu = torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-4

        mu = mu * self.action_bounds[1]
        # mu = torch.clamp(mu, self.action_bounds[0], self.action_bounds[1])

        return Normal(mu, sigma)


class Critic(nn.Module, ABC):
    def __init__(self, n_states, n_hiddens=128):
        super(Critic, self).__init__()
        self.n_states = n_states
        self.n_hiddens = n_hiddens

        self.hidden = nn.Linear(self.n_states, self.n_hiddens)
        self.value = nn.Linear(self.n_hiddens, 1)

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity="relu")
        self.hidden.bias.data.zero_()
        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.hidden(x))
        value = self.value(x)

        return value
