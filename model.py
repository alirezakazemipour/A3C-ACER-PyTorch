from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


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

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity="relu")
        self.hidden.bias.data.zero_()
        nn.init.xavier_uniform_(self.mu.weight)
        self.mu.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.hidden(x))
        mu = self.mu(x)
        # mu = mu * self.action_bounds[1]

        return MultivariateNormal(mu, 0.3 * torch.eye(mu.size()))


class SDNCritic(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hiddens=128):
        super(SDNCritic, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens

        self.hidden_value = nn.Linear(self.n_states, self.n_hiddens)
        self.value = nn.Linear(self.n_hiddens, 1)
        self.hidden_adv = nn.Linear(self.n_states + self.n_actions, self.n_hiddens)
        self.adv = nn.Linear(self.n_hiddens, 1)

        nn.init.kaiming_normal_(self.hidden_value.weight, nonlinearity="relu")
        self.hidden_value.bias.data.zero_()
        nn.init.kaiming_normal_(self.hidden_adv.weight, nonlinearity="relu")
        self.hidden_adv.bias.data.zero_()
        nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()
        nn.init.xavier_uniform_(self.adv.weight)
        self.adv.bias.data.zero_()

    def forward(self, inputs, a, u):
        x = inputs
        x = F.relu(self.hidden_value(x))
        value = self.value(x)

        x = F.relu(self.hidden_adv(torch.cat([inputs, a], dim=-1)))
        adv = self.adv(x)

        x = F.relu(self.hidden_adv(torch.cat([inputs, u], dim=-1)))
        adv_mean = self.adv(x).mean(1, keepdim=True)

        return value + adv - adv_mean
