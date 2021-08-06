from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


class Actor(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hiddens=128):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens

        self.hidden = nn.Linear(self.n_states, self.n_hiddens)
        self.logits = nn.Linear(self.n_hiddens, self.n_actions)

        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity="relu")
        self.hidden.bias.data.zero_()
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.hidden(x))
        probs = F.softmax(self.logits(x), dim=-1)

        return Categorical(probs)


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
