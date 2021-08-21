from abc import ABC
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class Model(nn.Module, ABC):

    def __init__(self, state_shape, n_actions):
        super(Model, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        c, w, h = state_shape
        #  https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = self.conv_shape(w, 8, 4)
        conv1_out_h = self.conv_shape(h, 8, 4)
        conv2_out_w = self.conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = self.conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = self.conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = self.conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size, out_features=512)
        self.q_value = nn.Linear(in_features=512, out_features=self.n_actions)
        self.logits = nn.Linear(in_features=512, out_features=self.n_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.q_value.weight)
        self.q_value.bias.data.zero_()
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        q_value = self.q_value(x)
        probs = F.softmax(self.logits(x), dim=1)
        dist = Categorical(probs)

        return dist, q_value, probs

    @staticmethod
    def conv_shape(input, kernel_size, stride, padding=0):
        return (input + 2 * padding - kernel_size) // stride + 1
