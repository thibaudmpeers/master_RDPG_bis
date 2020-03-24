import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

three_layers = False
size_layer = 128


def weight_init(size):
    random_params_layer = np.sqrt(2./(size[0] + size[1]))
    return torch.randn(size)*random_params_layer


def return_nn_arch_params():
    return 3 if three_layers else 2, size_layer


class Critic(nn.Module):

    def __init__(self, observation_dim, action_dim):
        super(Critic, self).__init__()

        self.size_rnn_layer = size_layer
        size_layer1 = size_layer
        size_layer2 = size_layer
        size_layer3 = size_layer
        size_last_layer = size_layer3 if three_layers else size_layer2

        self.rnn_layer = nn.RNNCell(observation_dim + action_dim + 1, self.size_rnn_layer, nonlinearity='tanh')
        self.rnn_layer.weight_hh.data = weight_init(self.rnn_layer.weight_hh.data.size())
        self.rnn_layer.weight_ih.data = weight_init(self.rnn_layer.weight_ih.data.size())

        self.hidden_layer1 = nn.Linear(self.size_rnn_layer + action_dim, size_layer1)
        self.hidden_layer1.weight.data = weight_init(self.hidden_layer1.weight.data.size())

        self.hidden_layer2 = nn.Linear(size_layer1, size_layer2)
        self.hidden_layer2.weight.data = weight_init(self.hidden_layer2.weight.data.size())

        if three_layers:
            self.hidden_layer3 = nn.Linear(size_layer2, size_layer3)
            self.hidden_layer3.weight.data = weight_init(self.hidden_layer3.weight.data.size())

        self.output_layer = nn.Linear(size_last_layer, 1)
        self.output_layer.weight.data = weight_init(self.output_layer.weight.data.size())

    def forward(self, inputs, action, h_critic):
        h_critic = self.rnn_layer(inputs, h_critic)

        x = torch.cat((h_critic, action), 1)

        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        if three_layers:
            x = F.relu(self.hidden_layer3(x))
        return self.output_layer(x), h_critic

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.size_rnn_layer, requires_grad=True)


class Actor(nn.Module):

    def __init__(self, observation_dim, action_dim):
        super(Actor, self).__init__()

        self.size_rnn_layer = size_layer
        size_layer1 = size_layer
        size_layer2 = size_layer
        size_layer3 = size_layer
        size_last_layer = size_layer3 if three_layers else size_layer2

        self.rnn_layer = nn.RNNCell(observation_dim + action_dim + 1, self.size_rnn_layer, nonlinearity='tanh')
        self.rnn_layer.weight_hh.data = weight_init(self.rnn_layer.weight_hh.data.size())
        self.rnn_layer.weight_ih.data = weight_init(self.rnn_layer.weight_ih.data.size())

        self.hidden_layer1 = nn.Linear(self.size_rnn_layer, size_layer1)
        self.hidden_layer1.weight.data = weight_init(self.hidden_layer1.weight.data.size())

        self.hidden_layer2 = nn.Linear(size_layer1, size_layer2)
        self.hidden_layer2.weight.data = weight_init(self.hidden_layer2.weight.data.size())

        if three_layers:
            self.hidden_layer3 = nn.Linear(size_layer2, size_layer3)
            self.hidden_layer3.weight.data = weight_init(self.hidden_layer3.weight.data.size())

        self.output_layer = nn.Linear(size_last_layer, action_dim)
        self.output_layer.weight.data = weight_init(self.output_layer.weight.data.size())

    def forward(self, inputs, h_actor):
        h_actor = self.rnn_layer(inputs, h_actor)

        x = F.relu(self.hidden_layer1(h_actor))
        x = F.relu(self.hidden_layer2(x))
        if three_layers:
            x = F.relu(self.hidden_layer3(x))
        return self.output_layer(x), h_actor

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.size_rnn_layer, requires_grad=True)
