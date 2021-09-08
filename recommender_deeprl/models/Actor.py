import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor
    """

    def __init__(self, input_dim, output_dim, hidden_size, init_w=2e-4):
        super(Actor, self).__init__()

        self.drop_layer = nn.Dropout(p=0.5)

        self.action_layer_1 = nn.Linear(input_dim, hidden_size)
        self.action_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.action_layer_3 = nn.Linear(hidden_size, output_dim)

        self.action_layer_3.weight.data.uniform_(-init_w, init_w)
        self.action_layer_3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, tanh=False):
        """
        :param state: state
        :param tanh: whether to use tahn as action activation
        :return: action
        """
        action = F.relu(self.action_layer_1(state))
        action = self.drop_layer(action)
        action = F.relu(self.action_layer_2(action))
        action = self.drop_layer(action)
        action = self.action_layer_3(action)
        if tanh:
            action = F.tanh(action)
        return action
