import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Critic
    """

    def __init__(self, input_dim, output_dim, hidden_size, init_w=3e-5):
        super(Critic, self).__init__()

        self.drop_layer = nn.Dropout(p=0.5)

        self.critic_layer_1 = nn.Linear(input_dim, output_dim)
        self.critic_layer_2 = nn.Linear(output_dim + output_dim, hidden_size)
        self.critic_layer_3 = nn.Linear(hidden_size, 1)

        self.critic_layer_3.weight.data.uniform_(-init_w, init_w)
        self.critic_layer_3.bias.data.uniform_(-init_w, init_w)

    # def __init__(self, input_dim, output_dim, hidden_size, init_w=3e-5):
    #     super(Critic, self).__init__()
    #
    #     self.drop_layer = nn.Dropout(p=0.5)
    #
    #     self.critic_layer_1 = nn.Linear(input_dim, hidden_size)
    #     self.critic_layer_2 = nn.Linear(hidden_size, hidden_size)
    #     self.critic_layer_3 = nn.Linear(hidden_size, 1)
    #
    #     self.critic_layer_3.weight.data.uniform_(-init_w, init_w)
    #     self.critic_layer_3.bias.data.uniform_(-init_w, init_w)


    # def forward(self, state, action):
    #     """"""
    #     value = torch.cat([state, action], 0)
    #     value = F.relu(self.critic_layer_1(value))
    #     value = self.drop_layer(value)
    #     value = F.relu(self.critic_layer_2(value))
    #     value = self.drop_layer(value)
    #     value = self.critic_layer_3(value)
    #     return value


    def forward(self, state, action):
        """"""
        value = F.relu(self.critic_layer_1(state))
        value = self.drop_layer(value)
        value = torch.cat([action, value], 1)
        value = F.relu(self.critic_layer_2(value))
        value = self.drop_layer(value)
        value = self.critic_layer_3(value)
        return value
