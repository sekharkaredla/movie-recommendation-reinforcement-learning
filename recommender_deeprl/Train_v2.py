import torch
import models
import numpy as np
import torch.optim as optim
from ML_100k.ReadUsers import get_user_data
from ML_100k.ReadItems import get_item_features
from datetime import datetime

class Train:
    def __init__(self):
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.user_states = {}
        self.buffer = models.ReplayBuffer(1000000000)



    def set_network_sizes(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

    def create_actor(self, learning_rate):
        self.actor = models.Actor(self.input_size, self.output_size, \
        self.hidden_size, learning_rate)

    def create_actor_target(self, learning_rate):
        self.actor_target = models.Actor(self.input_size, self.output_size, \
        self.hidden_size, learning_rate)

    def create_critic(self, learning_rate):
        self.critic = models.Critic(self.input_size, self.output_size, \
        self.hidden_size, learning_rate)

    def create_critic_target(self, learning_rate):
        self.critic_target = models.Critic(self.input_size, self.output_size, \
        self.hidden_size, learning_rate)


    def copy_parameters_between_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)


    def get_actor(self):
        return self.actor

    def get_critic(self):
        return self.critic


    def initialize_user_states(self, users_data):
        for each_user_id in users_data.keys():
            self.user_states[each_user_id] = models.UserState(each_user_id, \
            users_data[each_user_id]["embeddings"])

    def initialize_optimizers(self, actor_learning_rate, critic_learning_rate):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.critic_criterion = torch.nn.MSELoss()

    def get_user_state(self, user_id):
        return self.user_states[user_id].get_state_embedding()

    def add_item_to_user(self, user_id, item_embedding):
        self.user_states[user_id].add_item_embedding(item_embedding)


    def get_best_item_from_user_space_based_on_action(self, action, user_item_space, items_data):
        value = None
        selected_item_id = None
        for each_item_id in user_item_space:
            item_embeddings = items_data[each_item_id]
            multiplied = np.multiply(item_embeddings, action)
            if value is None or multiplied.sum(0) > value:
                value = multiplied.sum(0)
                selected_item_id = each_item_id
        return selected_item_id, value

    def calculate_reward_for_recommended_action_from_user(self, item_embedding, user_id):
        user_state = self.user_states[user_id]
        reward = 0.0
        if user_state.get_items() is not None:
            for each_item in user_state.get_items():
                reward = np.add(reward, each_item)
            reward = reward / reward.shape[0]
        else:
            reward = 1.0
        reward = np.multiply(reward, user_state.get_user_embeddings())
        reward = np.multiply(reward, item_embedding)
        return reward, np.sum(reward, axis = 0)


    def add_experience_to_replay_buffer(self, current_state, recommended_action, reward, next_state):
        self.buffer.push(current_state, recommended_action, reward, next_state, None)

    def update_networks(self, no_of_replay_samples, gamma, tau):
        states, actions, rewards, next_states, _ = self.buffer.sample(no_of_replay_samples)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)


        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

    def save_networks(self, path="./trained_models/implementation_v2"):
        now_time = datetime.now().strftime("%d%b%H%M")
        print(self.actor.state_dict())
        torch.save(self.actor, path + "/actor_"+str(now_time)+".pt")
        print(self.actor_target.state_dict())
        torch.save(self.actor_target, path + "/actor_target_"+str(now_time)+".pt")
        print(self.critic.state_dict())
        torch.save(self.critic, path + "/critic_"+str(now_time)+".pt")
        print(self.critic_target.state_dict())
        torch.save(self.critic_target, path + "/critic_target_"+str(now_time)+".pt")
