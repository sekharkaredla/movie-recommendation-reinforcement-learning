from Properties import Properties
from Train_v2 import Train
from ML_100k.ReadUsers import get_user_data
from ML_100k.ReadItems import get_item_features
import torch
from tqdm import tqdm

# ALGORITHM specific values

properties = Properties()
epochs = int(properties.get_epocs())
actor_learning_rate = float(properties.get_actor_learning_rate())
critic_learning_rate = float(properties.get_critic_learning_rate())
property_file = properties.get_property_file()
number_of_features = int(property_file['ALGORITHM']['k'])
input_size = 3 * number_of_features
hidden_size = 2 * number_of_features
output_size = 1 * number_of_features
tau = float(property_file['ALGORITHM']['tau'])
gamma = float(property_file['ALGORITHM']['gamma'])
no_of_replay_samples = int(property_file['ALGORITHM']['no_of_replay_samples'])




# Start of ALGORITHM
train = Train()

# create networks

train.set_network_sizes(input_size, output_size, hidden_size)
train.create_actor(actor_learning_rate)
train.create_actor_target(actor_learning_rate)
train.create_critic(critic_learning_rate)
train.create_critic_target(critic_learning_rate)
train.copy_parameters_between_networks()

users_data = get_user_data(properties.get_training_data_file(), properties.get_items_data_file())
items_data = get_item_features(properties.get_items_data_file())

train.initialize_optimizers(actor_learning_rate, critic_learning_rate)
for each_epoch in tqdm(range(epochs)):
    train.initialize_user_states(users_data)
    for each_user_id in tqdm(users_data.keys()):
        current_user_state = train.get_user_state(each_user_id)
        recommended_action = train.get_actor().forward(torch.from_numpy(current_user_state))
        # recommended_item = train.get_best_item_from_user_space_based_on_action(recommended_action.detach().numpy(),\
        # users_data[each_user_id]["items"].keys(), items_data)
        recommended_item = train.get_best_item_from_user_space_based_on_action(recommended_action.detach().numpy(),\
        items_data.keys(), items_data)
        # print(train.calculate_reward_for_recommended_action_from_user(recommended_action.detach().numpy(), each_user_id))
        reward_embedding, reward = train.calculate_reward_for_recommended_action_from_user(items_data[recommended_item[0]], each_user_id)
        current_user_state = train.get_user_state(each_user_id)
        if reward > 0.0:
            train.add_item_to_user(each_user_id, items_data[recommended_item[0]])
        next_user_state = train.get_user_state(each_user_id)
        train.add_experience_to_replay_buffer(current_user_state, recommended_action.detach().numpy(), reward_embedding, next_user_state)
        train.update_networks(no_of_replay_samples, gamma, tau)
train.save_networks()
