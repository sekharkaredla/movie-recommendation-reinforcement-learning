import models
from datetime import datetime
from ML_100k.ReadUsers import get_user_data
from ML_100k.ReadItems import get_item_features
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm

discount_factor = 0.8
batch_size = 20
frame_size = 10
initial_number_of_states = 3
M = 5



user_data = get_user_data()
items_data = get_item_features()


train_data = {}
with open("/Users/adithiloka/Desktop/recommender_deeprl/ml-100k/u.data", "rb") as file:
    for each_rating in file.readlines():
        data = each_rating.strip().split()
        user_id = data[0].decode("utf-8")
        item_id = data[1].decode("utf-8")
        rating = float(data[2].decode("utf-8")) - 2.5
        timestamp = datetime.fromtimestamp(int(data[3].decode("utf-8")))
        if user_id not in train_data.keys():
            train_data[user_id] = [{"timestamp" : timestamp, "item_id": item_id, "rating" : rating}]
        else:
            train_data[user_id].append({"timestamp" : timestamp, "item_id": item_id, "rating" : rating})


for each_user in train_data.keys():
    data = train_data[each_user]
    data = sorted(data, key = lambda i:i["timestamp"])
    train_data[each_user] = data



user_states = {}


for user_id in train_data:
    user_states[user_id] = models.UserState(user_id, user_data[user_id]["embeddings"])
    positive_interactions = 0
    counter = 0
    while positive_interactions < initial_number_of_states:
        initial_state = train_data[user_id][counter]
        if initial_state["rating"] < 0:
            counter += 1
            continue
        user_states[user_id].add_item_embedding(items_data[initial_state["item_id"]])
        train_data[user_id].pop(counter)
        counter += 1
        positive_interactions += 1



# print(train_data)
# print(user_states)
# print(str(user_states["924"]))
# for each_epoch in range(epochs):

#
# for user_id in train_data:
#     for each_interaction_of_user in train_data[user_id]:
#         if user_id not in user_states.keys():
#             user_states[user_id] = models.UserState(user_id, user_data[user_id]["embeddings"])
#             user_states[user_id].add_item_embedding(items_data[each_interaction_of_user["item_id"]])
#             continue




# recommendation = actor(torch.from_numpy(user_states["924"].get_state_embedding()))
# print(recommendation, recommendation.dim())
# value = critic(torch.from_numpy(user_states["924"].get_state_embedding()), recommendation)
# print(value, value.dim())


##########################################################

gamma = 0.99
tau = 1e-2
no_of_replay_experiences = 10


actor = models.Actor(57, 19, 38, 1e-4)
actor_target = models.Actor(57, 19, 38, 1e-4)
critic = models.Critic(57, 19, 38, 1e-3)
critic_target = models.Critic(57, 19, 38, 1e-3)

for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))


actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

critic_criterion = torch.nn.MSELoss()

buffer = models.ReplayBuffer(10000000)





def get_best_item_based_on_new_state(action):
    value = None
    selected_item_id = None
    for each_item in items_data.keys():
        item_embedding = items_data[each_item]
        multiplied = np.multiply(item_embedding, action)
        if value is None or multiplied.sum(0) > value:
            value = multiplied.sum(0)
            selected_item_id = each_item
    return (selected_item_id,value)



def get_reward_for_an_item(user_id, state, new_item_id):
    user_embeddings = user_data[user_id]["embeddings"]
    positive_interactions = state.get_items()
    avg_score = 0.0
    for each_interaction_of_user in positive_interactions:
        avg_score += np.multiply(each_interaction_of_user, user_embeddings)
    avg_score = avg_score / len(state.get_items())
    new_item_embeddings = items_data[new_item_id]
    new_item_score = np.multiply(new_item_embeddings, user_embeddings)
    return np.subtract(new_item_score, avg_score)



def update_networks():

    states, actions, rewards, next_states, _ = buffer.sample(no_of_replay_experiences)
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)

    # Critic loss
    Qvals = critic.forward(states, actions)
    next_actions = actor_target.forward(next_states)
    next_Q = critic_target.forward(next_states, next_actions.detach())
    Qprime = rewards + gamma * next_Q
    critic_loss = critic_criterion(Qvals, Qprime)

    # Actor loss
    policy_loss = -critic.forward(states, actor.forward(states)).mean()

    # update networks
    actor_optimizer.zero_grad()
    policy_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # update target networks
    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))







for user_id in tqdm(user_states):
    initial_state = user_states[user_id]
    for each_interaction_of_user in tqdm(train_data[user_id]):
        new_item_id = each_interaction_of_user["item_id"]
        new_state_embedding = initial_state.add_item_embedding(items_data[new_item_id])
        new_action = actor(torch.from_numpy(new_state_embedding))
        new_item_id, value = get_best_item_based_on_new_state(new_action.detach().numpy())
        user_reward = get_reward_for_an_item(user_id, initial_state, new_item_id)
        user_reward_score = user_reward.sum(0)
        new_state = None
        if user_reward_score > 0:
            initial_state.add_item_embedding(items_data[new_item_id])
        new_state = initial_state
        user_states[user_id] = new_state
        # print(initial_state.get_state_embedding(), new_action.detach().numpy(), user_reward, new_state.get_state_embedding(), None)
        buffer.push(initial_state.get_state_embedding(), new_action.detach().numpy(), user_reward, new_state.get_state_embedding(), None)
        update_networks()



now_time = datetime.now().strftime("%d%b%H%M")



print("actor-target")
print(actor.state_dict())
torch.save(actor, "./trained_models/actor_"+str(now_time)+".pt")
print(actor_target.state_dict())
torch.save(actor_target, "./trained_models/actor_target_"+str(now_time)+".pt")
print(critic.state_dict())
torch.save(critic, "./trained_models/critic_"+str(now_time)+".pt")
print(critic_target.state_dict())
torch.save(critic_target, "./trained_models/critic_target_"+str(now_time)+".pt")