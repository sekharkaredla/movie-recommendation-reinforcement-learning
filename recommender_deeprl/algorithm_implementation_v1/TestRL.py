from ML_100k.ReadItems import get_item_features
from ML_100k.ReadUsers import get_user_data
from datetime import datetime
from tqdm import tqdm
import numpy as np
import models
import torch
from models import Actor
from models import Critic



items_data = get_item_features()
user_data = get_user_data()



test_data = {}
with open("/Users/adithiloka/Desktop/recommender_deeprl/ml-100k/u1.test", "rb") as file:
    for each_rating in file.readlines():
        data = each_rating.strip().split()
        user_id = data[0].decode("utf-8")
        item_id = data[1].decode("utf-8")
        rating = float(data[2].decode("utf-8")) - 2.5
        timestamp = datetime.fromtimestamp(int(data[3].decode("utf-8")))
        if user_id not in test_data.keys():
            test_data[user_id] = [{"timestamp" : timestamp, "item_id": item_id, "rating" : rating}]
        else:
            test_data[user_id].append({"timestamp" : timestamp, "item_id": item_id, "rating" : rating})


# for each_user in test_data.keys():
#     data = test_data[each_user]
#     data = sorted(data, key = lambda i:i["timestamp"])
#     test_data[each_user] = data


user_states = {}
initial_number_of_states = 2




for user_id in test_data:
    user_states[user_id] = models.UserState(user_id, user_data[user_id]["embeddings"])
    positive_interactions = 0
    counter = 0
    while positive_interactions < initial_number_of_states:
        if counter >= len(test_data[user_id]):
            break
        initial_state = test_data[user_id][counter]
        if initial_state["rating"] < 0:
            counter += 1
            continue
        user_states[user_id].add_item_embedding(items_data[initial_state["item_id"]])
        test_data[user_id].pop(counter)
        counter += 1
        positive_interactions += 1


temp_data = {}
for each_user_id in test_data:
    if (user_states[each_user_id].get_items() is None or len(user_states[each_user_id].get_items())<initial_number_of_states):
        continue
    temp_data[each_user_id] = test_data[each_user_id]

test_data = temp_data


def get_best_item_based_on_new_state(user_restricted_space, action):
    value = None
    selected_item_id = None
    for each_item in user_restricted_space:
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




actor = torch.load("./trained_models/actor_v1.pt")
critic = torch.load("./trained_models/critic_v1.pt")

actor.train()
critic.train()

total = 0
correct = 0

for each_user_id in test_data.keys():
    user_state_space_with_rating = {}
    for data in test_data[each_user_id]:
        user_state_space_with_rating[data["item_id"]] = data["rating"]
    recommendation = actor(torch.from_numpy(user_states[each_user_id].get_state_embedding()))
    new_item_id = get_best_item_based_on_new_state(user_state_space_with_rating.keys(), recommendation.detach().numpy())
    reward = get_reward_for_an_item(each_user_id, user_states[each_user_id], new_item_id[0])
    print(reward.sum(0), "--->", user_state_space_with_rating[new_item_id[0]])
    if reward.sum(0) > 0 and user_state_space_with_rating[new_item_id[0]] > 0:
        correct += 1
    elif reward.sum(0) < 0 and user_state_space_with_rating[new_item_id[0]] < 0:
        correct += 1
    total += 1



print(total, correct, float(correct)/total)






