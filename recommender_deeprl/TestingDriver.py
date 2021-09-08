from ML_100k.ReadUsers import get_user_data
from ML_100k.ReadItems import get_item_features
from Properties import Properties
from models import *
from Train_v2 import Train
from tqdm import tqdm


properties = Properties()
users_data = get_user_data(properties.get_train_dataset_file(), properties.get_items_data_file())
items_data = get_item_features(properties.get_items_data_file())
property_file = properties.get_property_file()

test_data = get_user_data(properties.get_test_dataset_file(), properties.get_items_data_file())


actor = torch.load(property_file['EVALUATION']['actor_model'])
critic = torch.load(property_file['EVALUATION']['critic_model'])

actor.eval()
critic.eval()


train = Train()

train.initialize_user_states(users_data)

total = 0
count = 0

users_data = test_data

for each_user_id in tqdm(users_data.keys()):
    items_not_empty = True
    while items_not_empty:
        if len(users_data[each_user_id]["items"].keys()) == 0:
            items_not_empty = False
            break
        current_user_state = train.get_user_state(each_user_id)
        recommended_action = actor.forward(torch.from_numpy(current_user_state.astype(np.float32)))
        #print(recommended_action)
        reward_embedding, reward = train.calculate_reward_for_recommended_action_from_user(recommended_action.detach().numpy(), each_user_id)
        #print(reward_embedding, reward)
        recommended_item = train.get_best_item_from_user_space_based_on_action(recommended_action.detach().numpy(),\
        users_data[each_user_id]["items"].keys(), items_data)
        #print(recommended_item)
        rating = users_data[each_user_id]["items"][recommended_item[0]]
        # print(rating)
        #print(rating, reward)
        users_data[each_user_id]["items"].pop(recommended_item[0])

        if rating > 0 :
            train.add_item_to_user(each_user_id, items_data[recommended_item[0]])
        if rating >= 0.0 and reward > 0.0:
            count += 1


        elif rating < 0.0 and reward < 0.0:
            count += 1
        total += 1

print("-->", float(count)/total)
