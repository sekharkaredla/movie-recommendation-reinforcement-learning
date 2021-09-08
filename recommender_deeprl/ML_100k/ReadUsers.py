from tqdm import tqdm
from .ReadItems import get_item_features
import datetime
import numpy as np


def get_user_data(users_data_file, items_data_file):
    user_data = {}
    items = get_item_features(items_data_file)
    user_embeddings = {}
    with users_data_file as file:
        for each_rating in tqdm(file.readlines()):
            data = each_rating.strip().split()
            user_id = data[0].decode("utf-8")
            item_id = data[1].decode("utf-8")
            rating = float(data[2].decode("utf-8")) - 3.0
            timestamp = datetime.datetime.fromtimestamp(int(data[3].decode("utf-8")))

            if user_id not in user_data:
                user_data[user_id] = {}

            if "items" not in user_data[user_id].keys():
                user_data[user_id]["items"] = {item_id:rating}
            else:
                user_data[user_id]["items"][item_id] = rating

            if "embeddings" not in user_data[user_id].keys():
                user_data[user_id]["embeddings"] = rating * items[item_id]
            else:
                user_data[user_id]["embeddings"] = np.add(user_data[user_id]["embeddings"], rating * items[item_id])

            if "interactions" not in user_data[user_id].keys():
                user_data[user_id]["interactions"] = [{"timestamp":timestamp, "item_id": item_id}]
            else:
                user_data[user_id]["interactions"].append({"timestamp":timestamp, "item_id": item_id})

        file.close()
    for each_user_id in user_data:
        embeddings = user_data[each_user_id]["embeddings"]
        user_data[each_user_id]["embeddings"] = 2.0 * (embeddings - np.min(embeddings)) / np.ptp(embeddings) - 1

    for each_user_id in user_data:
        interactions = user_data[each_user_id]["interactions"]
        interactions = sorted(interactions, key= lambda i:i["timestamp"])
        user_data[each_user_id]["interactions"] = interactions

    return user_data
