
# Unknown |Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy |Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
# 19 features per movie

from tqdm import tqdm
import numpy as np

def get_item_features(items_data_file):
    items = {}
    with items_data_file as file:
        for each_item in tqdm(file.readlines()):
            data = each_item.strip().split(b"|")
            idx = data[0].decode("utf-8")
            features = data[-19:]
            features = [int(feature.decode("utf-8")) for feature in features]
            features = np.asarray(features, dtype=np.float32)
            items[idx] = features
        file.close()

    return items


# print(get_item_features())
