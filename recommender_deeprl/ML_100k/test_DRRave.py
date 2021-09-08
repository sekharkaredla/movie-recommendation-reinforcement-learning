from DRR_ave import CalculateDRRave
from ReadUsers import get_user_data
from ReadItems import get_item_features


user_data = get_user_data()
items_data = get_item_features()

test_user = user_data['941']

item_embeddings = [items_data[each] for each in test_user["items"]]

calculateDRRave = CalculateDRRave(test_user["embeddings"], item_embeddings)

print(calculateDRRave.get_state_embedding(items_data["240"]))