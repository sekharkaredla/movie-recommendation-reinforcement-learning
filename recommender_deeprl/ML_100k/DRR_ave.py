import numpy as np
from pprint import pprint

class CalculateDRRave:
    def __init__(self, user_embeddings, items = None):
        self.user_embeddings = user_embeddings
        self.items = items
        self.concatStateEmbedding = None


    def calculate_ave_pooling(self, item):
        if self.items is None:
            self.items = np.array([item])
        else:
            self.items = np.vstack([self.items, item])

        ave_pooling = np.sum(self.items, axis=0)
        return ave_pooling/self.items.shape[0]



    def get_state_embedding(self, item):
        ave_pooling = self.calculate_ave_pooling(item)
        user_ave_pooling = np.multiply(self.user_embeddings, ave_pooling)
        state_embedding = np.append(self.user_embeddings, user_ave_pooling)
        state_embedding = np.append(state_embedding, ave_pooling)
        self.concatStateEmbedding = state_embedding
        pprint(ave_pooling)
        pprint(user_ave_pooling)
        pprint(state_embedding)
        return state_embedding




