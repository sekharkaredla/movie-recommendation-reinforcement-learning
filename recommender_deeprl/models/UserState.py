import numpy as np

class UserState:
    def __init__(self, user_id, user_embeddings):
        self.user_id = user_id
        self.user_embeddings = user_embeddings
        self.items = None
        self.state_size = 3 * self.user_embeddings.shape[0]


    def add_item_embedding(self, item_embedding):
        if self.items is None:
            self.items = np.array([item_embedding])
        else:
            self.items = np.vstack([self.items, item_embedding])

        return self.get_state_embedding()


    def get_state_embedding(self):
        # ave_pooling = np.sum(self.items, axis=0)
        # ave_pooling = ave_pooling / self.items.shape[0]
        # user_ave_pooling = np.multiply(self.user_embeddings, ave_pooling)
        # state_embedding = np.append(self.user_embeddings, user_ave_pooling)
        # state_embedding = np.append(state_embedding, ave_pooling)
        # return state_embedding
        output = None
        if self.get_items() is None or len(self.get_items()) == 0:
            output = np.zeros((self.state_size,), dtype=np.float32)
            output[:self.user_embeddings.shape[0]] = self.user_embeddings
        elif len(self.get_items()) == 1:
            output = np.zeros((self.state_size,))
            output[:self.user_embeddings.shape[0]] = self.user_embeddings
            output[self.user_embeddings.shape[0]:2*self.user_embeddings.shape[0]] = self.get_items()[0]
        else:
            ave_pooling = np.sum(self.items, axis=0)
            ave_pooling = ave_pooling / self.items.shape[0]
            user_ave_pooling = np.multiply(self.user_embeddings, ave_pooling)
            output = np.append(self.user_embeddings, user_ave_pooling)
            output = np.append(output, ave_pooling)
        return output

    def get_items(self):
        return self.items

    def get_user_embeddings(self):
        return self.user_embeddings


    def __str__(self):
        return str(self.items) + " ->  " + str(self.items.shape) + ", user : " + \
               str(self.user_embeddings) + ", state_embedding : " + str(self.get_state_embedding())

    def __repr__(self):
        return self.__str__()
