import os.path
import numpy as np
from embeddings import Embeddings
from data_loading.data_utils import pickle_call,pickle_dump
"""
THIS IS A CHILD OF EMBEDDINGS!
"""

CHARACTER_NAMES = ['character_100', 'character_1000', 'character_10000']

class CharacterEmbeddings(Embeddings):

    def __init__(self, name):
        """
        This class calls the character one-hot-encoded vectors. These are therefore NOT EMBEDDINGS. The logic of
        how embeddings are called is used here in order to be able to leverage this framework
        """

        # Load the super class
        super(CharacterEmbeddings, self).__init__()

        # check which data set is to be loaded
        self.name = name

        if self.name == 'character_100':
            self.path = '../data/data_set/tokens_100'
            if not os.path.isfile(self.path):
                self.path = 'data/data_set/tokens_100'
                if not os.path.isfile(self.path):
                    raise Exception( self.path +
                        " tokens dont exist")

        if self.name == 'character_1000':
            self.path = '../data/data_set/tokens_1000'
            if not os.path.isfile(self.path):
                self.path = 'data/data_set/tokens_1000'
                if not os.path.isfile(self.path):
                    raise Exception(self.path +
                        "tokens dont exist")

        if self.name == 'character_10000':
            self.path = '../data/data_set/tokens_10000'
            if not os.path.isfile(self.path):
                self.path = 'data/data_set/tokens_10000'
                if not os.path.isfile(self.path):
                    raise Exception(self.path +
                        "tokens dont exist")


    def load_top_k(self, K, preload=False):
        """
        Option for loading strategy: Only load top k of embeddings assuming that they are ordered in frequency
        :param K: Number of top k embeddings to be retrieved
        :return:embeddings matrix as numpy
        """

        # This embedding dataset does not have PAD UNK START and END tokens pretrained that is why we initialize them
        # ourselves and only load K - 4 embeddings

        # we load the token dictionary from file
        token_dict = pickle_call(self.path)

        # to load only the top most frequent words, we sort them
        token_list = [[token, counts] for token, counts in token_dict.items()]
        token_list = sorted(token_list, key=lambda x: -x[1])

        # because we also need to add special embeddings, we reduce K by 4
        K = K - 4

        # if K is defined larger than what is available, set K to the maximum length
        K = min(K, len(token_list))

        embeddings = []

        # add all the special embeddings and define the one-hot-encoded positions
        self.add_term("<S>", preload=preload)
        embedding = np.zeros(K+4, dtype=np.float64)
        embedding[0] = 1.0
        embeddings.append(embedding)
        self.add_term("</S>", preload=preload)
        embedding = np.zeros(K+4, dtype=np.float64)
        embedding[1] = 1.0
        embeddings.append(embedding)
        self.add_term("<PAD>", preload=preload)
        embedding = np.zeros(K+4, dtype=np.float64)
        embedding[2] = 1.0
        embeddings.append(embedding)
        self.add_term("<UNK>", preload=preload)
        embedding = np.zeros(K+4, dtype=np.float64)
        embedding[3] = 1.0
        embeddings.append(embedding)

        # then loop through the sorted tokens and also define the one-hot-encoded vectors
        for k in range(K):
            token = token_list[k][0]
            self.add_term(token, preload=preload)
            embedding = np.zeros(K + 4, dtype=np.float64)
            embedding[k+4] = 1.0
            embeddings.append(embedding)

        # define the embedding size
        self.embedding_size = K + 4

        # return the embeddings
        return np.array(embeddings)

    def get_name(self):
        return self.name



