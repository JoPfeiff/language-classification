import os.path
import numpy as np
from embeddings import Embeddings
import io
import pdb
from data_loading.data_utils import pickle_call,pickle_dump
"""
THIS IS A CHILD OF EMBEDDINGS!
"""

CHARACTER_NAMES = ['character_100', 'character_1000', 'character_10000']

class CharacterEmbeddings(Embeddings):

    def __init__(self, name):
        """
        This class calls the FastText data
        """

        # Load the super class
        super(CharacterEmbeddings, self).__init__()

        # check if the FastText Data exisits
        self.name = name

        if self.name == 'character_100':
            self.path = '../data/data_set/tokens_100'
            if not os.path.isfile(self.path):
                self.path = 'data/data/data_set/tokens_100'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "tokens dont exist")

        if self.name == 'character_1000':
            self.path = '../data/data_set/tokens_1000'
            if not os.path.isfile(self.path):
                self.path = 'data/data/data_set/tokens_1000'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "tokens dont exist")

        if self.name == 'character_10000':
            self.path = '../data/data_set/tokens_10000'
            if not os.path.isfile(self.path):
                self.path = 'data/data/data_set/tokens_10000'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "tokens dont exist")


    def load_top_k(self, K, preload=False):
        """
        Option for loading strategy: Only load top k of embeddings assuming that they are ordered in frequency
        :param K: Number of top k embeddings to be retrieved
        :return:embeddings matrix as numpy
        """

        # This embedding dataset does not have PAD UNK START and END tokens pretrained that is why we initialize them
        # ourselves and only load K - 4 embeddings

        K = K - 4

        token_dict = pickle_call(self.path)

        token_list = [[token, counts] for token, counts in token_dict.items()]
        token_list = sorted(token_list, key=lambda x: -x[1])

        embeddings = []

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

        for k in range(K):
            token = token_list[k][0]
            self.add_term(token, preload=preload)
            embedding = np.zeros(K + 4, dtype=np.float64)
            embedding[k+4] = 1.0
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_name(self):
        return self.name



