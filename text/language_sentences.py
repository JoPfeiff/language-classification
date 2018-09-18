from text.text_data import TextData
from data_loading.data_utils import pad_positions, pickle_call
import os
from tqdm import *

class LanguageText(TextData):
    """
    This class is a child of TextData
    We load the billion word benchmark data set and store it in self.data_set
    Because of the size of this data set we do not store the raw data, but only the positions
    """
    def __init__(self, embeddings, data_params, labels):

        self.name = data_params['name']

        super(LanguageText, self).__init__( self.name, embeddings)

        self.labels = labels

        # Initialize the two dicts for train and dev buckets
        self.data_sets['train'] = {}
        self.data_sets['dev'] = {}
        self.data_sets['test'] = {}

        self.data_sets['train_buckets'] = {}
        self.data_sets['dev_buckets'] = {}
        self.data_sets['test_buckets'] = {}

        # TODO: Check the right bucket sizes
        self.buckets = [10,20,30,50,100,150, 200, 300,400,500,600,700,800,900,1000]

        # For testing purposes or to generate a smaller data set we can define the number of k files that we want to
        # load
        if 'k' in data_params:
            self.k = data_params['k']
        else:
            self.k = float('inf')

        # Define all variables for each bucket
        for bucket in self.buckets:
            # data_set + "_buckets"
            self.data_sets['train_buckets'][bucket] = {}
            self.data_sets['dev_buckets'][bucket] = {}
            self.data_sets['test_buckets'][bucket] = {}
            self.data_sets['train_buckets'][bucket]['data'] = []
            self.data_sets['dev_buckets'][bucket]['data'] = []
            self.data_sets['test_buckets'][bucket]['data'] = []
            self.data_sets['train_buckets'][bucket]['bucket_size'] = bucket
            self.data_sets['dev_buckets'][bucket]['bucket_size'] = bucket
            self.data_sets['test_buckets'][bucket]['bucket_size'] = bucket

            self.data_sets['train_buckets'][bucket]['length'] = 0
            self.data_sets['dev_buckets'][bucket]['length'] = 0
            self.data_sets['test_buckets'][bucket]['length'] = 0
            self.data_sets['train_buckets'][bucket]['position'] = 0
            self.data_sets['dev_buckets'][bucket]['position'] = 0
            self.data_sets['test_buckets'][bucket]['position'] = 0

            self.data_sets['train_buckets'][bucket]['buckets'] = bucket
            self.data_sets['dev_buckets'][bucket]['buckets'] = bucket
            self.data_sets['test_buckets'][bucket]['buckets'] = bucket


    def load_data(self, data_type='train', directory=None, initialize_term=False):
        """
        This function loads all the data defined by the data set name
        :param data_type: 'train', 'dev', 'test'
        :param directory: DEPRECATED
        :param initialize_term:
        :return:
        """
        if self.name == 'Language_Text_100':
            path = '../data/data_set/data_set_100_' + data_type
            if not os.path.isfile(path):
                path = 'data/data_set/data_set_100_' + data_type
                if not os.path.isfile(path):
                    raise Exception(
                        "tokens dont exist")

        if self.name == 'Language_Text_1000':
            path = '../data/data_set/data_set_1000_' + data_type
            if not os.path.isfile(path):
                path = 'data/data_set/data_set_1000_' + data_type
                if not os.path.isfile(path):
                    raise Exception(
                        "tokens dont exist")

        if self.name == 'Language_Text_10000':
            path = '../data/data_set/data_set_10000_' + data_type
            if not os.path.isfile(path):
                path = 'data/data_set/data_set_10000_' + data_type
                if not os.path.isfile(path):
                    raise Exception(
                        "tokens dont exist")

        # retrieve the data set from disc
        self.data_sets[data_type] = pickle_call(path)

        # loop through every sentence and encode it using the embedding object
        for elem in self.data_sets[data_type]:
            line = elem['sentence']
            elem['sentence_positions'] = self.embeddings.encode_sentence(line, initialize=initialize_term,
                                                             count_up=True)

        return self.data_sets[data_type]

    def bucketize_data(self, data_type, initialize):
        """
        Here we bucketize the sentences based on their length
        :param data_type:  'train', 'dev', 'test'
        :param initialize: if characters should be initialized or not
        :return:
        """

        # retrieve the padding position
        PAD_position = self.embeddings.get_pad_pos(initialize=initialize)

        # loop through the data set
        for elem in self.data_sets[data_type]:

            # calc the length of the sentence
            sent_length = len(elem['sentence_positions'])

            # loop through the bucket sizes to check which bucket the sentence fits in
            for bucket in self.buckets:

                # if the sentence fits in the bucket
                if sent_length <= bucket:

                    # get the length
                    elem['length'] = sent_length

                    # get the character positons
                    elem['label_pos'] = self.labels[elem['label']]

                    # pad the positions to the predefined bucket length
                    elem['sentence_positions'] = pad_positions(elem['sentence_positions'], PAD_position, bucket)

                    # and append to data set
                    self.data_sets[data_type + '_buckets'][bucket]['data'].append(elem)
                    self.data_sets[data_type + '_buckets'][bucket]['length'] += 1

                    # if this is done, we break because we do not want to add the data point to the next bucket
                    break






