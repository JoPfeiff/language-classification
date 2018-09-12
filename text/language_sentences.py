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

        if self.name == 'Language_Text_100':
            path = '../data/data_set/data_set_100_' + data_type
            if not os.path.isfile(path):
                path = 'data/data/data_set/data_set_100_' + data_type
                if not os.path.isfile(path):
                    raise Exception(
                        "tokens dont exist")

        if self.name == 'Language_Text_1000':
            path = '../data/data_set/data_set_1000_' + data_type
            if not os.path.isfile(path):
                path = 'data/data/data_set/data_set_1000_' + data_type
                if not os.path.isfile(path):
                    raise Exception(
                        "tokens dont exist")

        if self.name == 'Language_Text_10000':
            path = '../data/data_set/data_set_10000_' + data_type
            if not os.path.isfile(path):
                path = 'data/data/data_set/data_set_10000_' + data_type
                if not os.path.isfile(path):
                    raise Exception(
                        "tokens dont exist")

        self.data_sets[data_type] = pickle_call(path)

        for elem in self.data_sets[data_type]:
            line = elem['sentence']
            elem['sentence_positions'] = self.embeddings.encode_sentence(line, initialize=initialize_term,
                                                             count_up=True)

        return self.data_sets[data_type]

    def bucketize_data(self, data_type, initialize):

        PAD_position = self.embeddings.get_pad_pos(initialize=initialize)

        for elem in self.data_sets[data_type]:
            sent_length = len(elem['tokens'])
            for bucket in self.buckets:
                if sent_length <= bucket:
                    elem['length'] = sent_length
                    elem['label_pos'] = self.labels[elem['label']]
                    elem['sentence_positions'] = pad_positions(elem['sentence_positions'], PAD_position, bucket)
                    self.data_sets[data_type + '_buckets'][bucket]['data'].append(elem)
                    self.data_sets[data_type + '_buckets'][bucket]['length'] += 1
                    break


#
# if __name__ == '__main__':
#
#     bwd = BillionWordsData()
#
#     file_name = '../data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00000-of-00100'
#
#     bwd.load_billion_words(data_set='train', k=1)
#
#     print("done")



