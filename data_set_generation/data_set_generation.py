import os
import numpy as np
from data_loading.data_utils import pickle_dump, pickle_call
from tqdm import *
import random

def generate_data_set():
    path = '../data/raw/Big/'

    labels = []

    tokens_100 = {}
    tokens_1000 = {}
    tokens_10000 = {}

    data_set = [None, None, None]

    for filename in os.listdir(path):

        if filename.startswith("EUbookshop.raw."):

            label = filename[-2:]
            labels.append(label)

            full_file = []

            with open(path + filename, 'r') as file:

                for row in tqdm(file):

                    sentence = row.decode('utf-8').replace(u"\n",u"")

                    elem = {}
                    elem['sentence'] = sentence
                    elem['label'] = label

                    full_file.append(elem)

            full_file = np.array(full_file)

            selection_100 = np.random.choice(len(full_file), 100, replace=False)
            selection_1000 = np.random.choice(len(full_file), 1000, replace=False)
            selection_10000 = np.random.choice(len(full_file), 10000, replace=False)

            file_data_100 = full_file[selection_100]
            file_data_1000 = full_file[selection_1000]
            file_data_10000 = full_file[selection_10000]



            for i, [file_data_set, tokens] in enumerate([[file_data_100, tokens_100],
                                                    [file_data_1000, tokens_1000],
                                                    [file_data_10000, tokens_10000]]):
                for data_point in file_data_set:
                    data_point['tokens'] = list(data_point['sentence'])
                    for token in data_point['tokens']:
                        if token not in tokens:
                            tokens[token] = 1
                        else:
                            tokens[token] += 1

                if data_set[i] is None:
                    data_set[i] = file_data_set
                else:
                    data_set[i] = np.append(data_set[i], file_data_set)

    pickle_dump('../data/data_set/data_set_100',data_set[0])
    pickle_dump('../data/data_set/tokens_100',tokens_100)
    pickle_dump('../data/data_set/data_set_1000',data_set[1])
    pickle_dump('../data/data_set/tokens_1000',tokens_1000)
    pickle_dump('../data/data_set/data_set_10000',data_set[2])
    pickle_dump('../data/data_set/tokens_10000',tokens_10000)


def split_train_dev_test(path, nr_train, nr_dev, nr_test):

    data = pickle_call(path)

    random.shuffle(data)

    train = data[:nr_train]
    dev = data[nr_train:nr_train+nr_dev]
    test = data[nr_train+nr_dev:nr_train+nr_dev+nr_test]

    pickle_dump(path+"_train", train)
    pickle_dump(path+"_dev", dev)
    pickle_dump(path+"_test", test)


if __name__ == '__main__':
    # generate_data_set()
    split_train_dev_test('../data/data_set/data_set_100', 2160,270,270)
    split_train_dev_test('../data/data_set/data_set_1000', 21600,2700,2700)
    split_train_dev_test('../data/data_set/data_set_10000', 216000,27000,27000)




