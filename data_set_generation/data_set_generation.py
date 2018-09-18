import os
import numpy as np
from data_loading.data_utils import pickle_dump, pickle_call
from tqdm import *
import random

def generate_data_set():
    """
    This function loops through each file and samples 100, 1000 and 10000 lines which will be added to the overall
    data sets
    :return:
    """

    # Path where the raw data sets can be found
    path = '../data/raw/Big/'

    labels = []

    # dictionaries for the 3 different data sets
    tokens_100 = {}
    tokens_1000 = {}
    tokens_10000 = {}

    data_set = [None, None, None]

    # loop through all files in the path
    for filename in os.listdir(path):

        if filename.startswith("EUbookshop.raw."):

            # the label can be found as the last 2 chars in the file name
            label = filename[-2:]
            labels.append(label)

            full_file = []

            # loop through all rows in the file and add the sentence and label to the full_file list
            with open(path + filename, 'r') as file:
                for row in tqdm(file):

                    sentence = row.decode('utf-8').replace(u"\n",u"")
                    elem = {}
                    elem['sentence'] = sentence
                    elem['label'] = label

                    full_file.append(elem)

            # create a numpy so that we can sample from it
            full_file = np.array(full_file)

            # sample 100, 1000 and 10000 sentences
            selection_100 = np.random.choice(len(full_file), 100, replace=False)
            selection_1000 = np.random.choice(len(full_file), 1000, replace=False)
            selection_10000 = np.random.choice(len(full_file), 10000, replace=False)

            # and add them to the data set
            file_data_100 = full_file[selection_100]
            file_data_1000 = full_file[selection_1000]
            file_data_10000 = full_file[selection_10000]


            # loop through the three data sets and retrieve the characters which appear and save them in tokens
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

    # dump everything
    pickle_dump('../data/data_set/data_set_100',data_set[0])
    pickle_dump('../data/data_set/tokens_100',tokens_100)
    pickle_dump('../data/data_set/data_set_1000',data_set[1])
    pickle_dump('../data/data_set/tokens_1000',tokens_1000)
    pickle_dump('../data/data_set/data_set_10000',data_set[2])
    pickle_dump('../data/data_set/tokens_10000',tokens_10000)


def split_train_dev_test(path, nr_train, nr_dev, nr_test):
    """
    Here we split the data sets based on the defined number of data points
    :param path: path of the original data set
    :param nr_train: number of data points to train on
    :param nr_dev: number of data points to validate on
    :param nr_test: number of data points to test on
    :return:
    """

    # retrieve the data set from file
    data = pickle_call(path)

    #shuffle it
    random.shuffle(data)

    # split it based on the defined numbers
    train = data[:nr_train]
    dev = data[nr_train:nr_train+nr_dev]
    test = data[nr_train+nr_dev:nr_train+nr_dev+nr_test]

    # dump it to file
    pickle_dump(path+"_train", train)
    pickle_dump(path+"_dev", dev)
    pickle_dump(path+"_test", test)


if __name__ == '__main__':
    # Here we generate the data set by sampling the same amount of data points per class
    generate_data_set()
    # and split it into train, dev and test sets
    split_train_dev_test('../data/data_set/data_set_100', 2160,270,270)
    split_train_dev_test('../data/data_set/data_set_1000', 21600,2700,2700)
    split_train_dev_test('../data/data_set/data_set_10000', 216000,27000,27000)




