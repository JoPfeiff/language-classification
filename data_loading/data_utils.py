# import spacy
import os
import cPickle
# import pdb
import json


"""
functions that are used throughout data_loading scripts
"""
# spacy_tokenizer = spacy.load('en_core_web_sm', disable=["parser","tagger","ner"])
# nlp = spacy.load('en_core_web_sm')


def tokenize(sentence, start_token=None, end_token=None):
    # tokens = spacy_tokenizer(sentence)
    # tokens = [token.text for token in tokens]

    tokens = list(sentence)

    if start_token is not None and end_token is not None:
        tokens = [start_token] + tokens + [end_token]
    return tokens


def pickle_call(file_name):
    if os.path.isfile(file_name):
        with open(file_name, "rb") as input_file:
            return cPickle.load(input_file)
    else:
        # raise Exception("Pickles not available")
        return None


def pickle_dump(file_name, data):
    with open(file_name, "wb") as output_file:
        cPickle.dump(data, output_file)


def pad_positions(positions, pad_positions, length_bucket):
    """
    Pads the sentence with the defiend pad position for bucketized data sets
    :param positions: the sentence list of indexed ints
    :param pad_positions: the pad position in the embedding matrix
    :param length_bucket: the legnth of the buckets, so length_bucket - len(positions) = nr of padds
    :return:
    """
    length_sentence = len(positions)
    padding_length = length_bucket - length_sentence
    pad_array = [pad_positions] * padding_length
    return positions + pad_array


def create_model_path(directory, directory_name=None):
    """
    Here we create the path to our models if it doesnt exist yet
    :param directory:
    :param directory_name:
    :return:
    """
    data_models_path = '../data/models/'

    if not os.path.exists(data_models_path):
        data_models_path = 'data/models/'

        if not os.path.exists(data_models_path):
            raise Exception("models path does not exist")
    if directory_name is None:
        model_path = data_models_path + directory
    else:
        model_path = data_models_path + directory + "/" + directory_name

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return model_path + "/"

def load_params(file_name):
    with open(file_name) as handle:
        dictdump = json.loads(handle.read())
    return dictdump


if __name__ == "__main__":
    # tokens=    tokenize(u"Hello this, is Me.")
    #
    # print tokens
    data = pickle_call('../data/embeddings/polyglot-en.pkl')
    print "done"



