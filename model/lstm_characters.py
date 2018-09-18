import os
import json
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data_loading.data_loader import DataLoader
from data_loading.data_utils import create_model_path, load_params, pad_positions
import sys
from tqdm import *
import csv
reload(sys)
sys.setdefaultencoding('utf8')

if torch.cuda.is_available():
    USE_CUDA = True
    dtype = torch.cuda.FloatTensor
else:
    USE_CUDA= False
    dtype =  torch.FloatTensor


class LSTMCharacters(nn.Module):
    """
    This class defines the architecture for a character based LSTM for language prediction.
    Based on defined hyperparameters the depth of the hidden layers etc. is set up
    """
    def __init__(self, batch_size, hidden_size, vocab_size, embedding_size, n_layers, dropout, padding_idx,
                       num_classes, embedding , lr , optimizer_type, data_set_name, best_acc):
        """
        Initialize Model
        :param batch_size
        :param hidden_size: size of the hidden layers except input and output layers.
        :param vocab_size: NOT USED DEPRECATE
        :param embedding_size: size of the one-hot-encoded "embedding" of each character
        :param n_layers: number of layers of each LSTM unit
        :param dropout: dropout probability between layers
        :param padding_idx: the index which is used for padding
        :param num_classes: number languages to be predicted
        :param embedding: the embedding object
        :param lr: the initial learning rate
        :param optimizer_type: the optimizer that is used (e.g. adam)
        :param data_set_name: which of the 3 (small, medium, big) data sets is used (e.g Language_Text_1000)
        :param best_acc: the currently best accuracy that a model has achieved for this data set
        """

        # call super class
        super(LSTMCharacters, self).__init__()

        # Store all the hyperparameters in the object
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.num_directions = 1
        self.padding_idx = padding_idx
        self.num_classes = num_classes
        self.embedding = embedding
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.data_set_name = data_set_name
        self.best_acc = best_acc

        # initialized the embeddings and set gradients to false (we dont train them)
        self.embedding.weight.requires_grad = False

        # define the size of on LSTM unit
        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, dropout=self.dropout,batch_first=True)

        # define the output layer for prediction of the classes (languages)
        self.output_layer = nn.Linear(self.hidden_size, self.num_classes)

        # define the optimizer (not that this HAS to be defined after defining the LSTM and the output_layer
        self.optimizer = self.get_optimizer()

        # define the loging of the model
        self.create_hyps_and_hash()

    def create_hyps_and_hash(self):
        """
        In this function wie hack a way to define the name of a model by creating a dictionary of all the hyperparams,
        forming it to a string and hashing that. This is subsequently defined as the directory name for this model and
        here we store the logs of the model which includes epoch loss and accuracy of both training and dev sets.
        """

        # We first dump all the hyperparameters (AGAIN! I know... I said its a hack OK???) into a dict
        self.hyperparameters = {}
        self.hyperparameters['batch_size'] = self.batch_size
        self.hyperparameters['hidden_size'] = self.hidden_size
        self.hyperparameters['vocab_size'] = self.vocab_size
        self.hyperparameters['embedding_size'] = self.embedding_size
        self.hyperparameters['n_layers'] = self.n_layers
        self.hyperparameters['dropout'] = self.dropout
        self.hyperparameters['num_directions'] = self.num_directions
        self.hyperparameters['padding_idx'] = self.padding_idx
        self.hyperparameters['num_classes'] = self.num_classes
        self.hyperparameters['lr'] = self.lr
        self.hyperparameters['optimizer_type'] = self.optimizer_type
        self.hyperparameters['training_loss'] = []
        self.hyperparameters['training_acc'] = []
        self.hyperparameters['validation_loss'] = []
        self.hyperparameters['validation_acc'] = []
        self.hyperparameters['data_set_name'] = self.data_set_name
        self.hyperparameters['best_acc'] = self.best_acc

        # We then convert the dict into a string and hash it (to make it shorter)
        model_directory_name = str(hash(str(self.hyperparameters)))

        # we then create a directory with this name
        self.model_path = create_model_path(self.data_set_name, model_directory_name)

        # and also define where we want to store the best model
        self.model_type_path = create_model_path(self.data_set_name)

        # and we create a bunch of names for each of the logs
        self.store_losses =[]
        self.best_model_path = self.model_type_path + 'best_'+ self.data_set_name + 'model.pth.tar'
        self.best_model_log_name = self.model_type_path + 'best_'+ self.data_set_name + 'model.log'
        self.log_name = self.model_path + 'model.log'

    def forward(self, batch, length,  hidden=None):
        """
        This is the forward pass of the model
        :param batch: a tensor of indexes which are all padded to the same length
        :param length: the original length of each sentence without padding
        :param hidden: not needed because we don't forward the hidden state
        :return: output state for each class
        """

        # get dynamic batch size
        batch_size = len(batch)

        # Get embedded version
        embedded = self.embedding(batch)

        # Need to sort in decreasing order of length and keep track of indices so they can be concatenated together
        # after the LSTM
        sorted_lens, sorted_idx = torch.sort(length, descending=True)
        sorted = embedded[sorted_idx]
        _, sortedsorted_idx = sorted_idx.sort()

        # Convert to a packed sequence
        packed = torch.nn.utils.rnn.pack_padded_sequence(sorted, sorted_lens, batch_first=True)

        # Put packed sequence through the LSTM
        _, hidden = self.LSTM(packed, hidden)

        # This hidden has the last hidden state of each sequence according to their lengths (last non pad timestep)
        last_hidden_state = hidden[0].view(self.n_layers, self.num_directions, batch_size, self.hidden_size)[-1].squeeze(0)

        # First convert to batch_size X hidden_size. Time to reorder to original ordering
        unsorted_hidden = last_hidden_state[sortedsorted_idx]

        # Project to output space
        final_output = self.output_layer(unsorted_hidden)

        return final_output

    def get_optimizer(self):
        """
        What type of optimizer is chosen. Also initializes the learning rate
        :return:
        """
        if self.optimizer_type == 'adam':
            return optim.Adam(self.parameters(), lr=self.lr)
        if self.optimizer_type == 'sgd':
            return optim.SGD(self.parameters(), lr=self.lr)
        if self.optimizer_type == 'adagrad':
            return optim.Adagrad(self.parameters(), lr=self.lr)


    def step(self, x, length, y, train=True, predict=False, dl=None):
        """
        One step in each epoch. if train is True then back prop is triggered, else only the loss is returned
        :param batch: batch of embeddings
        :param neg: if negative sampling is chosen, this a batch of negative sampled embeddings. Else this is None
        :param train: if to be trained or not. if not, back prop is not triggered
        :return:
        """

        # set all gradients to 0
        self.optimizer.zero_grad()

        # create the loss function
        loss_function = nn.CrossEntropyLoss()

        # forward pass of the data
        output = self.forward(x, length)

        # get the loss
        loss = loss_function(output, y)

        # if we are not training, we don't need to compute the gradients and therefore return here
        if not train:
            return loss, output.cpu().data.numpy()

        # otherwise we compute the gradients
        loss.backward()

        # and update through the backward pass
        self.optimizer.step()

        return loss, output.cpu().data.numpy()

    def epoch(self, generator, train=True, store=False, epoch=-1, predict=False, dl=None):
        """
        loops through one epoch of data
        :param generator: the generator that yields randomized data points
        :param train: if we are trainging or validating
        :param store: if the data should be stored (should only be true if train==False) because we check if we have
                      trained a better model
        :param epoch: the epoch we are in
        :param dl: the data_loader object
        :return: accuracy of this epoch
        """

        # Set the model to train or not train
        self.train(train)

        # get the next batch
        batch, bucket = generator.next()

        # initialize the list of losses and accuracies
        step = 0
        epoch_loss = []
        steps_loss = []
        acc_list = []

        # as long as we have a batch we loop (if generator.next() returns None, None we break)
        while batch is not None:

            # we retrieve the batch data in a processable way. batch includes a list of dict elements which need to be
            # processed into a tensor
            x, length, y = get_data_from_batch(batch)

            # update steps
            step += 1
            if not predict:

                # do one step and get loss
                loss, output = self.step(x, length, y, train=train, predict=predict, dl=dl)

                # add the loss to the logging lists
                loss_output = loss.item()
                epoch_loss.append(loss_output)
                steps_loss.append(loss_output)
                acc_list = self.add_acc_array(acc_list, output, y)

                # print out loss at defined steps
                if step % 100 == 0:
                    print str(np.mean(steps_loss))
                    print("accuracy = ") + str(float(sum(acc_list))/float(len(acc_list)))
                    steps_loss = []

            # if we are not training, we will just print out info at the end (just assuming that this is less data)
            else:
                loss, output = self.step(x, length, y, train=train, predict=predict, dl=dl)
                acc_list = self.add_acc_array(acc_list, output, y)
                epoch_loss.append(loss.item())

            # get next batch. If no more data points are available, None, None is returned
            batch, bucket = generator.next()

        # When we have looped through one epoch, we return the mean of the epoch
        print("Mean epoch loss: " + str(np.mean(epoch_loss)))
        acc = float(sum(acc_list)) / float(len(acc_list))
        print "Epoch accuracy = " + str(acc)

        mean_loss = np.mean(epoch_loss)

        # if we are storing the data, we add the data to the validation set and check if we have beaten the already
        # trained models
        if store:
            self.hyperparameters['validation_loss'].append(float(mean_loss))
            self.hyperparameters['validation_acc'].append(float(acc))
            self.store_model(epoch=epoch, loss=mean_loss, acc=acc)

        # if we are not storing, we will assume that we are training, and add the loss and accuracy to the training logs
        else:
            self.hyperparameters['training_loss'].append(float(mean_loss))
            self.hyperparameters['training_acc'].append(float(acc))

        return acc

    def add_acc_array(self, acc_list, output, label_tens):
        """
        This function compares the predicted labels to the true labels and adds a boolean array to the current acc_list
        :param acc_list: a boolean list of values for if a label was predicted correctly or not
        :param output: the predicted labels
        :param label_tens: the true label tensor
        :return:
        """

        # we first retrieve the argmax of each output
        pred = np.argmax(output, axis=1)

        # we retrieve the true labels into a numpy array
        label = label_tens.cpu().data.numpy()

        # we generate a masked list of boolean values if each predicted label matches the true label and append it
        acc_list += list(np.array(pred) == np.array(label))

        return acc_list


    def store_model(self, epoch, loss, acc):
        """
        Dump model to file and log it
        :param epoch: the current epoch we are in
        :param loss: the loss of the epoch
        :param acc: the accuracy of the epoch
        :return:
        """

        # if the current loss is the best one, define it as the best one
        if min(self.hyperparameters['training_loss']) == loss:
            self.hyperparameters['best_score'] = float(loss)

        # if the current accuracy is the best one, define it as such
        if max(self.hyperparameters['validation_acc']) == acc:
            self.hyperparameters['best_acc'] = float(acc)

        # if the current accuracy is the best OVERALL (of all models), only then we dump it to file
        if acc > self.best_acc:
            state = {
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': self.state_dict(),
                'best_prec1': loss,
                'optimizer' : self.optimizer.state_dict(),
            }
            print "dumping new best model"
            # shutil.copyfile(self.checkpoint_model_name, self.model_name )
            torch.save(state, self.best_model_path)
            with open(self.best_model_log_name,  'w') as log:
                log.write(json.dumps(self.hyperparameters, indent=4, sort_keys=True))
            self.best_acc = acc

        # we store the current logs on file as a json
        with open(self.log_name, 'w') as log:
            log.write(json.dumps(self.hyperparameters, indent=4, sort_keys=True))

    def load_model(self, best=True):
        """
        Load model from file
        :param best:
        :return:
        """

        filename = self.best_model_path

        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            if USE_CUDA:
                checkpoint = torch.load(filename)
            else:
                checkpoint = torch.load(filename, map_location={'cuda:0': 'cpu'})

            # start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))


def get_data_from_batch(batch):
    """
    Transform one batch into data points
    :param batch: the batch of random data points
    :return: X tensor, lengths tensor, labels tensor
    """

    # Initialize the list of inputs and outputs
    x = []
    length = []
    y = []

    # loop through the batch and retrieve the relevant information
    for elem in batch:
        x.append(elem['sentence_positions'])
        length.append(elem['length'])
        y.append(elem['label_pos'])

    # define X and Y as tensors
    x, length = Variable(torch.LongTensor(x)), torch.IntTensor(length)
    y = torch.LongTensor(y)

    # cudarize if needed
    if USE_CUDA:
        x = x.cuda()
        y = y.cuda()

    return x, length, y


def train(parameters, dl, epochs=100, batch_size=32):
    """
    Here we train for the defined amount of epochs with early stopping by not having a better accuracy for 10 epochs
    :param parameters: hyperparameters defined
    :param dl: data_loading object
    :param epochs: nr of epochs max
    :param batch_size: batch size
    :return: the best accuracy of this model
    """

    # define the model
    model = LSTMCharacters(**parameters)

    # cudarize if needed
    if USE_CUDA:
        model.cuda()

    # initialize some loging stuff
    train_loss = []
    val_acc = []

    # loop through the epooch
    for i in range(epochs):

        print("\n\nEPOCH "+str(i+1))
        print("\nTrain")

        # get the training generator for one eopch
        gen = dl.get_generator('train', batch_size=batch_size, drop_last=True, initialize = False)

        # train for one epoch and append the accuracy
        train_loss.append(model.epoch(gen, train=True, dl=dl, epoch=i))

        print("\nValidation")

        # get the development generator for one epoch
        gen = dl.get_generator('dev', batch_size=100, drop_last=False, initialize = False)

        # calculate the accuracy of this epoch
        val_acc.append(model.epoch(gen, train=False, store=True, predict=True, dl=dl, epoch=i))

        # Early stopping if the accuracy has not increased for 10 epochs
        if np.argmax(np.array(val_acc)) < len(val_acc) - 10:
            print "\n\n\nEarly stopping triggered"
            print "Best Epoch = " + str(np.argmin(np.array(val_acc)))
            print "Validation loss = " + str(val_acc[np.argmax(np.array(val_acc))])
            break

    return model.best_acc


def random_walk(best_params_file, epochs=100, nr_samples=1, data_set_name='Language_Text_10000', embedding_name='character_10000', lrs=[0.001],
                batch_sizes=[32], hidden_sizes=[32], n_layers_list=[1], dropouts=[0.0], optimizer_types=['adam']):
    """
    This function randomly samples each hyperparameter and trains the model until early stopping
    :param best_params_file: location of the log file for the best model
    :param epochs: max number of epochs
    :param nr_samples: how many randomly sampled models should be trained
    :param data_set_name: which data set is to be chosen 'Language_Text_100', Language_Text_1000', 'Language_Text_10000'
    :param embedding_name: embeddings data set 'character_100', 'character_1000', or 'character_10000'
    :param lrs: possible learning rates
    :param batch_sizes: different batch sizes to be tested
    :param hidden_sizes: LSTM hidden sizes
    :param n_layers_list: number of layers for each LSTM timestep
    :param dropouts: dropout probability between LSTM layers
    :param optimizer_types: optimizer e.g. 'adam', 'sgd'
    :return: returns the overall best accuracy
    """

    # load the defined data loader data set, this can be shared for each model because nothing changes
    class_params = {'name': data_set_name}
    dl = DataLoader(data_set=data_set_name, embeddings_initial=embedding_name, embedding_loading='top_k',
                    K_embeddings=float('inf'), param_dict=class_params)
    dl.load('data/pickles/')
    dl.get_all_and_dump('data/pickles/')

    # if models have already been trained, get the best accuracy so that we don't overwrite our current best model
    try:
        best_params = load_params(best_params_file)
        best_acc = best_params['best_acc']

    # if not, we just define the accuracy the smallest possible
    except:
        best_acc = -float('inf')

    # we randomly sample #nr_samples models and train them until early stopping
    for i in range(nr_samples):

        print("\nCurrent best accuracy = " + str(best_acc)) + '\n'

        # randomly sample the hyperparams
        lr = np.random.choice(lrs)
        batch_size = np.random.choice(batch_sizes)
        hidden_size = np.random.choice(hidden_sizes)
        n_layers = np.random.choice(n_layers_list)
        optimizer_type = np.random.choice(optimizer_types)

        # if we only have one layer, theres no need for dropout (at least for this kind)
        if n_layers == 1:
            dropout = 0.0
        else:
            dropout = np.random.choice(dropouts)

        # batch_size = np.random.choice(batch_sizes)
        # hidden_size = np.random.choice([8,16,32,64,128,256])
        # n_layers = np.random.choice([1,2])
        # if n_layers == 1:
        #     dropout = 0.0
        # else:
        #     dropout = np.random.choice([0.0,0.3,0.6])

        # get the hyperparameters by initializing them and retrieving some parameters from data_loader
        hyperparameters, dl = define_hyperparams_and_load_data(dl=dl, data_set_name=data_set_name,
                                                               embedding_name = embedding_name,
                                                               batch_size=batch_size, hidden_size=hidden_size,
                                                               n_layers=n_layers, dropout=dropout, lr=lr ,
                                                               optimizer_type=optimizer_type, best_acc=best_acc)

        print ("Training with the following hyperparameters:")
        print hyperparameters

        # train until early stopping
        best_acc = train(hyperparameters, dl,  epochs=epochs, batch_size=batch_size)


def define_hyperparams_and_load_data(best_params=None, dl=None, data_set_name='Language_Text_100',
                                     embedding_name = 'character_100', batch_size=32, hidden_size=32, n_layers=1,
                                     dropout=0.0, lr=0.001 , optimizer_type='adam', best_acc=0.0):
    """
    here we define the hyperparams based on either defined settings, or based on the stored best json log
    If dl is not defined, we load it
    :param best_params: json file of best hyper parameters
    :param dl: data_loading object
    :param data_set_name: which data set ist to be loaded
    :param embedding_name: which embedding data set is to be loaded
    :param batch_size:
    :param hidden_size:
    :param n_layers:
    :param dropout:
    :param lr:
    :param optimizer_type:
    :param best_acc:
    :return:
    """

    # if we have passed the dictionary of bestparams we extract that information and set the hyperparameters accordingly
    if best_params is not None:
        if 'data_set_name' in best_params:
            data_set_name = best_params['data_set_name']
            if data_set_name.endswith("10000"):
                embedding_name = 'character_10000'
            elif  data_set_name.endswith("1000"):
                embedding_name = 'character_1000'
            else:
                embedding_name = 'character_100'
        if 'batch_size' in best_params:
            batch_size = best_params['batch_size']
        if 'hidden_size' in best_params:
            hidden_size = best_params['hidden_size']
        if 'n_layers' in best_params:
            n_layers = best_params['n_layers']
        if 'dropout' in best_params:
            dropout = best_params['dropout']
        if 'lr' in best_params:
            lr = best_params['lr']
        if 'optimizer_type' in best_params:
            optimizer_type = best_params['optimizer_type']
        if 'best_acc' in best_params:
            best_acc = best_params['best_acc']

    # if we have not passed a data_loader object, we call ist from file
    if dl is None:
        class_params = {'name': data_set_name}
        dl = DataLoader(data_set=data_set_name, embeddings_initial=embedding_name, embedding_loading='top_k',
                        K_embeddings=float('inf'), param_dict=class_params)
        dl.load('data/pickles/')
        dl.get_all_and_dump('data/pickles/')

    # define all the hyperparameters
    hyperparameters = {}
    hyperparameters['optimizer_type'] = optimizer_type
    hyperparameters['lr'] = lr
    hyperparameters['hidden_size'] = hidden_size
    hyperparameters['batch_size'] = batch_size
    hyperparameters['vocab_size'] = dl.embedding.get_vocab_size()
    hyperparameters['n_layers'] = n_layers
    hyperparameters['dropout'] = dropout
    hyperparameters['padding_idx'] = dl.embedding.get_pad_pos()
    hyperparameters['num_classes'] = len(dl.labels)
    hyperparameters['embedding'] = dl.embedding.get_embeddings()
    hyperparameters['embedding_size'] = dl.embedding.embedding_size
    hyperparameters['data_set_name'] = data_set_name
    hyperparameters['best_acc'] = best_acc

    return hyperparameters, dl


def predict_sentence(model, dl, sentence_list):
    """
    This function predicts a set of sentences by padding them
    Note that this does not bucketize them, so this is very inefficient
    :param model: LSTM model
    :param dl: data_loader object
    :param sentence_list: list of string sentences to be predicted
    :return: a list of predicted languages
    """

    # initialize the list of tokens
    sentence_tokens = []

    # the longest sentence
    max_len = 0

    # original length of the sentences
    lengths = []

    # padding position
    pad_position = dl.embedding.get_pad_pos()

    # for each sentence we encode the positoins and update the lists
    for sentence in sentence_list:
        encoded = dl.embedding.encode_sentence(sentence)
        sentence_tokens.append(encoded)
        length = len(encoded)
        max_len = max(length, max_len)
        lengths.append(length)

    # initialize the padding encodings
    padded_encodings = []

    # we pad each encoded sentence to the max length of the list of passed sentences
    for encodings in sentence_tokens:
        padded_encodings.append(pad_positions(encodings, pad_position, max_len))

    # we transform it into a Tensor
    sentence_encoding = torch.LongTensor(padded_encodings)

    # transform the lengths into tensors
    lengths = torch.LongTensor(lengths)

    # predict the sentences
    prediction = np.argmax(model.forward(sentence_encoding,lengths).cpu().data.numpy(), axis=1)

    # decode the languages
    pred_lang = list(np.array(dl.label_list)[prediction])

    return pred_lang


def predict_test_set(best_file_name):
    """
    Predict the held out test set
    :param best_file_name: The location of the best parameters
    :return:
    """

    # load the best parameters
    best_params = load_params(best_file_name)

    # define the hyperparameters and load the data loading object
    hyperparameters, dl = define_hyperparams_and_load_data(best_params=best_params)

    # define the model architecture
    model = LSTMCharacters(**hyperparameters)

    # load the model from file
    model.load_model()

    # generate the test set generator
    test_gen = dl.get_generator('test', drop_last=False, batch_size=64)

    # loop throught the data and get the accuracy
    acc = model.epoch(test_gen, train=False, predict=True, store=False)

    print acc


def command_line_prediction(best_params_file):
    """
    This is the demo case where you can enter a sentence in the command line and it predicts the language
    :param best_params_file: the best parameter file
    :return:
    """

    # get the bes parameters
    best_params = load_params(best_params_file)

    # define the hyperparameters and get the data loading object
    hyperparameters, dl = define_hyperparams_and_load_data(best_params=best_params)

    # build the model and load the parameters from file
    model = LSTMCharacters(**hyperparameters)
    model.load_model()

    print("\nData and model loading successful")
    print("\nDemo prediction ready! \n")

    # until the user force stopps we predict
    while True:

        # get the user input
        text = raw_input("Please write your sentence: ", ).decode('utf-8')

        # predict the sentence
        lang = predict_sentence(model, dl, [text])

        print "Your sentence was in " + lang[0]


def predict_doc_list(file_name, target_doc, best_params_file):
    """
    Based on a user defined text document where each line represents a sentence or document, this function predicts
    the language for each line and dumps the predicteion in the target_doc
    :param file_name: file with sentences/documents per line
    :param target_doc: destination of prediction
    :param best_params_file: file with the best model hyperparameters
    :return:
    """

    # load the best hyperparameters
    best_params = load_params(best_params_file)

    # load hyperparameters and data loading object
    hyperparameters, dl = define_hyperparams_and_load_data(best_params=best_params)

    # define architecture of model
    model = LSTMCharacters(**hyperparameters)

    # load the parameters from file
    model.load_model()

    # list of predictions and sentences
    pred_list = []
    sentence_list = []

    # loop through the file
    with open(file_name, 'r') as f:
        for sentence in tqdm(f):

            # append to list for a batch
            sentence_list.append(sentence)

            # if batch is 100 long we predict
            if len(sentence_list) == 100:

                # predict the sentence languages
                pred_list += predict_sentence(model, dl, sentence_list)
                sentence_list = []

    # also predict last batch
    if sentence_list != []:
        pred_list += predict_sentence(model, dl, sentence_list)

    # reshape predictions and dump to file
    pred_list = np.array(pred_list).reshape((-1, 1))
    with open(target_doc, "wb") as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerows(pred_list)

if __name__ == '__main__':
    pass

