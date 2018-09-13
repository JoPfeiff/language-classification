import os
import json
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data_loading.data_loader import DataLoader
import random
import time

from scipy.stats import spearmanr

if torch.cuda.is_available():
    USE_CUDA = True
    dtype = torch.cuda.FloatTensor
else:
    USE_CUDA= False
    dtype =  torch.FloatTensor

class LSTMCharactes(nn.Module):
    """
    This class maps original embeddings to retrofit embeddings. A multilayer perceptron of defined depth and defined
    hidden size calculates the difference between the actual retrofit embedding and predicted embedding
    A set of hyperparameters are choosable (depth, hidden size, activation function, loss function, etc.)
    """
    def __init__(self, batch_size, hidden_size, vocab_size, embedding_size, n_layers, dropout, padding_idx,
                       num_classes, embedding , lr , optimizer_type):
        """
        Initialize Model
        :param h_size: size of the hidden layers except input and output layers.
        :param depth: numper of hidden layers
        :param embedding_size: size of the embeddings
        :param activation_type: 'tanh', 'relu', 'swish'
        :param loss_function_type: 'cosine, 'mse', 'hinge-cosine'
        :param optimizer_type: 'adam', 'sgd', 'adagrad'
        :param lr: learning rate
        :param margin: if negative sampling, the margin for margin based hinge loss
        """

        # call super class
        super(LSTMCharactes, self).__init__()

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
        self.optimizer = self.get_optimizer()
        self.embedding.weight.requires_grad = False
        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, dropout=self.dropout,
                          batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, batch, length,  hidden=None):

        batch_size = len(batch)
        # Get embedded version
        embedded = self.embedding(batch)

        # Need to sort in decreasing order of length and keep track of indices so they can be concatenated together after the LSTM
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
        What type of optimizer is chosen. Also initializes the learningrate
        :return:
        """
        if self.optimizer_type == 'adam':
            return optim.Adam(self.parameters(), lr=self.lr)
        if self.optimizer_type == 'sgd':
            return optim.SGD(self.parameters(), lr=self.lr)
        if self.optimizer_type == 'adagrad':
            return optim.Adagrad(self.parameters(), lr=self.lr)



    # def step(self, batch, neg, train=True):
    def step(self, x, length, y, train=True, predict=False):
        """
        One step in each epoch. if train is True then back prop is triggered, else only the loss is returned
        :param batch: batch of embeddings
        :param neg: if negative sampling is chosen, this a batch of negative sampled embeddings. Else this is None
        :param train: if to be trained or not. if not, back prop is not triggered
        :return:
        """

        # set all gradients to 0
        self.optimizer.zero_grad()

        # get the defined loss function
        loss_function = nn.CrossEntropyLoss()

        # depending on the loss function different inputs have to be set
        output = self.forward(x, length)

        # softmax = nn.LogSoftmax(0)
        # logits = softmax(output)

        # loss = loss_function(logits, y)
        loss = loss_function(output, y)

        if not train:
            return loss

        loss.backward()

        self.optimizer.step()

        return loss

    def epoch(self, generator, train=True, store=False, epoch=-1, predict=False):
        """
        loops through one epoch of data
        :param generator: the generator that yields randomized data points
        :param train: if we are trainging or validating
        :return:
        """

        # Set the model to train or not train
        self.train(train)

        # get the next batch
        batch, bucket = generator.next()

        # initialize the list of losses
        step = 0
        epoch_loss = []
        steps_loss = []

        while batch is not None:

            x, length, y = get_data_from_batch(batch)
            # update steps
            step += 1
            if not predict:
                # do one step and get loss
                loss = self.step(x, length, y, train=train, predict=predict)
                loss_output = loss.item()
                epoch_loss.append(loss_output)
                steps_loss.append(loss_output)

                # print out loss at defined steps
                if step % 100 == 0:
                    print np.mean(steps_loss)
                    steps_loss = []
            else:
                loss = self.step(x, length, y, train=train, predict=predict)
                epoch_loss.append(loss.item())


            # get next batch. If no more data points are available, None, None is returned
            batch, bucket = generator.next()

        # When we have looped through one epoch, we return the mean of the epoch
        print("Mean epoch loss: " + str(np.mean(epoch_loss)))

        mean_loss = np.mean(epoch_loss)

        # if store:
        #     self.hyperparameters['validation_cosine'].append(float(mean_loss))
        #     self.store_model(epoch=epoch, loss=mean_loss)
        # else:
        #     self.hyperparameters['training_loss'].append(float(mean_loss))
        return mean_loss

    def store_model(self, epoch, loss):
        """
        Dump model to file and log it
        :param epoch:
        :param loss:
        :return:
        """

        self.store_losses.append(loss)

        state = {
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': self.state_dict(),
            'best_prec1': loss,
            'optimizer' : self.optimizer.state_dict(),
        }

        if min(self.store_losses) == loss:
            self.hyperparameters['best_score'] = float(loss)
        if loss < self.best_overall:
            # shutil.copyfile(self.checkpoint_model_name, self.model_name )
            torch.save(state, self.best_model_path)
            with open(self.best_model_log_name,  'w') as log:
                log.write(json.dumps(self.hyperparameters, indent=4, sort_keys=True))
            self.best_overall = loss
        with open(self.log_name, 'w') as log:
            log.write(json.dumps(self.hyperparameters, indent=4, sort_keys=True))

    def load_model(self, best=True):
        """
        Load model from file
        :param best:
        :return:
        """
        if best:
            filename = self.model_name

        else:
            filename = self.checkpoint_model_name

        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
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
    :param neg: if negative sampling is activated this is a set of random negative samples. else this is None
    :return:
    """

    # Initialize the list of inputs and outputs
    x = []
    length = []
    y = []

    for elem in batch:
        x.append(elem['sentence_positions'])
        length.append(elem['length'])
        y.append(elem['label_pos'])

    x, length = Variable(torch.LongTensor(x)), torch.IntTensor(length)
    y = torch.LongTensor(y)

    if USE_CUDA:
        x = x.cuda()

    return x, length, y


def train(parameters, dl, epochs=100, batch_size=32):

    # genClass = PLDataGenerator(small_data_set=small_data_set)

    model = LSTMCharactes(**parameters)
    if USE_CUDA:
        model.cuda()

    train_loss = []
    val_loss = []

    for i in range(epochs):

        print("\n\nEPOCH "+str(i+1))
        print("\nTrain")

        gen = dl.get_generator('train', batch_size=batch_size, drop_last=True, initialize = False)

        train_loss.append(model.epoch(gen, train=True))

        print("\nValidation")

        gen = dl.get_generator('dev', batch_size=100, drop_last=False, initialize = False)

        val_loss.append(model.epoch(gen, train=False, store=True, predict=True))

        # Early stopping
        if np.argmin(np.array(val_loss)) < len(val_loss) - 5:
            print "\n\n\nEarly stopping triggered"
            print "Best Epoch = " + str(np.argmin(np.array(val_loss)))
            print "Validation loss = " + str(val_loss[np.argmin(np.array(val_loss))])
            break
    return model.best_overall


def random_walk():

    epochs = 100

    class_params = {'name': 'Language_Text_100'}
    dl = DataLoader(data_set='Language_Text_100', embeddings_initial='character_100', embedding_loading='top_k',
                    K_embeddings=300, param_dict=class_params)
    dl.load('../data/pickles/')
    dl.get_all_and_dump('../data/pickles/')
    # dl.initialize_embeddings()

    lr = 0.1
    hidden_size = 10
    batch_size = 64
    vocab_size = dl.embedding.get_vocab_size()
    embedding_size = dl.embedding.embedding_size
    n_layers = 1
    dropout = 1.0
    padding_idx = dl.embedding.get_pad_pos()
    num_classes = len(dl.labels)
    embedding = dl.embedding.get_embeddings()

    hyperparameters = {}
    hyperparameters['optimizer_type'] = 'sgd'
    hyperparameters['lr'] = lr
    hyperparameters['hidden_size'] = hidden_size
    hyperparameters['batch_size'] = batch_size
    hyperparameters['vocab_size'] = vocab_size
    hyperparameters['embedding_size'] = embedding_size
    hyperparameters['n_layers'] = n_layers
    hyperparameters['dropout'] = dropout
    hyperparameters['padding_idx'] = padding_idx
    hyperparameters['num_classes'] = num_classes
    hyperparameters['embedding'] = embedding


    print hyperparameters

    best_overall = train(hyperparameters, dl,  epochs=epochs, batch_size=batch_size )


if __name__ == '__main__':
    random_walk()



