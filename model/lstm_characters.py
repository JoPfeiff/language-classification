import os
import json
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data_loading.data_loader import DataLoader
from data_loading.data_utils import create_model_path, load_params, pad_positions
import random
import time
import sys
from tqdm import *
import csv
reload(sys)
sys.setdefaultencoding('utf8')

from scipy.stats import spearmanr

if torch.cuda.is_available():
    USE_CUDA = True
    dtype = torch.cuda.FloatTensor
else:
    USE_CUDA= False
    dtype =  torch.FloatTensor


class LSTMCharacters(nn.Module):
    """
    This class maps original embeddings to retrofit embeddings. A multilayer perceptron of defined depth and defined
    hidden size calculates the difference between the actual retrofit embedding and predicted embedding
    A set of hyperparameters are choosable (depth, hidden size, activation function, loss function, etc.)
    """
    def __init__(self, batch_size, hidden_size, vocab_size, embedding_size, n_layers, dropout, padding_idx,
                       num_classes, embedding , lr , optimizer_type, data_set_name, best_acc):
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
        super(LSTMCharacters, self).__init__()

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
        self.embedding.weight.requires_grad = False
        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, dropout=self.dropout,
                          batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.num_classes)
        self.data_set_name = data_set_name
        self.best_acc = best_acc

        self.optimizer = self.get_optimizer()
        self.create_hyps_and_hash()

    def create_hyps_and_hash(self):
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

        model_directory_name = str(hash(str(self.hyperparameters)))

        self.model_type_path = create_model_path(self.data_set_name)

        self.model_path = create_model_path(self.data_set_name, model_directory_name)

        self.store_losses =[]
        # self.best_overall = float('inf')
        # self.best_acc = -float('inf')
        self.best_model_path = self.model_type_path + 'best_'+ self.data_set_name + 'model.pth.tar'
        self.best_model_log_name = self.model_type_path + 'best_'+ self.data_set_name + 'model.log'
        self.log_name = self.model_path + 'model.log'



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

        # get the defined loss function
        loss_function = nn.CrossEntropyLoss()

        # depending on the loss function different inputs have to be set
        output = self.forward(x, length)
        # print(output.data.numpy()[0])
        # print "model pred"
        # print dl.label_list[np.argmax(output.data.numpy()[0])]

        # softmax = nn.LogSoftmax(0)
        # logits = softmax(output)

        # loss = loss_function(logits, y)
        loss = loss_function(output, y)

        # train = False

        if not train:
            return loss, output.cpu().data.numpy()

        loss.backward()

        self.optimizer.step()

        return loss, output.cpu().data.numpy()

    def epoch(self, generator, train=True, store=False, epoch=-1, predict=False, dl=None):
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
        acc_list = []

        while batch is not None:

            x, length, y = get_data_from_batch(batch)

            # for i, elem in enumerate(batch):
            #     print ""
            #     print elem['sentence']
            #     print elem['label']
            #     # print x.data.numpy()[i]
            #     print "pred Sentence"
            #     print predict_sentence(self, dl, elem['sentence'])

            # update steps
            step += 1
            if not predict:
                # do one step and get loss
                loss, output = self.step(x, length, y, train=train, predict=predict, dl=dl)
                loss_output = loss.item()
                epoch_loss.append(loss_output)
                steps_loss.append(loss_output)
                acc_list = self.add_acc_array(acc_list, output, y)

                # print out loss at defined steps
                if step % 100 == 0:
                    print str(np.mean(steps_loss))
                    print("accuracy = ") + str(float(sum(acc_list))/float(len(acc_list)))
                    steps_loss = []
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

        if store:
            self.hyperparameters['validation_loss'].append(float(mean_loss))
            self.hyperparameters['validation_acc'].append(float(acc))
            self.store_model(epoch=epoch, loss=mean_loss, acc=acc)
        else:
            self.hyperparameters['training_loss'].append(float(mean_loss))
            self.hyperparameters['training_acc'].append(float(acc))
        return acc

    def add_acc_array(self, acc_list, output, label_tens):

        pred = np.argmax(output, axis=1)
        label = label_tens.cpu().data.numpy()

        acc_list += list(np.array(pred) == np.array(label))
        return acc_list


    def store_model(self, epoch, loss, acc):
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
        if max(self.hyperparameters['validation_acc']) == acc:
            self.hyperparameters['best_acc'] = float(acc)
        # if loss < self.best_overall:
        if acc > self.best_acc:
            print "dumping new best model"
            # shutil.copyfile(self.checkpoint_model_name, self.model_name )
            torch.save(state, self.best_model_path)
            with open(self.best_model_log_name,  'w') as log:
                log.write(json.dumps(self.hyperparameters, indent=4, sort_keys=True))
            self.best_acc = acc
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
        y = y.cuda()

    return x, length, y


def train(parameters, dl, epochs=100, batch_size=32, best_acc=-float('inf')):

    # genClass = PLDataGenerator(small_data_set=small_data_set)

    model = LSTMCharacters(**parameters)
    # model.load_model()
    if USE_CUDA:
        model.cuda()

    train_loss = []
    val_acc = []

    for i in range(epochs):

        print("\n\nEPOCH "+str(i+1))
        print("\nTrain")

        gen = dl.get_generator('train', batch_size=batch_size, drop_last=True, initialize = False)

        train_loss.append(model.epoch(gen, train=True, dl=dl))

        print("\nValidation")

        gen = dl.get_generator('dev', batch_size=100, drop_last=False, initialize = False)

        val_acc.append(model.epoch(gen, train=False, store=True, predict=True, dl=dl))

        # Early stopping
        if np.argmax(np.array(val_acc)) < len(val_acc) - 10:
            print "\n\n\nEarly stopping triggered"
            print "Best Epoch = " + str(np.argmin(np.array(val_acc)))
            print "Validation loss = " + str(val_acc[np.argmax(np.array(val_acc))])
            break
    return model.best_acc


def predict_sentence(model, dl, sentence_list):

    sentence_tokens = []

    max_len = 0
    lengths = []
    pad_position = dl.embedding.get_pad_pos()

    for sentence in sentence_list:
        encoded = dl.embedding.encode_sentence(sentence)
        sentence_tokens.append(encoded)
        length = len(encoded)
        max_len = max(length, max_len)
        lengths.append(length)

    padded_encodings = []

    for encodings in sentence_tokens:
        padded_encodings.append(pad_positions(encodings, pad_position, max_len))

    sentence_encoding = torch.LongTensor(padded_encodings)

    # print sentence_encoding.data.numpy()[0]

    lengths = torch.LongTensor(lengths)
    prediction = np.argmax(model.forward(sentence_encoding,lengths).cpu().data.numpy(), axis=1)

    pred_lang = list(np.array(dl.label_list)[prediction])

    return pred_lang


def random_walk():

    epochs = 500

    for data_set_name, embedding_name in [['Language_Text_100', 'character_100'],
                                          ['Language_Text_1000', 'character_1000'],
                                          ['Language_Text_10000', 'character_10000']]:

        class_params = {'name': data_set_name}
        dl = DataLoader(data_set=data_set_name, embeddings_initial=embedding_name, embedding_loading='top_k',
                        K_embeddings=float('inf'), param_dict=class_params)
        dl.load('data/pickles/')
        dl.get_all_and_dump('data/pickles/')

        best_acc = -float('inf')

        lr = 0.001

        for i in range(10):

            batch_size = np.random.choice([32,64])
            hidden_size = np.random.choice([8,16,32,64,128,256])
            n_layers = np.random.choice([1,2])
            if n_layers == 1:
                dropout = 0.0
            else:
                dropout = np.random.choice([0.0,0.3,0.6])

            hyperparameters, dl = define_hyperparams_and_load_data(dl=dl, data_set_name=data_set_name,
                                                                   embedding_name = embedding_name,
                                                                   batch_size=batch_size, hidden_size=hidden_size,
                                                                   n_layers=n_layers, dropout=dropout, lr=lr ,
                                                                   optimizer_type='adam', best_acc=best_acc)

            print hyperparameters

            best_acc = train(hyperparameters, dl,  epochs=epochs, batch_size=batch_size, best_acc=best_acc)


def predict_test_set():

    best_params = load_params('../data/models/Language_Text_10000/best_Language_Text_10000model.log')

    hyperparameters, dl = define_hyperparams_and_load_data(best_params=best_params)

    model = LSTMCharacters(**hyperparameters)

    model.load_model()

    test_gen = dl.get_generator('test', drop_last=False, batch_size=64)

    acc = model.epoch(test_gen, train=False)
    print acc


def define_hyperparams_and_load_data(best_params=None, dl=None, data_set_name='Language_Text_100', embedding_name = 'character_100', batch_size=32,
                       hidden_size=32, n_layers=1, dropout=0.0, lr=0.001 , optimizer_type='adam', best_acc=0.0):

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

    if dl is None:
        class_params = {'name': data_set_name}
        dl = DataLoader(data_set=data_set_name, embeddings_initial=embedding_name, embedding_loading='top_k',
                        K_embeddings=float('inf'), param_dict=class_params)
        dl.load('../data/pickles/')
        dl.get_all_and_dump('../data/pickles/')

    hyperparameters = {}
    hyperparameters['optimizer_type'] = optimizer_type
    hyperparameters['lr'] = lr
    hyperparameters['hidden_size'] = hidden_size
    hyperparameters['batch_size'] = batch_size
    hyperparameters['vocab_size'] = dl.embedding.get_vocab_size()
    hyperparameters['embedding_size'] = dl.embedding_size
    hyperparameters['n_layers'] = n_layers
    hyperparameters['dropout'] = dropout
    hyperparameters['padding_idx'] = dl.embedding.get_pad_pos()
    hyperparameters['num_classes'] = len(dl.labels)
    hyperparameters['embedding'] = dl.embedding.get_embeddings()
    hyperparameters['data_set_name'] = data_set_name
    hyperparameters['best_acc'] = best_acc

    return hyperparameters, dl


def command_line_prediction():

    best_params = load_params('../data/models/Language_Text_10000/best_Language_Text_10000model.log')

    hyperparameters, dl = define_hyperparams_and_load_data(best_params=best_params)

    model = LSTMCharacters(**hyperparameters)

    model.load_model()

    while True:

        text = raw_input("Please write your sentence: ", ).decode('utf-8')

        lang = predict_sentence(model, dl, [text])

        print "Your sentence was in " + lang[0]

def predict_doc_list(file_name, target_doc):

    best_params = load_params('../data/models/Language_Text_10000/best_Language_Text_10000model.log')

    hyperparameters, dl = define_hyperparams_and_load_data(best_params=best_params)

    model = LSTMCharacters(**hyperparameters)

    model.load_model()

    pred_list = []

    sentence_list = []

    with open(file_name, 'r') as f:
        for sentence in tqdm(f):

            sentence_list.append(sentence)

            if len(sentence_list) == 100:

                pred_list += predict_sentence(model, dl, sentence_list)
                sentence_list = []

    if sentence_list != []:
        pred_list += predict_sentence(model, dl, sentence_list)

    with open(target_doc, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(pred_list)

if __name__ == '__main__':
    # random_walk()
    # command_line_prediction()
    # predict_test_set()
    predict_doc_list('../data/raw/Small/English/Books.raw.en', '../data/raw/Small/English/pred')


