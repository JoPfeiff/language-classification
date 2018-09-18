from model.lstm_characters import random_walk, command_line_prediction, predict_test_set, predict_doc_list
import argparse

parser = argparse.ArgumentParser()

# Possible values: 'big', 'medium', 'small'
parser.add_argument("--data_set", default='big', type=str)

# Possible values: 'demo', 'train', 'predict', 'predict_test'
parser.add_argument("--task", default='demo', type=str)

# if predict set a file with sentences per line, and a destination where the predictions should be stored
parser.add_argument("--file", default=None, type=str)
parser.add_argument("--destination", default=None, type=str)

# if train we will define a random walk over the defined parameters
parser.add_argument("--nr_samples", default=1, type=int)
parser.add_argument("--batch_sizes", default=[32], nargs='+', type=int, dest='batch_sizes')
parser.add_argument("--hidden_sizes", default=[32], nargs='+', type=int, dest='hidden_sizes')
parser.add_argument("--n_layers_list", default=[1], nargs='+', type=int, dest='n_layers_list')
parser.add_argument("--dropouts", default=[0.0], nargs='+', type=float, dest='dropouts')
parser.add_argument("--lrs", default=[0.001], nargs='+', type=float, dest='lrs')
parser.add_argument("--optimizer_types", default=['adam'], nargs='+', type=str, dest='optimizer_types')
parser.add_argument("--epochs", default=100, type=int)

# parse the input configurations
parsed = vars(parser.parse_args())

# define the data set based on the config and set the location of the best model params
if parsed['data_set'] == 'small':
    data_set_name = 'Language_Text_100'
    embedding_name = 'character_100'
    best_param_file = 'data/models/Language_Text_100/best_Language_Text_100model.log'
elif parsed['data_set'] == 'medium':
    data_set_name = 'Language_Text_1000'
    embedding_name = 'character_1000'
    best_param_file = 'data/models/Language_Text_1000/best_Language_Text_1000model.log'
else:
    data_set_name = 'Language_Text_10000'
    embedding_name = 'character_10000'
    best_param_file = 'data/models/Language_Text_10000/best_Language_Text_10000model.log'

# In the case of demo, we call the command_line_prediction which predicts a user-input sentence
if parsed['task'] == 'demo':
    command_line_prediction(best_param_file)

# If the config was set to train, we will do a random walk over the set hyperparameters
# Note that nr_samples is the number of random walks we do
elif parsed['task'] == 'train':
    nr_samples = parsed['nr_samples']
    batch_sizes = parsed['batch_sizes']
    hidden_sizes = parsed['hidden_sizes']
    n_layers_list = parsed['n_layers_list']
    dropouts = parsed['dropouts']
    lrs = parsed['lrs']
    optimizer_types = parsed['optimizer_types']
    epochs = parsed['epochs']
    random_walk(best_param_file, epochs=epochs, nr_samples=nr_samples, data_set_name=data_set_name, embedding_name=embedding_name,
                lrs=lrs, batch_sizes=batch_sizes, hidden_sizes=hidden_sizes, n_layers_list=n_layers_list,
                dropouts=dropouts, optimizer_types=optimizer_types)

# If we predict, a file with sentences per line needs to be defined and the location file where the predictions
# should be stored
elif parsed['task'] == 'predict':
    file_name = parsed['file']
    destination = parsed['destination']
    predict_doc_list(file_name, destination, best_param_file)

elif parsed['task'] == 'predict_test':
    predict_test_set(best_param_file)


else:
    raise Exception("Task incorrectly defined")



