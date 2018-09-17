from model.lstm_characters import random_walk, command_line_prediction, predict_test_set, predict_doc_list
import argparse

parser = argparse.ArgumentParser()

# Possible values: 'big', 'medium', 'small'
parser.add_argument("--data_set", default='big', type=str)

# Possible values: 'demo', 'train', 'predict'
parser.add_argument("--task", default='demo', type=str)

# if predict set a file with sentences per line, and a destination where the predictions should be stored
parser.add_argument("--file", default=None, type=str)
parser.add_argument("--destination", default=None, type=str)

# if train we will define a random walk over the defined parameters
parser.add_argument("--nr_samples", default=1, type=int)
parser.add_argument("--batch_sizes", default=[32], type=list)
parser.add_argument("--hidden_sizes", default=[32], type=list)
parser.add_argument("--n_layers_list", default=[1], type=list)
parser.add_argument("--dropouts", default=[0.0], type=list)
parser.add_argument("--lrs", default=[0.001], type=list)
parser.add_argument("--optimizer_types", default=['adam'], type=list)
parser.add_argument("--epochs", default=100, type=int)


parsed = vars(parser.parse_args())

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

if parsed['task'] == 'demo':
    command_line_prediction(best_param_file)

elif parsed['task'] == 'train':
    nr_samples = parsed['nr_samples']
    batch_sizes = parsed['batch_sizes']
    hidden_sizes = parsed['hidden_sizes']
    n_layers_list = parsed['n_layers_list']
    dropouts = parsed['dropouts']
    lrs = parsed['lrs']
    optimizer_types = parsed['optimizer_types']
    epochs = parsed['epochs']
    random_walk(epochs=epochs, nr_samples=nr_samples, data_set_name=data_set_name, embedding_name=embedding_name,
                lrs=lrs, batch_sizes=batch_sizes, hidden_sizes=hidden_sizes, n_layers_list=n_layers_list,
                dropouts=dropouts, optimizer_types=optimizer_types)

elif parsed['task'] == 'predict':
    file_name = parsed['file']
    destination = parsed['destination']
    predict_doc_list(file_name, destination, best_param_file)

else:
    raise Exception("Task incorrectly defined")




# random_walk()
# command_line_prediction()
# predict_test_set()
# predict_doc_list('../data/raw/Small/English/Books.raw.en', '../data/raw/Small/English/pred')




