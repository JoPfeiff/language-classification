# Classification of Text Language

In this repository I have built a character based LSTM model that predicts the language of a sentence or a text document.
The data set is based on EU Bookshop data which includes sentences of 42 different languages. The raw data can be found here http://opus.nlpl.eu/EUbookshop.php. Note however that I only included languages for which more than 10000 sentences exist. <br/> 

This repository builds upon my data_loading framework which can be found here: https://github.com/JoPfeiff/nlp-data-loading-framework-

The data_set is generated using data_set_generation/data_set_generation.py <br/>
The data_set is built using data_loading/data_loader.py which gets as input the objects defined in  <br/>
 - embeddings/character_embedding.py <br/>
 - text/language_sentences.py <br/>
The model can be found in model/lstm_characters.py <br/>

## Data
The preprocessed training data can be found here: <br/>
https://www.dropbox.com/sh/mrw2gm9saj4fmy1/AAAvaf5rTzFLhDWe-BM3vUiFa?dl=0 <br/>
Please download the directory and store it under language-classification/ <br/>

The raw data links can be found in raw_data_links.txt <br/>
If you want to generate the data set from raw data please download all files and store them in data/raw/Big/ <br/>

## Installation
The code was written in python 2.7. <br/>
Please consider running the code in a virtual environment: <br/>
https://packaging.python.org/guides/installing-using-pip-and-virtualenv/  <br/>

run
```
$ python setup.py install
```
to install all dependencies

## Commandline Documentation
In order to run the demo, train or predict a set of command line arguments can be passed.
run
```
$ python start.py
```
to run the default demo settings with the best trained model. You can enter sentences for which the language is subsequently predicted. <br/>

### Demo 

The demo can explicitly be called using
```
$ python start.py --task demo
```
I have created three different data sets small, medium and large each consisting of 100, 1000, and 10000 sentences respectively per language. The data set to-be-used can be defined using e.g. `--data_set small`. The values are `small`, `medium` and `large`. Default is set to `large`.
```
$ python start.py --task demo --data_set small 
```
therefore calls the demo set using the model trained using the small data set. <br/>

### Training
A random walk over defined hyper parameters can be realized using `--task train`. <br/>
#### The following hyperparameters can be set as `ints`: <br/>
`--nr_samples`: number of random walks (sampled models) should be iterated over `default: 1` <br/>
`--epochs`: max number of epochs (early stopping is on)  `default: 100` <br/>
#### The following hyperparameters can be set as `list`:
`--batch_sizes`: sizes of the batch `default: [32]`<br/>
`--hidden_sizes`: sizes of the LSTM cell `default: [32]`<br/>
`--n_layers_list`: number of layers per LSTM step `default: [1]`<br/>
`--dropouts`: dropout probabilities between LSTM layers `default: [0.0]`<br/>
`--lrs`: initial learning rates `default: [0.001]`<br/>
`--optimizer_types`: optimizers `default: ['adam']`<br/>

The list hyperparameters are to be set using spaces. e.g.:

```
$ python start.py --data_set small --task train --nr_samples 5 --epochs 300 --batch_sizes 32 64 --hidden_sizes 8 16 32 64 128 256 --n_layers_list 1 2 --dropouts 0.0 0.3 0.6 --lrs 0.001 0.0001 --optimizer_types adam sgd
```

### Predict Testset
To calculate the accuracy of the test set using the best model this can be done using the task `--task predict_test`. e.g.:
```
$ python start.py --data_set small --task predict_test 
```

### Predict Document
There is also the possibility to predict the labels of text documents. This API assumes that one sentence or document ist written in each line. The task for this is `--task predict`. This function additionally needs a path to a document with the sentences `--file path/to/file.txt` and a destination file where the predictions are listed `--destination path/to/destination.txt`. Note that if the destination file exists, it will be overwritten. e.g.:

```
$ python start.py --data_set small --task predict  --file path/to/file.txt --destination path/to/destination.txt
```

# Data Set Generation
The three different data sets were built using 100, 1000, and 10000 sampled sentences for each language. The training set consists of 80% of the data whereas development and test set each consist of 10% of the data. <br/>
The script for generating the data sets can be found in `data_set_generation/data_set_generation.py`


# Model
We have trained a character based LSTM model. Each of the characters of a sentence is one-hot-encoded. Unseen characters are replaced with an `<UNK>` token. A start `<S>` token is added to the beginning and an end token `</S>` is added to the end of each sentence. For training all sentences are padded using a `<PAD>` token. <br/>

The last state of the LSTM is forwarded to a MLP for prediction. Each language is predicted using a softmax function. The loss is calculated using cross-entropy. 


