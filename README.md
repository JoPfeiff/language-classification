# Classification of Text Language

In this repository I have built a character based LSTM model that predicts the language of a sentence or a text document.

## Data
The preprocessed training data can be found here: <br/>
https://www.dropbox.com/sh/mrw2gm9saj4fmy1/AAAvaf5rTzFLhDWe-BM3vUiFa?dl=0 <br/>
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
The demo can explicitly be called using
```
$ python start.py --task demo
```
I have created three different data sets small, medium and big each consisting of 100, 1000, and 10000 sentences respectively per language. The data set to-be-used can be defined using e.g. `--data_set small`. The values are `small`, `medium` and `big`. Default is set to `big`.
```
$ python start.py --task demo --data_set small 
```
therefore calls the demo set using the model trained using the small data set. <br/>



