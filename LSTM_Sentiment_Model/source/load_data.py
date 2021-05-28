import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter

path = "/Users/nguyenquan/Desktop/mars_project/neuron_network/LSTM_RNN/data/IMDB_Dataset.csv"


def load_dataset(path):
    # return dataset dataframe
    df = pd.read_csv(path)
    X, y = df['review'].values, df['sentiment'].values
    # split data train and set
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y)
    # print("x_train", x_train.shape)
    # print("x_test", x_test.shape)
    return x_train, x_test, y_train, y_test


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)
    return s


def tockenize(x_train, y_train, x_val, y_val):
    word_list = []

    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label == 'positive' else 0 for label in y_train]
    encoded_test = [1 if label == 'positive' else 0 for label in y_val]
    return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(
        encoded_test), onehot_dict


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def data_preprocessing(path, seq_len=500):
    x_train, x_test, y_train, y_test = load_dataset(path)
    x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)
    x_train_pad = padding_(x_train, seq_len)
    x_test_pad = padding_(x_test, seq_len)
    return x_train_pad, x_test_pad, y_train, y_test, vocab


if __name__ == '__main__':
    x_train_pad, x_test_pad, y_train, y_test, vocab = data_preprocessing(path, 500)
    print(x_train_pad.shape)
    print(x_test_pad.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(len(vocab))
    dict_dataset = {"x_train": x_train_pad, "y_train": y_train, "x_test": x_test_pad, "y_test": y_test, "vocab": vocab }
    # save dataset into dict_dataset
    with open('/Users/nguyenquan/Desktop/mars_project/neuron_network/LSTM_RNN/data/data.p', 'wb') as fp:
        pickle.dump(dict_dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

