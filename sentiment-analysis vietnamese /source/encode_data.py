import pandas as pd
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import numpy as np
from data_processing import normalize_text, no_marks, diction_nag_pos_not


class DataSource(object):
    def _load_raw_data(self, filename, is_train=True):
        a = []
        b = []
        regex = 'train_'
        if not is_train:
            regex = 'test_'
        with open(filename, 'r', encoding='utf8') as file:
            for line in file:
                if regex in line:
                    b.append(a)
                    a = [line]
                elif line != '\n':
                    a.append(line)
        b.append(a)
        return b[1:]

    def _create_row(self, sample, is_train=True):
        d = {}
        d['id'] = sample[0].replace('\n', '')
        review = ""
        if is_train:
            for clause in sample[1:-1]:
                review += clause.replace('\n', '').strip()
            d['label'] = int(sample[-1].replace('\n', ''))
        else:
            for clause in sample[1:]:
                review += clause.replace('\n', '').strip()
        d['review'] = review
        return d

    def load_data(self, filename, is_train=True):

        raw_data = self._load_raw_data(filename, is_train)
        lst = []

        for row in raw_data:
            lst.append(self._create_row(row, is_train))

        return lst

    def transform_to_dataset(self, x_set, y_set):
        X, y = [], []
        for document, topic in zip(list(x_set), list(y_set)):
            document = normalize_text(document)
            X.append(document.strip())
            y.append(topic)
            # Augmentation bằng cách remove dấu tiếng Việt
            X.append(no_marks(document))
            y.append(topic)
        return X, y

    def return_data(self, file_name):
        ds = DataSource()
        data = pd.DataFrame(ds.load_data(file_name))
        '''
        new_data = []
        # Thêm mẫu bằng cách lấy trong từ điển Sentiment (nag/pos)
        nag_list, pos_list, not_list = diction_nag_pos_not()
        for index, row in enumerate(pos_list):
            new_data.append(['pos' + str(index), '0', row])
        for index, row in enumerate(nag_list):
            new_data.append(['nag' + str(index), '1', row])

        new_data = pd.DataFrame(new_data, columns=list(['id', 'label', 'review']))
        data = data.append(new_data, ignore_index=True)
        '''
        return data

def read_file():
    # doc file stop word Viet Nam
    file_name = "/Users/nguyenquan/Desktop/mars_project/neuron_network/sentiment_analysis/data/vietnamese_stopwords.txt"
    with open(file_name) as f:
        lines = f.readlines()
    stop_word = []
    for line in lines:
        stop_word.append(line.rstrip("\n"))
    return set(stop_word)


def load_dataset(file_name):
    ds = DataSource()
    data = ds.return_data(file_name)
    data_review, data_label = ds.transform_to_dataset(data.review, data.label)
    x_train, x_valid, y_train, y_valid = train_test_split(data_review, data_label, test_size=0.3,
                                                          random_state=42)
    return x_train, y_train, x_valid, y_valid


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s


def tokenize(x_train, y_train, x_valid, y_valid):
    word_list = []
    stop_words = read_file()

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
    onehot_dict["nan_word"] = 1001

    # tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_valid:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = y_train
    encoded_test = y_valid
    return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(
        encoded_test), onehot_dict


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def encode_data(file_name, seq_len):
    x_train, y_train, x_valid, y_valid = load_dataset(file_name)
    x_train, y_train, x_valid, y_valid, vocab = tokenize(x_train, y_train, x_valid, y_valid)
    x_train = padding_(x_train, seq_len)
    x_valid = padding_(x_valid, seq_len)
    return x_train, y_train, x_valid, y_valid, vocab



