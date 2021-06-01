from encode_data import encode_data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


class Experiment:
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, embedding_dim=64, hidden_dim=256,
                 output_dim=1, no_layers=2, vocab_size=1002, drop_prob=0.5, seq_len=100, clip = 5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        self.drop_prob = drop_prob
        self.seq_len = seq_len
        self.clip = clip

    def get_data(self):
        file_name = "/Users/nguyenquan/Desktop/mars_project/neuron_network/sentiment_analysis/data/train.crash"
        x_train, y_train, x_valid, y_valid, vocab = encode_data(file_name, self.seq_len)
        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        valid_data = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
        # make sure to SHUFFLE your data
        train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size, drop_last=True)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=self.batch_size, drop_last=True)
        return train_loader, valid_loader

    def train_and_eval(self):
        train_loader, valid_loader = self.get_data()
        # model and opt
        model = SentimentRNN(self.no_layers, self.vocab_size,
                             self.hidden_dim, self.embedding_dim, self.output_dim, self.drop_prob)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        valid_loss_min = np.Inf
        # train for some number of epochs
        epoch_tr_loss, epoch_vl_loss = [], []
        epoch_tr_acc, epoch_vl_acc = [], []
        for epoch in range(self.epochs):
            train_losses = []
            train_acc = 0.0
            model.train()
            # initialize hidden state
            h = model.init_hidden(self.batch_size)
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # print("inputs:", inputs)
                # print("inputs shape:", inputs.shape)
                # print("labels:", labels)
                # print("labels shape:", labels.shape)

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])
                # print("h:", h)
                # print("h shape:", len(h))
                model.zero_grad()
                output, h = model(inputs, h)

                # calculate the loss and perform backprop
                loss = model.loss(output.squeeze(), labels.float())
                loss.backward()
                train_losses.append(loss.item())
                # calculating accuracy
                accuracy = acc(output, labels)
                train_acc += accuracy
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                optimizer.step()

            val_h = model.init_hidden(self.batch_size)
            val_losses = []
            val_acc = 0.0
            model.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])

                inputs, labels = inputs.to(device), labels.to(device)

                output, val_h = model(inputs, val_h)
                val_loss = model.loss(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

                accuracy = acc(output, labels)
                val_acc += accuracy

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_train_acc = train_acc / len(train_loader.dataset)
            epoch_val_acc = val_acc / len(valid_loader.dataset)
            epoch_tr_loss.append(epoch_train_loss)
            epoch_vl_loss.append(epoch_val_loss)
            epoch_tr_acc.append(epoch_train_acc)
            epoch_vl_acc.append(epoch_val_acc)
            print(f'Epoch {epoch + 1}')
            print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
            print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')


if __name__ == '__main__':
    experiment = Experiment(learning_rate=0.001, epochs=10, batch_size=50, embedding_dim=64, hidden_dim=256,
                            output_dim=1, no_layers=2, vocab_size=1002, drop_prob=0.5, seq_len=100, clip=5)
    experiment.train_and_eval()
