# RNN for Inferring the thresholds for the decision to leave a review.
# The input data is the time-series of the histograms of reviews put over time
import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import math
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as utils_data

import pickle


import random
#from models_threshold_testing import *

import settings

random.seed()

SENTINEL = object()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class RNN(nn.Module):
    def __init__(self, input_size = settings.number_of_features,
                 hidden_size = settings.n_hidden, num_layers = settings.NUM_LAYERS):
        super(RNN, self).__init__()
        self.softmax = nn.Softmax()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rec = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )

        self.out_type = nn.Linear(hidden_size, settings.OUTPUT_SIZE)
        self.training_losses = []

    def forward(self, input, hidden):
        output, _ = self.rec(input, hidden)
        output_type = F.log_softmax(
            self.out_type(output[-1]))  # self.softmax(self.out(output[-1])) #F.log_softmax(self.out(output[-1]))
        return output_type

    def initHidden(self):
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))

    def initCell(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))

    def getSamplePrediction(self,torchSample):
        hidden0 = self.initHidden()
        # print(torchSample)
        if isinstance(torchSample,list) or isinstance(torchSample,tuple):
            feature = torchSample[1]
        else:
            feature = torchSample
        output_label = self.__call__(Variable(feature), hidden0)
        _, prediction = torch.topk(output_label, 1)
        return prediction.data[0][0], np.exp(output_label.data[0][prediction.data[0][0]])

    def evaluateAveragePerformance(self, dataset, n_iters = SENTINEL):
        random.shuffle(dataset)
        if n_iters is SENTINEL:
            n_iters = len(dataset)
        hidden0 = self.initHidden()
        count_iterations = 0
        count_corrects = 0
        for iter in dataset:
            label = iter[0]
            feature = iter[1]
            output_label = self.__call__(Variable(feature), hidden0)
            print(output_label)
            _, prediction = torch.topk(output_label, 1)
            if prediction.data[0][0] == label[0]:
                count_corrects += 1
            count_iterations += 1
            if count_iterations >= n_iters: break

        avg = count_corrects / count_iterations

        return avg

    def load_from_file(self,file_name = 'model.pkl'):
        loaded_obj = pickle.load(open('./data/'+ file_name, 'rb'))
        return loaded_obj

    def plot_losses(self, losses = SENTINEL,file_name = 'losses.png'):
        if losses == SENTINEL:
            plt.plot(self.training_losses)
            plt.savefig('./data/'+file_name)
        else:
            plt.plot(losses)
            plt.savefig('./data/'+file_name)

    def save_losses(self,losses = SENTINEL,file_name = 'losses.pkl'):
        if losses == SENTINEL:
            pickle.dump(losses, open('./data/'+file_name, 'wb'))
        else:
            losses = self.training_losses
            pickle.dump(losses, open('./data/' +file_name, 'wb'))

    def empty_losses(self):
        self.training_losses = []

    def doTraining(self, dataset, n_iters  = SENTINEL , batch_size=4, window_length_loss=64 , verbose = False ,
                   save = False , file_name = 'model.pkl'):

        if n_iters is SENTINEL:
            n_iters = len(dataset)

        features = []
        labels = []
        for iter in dataset:
            labels.append(iter[0])
            features.append(iter[1])

        training_samples = Simulations_Dataset(n_iters, features, labels)
        training_samples_loader = utils_data.DataLoader(training_samples, batch_size)


        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0)

        moving_avg_losses = []
        current_agg_loss = 0

        count_iterations = 0
        start = time.time()
        total_number_of_rounds = n_iters // batch_size

        for i, samples in enumerate(training_samples_loader):
            if i == total_number_of_rounds:
                break
            # Get each batch
            sampled_features, sampled_labels = samples  # features[i], labels[i] #samples
            # Convert tensors into Variables
            sampled_features, sampled_labels = Variable(sampled_features.permute(1, 0, 2)), Variable(sampled_labels)

            hidden = self.initHidden()
            output_type = self.__call__(sampled_features, hidden)

            optimizer.zero_grad()
            loss_type = criterion(output_type, sampled_labels.view(batch_size))

            loss_type.backward()

            optimizer.step()

            loss = loss_type.data[0]
            count_iterations += 1
            current_agg_loss += loss

            if count_iterations % window_length_loss == 0:
                current_avg_loss = current_agg_loss / window_length_loss
                moving_avg_losses.append(current_avg_loss)
                if verbose:
                    print('%d %d%% (%s) %.4f (avg %.4f) ' %
                        (count_iterations, float(count_iterations) / total_number_of_rounds * 100, timeSince(start), loss,
                         current_avg_loss))

                current_agg_loss = 0
        self.training_losses = self.training_losses + moving_avg_losses
        if save:
            model_fine_tuned = copy.deepcopy(self)
            pickle.dump(model_fine_tuned, open( './data/'+ file_name, 'wb'))
        return moving_avg_losses

class Simulations_Dataset(utils_data.Dataset):
    def __init__(self, n_iters, features, labels):
        self.ids_list = list(range(len(features)))
        self.ids_list = random.sample(self.ids_list,n_iters)
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        feature = self.features[self.ids_list[index]]
        label = self.labels[self.ids_list[index]]
        # print(feature)
        # print(label)
        return feature, label

    def __len__(self):
        return len(self.ids_list)
