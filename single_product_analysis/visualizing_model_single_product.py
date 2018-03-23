# visualizes the simulated training data



import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from models_single_product import *
import settings

import pylab as PL
import random as DR
import  numpy as np
import pandas as pd

import pickle

RD.seed()

import pycxsimulator

def init_viz():
    global time, time_series
    time = 0

def draw():
    global time, time_series
    PL.cla()
    if time == 0:
        normalized_histogram = time_series[time]
    else:
        normalized_histogram = list(np.array(time_series[time])/(0.01*time))

    bar_list = plt.bar(list(range(1, settings.params['number_of_rating_levels']+1)), normalized_histogram)
    bar_list[0].set_color('y')
    bar_list[1].set_color('y')
    bar_list[2].set_color('y')
    bar_list[3].set_color('y')
    bar_list[4].set_color('y')
    plt.ylim(ymin=0)
    plt.xticks(list(range(1, settings.params['number_of_rating_levels'] + 1)))
    plt.yticks(list(np.linspace(0, 100, num=5)))
    plt.ylabel('frequency (%)')
    plt.xlabel('reviews')
    PL.axis([0,6,0,120])
    #PL.savefig(str(customer_count)+'.png')

def step_viz():
    global time, time_series
    if time < len(time_series) - 1:
        time += 1

def process_observed_timeseries(observed_timeseries, input_type):
    if input_type == 'histograms':
        all_ratings = list(observed_timeseries['Rating'])
        current_histogram = [0] * settings.params['number_of_rating_levels']
        histogram_timeseries = [[0] * settings.params['number_of_rating_levels']]
        for rating in all_ratings:
            current_histogram[rating - 1] += 1
            append_histogram = copy.deepcopy(current_histogram)
            histogram_timeseries.append(append_histogram)
    df = pd.DataFrame(histogram_timeseries)
    torch_data = torch.FloatTensor(df.values[:, 0:5].astype(float))
    return torch_data

if __name__ == '__main__':

    dynamics = market(settings.params)
    print(settings.params)


    print('A Torch Sample')
    label_of_data, data = dynamics.genTorchSample()
    print(label_of_data)
    print(data)
    print(dynamics.params)

    # print('A Torch Dataset with two samples')
    # dataset = dynamics.genTorchDataset(2)
    # print(dataset)

    # network_time_series = pickle.load(open('./data/10000_net/' + 'network_time_series.pkl', 'rb'))

    # visualize time series
    global time_series

    observed_timeseries = pd.read_csv('./data/' + settings.tracked_product_ID +
                                      '_time_series.txt', sep='\t')
    time_series = process_observed_timeseries(observed_timeseries, settings.params['input_type'])

    # time_series = data

    pycxsimulator.GUI().start(func=[init_viz, draw, step_viz])

