# Processes the observed tomes series of reviews for one product and feeds the processed data to the trained network for
# classification

from inference_threshold_testing import *

import random as RD

import pandas as pd

import settings

import os

RD.seed()


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

    model = RNN()

    model = model.load_from_file('model_tuned_' + settings.tracked_product_ID + '.pkl')

    current_working_directory = os.getcwd()
    observed_timeseries = pd.read_csv(current_working_directory + '/data/' + settings.tracked_product_ID +
                                      '_time_series.txt', sep='\t')
    processed_timeseries = process_observed_timeseries(observed_timeseries, settings.params['input_type'])

    # observed_timeseries = pickle.load(open('./data/'+ 'asus_data.pkl', 'rb'))#model.load_from_file('asus_data.pkl')
    # cannot load pickle data in Python3 file if data is dumped in Python2

    print(processed_timeseries)

    print('The classifier output on the tablet reviews data is:')
    print(model.getSamplePrediction(processed_timeseries))
