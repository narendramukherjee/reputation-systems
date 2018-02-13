# simple test file for the classes defined in models.py

from models_threshold_testing import *



if __name__ == '__main__':

    params = {'product_tracked': 0,
              'prices': [10, 11, 12, 13],
              'product_features': [1, 2, 3, 4],
              'total_number_of_reviews': 20,
              'input_type': 'kurtosis',
              #'input_type': 'histograms',
              #'input_type': 'averages',
              'input_histograms_are_normalized':True,
              }

    dynamics = market(params)

    print('A Torch Sample')
    label_of_data, data = dynamics.genTorchSample()
    print(label_of_data)
    print(data)
    print(dynamics.params['rate_decision_threshold'])

    # print('A Torch Dataset with two samples')
    # dataset = dynamics.genTorchDataset(2)
    # print(dataset)

