# simple test file for the classes defined in models.py

from models_single_product import *
import settings

import matplotlib.pyplot as plt


if __name__ == '__main__':

    dynamics = market(settings.params)

    print(settings.params)

    dynamics.params['value_of_outside_option'] = 5

    print(dynamics.params)

    print('A Torch Sample')
    data, avg_reviews, percieved_qualities = dynamics.generateTimeseries(get_percieved_qualities_and_avg_reviews = True)
    print(data)
    print(avg_reviews)
    print(percieved_qualities)
    print(dynamics.params)

    plt.plot(avg_reviews)
    plt.title('avg_reviews')
    plt.show()

    plt.plot(percieved_qualities)
    plt.title('percieved_qualities')
    plt.show()


    # print('A Torch Dataset with two samples')
    # dataset = dynamics.genTorchDataset(2)
    # print(dataset)

