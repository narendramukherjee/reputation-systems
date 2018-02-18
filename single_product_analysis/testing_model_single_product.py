# simple test file for the classes defined in models.py

from models_single_product import *
import settings


if __name__ == '__main__':

    dynamics = market(settings.params)


    print('A Torch Sample')
    label_of_data, data = dynamics.genTorchSample()
    print(label_of_data)
    print(data)
    print(dynamics.params)

    # print('A Torch Dataset with two samples')
    # dataset = dynamics.genTorchDataset(2)
    # print(dataset)

