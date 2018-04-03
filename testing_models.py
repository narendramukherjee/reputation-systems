# simple test file for the classes defined in models.py

from models import *



if __name__ == '__main__':


    dynamics = market()
    print(dynamics.generateTimeseries())
    print(dynamics.purchase_count)