# simple test file for the classes defined in models.py


from model_multiple_products import *
import matplotlib.pyplot as plt


if __name__ == '__main__':




    dynamics = market()
    dynamics.params['total_number_of_reviews'] = 1000
    print(dynamics.generateTimeseries())
    print(dynamics.regret)
    print(dynamics.disappointment)

    plt.plot(dynamics.disappointment)
    plt.ylabel('disappointment')
    plt.xlabel('time (consumer ordinals)')
    plt.show()

    plt.plot(dynamics.regret)
    plt.ylabel('regret')
    plt.xlabel('time (consumer ordinals)')
    plt.show()