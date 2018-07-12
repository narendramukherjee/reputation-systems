# experiment on learning the true quality from perceived signal and avg reviews

from models_single_product_mcmc_replaced import *
import settings
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dynamics = market(settings.params)
    dynamics.params['testing_what'] = 'threshold_fixed'
    dynamics.params['true_quality'] = 5
    print(settings.params)

    dynamics.params['value_of_outside_option'] =5

    print(dynamics.params)

    print('A Torch Sample')
    data, avg_reviews_all_consumers, percieved_qualities,fit_of_customers_who_put_reviews = \
        dynamics.generateTimeseries(get_percieved_qualities_and_avg_reviews=True,
                                    get_fit_of_customers_who_put_reviews=True)
    print(dynamics.params)
    true_quality = [dynamics.params['true_quality']] * len(percieved_qualities)
    print(data)
    print(avg_reviews_all_consumers)
    print(percieved_qualities)
    print(dynamics.params)

    # plt.hold(True) the hold use is depreciated the default behavior is

    percieved_qualities_plot, = plt.plot(percieved_qualities, color='g', label='perceived')
    true_quality_plot,  = plt.plot(true_quality, color='r', label='true')
    avg_reviews_all_consumers_plot, = plt.plot(avg_reviews_all_consumers, color='b', label='observed_averages')
    plt.legend([percieved_qualities_plot,true_quality_plot,avg_reviews_all_consumers_plot],
               ["perceived", "true", "observed_averages"])
    plt.show()

    fit_of_customers_who_put_reviews_plot, = plt.plot(fit_of_customers_who_put_reviews, color='g', label='fit')
    # true_quality_plot, = plt.plot(true_quality, color='r', label='true')
    # avg_reviews_all_consumers_plot, = plt.plot(avg_reviews_all_consumers, color='b', label='observed_averages')
    # plt.legend([percieved_qualities_plot, true_quality_plot, avg_reviews_all_consumers_plot],
    #            ["perceived", "true", "observed_averages"])
    plt.show()