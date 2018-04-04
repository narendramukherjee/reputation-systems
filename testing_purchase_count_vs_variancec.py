# simple test file for the classes defined in models.py

from model_multiple_products import *
import matplotlib.pyplot as plt





if __name__ == '__main__':

    low_quality = 1.0
    high_quality = 5.0

    test_product_id = 1  # that is the product for which we vary the quality and plot the purchase counts versus variance
    fit_std_levels = np.linspace(1, 10, 5) # increase 5 to get more points, better plots

    number_of_averaging_iterations = 10 # each recorded value of the purchase count is the average of 10 random itertaions,
    # reduce this for faster code

    params = dict()
    params['total_number_of_reviews'] = 100 # the larger the better if too small then there is not enogh time to learn the
    # qualities from reviews, reduce this for faster code
    params['population_beta'] = {'feature_beta': [1.5416241306174765, 1]}
    params['population_alpha'] = [-2.8194872921927585, 1]
    params['true_qualities'] = [2.8877345395086067, 'will_be_replace_with_high_and_low',
                                3.7620897299370544, 4.8556626263703766]

    dynamics = market(params)

    # the low quality loop:
    low_quality_purchase_counts = []
    dynamics.params['true_qualities'][test_product_id] = low_quality
    for fit_std in fit_std_levels:
        dynamics.params['consumer_fit_std'][test_product_id] = fit_std
        # print(dynamics.params)
        # print(dynamics.generateTimeseries())
        # print(dynamics.purchase_count)

        product_purchases = []
        for iter in range(number_of_averaging_iterations):
            dynamics.generateTimeseries() # generate a time series
            all_counts = dynamics.purchase_count
            product_purchases += [all_counts[test_product_id]]
            print(all_counts)
            print(product_purchases)

        low_quality_purchase_counts += [np.mean(product_purchases)]

    print(low_quality_purchase_counts)
    plt.plot(fit_std_levels, low_quality_purchase_counts)
    plt.xlabel('Variance in ratings')
    plt.ylabel('Number of purchases')
    plt.show()
