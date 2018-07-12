# experiment on learning the true quality from perceived signal and avg reviews

from models_single_product_mcmc_replaced import *
import settings
import matplotlib.pyplot as plt

number_of_iterations = 6

if __name__ == '__main__':
    dynamics = market(settings.params)
    dynamics.params['testing_what'] = 'threshold_fixed'
    print(settings.params)
    # dynamics.params['value_of_outside_option'] = 0
    dynamics.params['total_number_of_reviews'] = 100

    print(dynamics.params)

    true_qualities = [1,2,3,4,5]
    outside_options = np.linspace(1, 7, 10)

    total_number_reviews = np.zeros([len(true_qualities), len(outside_options)])

    lower_errors = np.zeros([len(true_qualities), len(outside_options)])
    upper_errors = np.zeros([len(true_qualities), len(outside_options)])

    # lower_errors_perception = np.zeros([len(true_qualities), len(outside_options)])
    # upper_errors_perception = np.zeros([len(true_qualities), len(outside_options)])

    # asymmetric_errors_avg = [lower_errors_avg, upper_errors_avg]
    # asymmetric_errors_perception = [lower_errors_perception, upper_errors_perception]

    # observed averages loop:

    for k in range(len(true_qualities)):
        for i in range(len(outside_options)):
            dynamics.params['true_quality'] = true_qualities[k]
            dynamics.params['value_of_outside_option'] = outside_options[i]
            dummy_number_of_reviews = np.zeros(number_of_iterations)
            for j in range(number_of_iterations):
                review_data = \
                    dynamics.generateTimeseries(fix_population_size=True, population_size=1000,
                                                get_percieved_qualities_and_avg_reviews=False)
                dummy_number_of_reviews[j] = len(review_data)

            total_number_reviews[k][i] = np.mean(dummy_number_of_reviews)

            lower_errors[k][i] = abs(np.min(dummy_number_of_reviews) - total_number_reviews[k][i])
            upper_errors[k][i] = abs(np.max(dummy_number_of_reviews) - total_number_reviews[k][i])

        print(k)

    asymmetric_errors = [lower_errors, upper_errors]

    print(dynamics.params)

    # loop to plot averages
    clr = ['b', 'r', 'c', 'g', 'm']
    for k in range(len(true_qualities)):

        plot_labels_avg = []

        number_reviews_plot = plt.errorbar(outside_options, total_number_reviews[k],
                                                            yerr=[lower_errors[k],upper_errors[k]],
                                                            color=clr[k], label="true quality "+str(true_qualities[k]))

        # plot_labels_avg += [dummy_avg_reviews_all_consumers_plot]

        # plt.legend(dummy_avg_reviews_all_consumers_plot, ["true quality "+str(true_qualities[k])])

        # plot_labels_perceptions += [dummy_avg_reviews_all_consumers_plot]
    # print(plot_labels_avg)
    plt.legend() # plot_labels_avg,["true quality 1","true quality 2","true quality 3","true quality 4","true quality 5"]
    # plt.ylim(1.5, 5)
    plt.xlabel('outside option value')
    plt.ylabel('number of reviews')
    plt.show()