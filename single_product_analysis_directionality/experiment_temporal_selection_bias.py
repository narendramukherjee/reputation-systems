# experiment on learning the true quality from perceived signal and avg reviews

from models_single_product_mcmc_replaced import *
import settings
import matplotlib.pyplot as plt

number_of_iterations = 100

if __name__ == '__main__':
    dynamics = market(settings.params)
    dynamics.params['testing_what'] = 'threshold_fixed'
    print(settings.params)
    # dynamics.params['value_of_outside_option'] = 0
    dynamics.params['total_number_of_reviews'] = 100

    print(dynamics.params)

    true_qualities = [1,2,3,4,5]
    outside_options = np.linspace(0, 7, 10)

    # the differential stat is the difference between the mean of the first ten and the last ten

    differential_stat_ratings = np.zeros([len(true_qualities), len(outside_options)])

    differential_stat_perceptions = np.zeros([len(true_qualities), len(outside_options)])

    differential_stat_fits = np.zeros([len(true_qualities), len(outside_options)])

    # lower_errors_avg = np.zeros([len(true_qualities), len(outside_options)])
    # upper_errors_avg = np.zeros([len(true_qualities), len(outside_options)])
    #
    # lower_errors_perception = np.zeros([len(true_qualities), len(outside_options)])
    # upper_errors_perception = np.zeros([len(true_qualities), len(outside_options)])

    errors_ratings = np.zeros([len(true_qualities), len(outside_options)])

    errors_perception = np.zeros([len(true_qualities), len(outside_options)])

    errors_fits = np.zeros([len(true_qualities), len(outside_options)])

    # asymmetric_errors_avg = [lower_errors_avg, upper_errors_avg]
    # asymmetric_errors_perception = [lower_errors_perception, upper_errors_perception]

    # observed averages loop:

    for k in range(len(true_qualities)):
        for i in range(len(outside_options)):
            dynamics.params['true_quality'] = true_qualities[k]
            dynamics.params['value_of_outside_option'] = outside_options[i]
            dynamics.params['rate_decision_threshold_above'] = 2.0
            dynamics.params['rate_decision_threshold_below'] = dynamics.params['rate_decision_threshold_above']
            # true_quality = [dynamics.params['true_quality']] * len(perceived_qualities)
            # print(avg_reviews_all_consumers[-10:])
            # print(np.mean(avg_reviews_all_consumers[-10:]))
            dummy_ratings_differential_stat = np.zeros(number_of_iterations)

            dummy_perceptions_differential_stat = np.zeros(number_of_iterations)

            dummy_fits_differential_stat = np.zeros(number_of_iterations)

            for j in range(number_of_iterations):
                ratings, avg_reviews_all_consumers, perceived_qualities, fits = \
                    dynamics.generateTimeseries(fix_population_size=False,
                                                raw=True,
                                                population_size=None,
                                                get_percieved_qualities_and_avg_reviews=True,
                                                get_fit_of_customers_who_put_reviews=True,
                                                do_not_return_df=True)

                dummy_ratings_differential_stat[j] = (np.mean(ratings[:20]) -
                                                      np.mean(ratings[-20:]))

                dummy_perceptions_differential_stat[j] = (np.mean(perceived_qualities[:20]) -
                                                          np.mean(perceived_qualities[-20:]))

                dummy_fits_differential_stat[j] = (np.mean(fits[:20]) -
                                                   np.mean(fits[-20:]))

            differential_stat_ratings[k][i] = np.mean(dummy_ratings_differential_stat)

            differential_stat_perceptions[k][i] = np.mean(dummy_perceptions_differential_stat)

            differential_stat_fits[k][i] = np.mean(dummy_fits_differential_stat)

            errors_ratings[k][i] = np.std(dummy_ratings_differential_stat)

            errors_perception[k][i] = np.std(dummy_perceptions_differential_stat)

            errors_fits[k][i] = np.std(dummy_fits_differential_stat)

        print(k)

    # loop to plot ratings
    clr = ['b', 'r', 'c', 'g', 'm']
    for k in range(len(true_qualities)):

        print('differential_stat_ratings',differential_stat_ratings[k])

        ratings_plot = plt.errorbar(outside_options,
                                    differential_stat_ratings[k],
                                    yerr=errors_ratings[k],
                                    color=clr[k],
                                    label="true quality "+str(true_qualities[k]))

        plt.legend() #plot_labels_avg,["true quality 1","true quality 2","true quality 3","true quality 4","true quality 5"]
        # plt.ylim(1.5, 5)
        plt.xlabel('outside option value')
        plt.ylabel('difference in the first and last 15 points')
        plt.title('temporal differences in ratings')
        plt.show()

    # loop to plot perceptions
    clr = ['b', 'r', 'c', 'g', 'm']
    for k in range(len(true_qualities)):

        print('differential_stat_perceptions',differential_stat_perceptions[k])

        perceptions_plot = plt.errorbar(outside_options,
                                        differential_stat_perceptions[k],
                                        yerr=errors_perception[k],
                                        color=clr[k],
                                        label="true quality " + str(true_qualities[k]))

        plt.legend()  # plot_labels_avg,["true quality 1","true quality 2","true quality 3","true quality 4","true quality 5"]
        # plt.ylim(1.5, 5)
        plt.xlabel('outside option value')
        plt.ylabel('difference in the first and last 15 points')
        plt.title('temporal differences in perceptions')
        plt.show()

    clr = ['b', 'r', 'c', 'g', 'm']
    for k in range(len(true_qualities)):

        print('differential_stat_fits',differential_stat_fits[k])

        fits_plot = plt.errorbar(outside_options,
                                 differential_stat_fits[k],
                                 yerr=errors_fits[k],
                                 color=clr[k],
                                 label="true quality " + str(true_qualities[k]))

        plt.legend()  # plot_labels_avg,["true quality 1","true quality 2","true quality 3","true quality 4","true quality 5"]
        # plt.ylim(1.5, 5)
        plt.xlabel('outside option value')
        plt.ylabel('difference in the first and last 15 points')
        plt.title('temporal differences in customer fits')
        plt.show()