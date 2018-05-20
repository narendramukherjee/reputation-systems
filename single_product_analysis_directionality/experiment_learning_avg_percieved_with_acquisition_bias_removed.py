# experiment on learning the true quality from perceived signal and avg reviews

from models_single_product_mcmc_replaced import *
import settings
import matplotlib.pyplot as plt

number_of_iterations = 6

if __name__ == '__main__':
    dynamics = market(settings.params)
    dynamics.params['testing_what'] = 'acquisition_bias'
    print(settings.params)
    dynamics.params['value_of_outside_option'] = 5
    dynamics.params['total_number_of_reviews'] = 200

    print(dynamics.params)
    true_qualities = np.linspace(-5, 12, 40)
    final_avg_review = np.zeros(true_qualities.shape)
    final_perceptions = np.zeros(true_qualities.shape)

    lower_errors_avg = np.zeros(true_qualities.shape)
    upper_errors_avg = np.zeros(true_qualities.shape)

    lower_errors_perception = np.zeros(true_qualities.shape)
    upper_errors_perception = np.zeros(true_qualities.shape)

    asymmetric_errors_avg = [lower_errors_avg, upper_errors_avg]
    asymmetric_errors_perception = [lower_errors_perception, upper_errors_perception]

    # plt.figure()

    for i in range(len(true_qualities)):
        dynamics.params['true_quality'] = true_qualities[i]

        # true_quality = [dynamics.params['true_quality']] * len(perceived_qualities)
        # print(avg_reviews_all_consumers[-10:])
        # print(np.mean(avg_reviews_all_consumers[-10:]))
        dummy_final_avg_review  = np.zeros(number_of_iterations)
        dummy_final_perceptions = np.zeros(number_of_iterations)
        for j in range(number_of_iterations):
            _data_, avg_reviews_all_consumers, perceived_qualities = \
                dynamics.generateTimeseries(get_percieved_qualities_and_avg_reviews=True)
            dummy_final_avg_review[j] = np.mean(avg_reviews_all_consumers[-1:])
            dummy_final_perceptions[j] = np.mean(perceived_qualities[-1:])



        final_avg_review[i]  = np.mean(dummy_final_avg_review)

        final_perceptions[i] = np.mean(dummy_final_perceptions)

        lower_errors_avg[i] = abs(np.min(dummy_final_avg_review) - final_avg_review[i])
        upper_errors_avg[i] = abs(np.max(dummy_final_avg_review) - final_avg_review[i])

        lower_errors_perception[i] = abs(np.min(dummy_final_perceptions) - final_perceptions[i])
        upper_errors_perception[i] = abs(np.max(dummy_final_perceptions) - final_perceptions[i])

            # print(_data)
            # print(avg_reviews_all_consumers)
            # print(perceived_qualities)
            # print(dynamics.params)

            # plt.hold(True) the hold use is depreciated the default behavior is

    # true_qualities, mean_estimated_thetas, yerr=asymmetric_errors)
    # perceived_qualities_plot, = plt.plot(true_qualities,final_perceptions, color='r', label='perceived')

    asymmetric_errors_avg = [lower_errors_avg, upper_errors_avg]
    asymmetric_errors_perception = [lower_errors_perception, upper_errors_perception]

    perceived_qualities_plot = plt.errorbar(true_qualities, final_perceptions, yerr=asymmetric_errors_perception,
                                            color='r', label='perceived')
    true_quality_plot,  = plt.plot(true_qualities,true_qualities, color='g', linestyle=':', label='true')
    avg_reviews_all_consumers_plot = plt.errorbar(true_qualities,final_avg_review,yerr=asymmetric_errors_avg,
                                                  color='b', label='observed_averages')
    plt.legend([perceived_qualities_plot,true_quality_plot,avg_reviews_all_consumers_plot],
               ["perceived", "true", "observed_averages"])
    plt.xlabel('true quality')
    plt.show()