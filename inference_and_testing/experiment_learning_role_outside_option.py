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

    final_avg_review = np.zeros([len(true_qualities), len(outside_options)])

    final_perceptions = np.zeros([len(true_qualities), len(outside_options)])

    lower_errors_avg = np.zeros([len(true_qualities), len(outside_options)])
    upper_errors_avg = np.zeros([len(true_qualities), len(outside_options)])

    lower_errors_perception = np.zeros([len(true_qualities), len(outside_options)])
    upper_errors_perception = np.zeros([len(true_qualities), len(outside_options)])

    # asymmetric_errors_avg = [lower_errors_avg, upper_errors_avg]
    # asymmetric_errors_perception = [lower_errors_perception, upper_errors_perception]




    # observed averages loop:

    for k in range(len(true_qualities)):
        for i in range(len(outside_options)):
            dynamics.params['true_quality'] = true_qualities[k]
            dynamics.params['value_of_outside_option'] = outside_options[i]
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

            final_avg_review[k][i]  = np.mean(dummy_final_avg_review)

            final_perceptions[k][i] = np.mean(dummy_final_perceptions)

            lower_errors_avg[k][i] = abs(np.min(dummy_final_avg_review) - final_avg_review[k][i])
            upper_errors_avg[k][i] = abs(np.max(dummy_final_avg_review) - final_avg_review[k][i])

            lower_errors_perception[k][i] = abs(np.min(dummy_final_perceptions) - final_perceptions[k][i])
            upper_errors_perception[k][i] = abs(np.max(dummy_final_perceptions) - final_perceptions[k][i])

            # print(_data)
            # print(avg_reviews_all_consumers)
            # print(perceived_qualities)
            # print(dynamics.params)

            # plt.hold(True) the hold use is depreciated the default behavior is

        # true_qualities, mean_estimated_thetas, yerr=asymmetric_errors)
        # perceived_qualities_plot, = plt.plot(true_qualities,final_perceptions, color='r', label='perceived')
        print(k)
        # print('asymmetric_errors_avg', asymmetric_errors_avg)
        # print('asymmetric_errors_perception', asymmetric_errors_perception)
        # print('lower_errors_avg', lower_errors_avg)
        # print('upper_errors_avg', upper_errors_avg)
        # print('lower_errors_perception', lower_errors_perception)
        # print('upper_errors_perception', upper_errors_perception)
        # # asymmetric_errors_avg[k] = [lower_errors_avg[k], upper_errors_avg[k]]
        # asymmetric_errors_perception[k] = [lower_errors_perception[k], upper_errors_perception[k]]

        # print(asymmetric_errors_avg[0])
        # print(asymmetric_errors_perception[:][k])

    asymmetric_errors_avg = [lower_errors_avg, upper_errors_avg]

    asymmetric_errors_perception = [lower_errors_perception, upper_errors_perception]

    print(dynamics.params)



    # loop to plot averages
    clr = ['b', 'r', 'c', 'g', 'm']
    for k in range(len(true_qualities)):

        plot_labels_avg = []

        # dummy_label_perceived_qualities_plot = plt.errorbar(outside_options, final_perceptions,
        #                                                     yerr=asymmetric_errors_perception,
        #                                                     color='r', label='true quality ' + str(true_qualities[k]))


        # print("true quality "+str(true_qualities[k]))
        #
        # print(clr[k])
        #
        # print(lower_errors_avg[k])
        # print(upper_errors_avg[k])
        # print(outside_options)
        # print(final_avg_review)

        dummy_avg_reviews_all_consumers_plot = plt.errorbar(outside_options, final_avg_review[k],
                                                            yerr=[lower_errors_avg[k],upper_errors_avg[k]],
                                                            color=clr[k], label="true quality "+str(true_qualities[k]))

        # plot_labels_avg += [dummy_avg_reviews_all_consumers_plot]

        # plt.legend(dummy_avg_reviews_all_consumers_plot, ["true quality "+str(true_qualities[k])])

        # plot_labels_perceptions += [dummy_avg_reviews_all_consumers_plot]
    # print(plot_labels_avg)
    plt.legend() #plot_labels_avg,["true quality 1","true quality 2","true quality 3","true quality 4","true quality 5"]
    plt.ylim(1.5, 5)
    plt.xlabel('outside option value')
    plt.ylabel('final average of reviews')
    plt.show()

    # loop to plot perceptions
    clr = ['b', 'r', 'c', 'g', 'm']
    for k in range(len(true_qualities)):
        plot_labels_perceptions = []

        dummy_label_perceived_qualities_plot = plt.errorbar(outside_options, final_perceptions[k],
                                                            yerr=[lower_errors_perception[k], upper_errors_perception[k]],
                                                            color=clr[k],
                                                            label="true quality " + str(true_qualities[k]))

    plt.legend()  # plot_labels_avg,["true quality 1","true quality 2","true quality 3","true quality 4","true quality 5"]
    plt.ylim(0.5, 6)
    plt.xlabel('outside option value')
    plt.ylabel('final perception of quality')
    plt.show()


    # loop to plot overlay perceptions and averages

    clr = ['b', 'r', 'c', 'g', 'm']
    for k in [1,3,5]:
        plot_labels_perceptions = []

        dummy_label_perceived_qualities_plot = plt.errorbar(outside_options, final_perceptions[k-1],
                                                            yerr=[lower_errors_perception[k-1], upper_errors_perception[k-1]],
                                                            color=clr[k-1],
                                                            label="perceived quality, true quality =" + str(true_qualities[k-1]))
        dummy_label_avg_plot = plt.errorbar(outside_options, final_avg_review[k - 1],
                                                            yerr=[lower_errors_perception[k-1],
                                                                  upper_errors_perception[k-1]],
                                                            color=clr[k-1],linestyle=':',
                                                            label="average rating, true quality =" + str(true_qualities[k-1]))

    plt.legend()  # plot_labels_avg,["true quality 1","true quality 2","true quality 3","true quality 4","true quality 5"]
    plt.ylim(0.5,6)
    plt.xlabel('outside option value')
    plt.ylabel('final perception of quality/average rating')

    plt.show()


    #
    # for k in range((true_qualities)):
    #     plot_labels_avg = []
    #     plot_labels_perceptions = []
    #     for i in range(len(outside_options)):
    #         dynamics.params['true_quality'] = true_qualities[k]
    #         dynamics.params['value_of_outside_option'] = outside_options[i]
    #         # true_quality = [dynamics.params['true_quality']] * len(perceived_qualities)
    #         # print(avg_reviews_all_consumers[-10:])
    #         # print(np.mean(avg_reviews_all_consumers[-10:]))
    #         dummy_final_avg_review = np.zeros(number_of_iterations)
    #         dummy_final_perceptions = np.zeros(number_of_iterations)
    #         for j in range(number_of_iterations):
    #             _data_, avg_reviews_all_consumers, perceived_qualities = \
    #                 dynamics.generateTimeseries(get_percieved_qualities_and_avg_reviews=True)
    #             dummy_final_avg_review[j] = np.mean(avg_reviews_all_consumers[-1:])
    #             dummy_final_perceptions[j] = np.mean(perceived_qualities[-1:])
    #
    #         final_avg_review[i] = np.mean(dummy_final_avg_review)
    #
    #         final_perceptions[i] = np.mean(dummy_final_perceptions)
    #
    #         lower_errors_avg[i] = abs(np.min(dummy_final_avg_review) - final_avg_review[i])
    #         upper_errors_avg[i] = abs(np.max(dummy_final_avg_review) - final_avg_review[i])
    #
    #         lower_errors_perception[i] = abs(np.min(dummy_final_perceptions) - final_perceptions[i])
    #         upper_errors_perception[i] = abs(np.max(dummy_final_perceptions) - final_perceptions[i])
    #
    #         # print(_data)
    #         # print(avg_reviews_all_consumers)
    #         # print(perceived_qualities)
    #         # print(dynamics.params)
    #
    #         # plt.hold(True) the hold use is depreciated the default behavior is
    #
    #     # true_qualities, mean_estimated_thetas, yerr=asymmetric_errors)
    #     # perceived_qualities_plot, = plt.plot(true_qualities,final_perceptions, color='r', label='perceived')
    #
    #     asymmetric_errors_avg = [lower_errors_avg, upper_errors_avg]
    #     asymmetric_errors_perception = [lower_errors_perception, upper_errors_perception]
    #
    #     dummy_label_perceived_qualities_plot = plt.errorbar(outside_options, final_perceptions,
    #                                                         yerr=asymmetric_errors_perception,
    #                                                         color='r', label='true quality ' + str(true_qualities[k]))
    #     plot_labels_avg += [dummy_label_perceived_qualities_plot]
    #
    #     # true_quality_plot,  = plt.plot(true_qualities,true_qualities, color='g', linestyle=':', label='true')
    #     dummy_avg_reviews_all_consumers_plot = plt.errorbar(outside_options, final_avg_review,
    #                                                         yerr=asymmetric_errors_avg,
    #                                                         color='b', label='true quality ' + str(true_qualities[k]))
    #
    #     plot_labels_perceptions += [dummy_avg_reviews_all_consumers_plot]
    # plt.legend([perceived_qualities_plot, true_quality_plot, avg_reviews_all_consumers_plot],
    #            ["perceived", "true", "observed_averages"])
    # plt.xlabel('outside option value')
    # plt.show()