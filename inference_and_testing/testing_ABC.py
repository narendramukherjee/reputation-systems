# This is to test the performance of the estimators that use ABC posterior samples.
# We plot the estimated parameter versus the true values.

import settings
from inference_ABC import *



random.seed()


if __name__ == '__main__':

    true_thetas = [0.5,0.75,1, 1.25]#, 1, 1.25, 1.5, 1.75, 2]
    prior_theta = np.linspace(0, 2.5, 100)
    gen_model = ABC_GenerativeModel(params=settings.params, prior=prior_theta,
                                    conditioning=False, direction=None)

    # error, theta_estimates = eval_ABC_posterior(true_theta, gen_model, epsilon=0.03, n_posterior_samples=20, n_estimates=2,
    #                    estimator_type='posterior_mean', bin_size=10)

    # note: epsilons should be ABC
    estimator = Estimator(gen_model, epsilons=[0.05], n_posterior_samples=2, n_samples=1,
                 estimator_type='All', bin_size=10, error_type='MSE')

    estimator.get_estimates_for_true_thetas(true_thetas, do_plot=settings.do_plots, symmetric=True,
                                            verbose=True, do_hist=False,
                                            compute_estimates=False,
                                            save_estimates=False,
                                            load_estimates=True)

    # print('error:', error)
    # print('theta_estimates (posterior means):', theta_estimates)

    # probability_above = gen_model.compute_direction_probability(dataset_size=50)
    # print('probability_above is:', probability_above)
    # obsreved_data = gen_model.generate_data(true_theta)
    # print('obsreved_data', obsreved_data)
    # (posterior, distances,
    #  accepted_count, trial_count,
    #  epsilon) = basic_abc(gen_model, obsreved_data, epsilon=0.2, min_samples=10)
    # print('posterior',posterior)
    # print('distances', distances)
    # print('accepted_count', accepted_count)
    # print('trial_count', trial_count)
    # print('epsilon', epsilon)
    # print(np.mean(posterior))









