# ABC codes adopted from https://github.com/rcmorehead/simpleabc/blob/master/simple_abc.py

import settings

if settings.do_plots:
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


import random

import numpy as np
from models import market
import copy
import pickle
import pandas as pd
import numpy as np


random.seed()

SENTINEL = object()


class ABC_GenerativeModel(market):
    """
    Base class for constructing models for approximate bayesian computing
    inherits the market and implements the following additional methods:
    * Model.draw_theta
    * Model.generate_data
    * Model.summary_stats
    * Model.distance_function
    """

    def __init__(self, params={}, prior=np.linspace(0,2,100), conditioning=False, direction=None):
        """
        params dict for the generative model.
        if conditioning is True, then direction should be 'above' or 'below'
        """
        super(ABC_GenerativeModel, self).__init__(params)
        self.fixed_params['input_type'] = 'histograms'
        self.fixed_params['input_histograms_are_normalized'] = True
        self.params['input_type'] = 'histograms'
        self.params['input_histograms_are_normalized'] = True
        self.conditioning = conditioning
        self.direction = direction
        self.prior = prior

    def set_data(self, data):
        self.data = data # this is the observed data
        self.data_summary_stats = self.summary_stats(self.data)

    def set_epsilons(self, epsilons):
        """
        A method to give the model object the value of epsilon if your model
        code needs to know it.
        """
        if len(epsilons) > 1:
            assert len(epsilons) == settings.number_of_summaries,\
                "the length of epsilon does not match number_of_summaries"
        elif len(epsilons) == 1:
            epsilons = [epsilons[0]]*settings.number_of_summaries
        # print('epsilons:', epsilons)
        self.epsilons = epsilons

    def draw_theta(self):
        """
        Sub-classable method for drawing from a prior distribution.
        This method should return an array-like iterable that is a vector of
        proposed model parameters from your prior distribution.
        """
        # print('prior is:', self.prior)
        theta = np.random.choice(self.prior)
        # print('drawn theta:', theta)
        return theta

    def generate_data(self, theta):
        """
        Sub-classable method for generating synthetic data sets from forward
        model.
        This method should return an array/matrix/table of simulated data
        taking vector theta as an argument.
        """
        # self.fixed_params['rate_decision_threshold'] = theta
        data = self.generateTimeseries(theta, raw=True, get_percieved_qualities_and_avg_reviews=False,
                                       do_not_return_df=True)
        # print('synthetic', data)
        return data

    def summary_stats(self, data,number=settings.number_of_summaries,spacing=settings.space_between_summaries):
        """
        Sub-classable method for computing summary statistics.
        This method should return an array-like iterable of summary statistics
        taking an array/matrix/table as an argument
        """

        # print('input to summary stats:', data)
        # return np.asarray(data.iloc[-1])
        processed_data = self.process_raw_timeseries(data, processed_output='histograms')
        # print('processed_data in summary_states:', processed_data)
        summaries = np.zeros([number,settings.number_of_rating_levels])
        for num in range(number):
            summaries[num] = np.asarray(processed_data[-(1+num*spacing)])
        # summary = np.asarray(self.process_raw_timeseries(data, processed_output='histogram'))
        # print('summary', summary)
        return summaries

    def distance_function(self, summary_stats, summary_stats_synth):
        """
        Sub-classable method for computing a distance function.
        This method should return a distance D of for comparing to the
        acceptance tolerance (epsilon) taking two array-like iterables of
        summary statistics as an argument (nominally the observed summary
        statistics and .
        """
        # print('summary_stats:',summary_stats)
        # print('summary_stats_synth:',summary_stats_synth)
        assert summary_stats.shape == summary_stats_synth.shape, "summary_stats dimensions mismatch"
        number = summary_stats.shape[0]
        distances = np.zeros(number)
        for num in range(number):
            distances[num] = 0.2 * sum(abs(summary_stats[num] - summary_stats_synth[num]))
        # print('distances:',distances)
        return distances

    def process_raw_timeseries(self,raw_timeseries, processed_output='histograms'):
        """Returns the histogram can be applied to the the output of generateTimeseries(raw=True)
        should do all_ratings = list(raw_timeseries['Rating']) before applying process_raw_timeseries() on the data from
        tablets.
        """
        if self.conditioning:
            assert self.direction is not None, "The direction for conditioning is not provided."

            raw_averages = []
            for i in range(1,len(raw_timeseries)+1):
                raw_averages.append(np.mean(raw_timeseries[:i]))

            print('raw_averages' , raw_averages)

            conditioned_timeseries = [raw_timeseries[0]]

            for i in range(1, len(raw_timeseries)):
                if self.direction is 'above' and raw_timeseries[i] < raw_averages[i-1]:
                    # print('raw_timeseries[i]', raw_timeseries[i])
                    # print('raw_averages[i-1]', raw_averages[i-1])
                    continue
                elif self.direction is 'below' and raw_timeseries[i] > raw_averages[i-1]:
                    continue
                else:
                    conditioned_timeseries.append(raw_timeseries[i])

            print('conditioned_timeseries', conditioned_timeseries)

        if self.conditioning:
            all_ratings = conditioned_timeseries
        else:
            all_ratings = raw_timeseries

        if processed_output == 'histograms':
            current_histogram = [0] * self.params['number_of_rating_levels']
            histogram_timeseries = [[0] * settings.number_of_rating_levels]

            for rating in all_ratings:
                # print(rating)
                current_histogram[rating - 1] += 1
                append_histogram = copy.deepcopy(current_histogram)
                if self.params['input_histograms_are_normalized'] and (sum(append_histogram) > 0):
                    append_histogram = list(np.array(append_histogram) / (1.0 * sum(append_histogram)))
                histogram_timeseries.append(append_histogram)
        output_histograms_time_series = copy.deepcopy(histogram_timeseries)
        # print('output_histograms_time_series in process raw time series', output_histograms_time_series)
        return output_histograms_time_series

################################################################################
#########################    ABC Algorithms   ##################################
################################################################################

def basic_abc(model, data, epsilons=[1.0], min_samples=10,verbose=False):
    # ,
    #           parallel=False, n_procs='all', pmc_mode=False,
    #           weights='None', theta_prev='None', tau_squared='None'):
    """
    Perform Approximate Bayesian Computation (ABC) on a data set given a
    forward model.
    ABC is a likelihood-free method of Bayesian inference that uses simulation
    to approximate the true posterior distribution of a parameter. It is
    appropriate to use in situations where:
    The likelihood function is unknown or is too computationally
    expensive to compute.
    There exists a good forward model that can produce data sets
    like the one of interest.
    It is not a replacement for other methods when a likelihood
    function is available!
    Parameters
    ----------
    model : object
        A model that is a subclass of simpleabc.Model
    data  : object, array_like
        The "observed" data set for inference.
    epsilon : list of floats, optional
        The tolerance to accept parameter draws, default is 1.
    min_samples : int, optional
        Minimum number of posterior samples.
    parallel : bool, optional
        Run in parallel mode. Default is a single thread.
    n_procs : int, str, optional
        Number of subprocesses in parallel mode. Default is 'all' one for each
        available core.
    pmc_mode : bool, optional
        Population Monte Carlo mode on or off. Default is False. This is not
        meant to be called by the user, but is set by simple_abc.pmc_abc.
    weights : object, array_like, str, optional
        Importance sampling weights from previous PMC step. Used  by
        simple_abc.pmc_abc only.
    theta_prev : object, array_like, str, optional
        Posterior draws from previous PMC step.  Used by simple_abc.pmc_abc
        only.
    tau_squared : object, array_like, str, optional
        Previous Gaussian kernel variances. for importance sampling. Used by
        simple_abc.pmc_abc only.
    Returns
    -------
    posterior : numpy array
        Array of posterior samples.
    distances : object
        Array of accepted distances.
    accepted_count : float
        Number of  posterior samples.
    trial_count : float
        Number of total samples attempted.
    epsilon : float
        Distance tolerance used.
    weights : numpy array
        Importance sampling weights. Returns an array of 1s where
        size = posterior.size when not in pmc mode.
    tau_squared : numpy array
        Gaussian kernel variances. Returns an array of 0s where
        size = posterior.size when not in pmc mode.
    eff_sample : numpy array
        Effective sample size. Returns an array of 1s where
        size = posterior.size when not in pmc mode.
    Examples
    --------
    Forth coming.
    """
    posterior, rejected, accepted_distances = [], [], []
    trial_count, accepted_count = 0, 0
    print(data)

    data_summary_stats = model.summary_stats(data)
    # print(data_summary_stats)
    model.set_epsilons(epsilons)

    while accepted_count < min_samples:
        trial_count += 1

        # if pmc_mode:
        #     theta_star = theta_prev[:, np.random.choice(
        #                             xrange(0, theta_prev.shape[1]),
        #                             replace=True, p=weights/weights.sum())]
        #
        #     theta = stats.multivariate_normal.rvs(theta_star, tau_squared)
        #     if np.isscalar(theta) == True:
        #         theta = [theta]
        #
        #   
        # else:
        #     theta = model.draw_theta()

        theta = model.draw_theta()

        synthetic_data = model.generate_data(theta)

        synthetic_summary_stats = model.summary_stats(synthetic_data)

        distances = model.distance_function(data_summary_stats,
                                           synthetic_summary_stats)

        accept = True
        for i in range(settings.number_of_summaries):
            if distances[i] > model.epsilons[i]:
                accept = False

        if accept:
            accepted_count += 1
            posterior.append(theta)
            accepted_distances.append(distances)
            print('ACCEPTED!!',  accepted_count, 'out of', trial_count, theta)

        else:
            if verbose:
                print('REJECTED!!', trial_count - accepted_count, theta)
                # pass
            #rejected.append(theta)

    posterior = np.asarray(posterior).T

    if len(posterior.shape) > 1:
        n = posterior.shape[1]
    else:
        n = posterior.shape[0]


    # weights = np.ones(n)
    # tau_squared = np.zeros((posterior.shape[0], posterior.shape[0]))
    # eff_sample = n

    return (posterior, distances,
            accepted_count, trial_count,
            epsilons)#, weights, tau_squared, eff_sample)

# error type can be 'MSE' or 'MAE'
# estimator types can be 'posterior_mean', 'posterior_median', 'MAP'
class Estimator():
    def __init__(self, model, epsilons, n_posterior_samples=10, n_samples=10,
                 estimator_type='posterior_mean', bin_size=10, error_type='MSE'):
        self.model = model
        self.n_samples = n_samples # number of samples generated for each true theta to test the estimator for
        # the given theta
        self.n_posterior_samples = n_posterior_samples # number of samples from the posterior toi construct an estimator
        self.estimator_type=estimator_type # estimator_type can be 'posterior_mean', 'MAP', 'median'
        self.bin_size = bin_size
        self.error_type = error_type  # error type can be MSE or MAE
        self.epsilons = epsilons

    def get_estimates(self, true_theta, do_hist=False):
        if self.estimator_type != 'All':
            theta_estimates = np.zeros(self.n_samples)
        elif self.estimator_type == 'All':
            theta_estimates = np.zeros([3,self.n_samples])

        for i in range(self.n_samples):
            data = self.model.generate_data(true_theta)
            (posterior_samples, _, _, _, _) = basic_abc(self.model, data, self.epsilons, self.n_posterior_samples)
            posterior_samples = np.asarray(posterior_samples)
            # print('sampled_thetas:',sampled_thetas)
            if do_hist:
                plt.hist(posterior_samples)
                plt.title('Posterior Samples for theta = ' + str(true_theta))
                plt.show()
            # print(self.estimator_type)
            if self.estimator_type == 'posterior_mean':
                theta_estimates[i] = posterior_samples.mean()
            elif self.estimator_type == 'MAP':
                hist, bin_edges = np.histogram(posterior_samples, bins=self.bin_size)
                j = np.argmax(hist)
                theta_estimates[i] = (bin_edges[j] + bin_edges[j+1]) / 2.0
                # print(bin_edges)
            elif self.estimator_type == 'posterior_median':
                theta_estimates[i] = np.median(posterior_samples)
            elif self.estimator_type == 'All': # record all estimators in the order mean, median, MAP
                theta_estimates[0,i] = posterior_samples.mean()
                theta_estimates[1,i] = np.median(posterior_samples)
                hist, bin_edges = np.histogram(posterior_samples, bins=self.bin_size)
                j = np.argmax(hist)
                # print(bin_edges)
                theta_estimates[2,i] = (bin_edges[j] + bin_edges[j + 1]) / 2.0

        if self.estimator_type != 'All':
            error = 0
            if self.error_type == 'MSE':
                error = np.sum((selected_theta - np.mean(theta_estimates))**2 for selected_theta in theta_estimates)
            elif self.error_type == 'MAE':
                error = np.sum(abs(selected_theta - np.mean(theta_estimates)) for selected_theta in theta_estimates)
            error /= self.n_samples
        elif self.estimator_type == 'All': # record all errors in the order mean, median, MAP
            error = np.zeros(3)
            for j in range(3):
                if self.error_type == 'MSE':
                    error[j] = np.sum((selected_theta - np.mean(theta_estimates[j])) ** 2
                                      for selected_theta in theta_estimates[j])
                elif self.error_type == 'MAE':
                    error[j] = np.sum(abs(selected_theta - np.mean(theta_estimates[j]))
                                   for selected_theta in theta_estimates[j])
                error[j] /= self.n_samples
        return error, theta_estimates

    def get_estimates_for_true_thetas(self, true_thetas=[2,4,6], do_plot=True, do_hist=False,
                                      symmetric=False,verbose=True,
                                      compute_estimates=True,
                                      save_estimates=True,
                                      load_estimates=False):

        assert (not save_estimates) or compute_estimates, "cannot save estimates without computing them!"

        if compute_estimates:
            if self.estimator_type != 'All':
                estimated_thetas = []  # a list of theta_estimated for each true_theta
                mean_estimated_thetas = []
                errors = []
                for true_theta in true_thetas:
                    if verbose:
                        print('true theta:', true_theta)
                    theta_error, theta_estimates = self.get_estimates(true_theta, do_hist=do_hist)
                    estimated_thetas += [theta_estimates]
                    mean_estimated_thetas += [np.mean(theta_estimates)]
                    errors += [theta_error]
                    if verbose:
                        print('theta estimates:', theta_estimates)
                if save_estimates:
                    pickle.dump(mean_estimated_thetas, open('./data/mean_estimated_thetas.pkl', 'wb'))
                    pickle.dump(errors, open('./data/errors.pkl', 'wb'))

            elif self.estimator_type == 'All':
                mean_estimated_thetas = np.zeros([3, len(true_thetas)])  # a list of theta_estimated for each true_theta
                if symmetric:
                    errors = np.zeros([3, len(true_thetas)])
                    for ii in range(len(true_thetas)):
                        if verbose:
                            print('true theta:', true_thetas[ii])
                        theta_error, theta_estimates = self.get_estimates(true_thetas[ii], do_hist=do_hist)
                        for jj in range(3):
                            mean_estimated_thetas[jj, ii] = np.mean(theta_estimates[jj])
                            errors[jj, ii] = theta_error[jj]
                        if verbose:
                            print('theta estimates:', theta_estimates)
                    if save_estimates:
                        pickle.dump(mean_estimated_thetas, open('./data/mean_estimated_thetas.pkl', 'wb'))
                        pickle.dump(errors, open('./data/errors.pkl', 'wb'))
                elif not symmetric:
                    lower_errors = np.zeros([3, len(true_thetas)])
                    upper_errors = np.zeros([3, len(true_thetas)])
                    asymmetric_errors = np.zeros([3, 2, len(true_thetas)])
                    for ii in range(len(true_thetas)):
                        if verbose:
                            print('true theta:', true_thetas[ii])
                        theta_error, theta_estimates = self.get_estimates(true_thetas[ii], do_hist=do_hist)
                        for jj in range(3):
                            mean_estimated_thetas[jj, ii] = np.mean(theta_estimates[jj])
                            lower_errors[jj, ii] = abs(np.min(theta_estimates[jj])
                                                       - mean_estimated_thetas[jj, ii])
                            upper_errors[jj, ii] = abs(np.max(theta_estimates[jj])
                                                       - mean_estimated_thetas[jj, ii])
                            # print(asymmetric_errors[jj, 0, ii])
                            # print(asymmetric_errors[jj, 1, ii])
                            # print(lower_errors[jj,ii], upper_errors[jj,ii])
                            asymmetric_errors[jj, 0, ii] = lower_errors[jj, ii]
                            asymmetric_errors[jj, 1, ii] = upper_errors[jj, ii]
                            # print(asymmetric_errors)
                        if verbose:
                            print('theta estimates:', theta_estimates)
                    if save_estimates:
                        pickle.dump(mean_estimated_thetas, open('./data/mean_estimated_thetas.pkl', 'wb'))
                        pickle.dump(asymmetric_errors, open('./data/asymmetric_errors.pkl', 'wb'))
        if do_plot:
            if self.estimator_type != 'All':
                if symmetric:
                    if load_estimates:
                        mean_estimated_thetas = pickle.load(open('./data/mean_estimated_thetas.pkl', 'rb'))
                        errors = pickle.load(open('./data/errors.pkl', 'rb'))
                    plt.figure()
                    plt.errorbar(true_thetas, mean_estimated_thetas, yerr=errors)
                    plt.title(self.estimator_type + ' performance ')
                    plt.xlabel('true theta')
                    plt.ylabel(self.estimator_type)
                    plt.show()
                elif not symmetric:  # not symmetric
                    if load_estimates:
                        mean_estimated_thetas = pickle.load(open('./data/mean_estimated_thetas.pkl', 'rb'))
                        asymmetric_errors = pickle.load(open('./data/asymmetric_errors.pkl', 'rb'))
                    elif not load_estimates:
                        lower_errors = []
                        upper_errors = []
                        for i in range(len(true_thetas)):
                            theta_estimates = estimated_thetas[i]
                            lower_errors += [abs(np.min(theta_estimates) - mean_estimated_thetas[i])]
                            upper_errors += [abs(np.max(theta_estimates) - mean_estimated_thetas[i])]
                        asymmetric_errors = [lower_errors, upper_errors]
                        plt.figure()
                        plt.plot(true_thetas, true_thetas, color='g', linestyle=':', label='true $\\theta$')
                        plt.errorbar(true_thetas, mean_estimated_thetas, yerr=asymmetric_errors, color='r', label='MAP')
                        plt.title(self.estimator_type + ' estimation performance')
                        plt.xlabel('true value $(\\theta)$')
                        plt.ylabel('estimated $(\\hat{\\theta})$')
                        plt.show()
            elif self.estimator_type == 'All':
                if symmetric:
                    if load_estimates:
                        mean_estimated_thetas = pickle.load(open('./data/mean_estimated_thetas.pkl', 'rb'))
                        errors = pickle.load(open('./data/errors.pkl', 'rb'))
                    # print(mean_estimated_thetas[0])
                    plt.figure()
                    plt.errorbar(true_thetas, mean_estimated_thetas[0], yerr=errors[0],
                                 label='posterior mean', color='r')
                    plt.errorbar(true_thetas, mean_estimated_thetas[1], yerr=errors[1],
                                 label='posterior median', color='g')
                    plt.errorbar(true_thetas, mean_estimated_thetas[2], yerr=errors[2],
                                 label='MAP', color='b')
                    plt.plot(true_thetas, true_thetas, linestyle=':', label='true $\\theta$', color='c')
                    plt.title('estimation performance ')
                    plt.xlabel('true value $(\\theta)$')
                    plt.ylabel('estimated $(\\hat{\\theta})$')
                    plt.legend()
                    plt.show()
                elif not symmetric:  # not symmetric
                    if load_estimates:
                        mean_estimated_thetas = pickle.load(open('./data/mean_estimated_thetas.pkl', 'rb'))
                        asymmetric_errors = pickle.load(open('./data/asymmetric_errors.pkl', 'rb'))
                    plt.figure()
                    plt.errorbar(true_thetas, mean_estimated_thetas[0], yerr=asymmetric_errors[0],
                                 label='posterior mean', color='r')
                    plt.errorbar(true_thetas, mean_estimated_thetas[1], yerr=asymmetric_errors[1],
                                 label='posterior median', color='g')
                    plt.errorbar(true_thetas, mean_estimated_thetas[2], yerr=asymmetric_errors[2],
                                 label='MAP', color='b')
                    plt.plot(true_thetas, true_thetas, color='c', linestyle=':', label='true $\\theta$')
                    plt.title('estimation performance ')
                    plt.xlabel('true value $(\\theta)$')
                    plt.ylabel('estimated $(\\hat{\\theta})$')
                    plt.legend()
                    plt.show()

    # gen_model_params=settings.params,
    # conditioninal_posterior =False,
    # direction_of_conditioning=None,
    def get_estimates_for_observed_data(self,
                                        observed_data,
                                        do_hist=True,
                                        verbose=True,
                                        compute_posterior_samples=True,
                                        save_posterior_samples=True,
                                        load_posterior_samples=False):

        assert (not save_posterior_samples) or compute_posterior_samples, \
            "cannot save posterior samples without computing them!"

        if compute_posterior_samples:
            # gen_model = ABC_GenerativeModel(params=gen_model_params,
            #                                 conditioning=conditioninal_posterior,
            #                                 direction=direction_of_conditioning)
            # conditioning = False, direction = None
            # conditioning=True, direction='above'


            # observed_timeseries = pd.read_csv('./data/' + tracked_product_ID +
            #                                   '_time_series.txt', sep='\t')
            #
            # data = list(observed_timeseries['Rating'])
            # # all_ratings = list(raw_timeseries['Rating'])
            # # processed_timeseries = gen_model.process_raw_timeseries()
            #
            # print(len(data))

            (posterior_samples, distances,
             accepted_count, trial_count,
             epsilon) = basic_abc(self.model, observed_data, self.epsilons, self.n_posterior_samples)
            if verbose:
                print('posterior samples', posterior_samples)
                print('distances', distances)
                print('accepted_count', accepted_count)
                print('trial_count', trial_count)
                print('epsilon', epsilon)
                print('posterior mean:', np.mean(posterior_samples))
                print('posterior median:', np.median(posterior_samples))
                hist, bin_edges = np.histogram(posterior_samples, bins=self.bin_size)
                j = np.argmax(hist)
                MAP = (bin_edges[j] + bin_edges[j + 1]) / 2.0
                print('MAP:', MAP)
        #
            if save_posterior_samples:
                pickle.dump(posterior_samples, open('./data/'+settings.tracked_product_ID
                                                    +'_posterior_samples.pkl', 'wb'))
        if do_hist:
            if load_posterior_samples:
                posterior_samples = pickle.load(open('./data/'+settings.tracked_product_ID
                                                     +'_posterior_samples.pkl', 'rb'))
            plt.figure()
            plt.hist(posterior_samples)
            plt.title('Posterior Samples for ' + settings.tracked_product_ID)
            plt.show()