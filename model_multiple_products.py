# generative model for reviews
# this version is used to generate the histogram of reviews for a particular product over time
# based on the evolution of histograms over time we can infer if people have a non-zero threshold
# for deciding whether to leave a review on the platform or not.

import torch
import pymc as mc
import numpy as np
import random as RD
import scipy.stats as st
import pandas as pd
import copy
import pickle



RD.seed()


class product():
    def __init__(self):
        self.set_missing_product_params()

    def set_missing_product_params(self):
        if 'number_of_products' not in self.fixed_params:
            self.params['number_of_products'] = 4
        if 'product_indices' not in  self.fixed_params:
            self.params['product_indices'] = list(range(self.params['number_of_products']))
        if 'number_of_rating_levels' not in self.fixed_params:
            self.params['number_of_rating_levels'] = 5
        if 'prices' not in self.fixed_params:
            self.params['prices'] = [10,11,12,13]
        if 'product_features' not in self.fixed_params:
            self.params['product_features'] = dict.fromkeys(['feature'])
            self.params['product_features']['feature'] = [1,2,3,4]
        if 'neutral_qualities' not in self.fixed_params:
            self.params['neutral_population_qualities'] = [3]*self.params['number_of_products']
            #print(self.params['neutral_population_qualities'])
        if 'qualities_std' not in self.fixed_params:
            self.params['qualities_std'] = 1.5
        if 'true_qualities' not in self.fixed_params:
            self.params['true_qualities'] = list(np.random.normal(self.params['neutral_population_qualities'],
                                                                  self.params['qualities_std'],
                                                                  self.params['number_of_products']))
        if 'product_tracked' not in self.fixed_params:
            self.params['product_tracked'] = 0 # the product whose histograms we ae analyzing

        if 'input_type' not in self.fixed_params:
            self.params['input_type'] = 'multiple'  # train the network with the average of reviews rather than
            # the full histogram of reviews

        if 'input_histograms_are_normalized' not in self.fixed_params:
            self.params['input_histograms_are_normalized'] = False  # histograms are normalized to the frequencies rather
            # than showing the total counts

class consumer(product):
    def __init__(self):
        super(consumer, self).__init__()
        self.set_missing_consumer_params()

    def set_missing_consumer_params(self):
        if 'tendency_to_rate' not in self.fixed_params:
            self.params['tendency_to_rate'] = 0.2
        if 'number_of_rating_levels' not in self.fixed_params:
            self.params['number_of_rating_levels'] = 5
        if 'consumer_fit_std' not in self.fixed_params:
            self.params['consumer_fit_std'] = [4.5]*self.params['number_of_products']
        if 'consumer_fit_distributions' not in self.fixed_params:
            self.params['consumer_fit_distributions'] = dict.fromkeys(self.params['product_indices'])
            for product_index in self.params['consumer_fit_distributions'].keys():
                self.params['consumer_fit_distributions'][product_index] \
                    = st.norm(0, self.params['consumer_fit_std'][product_index])

    def init_consumer_private_parameters(self):
        self.consumer_private_fit = dict.fromkeys(self.params['product_indices'])
        for product_index in self.consumer_private_fit.keys():
            self.consumer_private_fit[product_index] = self.params['consumer_fit_distributions'][product_index].rvs()

        self.consumer_private_alpha = np.random.normal(self.params['population_alpha'][0], self.params['population_alpha'][1])
        self.consumer_private_beta = dict.fromkeys(self.params['population_beta'].keys())
        for i in self.params['population_beta'].keys():
            self.consumer_private_beta[i] = np.random.normal(self.params['population_beta'][i][0],
                                                             self.params['population_beta'][i][1])

    def make_purchase(self):

        features_utility = 0
        for i in self.params['product_features'].keys():
            features_utility += self.consumer_private_beta[i+'_beta']* np.array(self.params['product_features'][i])

        price_utility = self.consumer_private_alpha * np.array(self.params['prices'])

        expected_utilities = list(features_utility +#self.consumer_private_beta * np.array(self.params['product_features'])
                                  + price_utility #self.consumer_private_alpha * np.array(self.params['prices']) +
                                  + np.array(self.percieved_qualities) + np.array(list(self.consumer_private_fit.values())))
        product_index = np.argmax(expected_utilities)

        return product_index

    def evaluate_product(self,product_index):

        review_levels = [self.percieved_qualities[product_index] - 1.5, self.percieved_qualities[product_index] - 0.5,
                         self.percieved_qualities[product_index] + 0.5,
                         self.percieved_qualities[product_index] + 1.5]
        experienced_quality = self.params['true_qualities'][product_index] + self.consumer_private_fit[product_index]

        product_review = int(1 + sum(1.0*(experienced_quality >= np.array(review_levels))))

        return product_review

    def decide_to_rate(self,product_index,product_review):

        if np.random.binomial(1, self.params['tendency_to_rate']):
            decision = True
        elif self.avg_reviews[product_index]:  # it is not the first review
            decision = (abs(product_review - self.avg_reviews[product_index][-1]) > self.params['rate_decision_threshold']) \
                       and (np.random.binomial(1, min(3*self.params['tendency_to_rate'],1)))
        else:
            decision = False

        return decision



class market(consumer):
    def __init__(self, params ={}):
        self.fixed_params = copy.deepcopy(params)
        self.params = copy.deepcopy(params)
        self.set_missing_market_params()
        super(market, self).__init__()

    def set_missing_market_params(self):
        if 'population_beta' not in self.fixed_params:
            self.params['population_beta'] = dict.fromkeys(['feature_beta'])
            self.params['population_beta']['feature_beta'] = [np.random.uniform(1, 2),1]
        if 'population_alpha' not in self.fixed_params:
            self.params['population_alpha'] = [np.random.uniform(-3, -2), 1]
        if 'total_number_of_reviews' not in self.fixed_params:
            self.params['total_number_of_reviews'] = 20

    def set_random_params(self):
        """Randomly sets the parameters that are the subject of inference by the inference engine. The parameters are
        randomized according to the prior distributions"""
        if 'rate_decision_threshold' not in self.fixed_params:
            #self.params['rate_decision_threshold'] = RD.choice([-1.0,1.0])
            self.params['rate_decision_threshold'] = 1.0

    def init_reputation_dynamics(self):

        self.percieved_qualities = self.params['neutral_population_qualities']
        self.reviews = {key: [] for key in self.params['product_indices']}
        self.avg_reviews = {key: [3] for key in self.params['product_indices']}
        self.histogram_reviews = {key: [0] * self.params['number_of_rating_levels'] for key in self.params['product_indices']}

        self.customer_count = 0
        self.purchased_products = []
        self.purchase_count = [0] * self.params['number_of_products']

    def form_perception_of_quality(self): #### CHECK ###
        if self.avg_reviews:
            quality_anchors = list(map(lambda product: self.avg_reviews[product][-1], self.avg_reviews.keys()))

        else:
            quality_anchors = self.params['neutral_population_qualities']

        observed_histograms = list(map(lambda product: self.histogram_reviews[product], self.histogram_reviews.keys()))


        # infer_quality = mc.Normal('infer_quality', mu=self.params['neutral_quality'],
        #                          tau=self.params['quality_std'])  # this is the prior on the quality

        for product in self.params['product_indices']:
            infer_quality = np.linspace(-1.0, 8.0, 1000)  # assume quality is between -1 and 8 (1000 bins in between)
            # evaluate prior by plugging in above values to PDF (take log for ease of underflow errors: small numbers problem)
            log_prior = np.log(st.norm.pdf(infer_quality,
                                           loc=self.params['neutral_population_qualities'][product],
                                           scale=self.params['qualities_std']))

            data = observed_histograms[product]
            # quality anchor is fixed. 4 values of interest: quality anchor - (0.5 + infer_qual); quality anchor - (1.5 + infer_qual); quality anchor + (0.5 - infer_qual); quality anchor + (1.5 - infer_qual)
            # form two vectors, each with 2 rows, based on the above definition
            # then, evaluate the cdf part by part by utilising these values - reduces recomputing same estimates and speeds up calculation
            consumer_fit_test_vals1 = np.tile(np.array([0.5, 1.5]).reshape(2, 1), (1, infer_quality.shape[0])) + \
                                  np.tile(infer_quality.reshape(1, infer_quality.shape[0]), (2, 1))
            consumer_fit_test_vals2 = np.tile(np.array([0.5, 1.5]).reshape(2, 1), (1, infer_quality.shape[0])) - \
                                  np.tile(infer_quality.reshape(1, infer_quality.shape[0]), (2, 1))
            consumer_fit_cdf1 = self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] - consumer_fit_test_vals1)
            consumer_fit_cdf2 = self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] + consumer_fit_test_vals2)
            # if we take the vector log_prior, we can perform the operation of +- etc to the WHOLE vector instead of taking point by point evaluations and summing

            log_likelihood = np.log(consumer_fit_cdf1[1, :]) * data[0] + \
                         np.log(consumer_fit_cdf1[0, :] - consumer_fit_cdf1[1, :]) * data[1] + \
                         np.log(consumer_fit_cdf2[0, :] - consumer_fit_cdf1[0, :]) * data[2] + \
                         np.log(consumer_fit_cdf2[1, :] - consumer_fit_cdf2[0, :]) * data[3] + \
                         np.log(1 - consumer_fit_cdf2[1, :]) * data[4]
            posterior = np.exp(log_prior + log_likelihood)
            self.percieved_qualities[product] = np.sum(posterior * infer_quality / np.sum(posterior))

    # def form_perception_of_quality(self):
    #
    #
    #     # print('we are at the begining of perception', self.params['neutral_population_qualities'])
    #
    #     quality_anchors = list(map(lambda product: self.avg_reviews[product][-1], self.avg_reviews.keys()))
    #
    #     observed_histograms = list(map(lambda product: self.histogram_reviews[product], self.histogram_reviews.keys()))
    #
    #     for product in self.params['product_indices']:
    #         infer_quality = mc.Normal('infer_quality', mu=self.params['neutral_population_qualities'][product],
    #                                   tau=self.params['qualities_std'])  # this the prior on the quality
    #
    #         data = observed_histograms[product]
    #
    #         @mc.stochastic(observed=True)
    #         def histogram_mental_model(value=data, infer_quality=infer_quality):
    #             #np.log((1 - self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] + 1.5 - infer_quality)) ** value[4])
    #             #(self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] + 1.5 - infer_quality) -
    #             #  self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] + 0.5 - infer_quality)) ** value[3]
    #
    #             return np.sum(
    #                 np.log((self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] - 1.5 - infer_quality)) ** value[0]) +
    #                 np.log(
    #                     (self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] - 0.5 - infer_quality) -
    #                      self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] - 1.5 - infer_quality)) ** value[1]) +
    #                 np.log(
    #                     (self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] + 0.5 - infer_quality) -
    #                      self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] - 0.5 - infer_quality)) ** value[2]) +
    #                 np.log(
    #                     (self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] + 1.5 - infer_quality) -
    #                      self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] + 0.5 - infer_quality)) ** value[3]) +
    #                 np.log((1 - self.params['consumer_fit_distributions'][product].cdf(quality_anchors[product] + 1.5 - infer_quality)) ** value[4]))
    #
    #         model = mc.MCMC([infer_quality, histogram_mental_model])
    #         model.sample(iter=100, progress_bar=False)
    #         self.percieved_qualities[product] = np.mean(model.trace('infer_quality')[:])

        # print('we are at the end of perception', self.params['neutral_population_qualities'])
        # print('we are at the end of perception', self.percieved_qualities)

    def step(self):

        self.init_consumer_private_parameters()

        # fix the common prior on quality at the beginning of perception:
        if 'neutral_qualities' not in self.fixed_params:
            self.params['neutral_population_qualities'] = [3.0] * self.params['number_of_products']

        self.form_perception_of_quality()
        product_index = self.make_purchase()
        self.purchase_count[product_index] += 1
        self.purchased_products.append(product_index)
        product_review = self.evaluate_product(product_index)

        if self.decide_to_rate(product_index,product_review):
            self.reviews[product_index].append(product_review)
            self.avg_reviews[product_index].append(np.mean(self.reviews[product_index]))
            self.histogram_reviews[product_index][product_review - 1] += 1
            a_product_is_reviewed = True
        else:
            a_product_is_reviewed = False
        return a_product_is_reviewed

    def generateTimeseries(self):  # conditioned on the fixed_params
        self.set_random_params() # The random parameter that is the subject of inference is set here.
                                 # This parameter determines the true label for the generated time series (example).
                                 # The distribution according to which the parameter is randommized is our prior on it
        self.init_reputation_dynamics()
        timeseries = []

        while len(timeseries) < self.params['total_number_of_reviews']:
            a_product_is_reviewed = self.step()

            if a_product_is_reviewed:
                if self.params['input_type'] == 'averages':
                    timeseries.append(self.avg_reviews[self.params['product_tracked']][-1])
                elif self.params['input_type'] == 'histograms':
                    histogram = copy.deepcopy(self.histogram_reviews[self.params['product_tracked']])
                    if self.params['input_histograms_are_normalized'] and (sum(histogram) > 0):
                        histogram = list(np.array(histogram) / (1.0 * sum(histogram)))
                    timeseries.append(histogram)
                elif self.params['input_type'] == 'kurtosis':
                    histogram = copy.deepcopy(self.histogram_reviews[self.params['product_tracked']])
                    if (sum(histogram) > 0):
                        histogram = list(np.array(histogram) / (1.0 * sum(histogram)))
                    kurtosis = st.kurtosis(histogram, fisher=False, bias=True)
                    timeseries.append(kurtosis)
                elif self.params['input_type'] == 'multiple':
                    timeseries.append((self.purchased_products[-1] , self.reviews[self.purchased_products[-1]][-1]))

        df = pd.DataFrame(timeseries)
        return df

    def genTorchSample(self):
        df = self.generateTimeseries()
        data = torch.FloatTensor(df.values[:, 0:5].astype(float))
        label_of_data = torch.LongTensor([int(self.params['rate_decision_threshold']>0)])
        return label_of_data, data

    def genTorchDataset(self, dataset_size=1000,filename = 'dataset.pkl', LOAD=False, SAVE = False ):
        if LOAD:
            simulation_results = pickle.load(open('./data/'+filename,'rb'))
        else:
            simulation_results = []
            for iter in range(dataset_size):
                label_of_data, data = self.genTorchSample()
                simulation_results.append((label_of_data, data))
            if SAVE:
                pickle.dump(simulation_results, open('./data/'+filename, 'wb'))
        return simulation_results