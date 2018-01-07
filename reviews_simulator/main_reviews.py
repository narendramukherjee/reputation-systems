# Simple Dynamics simulator in Python
#
# *** Online Reputation ***
# Based on: Hiroki Sayama's Python file for Network Epidemics
# sayama@binghamton.edu

import matplotlib

matplotlib.use('TkAgg')

import pylab as PL
import pymc as mc
import matplotlib.pyplot as plt
import numpy as np
import random as RD
import scipy.stats as st
import pycxsimulator
RD.seed()




tendency_to_review = 0.3
J  = 4 # J different products
PopulationSize = 100 # market size (100 purchases)
L = 5 # L levels of reviews, {1,2,3,4,5,...,L}
view_product = 0 #np.random.choice(list(range(J)))
print(view_product)
population_beta = np.random.uniform(1,2) # population beta parameter to be inferred from the observed data
population_alpha =  np.random.uniform(-3,-2) # population alpha parameter to be inferred from the observed data
price = [10,11,12,13] # Four prices for four products j in {0,1,2,3}
product_features = [1,2,3,4] # four feature values for four products
neutral_population_quality = [3]*J # neutral product quality (noninformative prior mean quality)
population_quality_std = 1.5
true_population_quality = list(np.random.normal(neutral_population_quality,1.5,J)) # true product qualities (unknown)




def init():
    global reviews, avg_reviews, histogram_reviews, consumer_fit_dist, product_indices, expected_qualities,customer_count,purchase_count
    product_indices = list(range(J))
    expected_qualities = [3]*J
    purchase_count = [0]*J
    reviews = {key: [] for key in product_indices}
    avg_reviews = {key: [3] for key in product_indices}
    histogram_reviews= {key: [0]*L for key in product_indices}
    consumer_fit_dist = st.norm(0, 4.5)
    customer_count = 0



def step():
    global reviews, avg_review, histogram_reviews, consumer_beta, consumer_alpha, consumer_fit, \
        experienced_utility, product_index , product_review, consumer_fit_dist, customer_count,purchase_count
    customer_count += 1
    consumer_fit = list(consumer_fit_dist.rvs(J))
    consumer_alpha = np.random.normal(population_alpha,1)
    consumer_beta = np.random.normal(population_beta,1)

    form_quality_expectations()
    make_purchase()
    purchase_count[product_index] += 1
    experience_utility()
    evaluate_product()

    if decide_to_put_review():
        reviews[product_index].append(product_review)
        avg_reviews[product_index].append(np.mean(reviews[product_index]))
        histogram_reviews[product_index][product_review-1] += 1


def form_quality_expectations():
    global reviews, avg_review, histogram_reviews, consumer_beta, consumer_alpha, consumer_fit_dist, expected_qualities, product_indices

    quality_anchors = list(map(lambda product: avg_reviews[product][-1], avg_reviews.keys()))

    observed_histograms = list(map(lambda product: histogram_reviews[product], histogram_reviews.keys()))


    for product in product_indices:
        infer_quality = mc.Normal('infer_quality', mu=neutral_population_quality,
                                  tau=population_quality_std)  # this the prior on the quality
        data = observed_histograms[product]
        @mc.stochastic(observed=True)
        def histogram_mental_model(value=data, infer_quality = infer_quality):
            return np.sum(np.log((consumer_fit_dist.cdf(quality_anchors[product] - 1.5 - infer_quality)) ** value[0]) +
                          np.log((consumer_fit_dist.cdf(quality_anchors[product] - 0.5 - infer_quality) - consumer_fit_dist.cdf(
                              quality_anchors[product] - 1.5 - infer_quality)) ** value[1]) +
                          np.log((consumer_fit_dist.cdf(quality_anchors[product] + 0.5 - infer_quality) - consumer_fit_dist.cdf(
                              quality_anchors[product] - 0.5 - infer_quality)) ** value[2]) +
                          np.log((consumer_fit_dist.cdf(quality_anchors[product] + 1.5 - infer_quality) - consumer_fit_dist.cdf(
                              quality_anchors[product] + 0.5 - infer_quality)) ** value[3]) +
                          np.log((1 - consumer_fit_dist.cdf(quality_anchors[product] + 1.5 - infer_quality)) ** value[4]))

        model = mc.MCMC([infer_quality, histogram_mental_model])
        model.sample(iter=100,progress_bar=False)
        expected_qualities[product] = np.mean(model.trace('infer_quality')[:])


def make_purchase():
    global reviews, avg_review, histogram_reviews, consumer_beta, consumer_alpha, consumer_fit, \
        product_index, expected_qualities
    expected_utilities = list(consumer_beta * np.array(product_features) + consumer_alpha * np.array(price) +
                              np.array(expected_qualities) + np.array(consumer_fit))
    product_index = np.argmax(expected_utilities)

def experience_utility():
    global reviews, avg_review, histogram_reviews, consumer_beta, consumer_alpha, consumer_fit, \
        product_index, experienced_utility
    experienced_utility = consumer_beta*product_features[product_index] + consumer_alpha*price[product_index] + \
                          true_population_quality[product_index] + consumer_fit[product_index]

def evaluate_product():
    global experienced_utility, product_index, product_review, expected_qualities, consumer_beta, consumer_alpha

    review_levels = [expected_qualities[product_index] - 1.5, expected_qualities[product_index]-0.5,
                                                  expected_qualities[product_index]+0.5,
                                                  expected_qualities[product_index]+1.5]


    product_review = int(1 + sum(1.0 * ((true_population_quality[product_index] + consumer_fit[product_index])
                                        >= np.array(review_levels))))

def decide_to_put_review():
    global avg_reviews, product_review, product_index

    if np.random.binomial(1, tendency_to_review):
        decision = True
    elif avg_reviews[product_index]:  # it is not the first review
        decision = (abs(product_review - avg_reviews[product_index][-1]) > 0.5) and (np.random.binomial(1, min(3*tendency_to_review,1)))
    else:
        decision = False

    return  decision

def draw():
    global reviews, avg_reviews, histogram_reviews, customer_count,purchase_count
    reviewer_number = len(reviews[view_product])
    PL.cla()
    if reviewer_number == 0:
        normalized_histogram = histogram_reviews[view_product]
    else:
        normalized_histogram = list(np.array(histogram_reviews[view_product])/(0.01*reviewer_number))

    bar_list = plt.bar(list(range(1,L+1)),normalized_histogram)
    bar_list[0].set_color('y')
    bar_list[1].set_color('y')
    bar_list[2].set_color('y')
    bar_list[3].set_color('y')
    bar_list[4].set_color('y')
    plt.ylim(ymin=0)
    plt.xticks(list(range(1, L + 1)))
    plt.yticks(list(np.linspace(0, 100, num=5)))
    plt.ylabel('frequency (%)')
    plt.xlabel('reviews')
    PL.axis([0,6,0,120])
    PL.title(str(reviewer_number) + ' reviews from ' + str(purchase_count[view_product]) + ' purchases by '
             + str(customer_count) + ' customers')
    x = list(np.array(range(len(avg_reviews[view_product])))/20)
    PL.plot(x,list(20*np.array(avg_reviews[view_product])), 'c',
            label= 'Average of All Review Ratings (Scaled to Percentage of Five)' )
    PL.legend()
    #PL.savefig(str(customer_count)+'.png')


pycxsimulator.GUI().start(func=[init, draw, step])