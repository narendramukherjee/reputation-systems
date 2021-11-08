from pathlib import Path
from typing import Any, Deque, Dict, List

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.spatial.distance import cdist
from snpe.utils.statistics import review_histogram_means

from .simulator_class import *
from .simulator_class import HerdingSimulator


class MarketplaceSimulator(HerdingSimulator):
    def __init__(self, params: dict):
        self.num_products = params["num_products"]
        self.num_total_marketplace_reviews = params["num_total_marketplace_reviews"]
        self.consideration_set_size = params["consideration_set_size"]
        super(BaseMarketplaceSimulator, self).__init__(params)

    def load_embedding_density_estimators(self, dirname: Path) -> None:
        embedding_density_estimators = {"products": default_rng, "visitors": default_rng}
        product_embedding = embedding_density_estimators["products"].multivariate_normal(
            np.zeros(100), np.diag(np.ones(100))
        )
        visitor_embedding = embedding_density_estimators["visitors"].multivariate_normal(
            np.zeros(100), np.diag(np.ones(100))
        )
        assert (
            product_embedding.shape == visitor_embedding.shape
        ), f"""
            Product embedding generator produces embeddings of shape {product_embedding.shape} while visitor
            embedding generator produces embeddings of shape {visitor_embedding.shape}. They need to have the
            same shape to be able to calculate cosine similarities between them.
            """
        self.embedding_density_estimators = embedding_density_estimators

    def simulate_visitor_journey(self, simulated_reviews: np.ndarray, simulation_id: int) -> Union[int, None]:
        # Convolve the current simulated review distribution with the prior to get the posterior of reviews
        review_posterior = self.convolve_prior_with_existing_reviews(simulated_reviews)

        # Just make a single draw from the posterior Dirichlet dist of reviews to get the distribution
        # of the product experiences that the user expects
        expected_experience_dist = np.random.dirichlet(review_posterior)
        # Also get the mean "experience" that the user expects
        expected_experience_dist_mean = np.sum(expected_experience_dist * np.arange(1, 6))
        # Draw an experience from the user's expected distribution of experiences
        experience = np.where(np.random.multinomial(1, expected_experience_dist))[0][0] + 1.0

        # User's mismatch is the difference between their actual experience and the mean of the distribution
        # of experiences they expected
        delta = self.mismatch_calculator(experience, expected_experience_dist_mean)

        # Calculate the index of the rating the user wants to leave [0, 4]
        rating_index = self.rating_calculator(delta)
        # Get the decision to leave review (True/False)
        decision_to_rate = self.decision_to_leave_review(delta, simulation_id)

        # Add a review to the corresponding rating index if the user decided to rate
        if decision_to_rate:
            return rating_index
        else:
            return None

    def simulate_review_histogram(
        self, simulation_id: int, num_reviews_per_simulation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        raise NotImplementedError(
            f"""
            For {self.__class__.__name__}, cannot simulate product review histograms separately.
            Run simulate_marketplace instead
            """
        )

    def simulate(self, num_simulations: int, num_reviews_per_simulation: Optional[np.ndarray] = None) -> None:
        pass

    def multinomial_choice(self, consideration_set: np.ndarray, cos_sim: np.ndarray) -> int:
        # Pick a cosine similarity vs avg. rating adjuster for the visitor
        # This adjuster will be used to calculate a "score" for the products in the consideration set
        # The scores will be transformed to probabilities of choice using the logistic/softmax formula
        # Then a single multinomial choice of product will be made using these choice probabilities
        adj = np.random.random()
        consideration_set_cos_sim = cos_sim[consideration_set]
        consideration_set_reviews = np.array([simulated_reviews[prod][-1] for prod in consideration_set])
        consideration_set_avg_ratings = review_histogram_means(consideration_set_reviews)
        # Adjusting both cosine similarities and avg ratings to be on the same scale - 0 to 1
        consideration_set_cos_sim = (consideration_set_cos_sim + 1.0) / 2.0
        consideration_set_avg_ratings /= 5.0
        # Linear combination of cosine similarity and avg ratings using the adjuster
        scores = (adj * consideration_set_cos_sim) + ((1 - adj) * consideration_set_avg_ratings)
        # Calculate a shifted softmax to guard against over and underflow errors
        # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
        exps = np.exp(scores - np.max(scores))
        choice_p = exps/np.sum(exps)
        # Finally pick a product according to these choice probabilities
        choice_index = np.where(np.random.multinomial(1, choice_p))[0][0]
        return consideration_set[choice_index]

    def simulate_visitor_choice(self, product_embeddings: np.ndarray, simulated_reviews: List[Deque]) -> int:
        visitor_embedding = self.embedding_density_estimators["visitors"].multivariate_normal(
            np.zeros(100), np.diag(np.ones(100))
        )
        cos_sim = cdist(product_embeddings, visitor_embedding.reshape(1, -1), metric="cosine")
        # Faster way to pick the indexes of the top k elements in an array
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        consideration_set = np.argpartition(cos_sim, -self.consideration_set_size)[-self.consideration_set_size :]
        chosen_product = self.multinomial_choice(consideration_set, cos_sim)
        return chosen_product

    def simulate_marketplace(self) -> None:
        total_visitors = self.num_total_marketplace_reviews * 30
        simulated_reviews = [deque([np.zeros(5)], maxlen=total_visitors) for prod in range(self.num_products)]
        product_embeddings = self.embedding_density_estimators["products"].multivariate_normal(
            np.zeros(100), np.diag(np.ones(100), size=self.num_products)
        )

        for visitor in range(total_visitors):
            chosen_product = self.simulate_visitor_choice(product_embeddings, simulated_reviews)
            rating_index = self.simulate_visitor_journey(simulated_reviews[-1], simulation_id)
            if rating_index is not None:
                current_histogram = simulated_reviews[-1].copy()
                current_histogram[rating_index] += 1
                simulated_reviews.append(current_histogram)
            if np.sum(simulated_reviews[-1]) >= num_simulated_reviews:
                break

        simulated_reviews = np.array(simulated_reviews)

        # Return histogram or timeseries of review histograms based on simulation_type
        if self.simulation_type == "histogram":
            return simulated_reviews[-1, :]
        else:
            return simulated_reviews
