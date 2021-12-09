import multiprocessing as mp

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from numpy.random import default_rng
from scipy.spatial.distance import cdist
from snpe.utils.statistics import review_histogram_means
from snpe.utils.tqdm_utils import tqdm_joblib
from tqdm import tqdm

from .simulator_class import HerdingSimulator


class MarketplaceSimulator(HerdingSimulator):
    def __init__(self, params: dict):
        self.num_products = params["num_products"]
        self.num_total_marketplace_reviews = params["num_total_marketplace_reviews"]
        self.consideration_set_size = params["consideration_set_size"]
        super(MarketplaceSimulator, self).__init__(params)

    def load_embedding_density_estimators(self, dirname: Path = None) -> None:
        embedding_density_estimators = {"products": default_rng(), "visitors": default_rng()}
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
        assert (
            num_reviews_per_simulation is None
        ), f"""
            Num reviews per simulation is not needed for marketplace simulation
            """
        # num_simulations = number of marketplaces to be simulated
        # Total number of simulations = total number of marketplaces x num products per marketplace
        self.simulation_parameters = self.generate_simulation_parameters(num_simulations * self.num_products)
        self.load_embedding_density_estimators()
        with tqdm_joblib(tqdm(desc="Simulations", total=num_simulations)) as _:
            simulations = Parallel(n_jobs=mp.cpu_count())(
                delayed(self.simulate_marketplace)(i) for i in range(num_simulations)
            )
        self.simulations = np.array(simulations)

    def multinomial_choice(
        self, consideration_set: np.ndarray, cos_sim: np.ndarray, simulated_reviews: List[Deque]
    ) -> int:
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
        choice_p = exps / np.sum(exps)
        # Finally pick a product according to these choice probabilities
        choice_index = np.where(np.random.multinomial(1, choice_p))[0][0]
        return consideration_set[choice_index]

    def simulate_visitor_choice(self, product_embeddings: np.ndarray, simulated_reviews: List[Deque]) -> int:
        visitor_embedding = self.embedding_density_estimators["visitors"].multivariate_normal(
            np.zeros(100), np.diag(np.ones(100))
        )
        # cdist calculates the "cosine distance", i.e, 1 - cos_sim. So cos_sim = 1 - cdist
        cos_sim = 1.0 - cdist(product_embeddings, visitor_embedding.reshape(1, -1), metric="cosine")
        cos_sim = cos_sim.flatten()  # can flatten as only 1 visitor was compared to all products
        # Faster way to pick the indexes of the top k elements in an array
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        consideration_set = np.argpartition(cos_sim, -self.consideration_set_size)[-self.consideration_set_size :]
        chosen_product = self.multinomial_choice(consideration_set, cos_sim, simulated_reviews)
        return chosen_product

    def simulate_marketplace(self, marketplace_id: int) -> np.ndarray:
        total_visitors = self.num_total_marketplace_reviews * 30
        # Each product gets 5 reviews, 1 for each star to start off. This is necessary to prevent the
        # "cold start" problem when avg. rating needs to be calculated for products that have not accumulated
        # ratings yet
        simulated_reviews = [deque([np.ones(5)], maxlen=total_visitors) for prod in range(self.num_products)]
        product_embeddings = self.embedding_density_estimators["products"].multivariate_normal(
            np.zeros(100), np.diag(np.ones(100)), size=self.num_products
        )
        # Maintain a counter of total ratings on the platform - this total ignores the 1st 5 reviews
        # added to all products by default
        current_total_marketplace_reviews = 0

        for visitor in range(total_visitors):
            chosen_product = self.simulate_visitor_choice(product_embeddings, simulated_reviews)
            # The chosen_product and marketplace_id together determine the parameters (rho, h_p)
            # that will be used in the rating decision
            # In previous versions, we could pass simulation_id directly to pick the right parameters
            # Now we need to calculate the right simulation_id based on the product and marketplace
            # as in the marketplace simulation, parameters run from 0 to num_marketplaces X num_products
            simulation_id = (marketplace_id * self.num_products) + chosen_product
            rating_index = self.simulate_visitor_journey(simulated_reviews[chosen_product][-1], simulation_id, False)
            if rating_index is not None:
                current_histogram = simulated_reviews[chosen_product][-1].copy()
                current_histogram[rating_index] += 1
                simulated_reviews[chosen_product].append(current_histogram)
                current_total_marketplace_reviews += 1
            if current_total_marketplace_reviews >= self.num_total_marketplace_reviews:
                break

        # simulated_reviews = np.array([np.array(timeseries) for timeseries in simulated_reviews])

        # Return final histogram or timeseries of review histograms for all products based on simulation_type
        if self.simulation_type == "histogram":
            return np.array([np.array(timeseries[-1]) for timeseries in simulated_reviews])
        else:
            return np.array([np.array(timeseries) for timeseries in simulated_reviews])
