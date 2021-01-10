import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from snpe.utils.tqdm_utils import tqdm_joblib
from tqdm import tqdm


class BaseSimulator:
    def __init__(self, params: dict):
        self.review_prior = params.pop("review_prior")
        assert (
            len(self.review_prior) == 5
        ), f"""
        Prior Dirichlet distribution of simulated reviews needs to have 5 parameters,
        but found {len(self.review_prior)}
        """
        self.tendency_to_rate = params.pop("tendency_to_rate")
        self.params = params

    def generate_simulation_parameters(self, num_simulations: int) -> dict:
        raise NotImplementedError

    def convolve_prior_with_existing_reviews(self, simulated_reviews: np.array) -> np.array:
        assert (
            self.review_prior.shape == simulated_reviews.shape
        ), "Prior and simulated distributions of reviews should have the same shape"
        return self.review_prior + simulated_reviews

    def simulate_visitor_journey(self, simulated_reviews: np.array, simulation_id: int) -> np.array:
        raise NotImplementedError

    def simulate_review_histogram(
        self, simulation_id: int, num_reviews_per_simulation: Optional[np.array] = None
    ) -> np.array:
        raise NotImplementedError

    def mismatch_calculator(self, experience: float, expected_experience_dist_mean: float) -> float:
        raise NotImplementedError

    def rating_calculator(self, delta: float) -> int:
        raise NotImplementedError

    def decision_to_leave_review(self, delta: float, simulation_id: int) -> bool:
        raise NotImplementedError

    def simulate(self, num_simulations: int, num_reviews_per_simulation: Optional[np.array] = None) -> None:
        if num_reviews_per_simulation is not None:
            assert (
                len(num_reviews_per_simulation) == num_simulations
            ), f"""
            {num_simulations} simulations to be done,
            but {len(num_reviews_per_simulation)} review counts per simulation provided
            """

        self.simulation_parameters = self.generate_simulation_parameters(num_simulations)
        with tqdm_joblib(tqdm(desc="Simulations", total=num_simulations)) as progress_bar:
            simulations = Parallel(n_jobs=mp.cpu_count())(
                delayed(self.simulate_review_histogram)(i, num_reviews_per_simulation) for i in range(num_simulations)
            )
        self.simulations = np.array(simulations)

    def save_simulations(self, dirname: Path) -> None:
        simulation_dict = {
            "simulation_parameters": self.simulation_parameters,
            "simulations": self.simulations,
            "tendency_to_rate": self.tendency_to_rate,
            "review_prior": self.review_prior,
            "other_params": self.params,
        }
        with open(dirname / (self.__class__.__name__ + ".pkl"), "wb") as f:
            pickle.dump(simulation_dict, f)

    def load_simulator(self, dirname: Path) -> None:
        with open(dirname / (self.__class__.__name__ + ".pkl"), "rb") as f:
            simulation_dict = pickle.load(f)
        for key in simulation_dict:
            setattr(self, key, simulation_dict[key])


class SingleRhoSimulator(BaseSimulator):
    def __init__(self, params: dict):
        super(SingleRhoSimulator, self).__init__(params)

    def generate_simulation_parameters(self, num_simulations) -> dict:
        return {"rho": np.random.random(size=num_simulations) * 4}

    def simulate_visitor_journey(self, simulated_reviews: np.array, simulation_id: int) -> int:
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
            simulated_reviews[rating_index] += 1

        return simulated_reviews

    def mismatch_calculator(self, experience: float, expected_experience_dist_mean: float) -> float:
        assert experience in np.arange(
            1, 6, 1
        ), f"User's experience should be a whole number in [1, 5], got {experience} instead"
        assert (
            expected_experience_dist_mean >= 1.0 and expected_experience_dist_mean <= 5.0
        ), f"""
        Mean of user's expected distribution of experiences is a float in [1, 5],
        got {expected_experience_dist_mean} instead
        """
        return experience - expected_experience_dist_mean

    def rating_calculator(self, delta: float) -> int:
        if delta <= -1.5:
            return 0
        elif delta > -1.5 and delta <= -0.5:
            return 1
        elif delta > -0.5 and delta <= 0.5:
            return 2
        elif delta > 0.5 and delta <= 1.5:
            return 3
        else:
            return 4

    def decision_to_leave_review(self, delta: float, simulation_id: int) -> bool:
        # Pull out the single rho which will be used in the decision to rate
        rho = self.simulation_parameters["rho"][simulation_id]
        # Return the review only if mismatch is higher than rho
        # Tendency to rate governs baseline probability of returning review
        if np.random.random() <= self.tendency_to_rate:
            return True
        elif np.abs(delta) >= rho:
            return True
        else:
            return False

    def simulate_review_histogram(
        self, simulation_id: int, num_reviews_per_simulation: Optional[np.array] = None
    ) -> np.array:
        if num_reviews_per_simulation is None:
            num_simulated_reviews = np.random.randint(low=20, high=3001)
        else:
            num_simulated_reviews = int(num_reviews_per_simulation[simulation_id])

        total_visitors = num_simulated_reviews * 30
        simulated_reviews = np.zeros(5)

        for visitor in range(total_visitors):
            simulated_reviews = self.simulate_visitor_journey(simulated_reviews, simulation_id)
            if np.sum(simulated_reviews) >= num_simulated_reviews:
                break

        return simulated_reviews


class DoubleRhoSimulator(SingleRhoSimulator):
    def __init__(self, params):
        super(DoubleRhoSimulator, self).__init__(params)

    def generate_simulation_parameters(self, num_simulations) -> dict:
        return {
            "rho": np.vstack((np.random.random(size=num_simulations) * 4, np.random.random(size=num_simulations) * 4)).T
        }

    def decision_to_leave_review(self, delta: float, simulation_id: int) -> bool:
        # Pull out the single rho which will be used in the decision to rate
        rho = self.simulation_parameters["rho"][simulation_id]
        # Return the review only if mismatch is higher than rho
        # We use two rhos here - rho[0] is for negative mismatch, and rho[1] for positive mismatch
        assert rho.shape == (2,), f"Expecting shape (2,) for rho, got {rho.shape} instead"
        # Tendency to rate governs baseline probability of returning review
        if np.random.random() <= self.tendency_to_rate:
            return True
        elif delta < 0 and np.abs(delta) >= rho[0]:
            return True
        elif delta >= 0 and np.abs(delta) >= rho[1]:
            return True
        else:
            return False
