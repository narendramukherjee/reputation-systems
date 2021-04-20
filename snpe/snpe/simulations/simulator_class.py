import multiprocessing as mp
import pickle

from collections import deque
from pathlib import Path
from typing import Optional, Union

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
        self.simulation_type = params.pop("simulation_type")
        assert self.simulation_type in [
            "histogram",
            "timeseries",
        ], f"Can only simulate review histogram or timeseries, got simulation_type={self.simulation_type}"
        self.params = params

    @classmethod
    def generate_simulation_parameters(cls, num_simulations: int) -> dict:
        raise NotImplementedError

    def convolve_prior_with_existing_reviews(self, simulated_reviews: np.ndarray) -> np.ndarray:
        assert (
            self.review_prior.shape == simulated_reviews.shape
        ), "Prior and simulated distributions of reviews should have the same shape"
        return self.review_prior + simulated_reviews

    def simulate_visitor_journey(self, simulated_reviews: np.ndarray, simulation_id: int) -> np.ndarray:
        raise NotImplementedError

    def simulate_review_histogram(
        self, simulation_id: int, num_reviews_per_simulation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        raise NotImplementedError

    def mismatch_calculator(self, experience: float, expected_experience_dist_mean: float) -> float:
        raise NotImplementedError

    def rating_calculator(self, delta: float) -> int:
        raise NotImplementedError

    def decision_to_leave_review(self, delta: float, simulation_id: int) -> bool:
        raise NotImplementedError

    def simulate(self, num_simulations: int, num_reviews_per_simulation: Optional[np.ndarray] = None) -> None:
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
            "simulation_type": self.simulation_type,
            "simulation_parameters": self.simulation_parameters,
            "simulations": self.simulations,
            "tendency_to_rate": self.tendency_to_rate,
            "review_prior": self.review_prior,
            "params": self.params,
        }
        with open(dirname / f"{self.__class__.__name__}_{self.simulation_type}.pkl", "wb") as f:
            pickle.dump(simulation_dict, f)

    def load_simulator(self, dirname: Path) -> None:
        with open(dirname / f"{self.__class__.__name__}_{self.simulation_type}.pkl", "rb") as f:
            simulation_dict = pickle.load(f)
        for key in simulation_dict:
            setattr(self, key, simulation_dict[key])


class SingleRhoSimulator(BaseSimulator):
    def __init__(self, params: dict):
        super(SingleRhoSimulator, self).__init__(params)

    @classmethod
    def generate_simulation_parameters(cls, num_simulations) -> dict:
        return {"rho": np.random.random(size=num_simulations) * 4}

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
        # Right now, we don't make the very first user always leave a review - maybe change later
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
        self, simulation_id: int, num_reviews_per_simulation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if num_reviews_per_simulation is None:
            num_simulated_reviews = np.random.randint(low=20, high=5001)
        else:
            num_simulated_reviews = int(num_reviews_per_simulation[simulation_id])

        total_visitors = num_simulated_reviews * 30
        simulated_reviews = deque([np.zeros(5)], maxlen=total_visitors)

        for visitor in range(total_visitors):
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


class DoubleRhoSimulator(SingleRhoSimulator):
    def __init__(self, params: dict):
        super(DoubleRhoSimulator, self).__init__(params)

    @classmethod
    def generate_simulation_parameters(cls, num_simulations) -> dict:
        return {
            "rho": np.vstack((np.random.random(size=num_simulations) * 4, np.random.random(size=num_simulations) * 4)).T
        }

    def decision_to_leave_review(self, delta: float, simulation_id: int) -> bool:
        # Right now, we don't make the very first user always leave a review - maybe change later
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


class HerdingSimulator(DoubleRhoSimulator):
    def __init__(self, params: dict):
        self.previous_rating_measure = params["previous_rating_measure"]
        self.min_reviews_for_herding = params["min_reviews_for_herding"]
        assert self.previous_rating_measure in [
            "mean",
            "mode",
            "latest",
        ], f"Can only use mean/mode/latest rating as previous rating, provided {self.previous_rating_measure} instead"
        assert (
            self.min_reviews_for_herding >= 1
        ), f"At least 1 review has to exist before herding can happen, found {self.min_reviews_for_herding} instead"
        super(HerdingSimulator, self).__init__(params)

    @classmethod
    def generate_simulation_parameters(cls, num_simulations) -> dict:
        return {
            "rho": np.vstack(
                (np.random.random(size=num_simulations) * 4, np.random.random(size=num_simulations) * 4)
            ).T,
            "h_p": np.random.random(size=num_simulations),
        }

    def simulate_visitor_journey(self, simulated_reviews: np.ndarray, simulation_id: int) -> Union[int, None]:
        # Run the visitor journey the same way at first
        rating_index = super(HerdingSimulator, self).simulate_visitor_journey(simulated_reviews, simulation_id)

        # If the decision to rate was true, modify the rating index according to the herding procedure
        # Don't initiate the herding procedure till at least the minimum number of reviews have come
        if (rating_index is not None) and (np.sum(simulated_reviews[-1]) >= self.min_reviews_for_herding):
            herded_rating_index = self.herding(rating_index, simulated_reviews, simulation_id)
            return herded_rating_index
        # Otherwise just return the original rating index (which is = None in this case)
        else:
            return rating_index

    def herding(self, rating_index: int, simulated_reviews: np.ndarray, simulation_id: int) -> int:
        # Pull out the herding parameter which will be used in this simulation
        h_p = self.simulation_parameters["h_p"][simulation_id]
        assert isinstance(h_p, float), f"Expecting a scalar value for the herding parameter, got {h_p} instead"
        # For the user whose decision to rate is being simulated, generate a herding parameter h_u
        h_u = np.random.random()
        # The final herding probability is the product of h_p and h_u
        # So this user will herd with p=h_p*h_u and not with 1-p
        if np.random.random() <= h_p * h_u:
            # Herding happening
            if self.previous_rating_measure == "mean":
                # Mean calculation from review histogram - using the indices (0-4) instead of actual ratings (1-5)
                previous_rating_index = (
                    np.sum(np.array(simulated_reviews[-1]) * np.arange(5)) / np.array(simulated_reviews[-1]).sum()
                )
            elif self.previous_rating_measure == "mode":
                # WARNING: If the histogram has more than 1 mode, argmax will ONLY RETURN THE FIRST ONE
                previous_rating_index = np.argmax(np.array(simulated_reviews[-1]))
            else:
                # Latest rating index finding by subtracting the last 2 review histograms
                previous_rating_index = np.where(np.array(simulated_reviews[-1]) - np.array(simulated_reviews[-2]))[0][
                    0
                ]
            # Numpy inherits from the built-in float type, but not built-in int type. Therefore, we could check if h_p
            # was an instance of float at the start of this method, but can't use
            # isinstance(previous_rating_index, (float, int)) here.
            # Check: https://numpy.org/doc/stable/reference/arrays.scalars.html
            assert np.issubdtype(
                previous_rating_index, np.number
            ), f"Previous rating index should be a number, found {type(previous_rating_index)} instead"
            assert (
                np.sum(simulated_reviews[-1]) >= 1
            ), f"Herding cannot be done when only {np.sum(simulated_reviews[-1])} reviews exist"
            # Return the average of the currently calculated rating and the previous rating measure
            # Convert to integer because this is used to index the rating histogram
            return int((rating_index + previous_rating_index) / 2)
        else:
            # Herding not happening
            return rating_index
