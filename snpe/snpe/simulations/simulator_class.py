import multiprocessing as mp
import pickle

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Union

import numpy as np

from joblib import Parallel, delayed
from snpe.utils.functions import check_existing_reviews, check_simulation_parameters
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

    def simulate_visitor_journey(self, simulated_reviews: Deque, simulation_id: int, **kwargs) -> Union[int, None]:
        raise NotImplementedError

    def simulate_review_histogram(
        self, simulation_id: int, num_reviews_per_simulation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        raise NotImplementedError

    def mismatch_calculator(self, experience: float, expected_experience: float) -> float:
        raise NotImplementedError

    def rating_calculator(self, delta: float, simulation_id: int) -> int:
        raise NotImplementedError

    def decision_to_leave_review(self, delta: float, simulation_id: int) -> bool:
        raise NotImplementedError

    def simulate(
        self,
        num_simulations: int,
        num_reviews_per_simulation: Optional[np.ndarray] = None,
        simulation_parameters: Optional[dict] = None,
        existing_reviews: Optional[List[np.ndarray]] = None,
        **kwargs,
    ) -> None:
        if existing_reviews is not None:
            assert (
                simulation_parameters is not None
            ), f"""
            Existing reviews for products supplied, but no simulation parameters given
            """
            assert (
                num_reviews_per_simulation is not None
            ), f"""
            Existing reviews for products supplied,but num_reviews_per_simulation not given. This gives the number of
            TOTAL reviews per product desired
            """
            # Run checks on the shape and initial values of the review timeseries provided. These checks remove
            # the first value in the timeseries before returning it (as that first value is automatically re-appended
            # during simulations)
            existing_reviews = check_existing_reviews(existing_reviews)
            # Also pick num_products = num_simulations from the provided existing reviews if the tests succeed. The
            # provided num_simulations will then be ignored
            num_simulations = len(existing_reviews)

        if num_reviews_per_simulation is not None:
            assert (
                len(num_reviews_per_simulation) == num_simulations
            ), f"""
            {num_simulations} simulations to be done,
            but {len(num_reviews_per_simulation)} review counts per simulation provided
            """

        if simulation_parameters is not None:
            # Check that the provided simulation parameters have all the parameters (i.e, dict keys)
            # that should be there. This is done by comparing to a dummy set of generated parameters
            dummy_parameters = self.generate_simulation_parameters(10)
            assert set(simulation_parameters) == set(
                dummy_parameters
            ), f"""
            Found parameters {simulation_parameters.keys()} in the provided parameters; expected
            {dummy_parameters.keys()} as simulation parameters instead
            """
        else:
            simulation_parameters = self.generate_simulation_parameters(num_simulations)
        # Run shape checks on the input dict of simulation parameters
        # Store the number of distribution samples per parameter if the checks succeed
        self.params["num_dist_samples"] = check_simulation_parameters(simulation_parameters, num_simulations)
        self.simulation_parameters = simulation_parameters

        with tqdm_joblib(tqdm(desc="Simulations", total=num_simulations)) as progress_bar:
            simulations = Parallel(n_jobs=mp.cpu_count())(
                delayed(self.simulate_review_histogram)(i, num_reviews_per_simulation, existing_reviews)
                for i in range(num_simulations)
            )
        self.simulations = np.array(simulations)

    # Helper function to sample the distribution of parameters to yield a set of simulation parameters for
    # every visitor. Our simulations are at the product level, thus the simulation ids run from 0 to n-1 if we are
    # simulation n products. Each of these n products has a distribution of simulation parameters (which is the inferred
    # set of posterior distributions if we are trying to simulate new data after performing inference from observed products)
    def yield_simulation_param_per_visitor(self, simulation_id: int, param_to_yield: str) -> Union[float, np.ndarray]:
        return self.simulation_parameters[param_to_yield][
            np.random.randint(self.params["num_dist_samples"]), simulation_id
        ]

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
    def generate_simulation_parameters(cls, num_simulations: int) -> dict:
        # This is the basic simulation parameter generator and should be used when simulations are being done
        # to train SNPE model. If supplying already inferred posterior for simulations, this method should not be used
        # NOTE: The simulation models expect a distribution over simulation parameters for every simulation id.
        # This method hacks the distribution requirement by creating distribution samples that are the same value
        # Of course, the parameters are uniformly distributed across simulations, but there is no variance in the
        # distributions "per simulation"
        # That is why the num_dist_samples is only 10 here (just any small number would do) as all the values (per
        # simulation id) are exactly equal
        simulation_parameters = {"rho": np.tile((np.random.random(size=num_simulations) * 4)[None, :], (10, 1))}
        return simulation_parameters

    def get_actual_experience(self, expected_experience_dist: np.ndarray, **kwargs) -> int:
        # This method is separated out so that during marketplace simulations, a more
        # involved process of getting the actual experience (through product embeddings) can be used
        # For the general single rho simulator, actual experience is just a draw from the expected
        # distribution of experiences
        return np.where(np.random.multinomial(1, expected_experience_dist))[0][0] + 1.0

    def simulate_visitor_journey(self, simulated_reviews: Deque, simulation_id: int, **kwargs) -> Union[int, None]:
        # Convolve the current simulated review distribution with the prior to get the posterior of reviews
        review_posterior = self.convolve_prior_with_existing_reviews(simulated_reviews[-1])

        # Just make a single draw from the posterior Dirichlet dist of reviews to get the distribution
        # of the product experiences that the user expects
        # Thus the expected experience is built out of the current distribution of reviews the user can see
        expected_experience_dist = np.random.dirichlet(review_posterior)
        # Also get the mean "experience" that the user expects
        expected_experience = np.sum(expected_experience_dist * np.arange(1, 6))
        # Get the user's actual experience
        experience = self.get_actual_experience(expected_experience_dist, **kwargs)

        # User's mismatch is the difference between their actual experience and the mean of the distribution
        # of experiences they expected
        delta = self.mismatch_calculator(experience, expected_experience)

        # Calculate the index of the rating the user wants to leave [0, 4]
        rating_index = self.rating_calculator(delta, simulation_id)
        # Get the decision to leave review (True/False)
        decision_to_rate = self.decision_to_leave_review(delta, simulation_id)

        # Add a review to the corresponding rating index if the user decided to rate
        if decision_to_rate:
            return rating_index
        else:
            return None

    def mismatch_calculator(self, experience: float, expected_experience: float) -> float:
        assert experience in np.arange(
            1, 6, 1
        ), f"User's experience should be a whole number in [1, 5], got {experience} instead"
        assert (
            expected_experience >= 1.0 and expected_experience <= 5.0
        ), f"""
        Mean of user's expected distribution of experiences is a float in [1, 5],
        got {expected_experience} instead
        """
        return experience - expected_experience

    def rating_calculator(self, delta: float, simulation_id: int) -> int:
        # Simulation id is ignored here as the cutoffs for delta to determine different star ratings are
        # fixed in this case. However, that will not be the case for the RatingScaleSimulator
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
        rho = self.yield_simulation_param_per_visitor(simulation_id, "rho")
        # Return the review only if mismatch is higher than rho
        # Tendency to rate governs baseline probability of returning review
        if np.random.random() <= self.tendency_to_rate:
            return True
        elif np.abs(delta) >= rho:
            return True
        else:
            return False

    def simulate_review_histogram(
        self,
        simulation_id: int,
        num_reviews_per_simulation: Optional[np.ndarray] = None,
        existing_reviews: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        if num_reviews_per_simulation is None:
            num_simulated_reviews = np.random.randint(low=20, high=5001)
        else:
            num_simulated_reviews = int(num_reviews_per_simulation[simulation_id])

        total_visitors = num_simulated_reviews * 30
        # Give the product 5 reviews to start with, one for each rating. This is only so that the review timeseries
        # looks similar to that produced by the more complex marketplace simulations
        simulated_reviews = deque([np.ones(5)], maxlen=total_visitors)
        # If existing reviews have been supplied, unravel them into the simulated reviews deque
        if existing_reviews is not None:
            product_reviews = existing_reviews[simulation_id]
            for review in product_reviews:
                # We use the same manner of appending existing review histograms to simulated_reviews as if
                # those histograms were actually produced during simulations. This ensures that the same dtype is
                # appended to the deque always and keeps the size of the deque as small as possible
                current_histogram = simulated_reviews[-1].copy()
                rating_index = np.where(review - current_histogram)[0][0]
                current_histogram[rating_index] += 1
                simulated_reviews.append(current_histogram)
                if len(simulated_reviews) > 1:
                    assert (
                        np.sum(simulated_reviews[-1]) - np.sum(simulated_reviews[-2])
                    ) == 1, """
                    Please check the histograms provided in the array of existing reviews. These should be in the form
                    of cumulative histograms and should only add 1 rating at a time
                    """
                total_visitors -= 1

        for visitor in range(total_visitors):
            rating_index = self.simulate_visitor_journey(simulated_reviews, simulation_id)
            if rating_index is not None:
                current_histogram = simulated_reviews[-1].copy()
                current_histogram[rating_index] += 1
                simulated_reviews.append(current_histogram)
            if np.sum(simulated_reviews[-1]) >= num_simulated_reviews:
                break

        simulated_reviews_array = np.array(simulated_reviews)

        # Return histogram or timeseries of review histograms based on simulation_type
        if self.simulation_type == "histogram":
            return simulated_reviews_array[-1, :]
        else:
            return simulated_reviews_array


class DoubleRhoSimulator(SingleRhoSimulator):
    def __init__(self, params: dict):
        super(DoubleRhoSimulator, self).__init__(params)

    @classmethod
    def generate_simulation_parameters(cls, num_simulations: int) -> dict:
        # This is the basic simulation parameter generator and should be used when simulations are being done
        # to train SNPE model. If supplying already inferred posterior for simulations, this method should not be used
        # NOTE: The simulation models expect a distribution over simulation parameters for every simulation id.
        # This method hacks the distribution requirement by creating distribution samples that are the same value
        # Of course, the parameters are uniformly distributed across simulations, but there is no variance in the
        # distributions "per simulation"
        # That is why the num_dist_samples is only 10 here (just any small number would do) as all the values (per
        # simulation id) are exactly equal
        rho_array = np.vstack(
            (np.random.random(size=num_simulations) * 4, np.random.random(size=num_simulations) * 4)
        ).T
        rho_array = np.tile(rho_array[None, :, :], (10, 1, 1))
        simulation_parameters = {"rho": rho_array}
        return simulation_parameters

    def decision_to_leave_review(self, delta: float, simulation_id: int) -> bool:
        # Right now, we don't make the very first user always leave a review - maybe change later
        # Pull out the single rho which will be used in the decision to rate
        rho = self.yield_simulation_param_per_visitor(simulation_id, "rho")
        # Return the review only if mismatch is higher than rho
        # We use two rhos here - rho[0] is for negative mismatch, and rho[1] for positive mismatch
        assert isinstance(rho, np.ndarray), f"Expected np.ndarray type for rho, found {type(rho)} instead"
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
            "mode of latest",
        ], f"Can only use mean/mode/mode of latest as previous rating, provided {self.previous_rating_measure} instead"
        if self.previous_rating_measure == "mode of latest":
            assert (
                "num_latest_reviews_for_herding" in params
            ), """
            Number of latest reviews to calculate mode needed if mode of latest is being used for herding
            """
            self.num_latest_reviews_for_herding = params["num_latest_reviews_for_herding"]
            assert (
                self.num_latest_reviews_for_herding < self.min_reviews_for_herding
            ), f"""
            Minimum {self.min_reviews_for_herding} required before herding can be done, but
            {self.num_latest_reviews_for_herding} latest reviews to be actually used for mode calculation during herding,
            so herding cannot actually be done before {self.num_latest_reviews_for_herding + 1} reviews accumulate
            """
        assert (
            self.min_reviews_for_herding >= 1
        ), f"At least 1 review has to exist before herding can happen, found {self.min_reviews_for_herding} instead"
        super(HerdingSimulator, self).__init__(params)

    @classmethod
    def generate_simulation_parameters(cls, num_simulations: int) -> dict:
        # This method gets the rho parameters by calling the parameter generating classmethod of the DoubleRhoSimulator
        # Then it just adds the herding parameter on top
        simulation_parameters = DoubleRhoSimulator.generate_simulation_parameters(num_simulations)
        simulation_parameters["h_p"] = np.tile(
            np.random.random(size=num_simulations)[None, :], (simulation_parameters["rho"].shape[0], 1)
        )
        return simulation_parameters

    def simulate_visitor_journey(
        self, simulated_reviews: Deque, simulation_id: int, use_h_u: bool = False, **kwargs
    ) -> Union[int, None]:
        # Run the visitor journey the same way at first
        rating_index = super(HerdingSimulator, self).simulate_visitor_journey(
            simulated_reviews, simulation_id, **kwargs
        )

        # If the decision to rate was true, modify the rating index according to the herding procedure
        # Don't initiate the herding procedure till at least the minimum number of reviews have come
        # Also since we add 5 reviews right at the beginning (timeseries begins with np.ones(5)), we have to
        # subtract those from the total count to see if we have enough reviews to start herding
        if (rating_index is not None) and (
            np.sum(simulated_reviews[-1]) - np.sum(simulated_reviews[0]) >= self.min_reviews_for_herding
        ):
            herded_rating_index = self.herding(rating_index, simulated_reviews, simulation_id, use_h_u)
            return herded_rating_index
        # Otherwise just return the original rating index (which is = None in this case)
        else:
            return rating_index

    def choose_herding_parameter(self, rating_index: int, simulated_reviews: Deque, simulation_id: int) -> float:
        h_p = self.yield_simulation_param_per_visitor(simulation_id, "h_p")
        assert isinstance(h_p, float), f"Expecting a scalar value for the herding parameter, got {h_p} instead"
        return h_p

    def herding(self, rating_index: int, simulated_reviews: Deque, simulation_id: int, use_h_u: bool = False) -> int:
        assert (
            np.sum(simulated_reviews[-1]) - np.sum(simulated_reviews[0]) >= self.min_reviews_for_herding
        ), f"""
        Minimum {self.min_reviews_for_herding} reviews need to have been obtained for herding to happen,
        found only {np.sum(simulated_reviews[-1]) - np.sum(simulated_reviews[0])} instead
        """
        # Check that the whole timeseries of simulated_reviews has been supplied
        if len(simulated_reviews) == 1:
            np.testing.assert_array_equal(simulated_reviews[0], np.ones(5))
        else:
            np.testing.assert_array_equal(
                np.array([review.shape[0] for review in simulated_reviews]), 5 * np.ones(len(simulated_reviews))
            )
        # Pull out the herding parameter which will be used in this simulation
        # This step is trivial when using a single herding h_p, but becomes important when using 2
        h_p = self.choose_herding_parameter(rating_index, simulated_reviews, simulation_id)
        # If an additional user-specific herding parameter is being used, generate it
        if use_h_u:
            # The final herding probability is the product of h_p and h_u
            # So this user will herd with p=h_p*h_u and not with 1-p
            h_u = np.random.random()
            herding_prob = h_p * h_u
        else:
            # Otherwise there is only the product-specific herding probability h_p
            herding_prob = h_p
        # Simulate the herding process
        if np.random.random() <= herding_prob:
            # Herding happening
            if self.previous_rating_measure == "mean":
                # Mean calculation from review histogram - using the indices (0-4) instead of actual ratings (1-5)
                previous_rating_index = (
                    np.sum(np.array(simulated_reviews[-1]) * np.arange(5)) / np.array(simulated_reviews[-1]).sum()
                )
            elif self.previous_rating_measure == "mode":
                # WARNING: If the histogram has more than 1 mode, argmax will ONLY RETURN THE FIRST ONE
                previous_rating_index = np.argmax(np.array(simulated_reviews[-1]))
            elif self.previous_rating_measure == "mode of latest":
                # Get the histogram of latest num_latest_reviews_for_herding, and pick the mode
                # WARNING: If the histogram has more than 1 mode, argmax will ONLY RETURN THE FIRST ONE
                latest_histogram = simulated_reviews[-1] - simulated_reviews[-self.num_latest_reviews_for_herding - 1]
                previous_rating_index = np.argmax(latest_histogram)
            else:
                f"Can only use mean/mode/mode of latest as previous rating, provided {self.previous_rating_measure} instead"
            # Numpy inherits from the built-in float type, but not built-in int type. Therefore, we could check if h_p
            # was an instance of float at the start of this method, but can't use
            # isinstance(previous_rating_index, (float, int)) here.
            # Check: https://numpy.org/doc/stable/reference/arrays.scalars.html
            assert np.issubdtype(
                previous_rating_index, np.number
            ), f"Previous rating index should be a number, found {type(previous_rating_index)} instead"
            # Return the average of the currently calculated rating and the previous rating measure
            # Convert to integer because this is used to index the rating histogram
            return int((rating_index + previous_rating_index) / 2)
        else:
            # Herding not happening
            return rating_index


class DoubleHerdingSimulator(HerdingSimulator):
    # Simulates herding with 2 product-specific herding parameters (h_p): one is used when the visitor's intended rating
    # is above a metric of existing ratings (mean or mode) and the other when it is below
    def __init__(self, params: dict):
        self.herding_differentiating_measure = params["herding_differentiating_measure"]
        assert self.herding_differentiating_measure in [
            "mean",
            "mode",
        ], f"""
        Can only use mean/mode of the existing ratings to choose the h_p to use,
        provided {self.herding_differentiating_measure} instead
        """
        super(DoubleHerdingSimulator, self).__init__(params)

    @classmethod
    def generate_simulation_parameters(cls, num_simulations) -> dict:
        # Same strategy as in the HerdingSimulator
        simulation_parameters = DoubleRhoSimulator.generate_simulation_parameters(num_simulations)
        h_p_array = np.vstack((np.random.random(size=num_simulations), np.random.random(size=num_simulations))).T
        h_p_array = np.tile(h_p_array[None, :, :], (simulation_parameters["rho"].shape[0], 1, 1))
        simulation_parameters["h_p"] = h_p_array
        return simulation_parameters

    def choose_herding_parameter(self, rating_index, simulated_reviews: Deque, simulation_id: int) -> float:
        # Pull out the (2 valued) h_p corresponding to this simulation id
        h_p = self.yield_simulation_param_per_visitor(simulation_id, "h_p")
        # Confirm that h_p is 2-dimensional array
        assert isinstance(h_p, np.ndarray), f"Expected np.ndarray type for h_p, found {type(h_p)} instead"
        assert h_p.shape == (2,), f"Expecting shape (2,) for h_p, got {h_p.shape} instead"
        # Pick which h_p to use based on the rating_index that the visitor picked
        # If it is greater than the mean/mode of existing ratings, pick h_p[1], else pick h_p[0]
        if self.herding_differentiating_measure == "mean":
            metric = np.sum(np.array(simulated_reviews[-1]) * np.arange(5)) / np.array(simulated_reviews[-1]).sum()
        elif self.herding_differentiating_measure == "mode":
            # WARNING: If the histogram has more than 1 mode, argmax will ONLY RETURN THE FIRST ONE
            metric = np.argmax(np.array(simulated_reviews[-1]))
        else:
            raise ValueError(
                f"""
                herding_differentiating_measure has to be one of mean or mode,
                found {self.herding_differentiating_measure} instead
                """
            )

        if rating_index <= metric:
            return h_p[0]
        else:
            return h_p[1]


class RatingScaleSimulator(HerdingSimulator):
    def __init__(self, params: dict):
        # The highest value that limits 5 star ratings
        # The actual limit can be lower than this value, this is the upper bound of the limit
        # Delta (expected-experience) will be compared to this limit, and if delta > limit, a 5 star rating is left
        # The actual limit of 5 star ratings will lie between 0.5*five_star_highest_limit and five_star_highest_limit
        self.five_star_highest_limit = params["five_star_highest_limit"]
        # Similarly, we have a lowest value that limits 1 star ratings
        # The actual limit can be higher than this value, this is the lower bound of the limit
        # Delta (expected-experience) will be compared to this limit, and if delta < limit, a 1 star rating is left
        # The actual limit of 1 star ratings will lie between one_star_lowest_limit and 0.5*one_star_lowest_limit
        self.one_star_lowest_limit = params["one_star_lowest_limit"]
        # Limit of 5 star ratings should be positive, and vice-versa for 1 star ratings
        assert (
            self.five_star_highest_limit > 0.0 and self.five_star_highest_limit < 4.0
        ), f"""
        The highest limit of delta for 5 star ratings should be positive and less than 4,
        found {self.five_star_highest_limit}
        """
        assert (
            self.one_star_lowest_limit < 0.0 and self.one_star_lowest_limit > -4.0
        ), f"""
        The lowest limit of delta for 1 star ratings should be negative and more than -4,
        found {self.one_star_lowest_limit}
        """
        # We also need to supply the max probability of 5 star bias across all simulated products
        # Based on this probability, a visitor leaves a 5 star rating irrespective of their experience, +ive or -ive
        self.max_bias_5_star = params["max_bias_5_star"]
        assert (
            self.max_bias_5_star >= 0 and self.max_bias_5_star <= 1
        ), f"""
        The max 5 star bias across all simulated products is a maximum probability value. So it should be
        between 0 and 1, but found {self.max_bias_5_star} instead
        """
        super(RatingScaleSimulator, self).__init__(params)

    @classmethod
    def generate_simulation_parameters(cls, num_simulations) -> dict:
        # This generates the rho and h_p herding parameters from the classmethod of the double herding simulator
        # Then we will add the p values that define the rating scales on top
        # These p values determine how we will split up the space from one_star_lowest_limit to five_star_lowest_limit
        # over the scale on which delta values are compared and star ratings are determined
        simulation_parameters = HerdingSimulator.generate_simulation_parameters(num_simulations)
        # p_5 determines the actual limit to which delta is compared to get 5 star ratings
        # That limit = five_star_highest_limit * p_5
        # So if p_5 = 1 (highest value), limit = five_star_highest_limit
        # If p_5 = 0.5 (lowest value), limit = 0.5 * five_star_highest_limit
        # p_4 determines the actual limit to which delta is compared to get 4 star ratings
        # That limit = five_star_highest_limit * p_5 * p_4
        # So 3 star ratings come from 0 to five_star_highest_limit * p_5 * p_4
        # And 4 star ratings come from five_star_highest_limit * p_5 * p_4 to five_star_highest_limit * p_5
        p_5 = 0.5 * np.random.random(size=num_simulations) + 0.5
        p_4 = 0.5 * np.random.random(size=num_simulations) + 0.25
        # p_1 determines the actual limit to which delta is compared to get 1 star ratings
        # That limit = one_star_lowest_limit * p_1
        # So if p_1 = 1 (highest value), limit = one_star_lowest_limit
        # If p_1 = 0.5 (lowest value), limit = 0.5 * one_star_lowest_limit
        # p_2 determines the actual limit to which delta is compared to get 2 star ratings
        # That limit = one_star_lowest_limit * p_1 * p_2
        # So 3 star ratings come from one_star_lowest_limit * p_1 * p_2 to 0
        # And 2 star ratings come from one_star_lowest_limit * p_1 to one_star_lowest_limit * p_1 * p_2
        p_1 = 0.5 * np.random.random(size=num_simulations) + 0.5
        p_2 = 0.5 * np.random.random(size=num_simulations) + 0.25
        # A final bias parameter that encodes bias towards 5 star ratings
        # A user leaves a 5 star rating on the product (irrespective of experience) with this probability
        bias_5_star = np.random.random(size=num_simulations)
        simulation_parameters["p_5"] = np.tile(p_5[None, :], (simulation_parameters["rho"].shape[0], 1))
        simulation_parameters["p_4"] = np.tile(p_4[None, :], (simulation_parameters["rho"].shape[0], 1))
        simulation_parameters["p_2"] = np.tile(p_2[None, :], (simulation_parameters["rho"].shape[0], 1))
        simulation_parameters["p_1"] = np.tile(p_1[None, :], (simulation_parameters["rho"].shape[0], 1))
        simulation_parameters["bias_5_star"] = np.tile(bias_5_star[None, :], (simulation_parameters["rho"].shape[0], 1))
        return simulation_parameters

    def rating_calculator(self, delta: float, simulation_id: int) -> int:
        # Pull out the rating scale related parameters that correspond to this simulation id
        p_1 = self.yield_simulation_param_per_visitor(simulation_id, "p_1")
        p_2 = self.yield_simulation_param_per_visitor(simulation_id, "p_2")
        p_4 = self.yield_simulation_param_per_visitor(simulation_id, "p_4")
        p_5 = self.yield_simulation_param_per_visitor(simulation_id, "p_5")
        if delta <= (self.one_star_lowest_limit * p_1):
            return 0
        elif delta > (self.one_star_lowest_limit * p_1) and delta <= (self.one_star_lowest_limit * p_1 * p_2):
            return 1
        elif delta > (self.one_star_lowest_limit * p_1 * p_2) and delta <= (self.five_star_highest_limit * p_5 * p_4):
            return 2
        elif delta > (self.five_star_highest_limit * p_5 * p_4) and delta <= (self.five_star_highest_limit * p_5):
            return 3
        else:
            return 4

    def simulate_visitor_journey(
        self, simulated_reviews: Deque, simulation_id: int, use_h_u: bool = False, **kwargs
    ) -> Union[int, None]:
        # Run the visitor journey the same way at first
        rating_index = super(RatingScaleSimulator, self).simulate_visitor_journey(
            simulated_reviews, simulation_id, use_h_u, **kwargs
        )

        # A user simply returns a 5 star rating with probability = bias_5_star
        bias_5_star = self.yield_simulation_param_per_visitor(simulation_id, "bias_5_star")
        if np.random.random() <= self.max_bias_5_star * bias_5_star:
            return 4
        else:
            return rating_index
