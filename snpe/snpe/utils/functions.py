import copy
import subprocess

from typing import List

import numpy as np
import torch


# A utility function to execute a command on the terminal using subprocess
# and print the outputs from the terminal in jupyter/python
# Use it as:
# for path in terminal_execute(command):
#   print(path, end="")
# https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def terminal_execute(cmd: str) -> None:
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# A utility function to check the convergence of a torch model tranining process. If the validation loss does not decrease
# for a number of epochs, training is stopped
# The torch model does need to have a best_validation_loss and epochs since last improvement
# defined in __init__ for this to work
def nn_converged(epoch: int, stop_after_epochs: int, validation_loss: torch.Tensor, model: torch.nn.Module) -> bool:
    converged = False
    # (Re)-start the epoch count with the first epoch or any improvement.
    if epoch == 0 or validation_loss < model.best_validation_loss:
        model.best_validation_loss = validation_loss
        model.epochs_since_last_improvement = 0
        model.best_model = copy.deepcopy(model.net.state_dict())
    else:
        model.epochs_since_last_improvement += 1

    # If no validation improvement over many epochs, stop training.
    if model.epochs_since_last_improvement > stop_after_epochs - 1:
        model.net.load_state_dict(model.best_model)
        converged = True
    return converged


# A utility function to check that all the params in the dict of simulation parameters are arrays of the right shape
# For each simulation id, we have a distribution for each simulation parameter
# So the shape of the simulation parameter arrays should be (num_dist_samples X num_simulations X parameter dims)
def check_simulation_parameters(simulation_parameters: dict, num_simulations: int) -> int:
    # Assert that the shape of the simulation parameter arrays is (num_dist_samples X num_simulations X param dims)
    # So we check that all parameter arrays have shape >= 2
    assert np.all(
        [len(val.shape) >= 2 for (key, val) in simulation_parameters.items()]
    ), f"""
    Expected shape (num_dist_samples X num_simulations X param dims) for all simulation parameter arrays, \n
    Found shapes {[key + ": " + str(val.shape) for (key, val) in simulation_parameters.items()]}
    """
    # Also assert that the number of parameters provided is equal to the number of simulations desired
    # Remember that number of simulations forms the 2nd dim of the array of simulation parameters
    assert np.array_equal(
        [val.shape[1] for (key, val) in simulation_parameters.items()],
        np.repeat(num_simulations, len(simulation_parameters)),
    ), f"""
    {num_simulations} simulations to be done, but number of provided parameters are: \n
    {[key + ": " + str(val.shape[1]) for (key, val) in simulation_parameters.items()]}
    \n Leave as None to generate parameters during simulation, or provide {num_simulations} for each.
    """
    # Finally check that all simulation parameters have the same number of distribution samples (i.e their 1st dim
    # is equal). Easy way is to take num_dist_samples from one of the parameters and see if all the other parameters
    # have the same num_dist_samples
    num_dist_samples = list(simulation_parameters.values())[0].shape[0]
    assert np.all(
        [val.shape[0] == num_dist_samples for (key, val) in simulation_parameters.items()]
    ), f"""
    Found unequal number of distribution samples for the simulation parameters as follows:
    {[key + ": " + str(val.shape[0]) for (key, val) in simulation_parameters.items()]}
    """
    # If all tests passed return the num_dist_samples that was found
    return num_dist_samples


# A utility function to check that the shape and starting values of the array-like of existing reviews provided during
# simulations is correct
def check_existing_reviews(existing_reviews: List[np.ndarray]) -> List[np.ndarray]:
    # Run shape checks on the list/array of existing reviews. The existing reviews need to be provided as
    # timeseries as the simulations themselves are run in timeseries form and converted to histograms later if needed
    # The existing reviews are expected to be of shape (num_products X num_reviews X 5)
    # for timeseries simulations
    # With marketplace simulations, we assume that this set of existing reviews is from a single marketplace
    np.testing.assert_array_equal(
        [product.shape[1] for product in existing_reviews], np.repeat(5, len(existing_reviews))
    )
    # Also need to drop the first review (represented by a 5D histogram) for each product
    # Before doing this, check if the review timeseries begins with [1, 1, 1, 1, 1]. The simulations use this
    # as the first timeseries value, and in case we start with [0, 0, 0, 0, 0], we need to add those first 5
    # ratings to every review in the timeseries
    for product in range(len(existing_reviews)):
        if np.array_equal(existing_reviews[product][0, :], np.zeros(5)):
            existing_reviews[product] += np.ones(5)[None, :]
        elif np.array_equal(existing_reviews[product][0, :], np.ones(5)):
            None
        else:
            raise ValueError(
                f"""
                For product {product}, found unexpected first histogram in review timseries:
                \n {existing_reviews[product][0, :]}
                """
            )
        existing_reviews[product] = existing_reviews[product][1:, :]
    return existing_reviews
