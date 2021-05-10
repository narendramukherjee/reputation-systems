import multiprocessing as mp

from collections import deque
from pathlib import Path
from typing import Dict

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import sbi
import sbi.utils as sbi_utils
import seaborn as sns
import statsmodels.formula.api as smf
import torch

from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from snpe.inference import inference_class
from snpe.simulations import simulator_class
from snpe.utils.tqdm_utils import tqdm_joblib
from tqdm import tqdm

# Set plotting parameters
sns.set(style="white", context="talk", font_scale=2.5)
sns.set_color_codes(palette="colorblind")
sns.set_style("ticks", {"axes.linewidth": 2.0})
plt.ion()

ARTIFACT_PATH = Path("./artifacts/hyperparameter_tuning/inference_with_herding")


def generate_and_save_simulations(
    num_simulations: int,
    review_prior: np.array,
    tendency_to_rate: float,
    simulation_type: str,
    previous_rating_measure: str,
    min_reviews_for_herding: int,
) -> None:
    params = {
        "review_prior": review_prior,
        "tendency_to_rate": tendency_to_rate,
        "simulation_type": simulation_type,
        "previous_rating_measure": previous_rating_measure,
        "min_reviews_for_herding": min_reviews_for_herding,
    }
    simulator = simulator_class.HerdingSimulator(params)
    simulator.simulate(num_simulations=num_simulations)
    simulator.save_simulations(ARTIFACT_PATH)


def infer_and_save_posterior(device: str, simulator_type: str, simulation_type: str, params: Dict) -> None:
    parameter_prior = sbi_utils.BoxUniform(
        low=torch.tensor([0.0, 0.0, 0.0]).type(torch.FloatTensor),
        high=torch.tensor([4.0, 4.0, 1.0]).type(torch.FloatTensor),
    )
    inferrer = inference_class.TimeSeriesInference(parameter_prior=parameter_prior, device=device)
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type, simulation_type=simulation_type)
    batch_size = params.pop("batch_size")
    learning_rate = params.pop("learning_rate")
    hidden_features = params.pop("hidden_features")
    num_transforms = params.pop("num_transforms")
    inferrer.infer_snpe_posterior(
        embedding_net_conf=params,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
    )
    inferrer.save_inference(ARTIFACT_PATH)


def sample_posterior_with_observed(
    observations: np.array, num_samples: int, simulator_type: str, simulation_type: str
) -> np.array:
    # The parameter prior doesn't matter here as it will be overridden by that of the loaded inference object
    parameter_prior = sbi.utils.BoxUniform(
        low=torch.tensor([0.0, 0.0, 0.0]).type(torch.FloatTensor),
        high=torch.tensor([4.0, 4.0, 1.0]).type(torch.FloatTensor),
    )
    inferrer = inference_class.TimeSeriesInference(parameter_prior=parameter_prior)
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type, simulation_type=simulation_type)
    inferrer.load_inference(dirname=ARTIFACT_PATH)
    posterior_samples = inferrer.get_posterior_samples(observations, num_samples=num_samples)
    return posterior_samples


def plot_simulated_vs_actual_histogram_test(
    observed_histograms: np.array,
    posterior_samples: np.array,
    products_to_test: np.array,
    previous_rating_measure: str,
    min_reviews_for_herding: int,
) -> None:
    print(posterior_samples.shape)
    products_to_test = products_to_test.astype("int")
    simulated_histograms = np.zeros((posterior_samples.shape[0], len(products_to_test), 5))
    # Get the total number of reviews of the products we want to test
    # We will simulate as many reviews for each products as exist in their observed histograms
    # total_reviews = np.sum(observed_histograms[products_to_test, :], axis=1)

    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "histogram",
        "previous_rating_measure": previous_rating_measure,
        "min_reviews_for_herding": min_reviews_for_herding,
    }
    simulator = simulator_class.HerdingSimulator(params)
    # Take posterior samples of the products we want to test
    # We will simulate distributions using these posterior samples as parameters
    parameters = np.swapaxes(posterior_samples[:, products_to_test, :], 0, 1).reshape((-1, 3))
    # We need to expand total reviews to be same number as the number of simulations to be run
    # total_reviews = np.tile(total_reviews[:, None], (1, posterior_samples.shape[0])).flatten()
    simulator.simulation_parameters = {"rho": parameters[:, :2], "h_p": parameters[:, 2]}

    with tqdm_joblib(tqdm(desc="Simulations", total=parameters.shape[0])) as progress_bar:
        simulations = Parallel(n_jobs=mp.cpu_count())(
            delayed(simulator.simulate_review_histogram)(i) for i in range(parameters.shape[0])
        )
    simulations = np.array(simulations)
    simulated_histograms[:, :, :] = simulations.reshape((-1, len(products_to_test), 5), order="F")
    simulated_histograms /= np.sum(simulated_histograms, axis=-1)[:, :, None]

    for i in range(len(products_to_test)):
        plt.figure()
        plt.plot(
            np.arange(5) + 1,
            observed_histograms[i, :] / np.sum(observed_histograms[i, :]),
            linewidth=4.0,
            color="black",
        )
        # Get the HPDs of the simulated histograms
        hpd = arviz.hdi(simulated_histograms[:, i, :], hdi_prob=0.95)
        plt.fill_between(np.arange(5) + 1, hpd[:, 0], hpd[:, 1], color="black", alpha=0.4)
        plt.ylim([0, 1])


def plot_test_parameter_recovery(parameters: np.array, num_posterior_samples: int, simulator_type: str) -> None:
    # Simulate review histograms using provided parameters
    params = {"review_prior": np.ones(5), "tendency_to_rate": 0.05}
    simulator = simulator_class.DoubleRhoSimulator(params)
    simulator.simulation_parameters = {"rho": parameters}
    with tqdm_joblib(tqdm(desc="Simulations", total=parameters.shape[0])) as progress_bar:
        simulations = Parallel(n_jobs=mp.cpu_count())(
            delayed(simulator.simulate_review_histogram)(i) for i in range(parameters.shape[0])
        )
    simulations = np.array(simulations)

    # The parameter prior doesn't matter here as it will be overridden by that of the loaded inference object
    parameter_prior = sbi.utils.BoxUniform(
        low=torch.tensor([0.0, 0.0]).type(torch.FloatTensor), high=torch.tensor([4.0, 4.0]).type(torch.FloatTensor)
    )
    inferrer = inference_class.BaseInference(parameter_prior=parameter_prior)
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type)
    inferrer.load_inference(dirname=ARTIFACT_PATH)
    posterior_samples = inferrer.get_posterior_samples(simulations, num_samples=num_posterior_samples)

    # Plot the posterior samples inferred for the simulated data
    # We will plot upto 4 plots in one row of the panel
    if len(parameters) <= 4:
        fig, ax = plt.subplots(1, len(parameters), squeeze=False)
    else:
        fig, ax = plt.subplots((len(parameters) + 1) // 4, 4, squeeze=False)
    row_index = 0
    for i in range(len(parameters)):
        if len(parameters) > 4:
            row_index = i // 4
        ax[row_index, i % 4].hist(posterior_samples[:, i, 0], color="black", alpha=0.5, bins=10, label=r"$\rho_{-}$")
        ax[row_index, i % 4].axvline(x=parameters[i, 0], linewidth=3.0, color="black", linestyle="--")
        ax[row_index, i % 4].hist(posterior_samples[:, i, 1], color="red", alpha=0.5, bins=10, label=r"$\rho_{+}$")
        ax[row_index, i % 4].axvline(x=parameters[i, 1], linewidth=3.0, color="red", linestyle="--")
        ax[row_index, i % 4].set_xlim([0, 4])
        ax[row_index, i % 4].set_xticks([0, 1, 2, 3, 4])
        ax[row_index, i % 4].legend()


def main() -> None:
    torch.set_num_threads(8)
    torch.get_num_threads()

    # Simulate, infer posterior and save everything in the artifact directory
    generate_and_save_simulations(20_000, np.ones(5), 0.05, "timeseries", "mode", 5)
    inference_params = {
        "batch_size": 128,
        "learning_rate": 1.25e-3,
        "hidden_features": 50,
        "num_transforms": 5,
        "num_conv_layers": 3,
        "num_channels": 7,
        "conv_kernel_size": 3,
        "maxpool_kernel_size": 5,
        "num_dense_layers": 2,
    }
    infer_and_save_posterior("gpu", "herding", "timeseries", inference_params)

    # Load up the observed data of review timeseries and product features
    speakers_timeseries = pyreadr.read_r("./artifacts/hyperparameter_tuning/cnn_tuning/rating_speakers_SNPE.Rds")
    speakers_timeseries = speakers_timeseries[None]
    speakers_features = pd.read_csv(ARTIFACT_PATH / "speakers_prod_price_brand_snpe.txt", sep="\t")
    # Both loaded dataframes should have the same number of products
    assert speakers_timeseries.asin.unique().shape[0] == len(
        speakers_features
    ), f"""
    Loaded timeseries df has {speakers_timeseries.asin.unique().shape[0]} products
    while features has shape {speakers_features.shape}
    """

    # Check that both DFs are sorted properly
    pd.testing.assert_frame_equal(speakers_timeseries, speakers_timeseries.sort_values(["asin", "unixReviewTime"]))
    pd.testing.assert_frame_equal(speakers_features, speakers_features.sort_values("asin"))

    # Pull out the ratings from the timeseries DF and convert them into a format
    # that can be fed into the inference engine
    timeseries_data = []
    for product in speakers_features.asin:
        # We only simulate upto 5001 reviews, so we cut off all observed timeseries at that number as well
        timeseries = deque([np.zeros(5)], maxlen=5001)
        ratings = np.array(speakers_timeseries.loc[speakers_timeseries.asin == product, "overall"])
        ratings = ratings[:5000]
        for rating in ratings:
            current_histogram = timeseries[-1].copy()
            current_histogram[int(rating - 1)] += 1
            timeseries.append(current_histogram)
        timeseries_data.append(np.array(timeseries))
    timeseries_data = np.array(timeseries_data, dtype="object")

    # Get samples from the posteriors
    posterior_samples = sample_posterior_with_observed(timeseries_data, 10_000, "herding", "timeseries")

    # To do posterior predictive checks on histograms, first get histograms from timeseries data
    histogram_data = np.array([ts[-1] for ts in timeseries_data])
    assert (
        histogram_data.shape[1] == 5
    ), f"Should have shape 5 on axis 1 of histograms, got shape {histogram_data.shape} instead"

    plot_simulated_vs_actual_histogram_test(
        histogram_data,
        posterior_samples,
        products_to_test=np.array([1, 7, 11, 500]),
        previous_rating_measure="mode",
        min_reviews_for_herding=5,
    )
