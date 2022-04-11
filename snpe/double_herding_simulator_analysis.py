import multiprocessing as mp

from collections import deque
from pathlib import Path
from typing import Dict, Optional

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
from snpe.utils.statistics import review_histogram_correlation
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
    herding_differentiating_measure: str,
) -> None:
    params = {
        "review_prior": review_prior,
        "tendency_to_rate": tendency_to_rate,
        "simulation_type": simulation_type,
        "previous_rating_measure": previous_rating_measure,
        "min_reviews_for_herding": min_reviews_for_herding,
        "herding_differentiating_measure": herding_differentiating_measure,
    }
    simulator = simulator_class.DoubleHerdingSimulator(params)
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
    plot_histograms: bool = False,
    return_raw_simulations: bool = True,
) -> np.ndarray:
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

    if plot_histograms:
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

    if return_raw_simulations:
        return simulations
    else:
        return simulated_histograms


def plot_test_parameter_recovery(
    parameters: np.array,
    num_posterior_samples: int,
    simulator_type: str,
    simulation_type: str,
    previous_rating_measure: str,
    min_reviews_for_herding: int,
    plot_posteriors: bool = False,
    get_stats: bool = False,
    rho_posterior_prob_band: Optional[float] = None,
    herding_posterior_prob_band: Optional[float] = None,
) -> np.ndarray:
    # Simulate review histograms using provided parameters
    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": simulation_type,
        "previous_rating_measure": previous_rating_measure,
        "min_reviews_for_herding": min_reviews_for_herding,
    }
    simulator = simulator_class.HerdingSimulator(params)
    simulator.simulation_parameters = {"rho": parameters[:, :2], "h_p": parameters[:, 2]}
    with tqdm_joblib(tqdm(desc="Simulations", total=parameters.shape[0])) as progress_bar:
        simulations = Parallel(n_jobs=mp.cpu_count())(
            delayed(simulator.simulate_review_histogram)(i) for i in range(parameters.shape[0])
        )
    simulations = np.array(simulations)

    # The parameter prior doesn't matter here as it will be overridden by that of the loaded inference object
    parameter_prior = sbi.utils.BoxUniform(
        low=torch.tensor([0.0, 0.0, 0.0]).type(torch.FloatTensor),
        high=torch.tensor([4.0, 4.0, 1.0]).type(torch.FloatTensor),
    )
    inferrer = inference_class.TimeSeriesInference(parameter_prior=parameter_prior)
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type, simulation_type=simulation_type)
    inferrer.load_inference(dirname=ARTIFACT_PATH)
    posterior_samples = inferrer.get_posterior_samples(simulations, num_samples=num_posterior_samples)

    # Plot the posterior samples inferred for the simulated data
    # We will plot upto 4 plots in one row of the panel
    if plot_posteriors:
        if len(parameters) <= 4:
            fig, ax = plt.subplots(2, len(parameters), squeeze=False)
        else:
            fig, ax = plt.subplots(2 * ((len(parameters) + 1) // 4), 4, squeeze=False)
        row_index = 0
        for i in range(len(parameters)):
            if len(parameters) > 4:
                row_index = 2 * (i // 4)
            ax[row_index, i % 4].hist(
                posterior_samples[:, i, 0], color=sns.xkcd_rgb["cerulean"], alpha=0.5, bins=10, label=r"$\rho_{-}$"
            )
            ax[row_index, i % 4].axvline(
                x=parameters[i, 0], linewidth=3.0, color=sns.xkcd_rgb["cerulean"], linestyle="--"
            )
            ax[row_index, i % 4].hist(
                posterior_samples[:, i, 1], color=sns.xkcd_rgb["dark orange"], alpha=0.5, bins=10, label=r"$\rho_{+}$"
            )
            ax[row_index, i % 4].axvline(
                x=parameters[i, 1], linewidth=3.0, color=sns.xkcd_rgb["dark orange"], linestyle="--"
            )
            ax[row_index + 1, i % 4].hist(
                posterior_samples[:, i, 2], color=sns.xkcd_rgb["black"], alpha=0.5, bins=10, label=r"$h_{p}$"
            )
            ax[row_index + 1, i % 4].axvline(
                x=parameters[i, 2], linewidth=3.0, color=sns.xkcd_rgb["black"], linestyle="--"
            )
            ax[row_index, i % 4].set_xlim([0, 4])
            ax[row_index + 1, i % 4].set_xlim([0, 1])
            ax[row_index, i % 4].set_xticks([0, 1, 2, 3, 4])
            ax[row_index + 1, i % 4].set_xticks([0, 0.5, 1])
            ax[row_index + 1, i % 4].set_xticklabels(["0", "0.5", "1"])
            ax[row_index, i % 4].tick_params(axis="y", labelsize=17)
            ax[row_index + 1, i % 4].tick_params(axis="y", labelsize=17)
            ax[row_index, i % 4].legend(fontsize=20)
            ax[row_index + 1, i % 4].legend(fontsize=20)
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
        # plt.xlabel(r"$\rho_{-}, \rho_{+}$")
        plt.ylabel("Number of samples")

    # If asked, print how many of the provided parameters are recovered by the inference engine
    # i.e, how often do the supplied parameters lie within the 95% HPD of the posterior
    if get_stats:
        f = open(ARTIFACT_PATH / "stats_parameter_recovery.txt", "w")
        assert (
            posterior_samples.shape == (num_posterior_samples,) + parameters.shape
        ), f"""
        Expected shape {(num_posterior_samples,) + parameters.shape} for array of posterior samples,
        but got {posterior_samples.shape} instead
        """
        # First get the HPD of each recovered posterior distribution
        hpd = np.array([arviz.hdi(posterior_samples[:, i, :], hdi_prob=0.95) for i in range(parameters.shape[0])])
        assert hpd.shape == parameters.shape + (2,), f"Found shape {hpd.shape} for hpd"
        # See how many of the supplied rho_- and rho_+ are contained in these HPDs
        contained_rho_0 = [
            True if (parameters[i, 0] <= hpd[i, 0, 1] and parameters[i, 0] >= hpd[i, 0, 0]) else False
            for i in range(parameters.shape[0])
        ]
        contained_rho_1 = [
            True if (parameters[i, 1] <= hpd[i, 1, 1] and parameters[i, 1] >= hpd[i, 1, 0]) else False
            for i in range(parameters.shape[0])
        ]
        contained_h_p = [
            True if (parameters[i, 2] <= hpd[i, 2, 1] and parameters[i, 2] >= hpd[i, 2, 0]) else False
            for i in range(parameters.shape[0])
        ]
        print(
            f"""
        rho- is recovered {np.sum(contained_rho_0)} times out of {parameters.shape[0]}
        = {100*(np.sum(contained_rho_0) / parameters.shape[0]):0.2f}%"
        """,
            file=f,
        )
        print(
            f"""
        rho+ is recovered {np.sum(contained_rho_1)} times out of {parameters.shape[0]}
        = {100*(np.sum(contained_rho_1) / parameters.shape[0]):0.2f}%"
        """,
            file=f,
        )
        print(
            f"""
        Herding parameter h_p is recovered {np.sum(contained_h_p)} times out of {parameters.shape[0]}
        = {100*(np.sum(contained_h_p) / parameters.shape[0]):0.2f}%"
        """,
            file=f,
        )
        print("=======================================================", file=f)
        # Now get the probability that the posterior distribution puts in a band/region around
        # the passed parameter values. For good parameter recovery, this number should be high
        assert (
            rho_posterior_prob_band is not None and herding_posterior_prob_band is not None
        ), f"""
        Posterior probability band around parameter values need to be passed if stats are needed
        """
        rho_band_low = parameters[:, :2] - rho_posterior_prob_band
        rho_band_high = parameters[:, :2] + rho_posterior_prob_band
        h_p_band_low = parameters[:, 2] - herding_posterior_prob_band
        h_p_band_high = parameters[:, 2] + herding_posterior_prob_band
        rho_0_probs = (posterior_samples[:, :, 0] >= rho_band_low[None, :, 0]) * (
            posterior_samples[:, :, 0] <= rho_band_high[None, :, 0]
        )
        rho_0_probs = np.mean(rho_0_probs, axis=0)
        rho_1_probs = (posterior_samples[:, :, 1] >= rho_band_low[None, :, 1]) * (
            posterior_samples[:, :, 1] <= rho_band_high[None, :, 1]
        )
        rho_1_probs = np.mean(rho_1_probs, axis=0)
        h_p_probs = (posterior_samples[:, :, 2] >= h_p_band_low[None, :]) * (
            posterior_samples[:, :, 2] <= h_p_band_high[None, :]
        )
        h_p_probs = np.mean(h_p_probs, axis=0)
        print(
            f"""
        In {100*np.mean(rho_0_probs>=0.5):0.2f}% of cases, the inferred posterior places more than 50% probability
        in a band of {2*rho_posterior_prob_band} around the true value of rho-
        """,
            file=f,
        )
        print(
            f"""
        In {100*np.mean(rho_1_probs>=0.5):0.2f}% of cases, the inferred posterior places more than 50% probability
        in a band of {2*rho_posterior_prob_band} around the true value of rho+
        """,
            file=f,
        )
        print(
            f"""
        In {100*np.mean(h_p_probs>=0.5):0.2f}% of cases, the inferred posterior places more than 50% probability
        in a band of {2*herding_posterior_prob_band} around the true value of herding parameter h_p
        """,
            file=f,
        )
        f.close()
        # Finally, plot the distribution of the posterior probability the inference engine places in a
        # band around the true value of rho- and rho+
        plt.figure()
        plt.hist(rho_0_probs, alpha=0.5, label=r"$\rho_{-}$")
        plt.hist(rho_1_probs, alpha=0.5, label=r"$\rho_{+}$")
        plt.hist(h_p_probs, alpha=0.5, label=r"$h_{p}$")
        plt.legend(fontsize=20)

    return posterior_samples


def plot_mean_posteriors_for_products(posterior_samples: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, squeeze=False)
    ax[0, 0].hist(
        np.mean(posterior_samples[:, :, 0], axis=0),
        color=sns.xkcd_rgb["cerulean"],
        alpha=0.5,
        bins=10,
        label=r"$\rho_{-}$",
    )
    ax[0, 0].hist(
        np.mean(posterior_samples[:, :, 1], axis=0),
        color=sns.xkcd_rgb["dark orange"],
        alpha=0.5,
        bins=10,
        label=r"$\rho_{+}$",
    )
    ax[0, 1].hist(
        np.mean(posterior_samples[:, :, 2], axis=0), color=sns.xkcd_rgb["black"], alpha=0.5, bins=10, label=r"$h_p$"
    )
    ax[0, 0].legend(fontsize=20)
    ax[0, 1].legend(fontsize=20)
    ax[0, 0].set_xticks([0, 1, 2, 3, 4])
    ax[0, 1].set_xticks([0, 0.5, 1])
    ax[0, 0].tick_params(axis="both", labelsize=23)
    ax[0, 1].tick_params(axis="both", labelsize=23)
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    # plt.xlabel(r"$\rho_{-}, \rho_{+}$")
    plt.ylabel(f"Number of products (Total = {posterior_samples.shape[1]})", fontsize=28)


def main() -> None:
    torch.set_num_threads(mp.cpu_count())
    torch.get_num_threads()

    # Simulate, infer posterior and save everything in the artifact directory
    generate_and_save_simulations(20_000, np.ones(5), 0.05, "timeseries", "mode", 5, "mean")
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
