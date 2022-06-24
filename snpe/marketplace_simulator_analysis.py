import itertools
import multiprocessing as mp
import pickle

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
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from snpe.inference import inference_class
from snpe.simulations import marketplace_simulator_class, simulator_class
from snpe.utils.statistics import review_histogram_correlation
from snpe.utils.tqdm_utils import tqdm_joblib
from tqdm import tqdm

# Set plotting parameters
sns.set(style="white", context="talk", font_scale=2.5)
sns.set_color_codes(palette="colorblind")
sns.set_style("ticks", {"axes.linewidth": 2.0})
plt.ion()

ARTIFACT_PATH = Path("/data/reputation-systems/snpe/artifacts/marketplace")


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


def plot_simulated_histogram_variety_test(
    simulations: np.ndarray, simulation_parameters: np.ndarray, params_to_plot: np.ndarray
) -> None:
    # This method searches in the array of simulation_parameters to find the parameter set that is
    # closest in Euclidean space to the parameter combination we want to plot
    distances = cdist(params_to_plot, simulation_parameters, metric="euclidean")
    sim_idx = np.argmin(distances, axis=1).astype("int")

    if len(params_to_plot) <= 4:
        fig, ax = plt.subplots(len(params_to_plot), squeeze=False)
    else:
        fig, ax = plt.subplots((len(params_to_plot) + 1) // 4, 4, squeeze=False)

    for i, idx in enumerate(sim_idx):
        row_index = i // 4
        ax[row_index, i % 4].bar(
            [1, 2, 3, 4, 5],
            simulations[idx][-1, :] / np.sum(simulations[idx][-1, :]),
            width=0.5,
            color=sns.xkcd_rgb["grey"],
            label=r"$\rho_{-}=$"
            + f"{simulation_parameters[idx, 0]:0.2f}, "
            + "\n"
            + r"$\rho_{+}=$"
            + f"{simulation_parameters[idx, 1]:0.2f}, "
            + "\n"
            + r"$h_{p}=$"
            + f"{simulation_parameters[idx, 2]:0.2f}",
        )
        ax[row_index, i % 4].spines["top"].set_visible(False)
        ax[row_index, i % 4].spines["right"].set_visible(False)
        ax[row_index, i % 4].legend(fontsize=15)
        # If this is the bottom row of plots, add the xticks, otherwise hide them
        if row_index == (len(sim_idx) - 1) // 4:
            ax[row_index, i % 4].set_xticks([1, 2, 3, 4, 5])
            ax[row_index, i % 4].set_xticklabels(["1", "2", "3", "4", "5"])
            ax[row_index, i % 4].tick_params(axis="x", labelsize=20)
        else:
            ax[row_index, i % 4].set_xticks([1, 2, 3, 4, 5])
            ax[row_index, i % 4].set_xticklabels([])
        # If this is the leftmost plot, add y-axis labels, otherwise hide them
        ax[row_index, i % 4].set_ylim([0, 1])
        if i % 4 == 0:
            ax[row_index, i % 4].set_yticks([0, 0.5, 1])
            ax[row_index, i % 4].set_yticklabels(["0", "0.5", "1"])
        else:
            ax[row_index, i % 4].set_yticklabels([])

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    # plt.xlabel(r"$\rho_{-}, \rho_{+}$")
    plt.xlabel("Star Rating")
    plt.ylabel("Fraction of total reviews")


def plot_simulated_vs_actual_total_reviews(simulations: np.ndarray, observations: pd.DataFrame) -> None:
    # Plot the distribution of the total number of reviews per product in a separate figure
    plt.figure()
    total_num_reviews = np.array([np.sum(sim[-1]) for sim in simulations])
    bins = np.histogram_bin_edges(total_num_reviews, bins='auto')
    plt.hist(total_num_reviews, color=sns.xkcd_rgb["cerulean"], alpha=0.5, label="Simulated", density=True, bins=bins)
    obs_total_num_reviews = np.array(observations.asin.value_counts())
    plt.hist(obs_total_num_reviews, color=sns.xkcd_rgb["dark orange"], alpha=0.5, label="Actual", density=True, bins=bins)
    plt.ylabel(f"Probability density")
    plt.xlabel("Total number of reviews per product")
    plt.legend(fontsize=27)


def plot_test_parameter_recovery(
    posterior_samples: np.ndarray,
    simulation_parameters: dict,
    simulation_idx_to_plot: Optional[np.ndarray] = None,
    plot_posteriors: bool = False,
    get_stats: bool = False,
    rho_posterior_prob_band: Optional[float] = None,
    herding_posterior_prob_band: Optional[float] = None,
) -> None:
    # Plot the posterior samples inferred for the simulated data
    # We will plot upto 4 plots in one row of the panel
    if plot_posteriors:
        assert simulation_idx_to_plot is not None, "Simulation IDs to be plotted needed for posterior plotting"
        if len(simulation_idx_to_plot) <= 4:
            fig, ax = plt.subplots(2, len(simulation_idx_to_plot), squeeze=False)
        else:
            fig, ax = plt.subplots(2 * ((len(simulation_idx_to_plot) + 1) // 4), 4, squeeze=False)
        row_index = 0
        for i in range(len(simulation_idx_to_plot)):
            if len(simulation_idx_to_plot) > 4:
                row_index = 2 * (i // 4)
            ax[row_index, i % 4].hist(
                posterior_samples[:, simulation_idx_to_plot[i], 0],
                color=sns.xkcd_rgb["cerulean"],
                alpha=0.5,
                bins=10,
                label=r"$\rho_{-}$",
            )
            ax[row_index, i % 4].axvline(
                x=simulation_parameters[simulation_idx_to_plot[i], 0],
                linewidth=3.0,
                color=sns.xkcd_rgb["cerulean"],
                linestyle="--",
            )
            ax[row_index, i % 4].hist(
                posterior_samples[:, simulation_idx_to_plot[i], 1],
                color=sns.xkcd_rgb["dark orange"],
                alpha=0.5,
                bins=10,
                label=r"$\rho_{+}$",
            )
            ax[row_index, i % 4].axvline(
                x=simulation_parameters[simulation_idx_to_plot[i], 1],
                linewidth=3.0,
                color=sns.xkcd_rgb["dark orange"],
                linestyle="--",
            )
            ax[row_index + 1, i % 4].hist(
                posterior_samples[:, simulation_idx_to_plot[i], 2],
                color=sns.xkcd_rgb["black"],
                alpha=0.5,
                bins=10,
                label=r"$h_{p}$",
            )
            ax[row_index + 1, i % 4].axvline(
                x=simulation_parameters[simulation_idx_to_plot[i], 2],
                linewidth=3.0,
                color=sns.xkcd_rgb["black"],
                linestyle="--",
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
        # First get the HPD of each recovered posterior distribution
        hpd = np.array(
            [arviz.hdi(posterior_samples[:, i, :], hdi_prob=0.95) for i in range(simulation_parameters.shape[0])]
        )
        assert hpd.shape == simulation_parameters.shape + (2,), f"Found shape {hpd.shape} for hpd"
        # See how many of the supplied rho_- and rho_+ are contained in these HPDs
        contained_rho_0 = [
            True
            if (simulation_parameters[i, 0] <= hpd[i, 0, 1] and simulation_parameters[i, 0] >= hpd[i, 0, 0])
            else False
            for i in range(simulation_parameters.shape[0])
        ]
        contained_rho_1 = [
            True
            if (simulation_parameters[i, 1] <= hpd[i, 1, 1] and simulation_parameters[i, 1] >= hpd[i, 1, 0])
            else False
            for i in range(simulation_parameters.shape[0])
        ]
        contained_h_p = [
            True
            if (simulation_parameters[i, 2] <= hpd[i, 2, 1] and simulation_parameters[i, 2] >= hpd[i, 2, 0])
            else False
            for i in range(simulation_parameters.shape[0])
        ]
        print(
            f"""
        rho- is recovered {np.sum(contained_rho_0)} times out of {simulation_parameters.shape[0]}
        = {100*(np.sum(contained_rho_0) / simulation_parameters.shape[0]):0.2f}%"
        """,
            file=f,
        )
        print(
            f"""
        rho+ is recovered {np.sum(contained_rho_1)} times out of {simulation_parameters.shape[0]}
        = {100*(np.sum(contained_rho_1) / simulation_parameters.shape[0]):0.2f}%"
        """,
            file=f,
        )
        print(
            f"""
        Herding parameter h_p is recovered {np.sum(contained_h_p)} times out of {simulation_parameters.shape[0]}
        = {100*(np.sum(contained_h_p) / simulation_parameters.shape[0]):0.2f}%"
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
        rho_band_low = simulation_parameters[:, :2] - rho_posterior_prob_band
        rho_band_high = simulation_parameters[:, :2] + rho_posterior_prob_band
        h_p_band_low = simulation_parameters[:, 2] - herding_posterior_prob_band
        h_p_band_high = simulation_parameters[:, 2] + herding_posterior_prob_band
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


def main() -> None:
    torch.set_num_threads(16)
    torch.get_num_threads()

    # Load posterior samples (everything before this would ideally have been done in the ipynb notebook on Google cloud)
    posterior_samples = np.load(ARTIFACT_PATH / "posterior_samples_bazaarvoice.npy")
    # Plot the distribution of the mean of the posteriors across products
    plot_mean_posteriors_for_products(posterior_samples)

    # Load up the simulator and use the simulations within it to plot a few to show the variety of histograms
    # that can be generated in the simulations
    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "timeseries",
        "previous_rating_measure": "mode",
        "min_reviews_for_herding": 5,
        "num_products": 1400,
        "num_total_marketplace_reviews": 300_000,
        "consideration_set_size": 5,
    }
    simulator = marketplace_simulator_class.MarketplaceSimulator(params)
    simulator.load_simulator(ARTIFACT_PATH)
    simulations = np.array(list(itertools.chain.from_iterable(simulator.simulations)), dtype=object)
    simulation_parameters = np.concatenate(
        (simulator.simulation_parameters["rho"], simulator.simulation_parameters["h_p"][:, None]), axis=1
    )
    # Produce histogram plots to test that simulation model can generate all sorts of review distributions
    plot_simulated_histogram_variety_test(
        simulations,
        simulation_parameters,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 3.5, 0.0],
                [1.0, 1.0, 0.0],
                [3.5, 1.5, 0.0],
                [0.0, 0.0, 0.3],
                [1.5, 3.5, 0.4],
                [1.0, 1.0, 0.5],
                [3.5, 1.5, 0.8],
            ]
        ),
    )
    # Load up the real data and plot hist of total number of reviews per product to show
    # that it looks like what is produced in the simulation
    reviews = pyreadr.read_r(ARTIFACT_PATH / "reviews_bazaarvoice_main_vars.Rds")
    reviews = reviews[None]
    plot_simulated_vs_actual_total_reviews(simulations, reviews)
    # Next load up the simulations done from the posteriors of real products, and compare those to
    # the real data
    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "timeseries",
        "previous_rating_measure": "mode",
        "min_reviews_for_herding": 5,
        "num_products": 1400,
        "num_total_marketplace_reviews": 140_000,
        "consideration_set_size": 5,
    }
    simulator = marketplace_simulator_class.MarketplaceSimulator(params)
    simulator.load_simulator(ARTIFACT_PATH / "simulations_from_posterior")
    simulations = np.array(list(itertools.chain.from_iterable(simulator.simulations)), dtype=object)
    simulation_parameters = np.concatenate(
        (simulator.simulation_parameters["rho"], simulator.simulation_parameters["h_p"][:, None]), axis=1
    )
    plot_simulated_vs_actual_total_reviews(simulations, reviews)

    # Load up the posterior samples for simulations to look at parameter recovery
    with open(ARTIFACT_PATH / "posterior_inference_on_simulations.pkl", "rb") as f:
        posterior_inference_on_simulations = pickle.load(f)
    # Pull out the simulation parameters and the posterior samples for the simulations from the dict
    # These should not be the same as the ones used above as this test needs to be done on a separate
    # set of marketplace simulations that has not been used during SNPE training
    posterior_samples_simulation = posterior_inference_on_simulations["posterior_simulations"]
    simulation_parameters = posterior_inference_on_simulations["simulation_parameters"]
    simulation_parameters = np.concatenate(
        (simulation_parameters["rho"], simulation_parameters["h_p"][:, None]), axis=1
    )
    # Now look at the parameter recovery for a few simulations
    plot_test_parameter_recovery(
        posterior_samples_simulation,
        simulation_parameters,
        np.array([0, 789]).astype("int"),
        plot_posteriors=True,
        get_stats=False,
    )
    # Next get the statistics on parameter recovery across all the simulations in the marketplace
    plot_test_parameter_recovery(
        posterior_samples_simulation,
        simulation_parameters,
        np.array([0, 789]).astype("int"),
        plot_posteriors=False,
        get_stats=True,
        rho_posterior_prob_band=0.5,
        herding_posterior_prob_band=0.125,
    )
