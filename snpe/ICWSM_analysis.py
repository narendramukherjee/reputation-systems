import multiprocessing as mp

from pathlib import Path
from typing import Optional

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
sns.set(style="white", context="talk", font_scale=3.5)
sns.set_color_codes(palette="colorblind")
sns.set_style("ticks", {"axes.linewidth": 2.0})
plt.ion()

ARTIFACT_PATH = Path("./artifacts/ICWSM")


def generate_and_save_simulations(
    num_simulations: int, review_prior: np.array, tendency_to_rate: float, simulation_type: str
) -> None:
    params = {"review_prior": review_prior, "tendency_to_rate": tendency_to_rate, "simulation_type": simulation_type}
    simulator = simulator_class.DoubleRhoSimulator(params)
    simulator.simulate(num_simulations=num_simulations)
    simulator.save_simulations(ARTIFACT_PATH)


def infer_and_save_posterior(device: str, simulator_type: str, simulation_type: str) -> None:
    parameter_prior = sbi_utils.BoxUniform(
        low=torch.tensor([0.0, 0.0]).type(torch.FloatTensor), high=torch.tensor([4.0, 4.0]).type(torch.FloatTensor)
    )
    inferrer = inference_class.HistogramInference(parameter_prior=parameter_prior, device=device)
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type, simulation_type=simulation_type)
    inferrer.infer_snpe_posterior()
    inferrer.save_inference(ARTIFACT_PATH)


def sample_posterior_with_observed(
    observed_histograms: np.array, num_samples: int, simulator_type: str, simulation_type: str
) -> np.array:
    # The parameter prior doesn't matter here as it will be overridden by that of the loaded inference object
    parameter_prior = sbi.utils.BoxUniform(
        low=torch.tensor([0.0, 0.0]).type(torch.FloatTensor), high=torch.tensor([4.0, 4.0]).type(torch.FloatTensor)
    )
    inferrer = inference_class.HistogramInference(parameter_prior=parameter_prior)
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type, simulation_type=simulation_type)
    inferrer.load_inference(dirname=ARTIFACT_PATH)
    posterior_samples = inferrer.get_posterior_samples(observed_histograms, num_samples=num_samples)
    return posterior_samples


def plot_simulated_histogram_variety_test(parameters: np.array) -> None:
    params = {"review_prior": np.ones(5), "tendency_to_rate": 0.05}
    simulator = simulator_class.DoubleRhoSimulator(params)
    simulator.simulation_parameters = {"rho": parameters}

    with tqdm_joblib(tqdm(desc="Simulations", total=parameters.shape[0])) as progress_bar:
        simulations = Parallel(n_jobs=mp.cpu_count())(
            delayed(simulator.simulate_review_histogram)(i) for i in range(parameters.shape[0])
        )
    simulations = np.array(simulations)

    for i in range(parameters.shape[0]):
        plt.figure()
        plt.bar(
            [1, 2, 3, 4, 5],
            simulations[i, :],
            width=0.5,
            color=sns.xkcd_rgb["grey"],
            label=r"$\rho_{-}=$" + f"{parameters[i, 0]}, " + r"$\rho_{+}=$" + f"{parameters[i, 1]}",
        )
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xlabel("Rating")
        plt.ylabel("Number of ratings")
        plt.legend(loc="upper left", fontsize=20)


def plot_simulated_vs_actual_histogram_test(
    observed_histograms: np.array,
    posterior_samples: np.array,
    products_to_test: np.array,
    plot_histograms: bool = False,
    return_raw_simulations: bool = True,
) -> np.ndarray:
    print(posterior_samples.shape)
    products_to_test = products_to_test.astype("int")
    simulated_histograms = np.zeros((posterior_samples.shape[0], len(products_to_test), 5))
    # Get the total number of reviews of the products we want to test
    # We will simulate as many reviews for each products as exist in their observed histograms
    # total_reviews = np.sum(observed_histograms[products_to_test, :], axis=1)

    params = {"review_prior": np.ones(5), "tendency_to_rate": 0.05, "simulation_type": "histogram"}
    simulator = simulator_class.DoubleRhoSimulator(params)
    # Take posterior samples of the products we want to test
    # We will simulate distributions using these posterior samples as parameters
    parameters = np.swapaxes(posterior_samples[:, products_to_test, :], 0, 1).reshape((-1, 2))
    # We need to expand total reviews to be same number as the number of simulations to be run
    # total_reviews = np.tile(total_reviews[:, None], (1, posterior_samples.shape[0])).flatten()
    simulator.simulation_parameters = {"rho": parameters}

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
    plot_posteriors: bool = False,
    get_stats: bool = False,
    param_posterior_prob_band: Optional[float] = None,
) -> np.ndarray:
    # Simulate review histograms using provided parameters
    params = {"review_prior": np.ones(5), "tendency_to_rate": 0.05, "simulation_type": simulation_type}
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
    inferrer = inference_class.HistogramInference(parameter_prior=parameter_prior)
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type, simulation_type=simulation_type)
    inferrer.load_inference(dirname=ARTIFACT_PATH)
    posterior_samples = inferrer.get_posterior_samples(simulations, num_samples=num_posterior_samples)

    # Plot the posterior samples inferred for the simulated data
    # We will plot upto 4 plots in one row of the panel
    if plot_posteriors:
        if len(parameters) <= 4:
            fig, ax = plt.subplots(1, len(parameters), squeeze=False)
        else:
            fig, ax = plt.subplots((len(parameters) + 1) // 4, 4, squeeze=False)
        row_index = 0
        for i in range(len(parameters)):
            if len(parameters) > 4:
                row_index = i // 4
            ax[row_index, i % 4].hist(
                posterior_samples[:, i, 0], color="black", alpha=0.5, bins=10, label=r"$\rho_{-}$"
            )
            ax[row_index, i % 4].axvline(x=parameters[i, 0], linewidth=3.0, color="black", linestyle="--")
            ax[row_index, i % 4].hist(posterior_samples[:, i, 1], color="red", alpha=0.5, bins=10, label=r"$\rho_{+}$")
            ax[row_index, i % 4].axvline(x=parameters[i, 1], linewidth=3.0, color="red", linestyle="--")
            ax[row_index, i % 4].set_xlim([0, 4])
            ax[row_index, i % 4].set_xticks([0, 1, 2, 3, 4])
            ax[row_index, i % 4].legend()
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
        plt.xlabel(r"$\rho_{-}, \rho_{+}$")
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
            True if (parameters[i, 0] < hpd[i, 0, 1] and parameters[i, 0] > hpd[i, 0, 0]) else False
            for i in range(parameters.shape[0])
        ]
        contained_rho_1 = [
            True if (parameters[i, 1] < hpd[i, 1, 1] and parameters[i, 1] > hpd[i, 1, 0]) else False
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
        print("=======================================================", file=f)
        # Now get the probability that the posterior distribution puts in a band/region around
        # the passed parameter values. For good parameter recovery, this number should be high
        assert (
            param_posterior_prob_band is not None
        ), f"""
        Posterior probability band around parameter values need to be passed if stats are needed
        """
        param_band_low = parameters - param_posterior_prob_band
        param_band_high = parameters + param_posterior_prob_band
        rho_0_probs = (posterior_samples[:, :, 0] >= param_band_low[None, :, 0]) * (
            posterior_samples[:, :, 0] <= param_band_high[None, :, 0]
        )
        rho_0_probs = np.mean(rho_0_probs, axis=0)
        rho_1_probs = (posterior_samples[:, :, 1] >= param_band_low[None, :, 1]) * (
            posterior_samples[:, :, 1] <= param_band_high[None, :, 1]
        )
        rho_1_probs = np.mean(rho_1_probs, axis=0)
        print(
            f"""
        In {100*np.mean(rho_0_probs>=0.5):0.2f}% of cases, the inferred posterior places more than 50% probability
        in a band of {2*param_posterior_prob_band} around the true value of rho-
        """,
            file=f,
        )
        print(
            f"""
        In {100*np.mean(rho_1_probs>=0.5):0.2f}% of cases, the inferred posterior places more than 50% probability
        in a band of {2*param_posterior_prob_band} around the true value of rho+
        """,
            file=f,
        )
        f.close()
        # Finally, plot the distribution of the posterior probability the inference engine places in a
        # band around the true value of rho- and rho+
        plt.figure()
        plt.hist(rho_0_probs, alpha=0.5, label=r"$\rho_{-}$")
        plt.hist(rho_1_probs, alpha=0.5, label=r"$\rho_{+}$")
        plt.legend()
        plt.title(
            f"Posterior probability placed by inference engine in a band of {2*param_posterior_prob_band}"
            + f"\n around the true value of the parameters ({parameters.shape[0]} trials)",
            fontsize=24.0,
        )

    return posterior_samples


def plot_mean_rho_vs_features(features: pd.DataFrame, features_to_plot: dict) -> None:
    for feature, x_label in features_to_plot.items():
        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(features[feature], features["rho_0"], alpha=0.8, s=3.0)
        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel("Posterior mean of " + r"$\rho_{-}$")
        ax[0].set_ylim([0, 4])

        ax[1].scatter(features[feature], features["rho_1"], alpha=0.8, s=3.0)
        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel("Posterior mean of " + r"$\rho_{+}$")
        ax[1].set_ylim([0, 4])

    # Also plot the distribution of mean rhos across all products
    plt.figure()
    plt.hist(features["rho_0"], color="black", bins=10, alpha=0.5, label=r"$\rho_{-}$")
    plt.hist(features["rho_1"], color="red", bins=10, alpha=0.5, label=r"$\rho_{+}$")
    plt.xlim([0, 4])
    plt.legend()


def stats_mean_rho_binary_features(features: pd.DataFrame, features_to_test: list) -> None:
    with open(ARTIFACT_PATH / "stats_binary_features.txt", "w") as f:
        for feature in features_to_test:
            print(10 * "*", f"Feature = {feature}", 10 * "*", file=f)
            print("\n", file=f)
            print("Regression results for rho_0:", file=f)
            model = smf.ols(formula=f"rho_0 ~ {feature}", data=features)
            res = model.fit()
            print(res.summary(), file=f)
            print("\n", file=f)
            print("Regression results for rho_1:", file=f)
            model = smf.ols(formula=f"rho_1 ~ {feature}", data=features)
            res = model.fit()
            print(res.summary(), file=f)
            print("\n", file=f)
            print("Independent t-test results for rho_0:", file=f)
            t, p = ttest_ind(features["rho_0"][features[feature] == "Y"], features["rho_0"][features[feature] == "N"])
            print(f"T statistic: {t}, p-value: {p}", file=f)
            print("\n", file=f)
            print("Independent t-test results for rho_1:", file=f)
            t, p = ttest_ind(features["rho_1"][features[feature] == "Y"], features["rho_1"][features[feature] == "N"])
            print(f"T statistic: {t}, p-value: {p}", file=f)


def main() -> None:
    torch.set_num_threads(16)
    torch.get_num_threads()

    # Simulate, infer posterior and save everything in the artifact directory
    generate_and_save_simulations(20_000, np.ones(5), 0.05, "histogram")
    infer_and_save_posterior("gpu", "double_rho", "histogram")

    # Load up the observed data of review histograms and product features
    ratings = pd.read_csv(ARTIFACT_PATH / "rating_hist_speakers_snpe.txt", sep="\t")
    features = pd.read_csv(ARTIFACT_PATH / "prod_price_brand_snpe.txt", sep="\t")
    # Both loaded dataframes should have the same number of products
    assert len(ratings) == len(
        features
    ), f"Loaded ratings df has shape {ratings.shape} while features has shape {features.shape}"

    # Align the ratings and features DFs to have the same order of products
    features = features.set_index("asin")
    features = features.reindex(index=ratings["asin"])
    features = features.reset_index()
    assert np.all(
        features["asin"] == ratings["asin"]
    ), f"Features and ratings DFs do not have the same ordering of products"

    # Add additional features
    features["num_reviews"] = np.array(ratings.iloc[:, 1:]).sum(axis=1)
    features["log_num_reviews"] = np.log(features["num_reviews"] + 1)
    features["log_val"] = np.log(features["val"] + 1)
    features["top_10"] = features["brand"].isin(np.array(features["brand"].value_counts()[:10].index))
    features["top_10"] = ["Y" if i else "N" for i in features["top_10"]]

    # Get samples from the posteriors of all products, using their observed review histograms
    observed_histograms = np.array(ratings.iloc[:, 1:], dtype=np.float64)
    posterior_samples = sample_posterior_with_observed(observed_histograms, 10_000, "double_rho", "histogram")

    # Produce histogram plots to test that model can generate all sorts of review distributions
    plot_simulated_histogram_variety_test(np.array([[0.0, 0.0], [1.5, 3.5], [1.0, 1.0], [3.5, 1.5]]))

    # Check parameter recovery by model for a handful of parameter values
    recovered_posterior = plot_test_parameter_recovery(
        np.array([[3.5, 1.0], [1.0, 1.0], [1.5, 2.5]]), 10_000, "double_rho", "histogram", plot_posteriors=True
    )
    # Do the same for a larger number of parameter values, but this time don't plot the posteriors
    parameters = np.vstack((np.random.random(size=1000) * 4, np.random.random(size=1000) * 4)).T
    recovered_posteriors = plot_test_parameter_recovery(
        parameters,
        10_000,
        "double_rho",
        "histogram",
        plot_posteriors=False,
        get_stats=True,
        param_posterior_prob_band=0.5,
    )

    # Compare simulated and observed histograms for a subset of products
    simulated_histograms = plot_simulated_vs_actual_histogram_test(
        observed_histograms,
        posterior_samples[::500, :, :],
        products_to_test=np.array([1, 7, 11, 500]),
        plot_histograms=True,
        return_raw_simulations=False,
    )
    # Get simulated histograms for all products
    simulated_histograms = plot_simulated_vs_actual_histogram_test(
        observed_histograms,
        posterior_samples[::500, :, :],
        products_to_test=np.arange(observed_histograms.shape[0]),
        return_raw_simulations=False,
    )
    # simulated_histograms = raw_simulations.reshape((-1, observed_histograms.shape[0], 5), order="F")
    # simulated_histograms /= np.sum(simulated_histograms, axis=-1)[:, :, None]
    np.save(ARTIFACT_PATH / "posterior_predictive_simulations.npy", simulated_histograms)
    # Then get the correlations of the mean and HPD limits of the simulated histograms with the observed
    # aka posterior predictive check (PPC)
    ppc_corr = review_histogram_correlation(observed_histograms, simulated_histograms)
    np.save(ARTIFACT_PATH / "ppc_correlations.npy", ppc_corr)

    # Add mean and sd of rho to the features DF
    features["rho_0"] = np.mean(posterior_samples[:, :, 0], axis=0)
    features["rho_1"] = np.mean(posterior_samples[:, :, 1], axis=0)
    features["sd_rho_0"] = np.std(posterior_samples[:, :, 0], axis=0)
    features["sd_rho_1"] = np.std(posterior_samples[:, :, 1], axis=0)
    # Now plot rho vs these features
    plot_mean_rho_vs_features(features, {"log_val": "log(Price in $)", "log_num_reviews": "log(Number of reviews)"})

    # Create 2 binary features: above/below mean(log_price) and Y/N top_10 brand product
    features["top_10"] = features["brand"].isin(np.array(features["brand"].value_counts()[:10].index))
    features["top_10"] = ["Y" if i else "N" for i in features["top_10"]]
    features["log_price_above_mean"] = ["Y" if i >= np.mean(features["log_val"]) else "N" for i in features["log_val"]]
    # Run statistics on these binary features
    stats_mean_rho_binary_features(features, ["top_10", "log_price_above_mean"])


if __name__ == "__main__":
    main()
