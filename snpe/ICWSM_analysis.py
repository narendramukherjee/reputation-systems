import multiprocessing as mp
from pathlib import Path

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
from snpe.utils.tqdm_utils import tqdm_joblib
from tqdm import tqdm

# Set plotting parameters
sns.set(style="white", context="talk", font_scale=3.5)
sns.set_color_codes(palette="colorblind")
sns.set_style("ticks", {"axes.linewidth": 2.0})
plt.ion()

ARTIFACT_PATH = Path("./artifacts/ICWSM")


def generate_and_save_simulations(num_simulations: int, review_prior: np.array, tendency_to_rate: float) -> None:
    params = {"review_prior": review_prior, "tendency_to_rate": tendency_to_rate}
    simulator = simulator_class.DoubleRhoSimulator(params)
    simulator.simulate(num_simulations=num_simulations)
    simulator.save_simulations(ARTIFACT_PATH)


def infer_and_save_posterior(simulator_type: str) -> None:
    parameter_prior = sbi_utils.BoxUniform(
        low=torch.tensor([0.0, 0.0]).type(torch.FloatTensor), high=torch.tensor([4.0, 4.0]).type(torch.FloatTensor)
    )
    inferrer = inference_class.BaseInference(parameter_prior=parameter_prior, device="cpu")
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type)
    inferrer.infer_snpe_posterior()
    inferrer.save_inference(ARTIFACT_PATH)


def sample_posterior_with_observed(observed_histograms: np.array, num_samples: int, simulator_type: str) -> np.array:
    # The parameter prior doesn't matter here as it will be overridden by that of the loaded inference object
    parameter_prior = sbi.utils.BoxUniform(
        low=torch.tensor([0.0, 0.0]).type(torch.FloatTensor), high=torch.tensor([4.0, 4.0]).type(torch.FloatTensor)
    )
    inferrer = inference_class.BaseInference(parameter_prior=parameter_prior)
    inferrer.load_simulator(dirname=ARTIFACT_PATH, simulator_type=simulator_type)
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
    observed_histograms: np.array, posterior_samples: np.array, products_to_test: np.array
) -> None:
    print(posterior_samples.shape)
    products_to_test = products_to_test.astype("int")
    simulated_histograms = np.zeros((posterior_samples.shape[0], len(products_to_test), 5))
    # Get the total number of reviews of the products we want to test
    # We will simulate as many reviews for each products as exist in their observed histograms
    # total_reviews = np.sum(observed_histograms[products_to_test, :], axis=1)

    params = {"review_prior": np.ones(5), "tendency_to_rate": 0.05}
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
    # Simulate, infer posterior and save everything in the artifact directory
    generate_and_save_simulations(20_000, np.ones(5), 0.05)
    infer_and_save_posterior("double_rho")

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
    posterior_samples = sample_posterior_with_observed(observed_histograms, 10_000, "double_rho")

    # Produce histogram plots to test that model can generate all sorts of review distributions
    plot_simulated_histogram_variety_test(np.array([[0.0, 0.0], [3.5, 1.0], [1.0, 1.0], [3.5, 1.5]]))

    # Check parameter recovery by model
    plot_test_parameter_recovery(np.array([[3.5, 1.0], [1.0, 1.0], [1.5, 2.5]]), 10_000, "double_rho")

    # Compare simulated and observed histograms for a subset of products
    plot_simulated_vs_actual_histogram_test(
        observed_histograms,
        posterior_samples[::20, :, :],
        products_to_test=np.random.randint(observed_histograms.shape[0], size=5),
    )

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
