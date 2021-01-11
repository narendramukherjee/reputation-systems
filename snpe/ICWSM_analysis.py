import multiprocessing as mp
from pathlib import Path

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sbi
import sbi.utils as sbi_utils
import seaborn as sns
import torch
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
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
        plt.legend()


def plot_simulated_vs_actual_histogram_test(
    observed_histograms: np.array, posterior_samples: np.array, products_to_test: np.array
) -> None:
    print(posterior_samples.shape)
    products_to_test = products_to_test.astype("int")
    simulated_histograms = np.zeros((posterior_samples.shape[0], len(products_to_test), 5))
    # Get the total number of reviews of the products we want to test
    # We will simulate as many reviews for each products as exist in their observed histograms
    total_reviews = np.sum(observed_histograms[products_to_test, :], axis=1)

    params = {"review_prior": np.ones(5), "tendency_to_rate": 0.05}
    simulator = simulator_class.DoubleRhoSimulator(params)
    # Take posterior samples of the products we want to test
    # We will simulate distributions using these posterior samples as parameters
    parameters = posterior_samples[:, products_to_test, :].reshape((-1, 2))
    # We need to expand total reviews to be same number as the number of simulations to be run
    total_reviews = np.tile(total_reviews[None, :], (posterior_samples.shape[0], 1)).flatten()
    simulator.simulation_parameters = {"rho": parameters}

    with tqdm_joblib(tqdm(desc="Simulations", total=parameters.shape[0])) as progress_bar:
        simulations = Parallel(n_jobs=mp.cpu_count())(
            delayed(simulator.simulate_review_histogram)(i, total_reviews) for i in range(parameters.shape[0])
        )
    simulations = np.array(simulations)
    simulated_histograms[:, :, :] = simulations.reshape((-1, len(products_to_test), 5))

    for i in range(len(products_to_test)):
        plt.figure()
        plt.plot(np.arange(5) + 1, observed_histograms[i, :], linewidth=4.0, color="black")
        # Get the HPDs of the simulated histograms
        hpd = arviz.hdi(simulated_histograms[:, i, :], hdi_prob=0.95)
        plt.fill_between(np.arange(5) + 1, hpd[:, 0], hpd[:, 1], color="black", alpha=0.4)


def main():
    # Simulate, infer posterior and save everything in the artifact directory
    generate_and_save_simulations(15_000, np.ones(5), 0.05)
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
    posterior_samples = sample_posterior_with_observed(observed_histograms, 5_000, "double_rho")

    # Produce histogram plots to test that model can generate all sorts of review distributions
    plot_simulated_histogram_variety_test(np.array([[0.0, 0.0], [3.5, 1.0], [1.0, 1.0], [3.5, 1.5]]))

    plot_simulated_vs_actual_histogram_test(
        observed_histograms, posterior_samples[::10, :, :], products_to_test=np.arange(5)
    )


if __name__ == "__main__":
    main()
