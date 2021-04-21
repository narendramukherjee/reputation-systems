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
