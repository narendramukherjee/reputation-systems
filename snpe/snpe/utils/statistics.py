"""
Statistical/metric calculation related utils
"""

import arviz
import numpy as np

from scipy.stats import pearsonr


def dirichlet_kl_divergence() -> np.ndarray:
    # Calculates the KL divergences between 2 sets of dirichlet distributions describing review histograms
    raise NotImplementedError


def review_histogram_correlation(observed_histograms: np.ndarray, simulated_histograms: np.ndarray) -> np.ndarray:
    # Calculates the pearson/linear correlation between observed and simulated review histograms
    # Each histogram is 5 numbers (1 for each rating) - this calculates the correlation between those 5
    # numbers in the observed and simulated histograms
    # Calculates 3 corr. coeffs. in each comparison, using the mean, and the 95% HPD limits of the
    # simulated histograms respectively
    assert (
        observed_histograms.shape[0] == simulated_histograms.shape[1]
    ), f"""
    Observed histograms have {observed_histograms.shape[0]} products
    while simulated histograms have {simulated_histograms.shape[1]} products. Need to be equal
    """
    assert (
        observed_histograms.shape[1] == 5
    ), f"Observed review histograms need to be 5D, found shape {observed_histograms.shape} instead"
    assert (
        simulated_histograms.shape[2] == 5
    ), f"Simulated review histograms need to be 5D, found shape {simulated_histograms.shape} instead"
    # Calculate mean and 95% HPD of the simulated histograms
    simulation_mean = np.mean(simulated_histograms, axis=0)
    assert (
        observed_histograms.shape == simulation_mean.shape
    ), f"""
    Mean of all simulated histograms for the products should have the same shape
    as the set of observed histograms of products
    """
    hpd = np.array(
        [arviz.hdi(simulated_histograms[:, i, :], hdi_prob=0.95) for i in range(observed_histograms.shape[0])]
    )
    assert hpd.shape == observed_histograms.shape + tuple(
        (2,)
    ), f"""
    Shape of hpd array should be {observed_histograms.shape + (2,)}, found {hpd.shape} instead
    """
    # Will store correlations in the order of HPD_0, mean, HPD_1
    correlations = []
    for product in range(hpd.shape[0]):
        r_0, p_0 = pearsonr(observed_histograms[product, :], hpd[product, :, 0])
        r_mean, p_mean = pearsonr(observed_histograms[product, :], simulation_mean[product, :])
        r_1, p_1 = pearsonr(observed_histograms[product, :], hpd[product, :, 1])
        correlations.append([r_0, r_mean, r_1])

    return np.array(correlations)
