"""
Statistical/metric calculation related utils
"""

import arviz
import numpy as np
import torch

from scipy.stats import pearsonr
from torch.special import digamma, gammaln


def dirichlet_kl_divergence(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # Calculates the KL divergences between 2 sets of dirichlet distributions describing review histograms
    # This is implemented in torch as it is used while training a NN to predict review histograms from product embeddings
    # In case alpha and beta have only 1 dimension, raise an assertion error as this method expects 2-D tensors
    # Dim 0 contains separate dirichlet distributions, while dim 1 has the conc. parameters of each of those separate dists
    assert len(alpha.size()) == 2, f"Expected 2 dims in alpha, found: {alpha.size()}"
    assert len(beta.size()) == 2, f"Expected 2 dims in beta, found: {beta.size()}"
    # https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    return (
        gammaln(alpha.sum(axis=1))
        - torch.sum(gammaln(alpha), axis=1)
        - gammaln(beta.sum(axis=1))
        + torch.sum(gammaln(beta), axis=1)
        + torch.sum((alpha - beta) * (digamma(alpha) - digamma(alpha.sum(axis=1))[:, None]), axis=1,)
    )


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


def review_histogram_means(review_histograms: np.ndarray) -> float:
    # Calculates the average ratings of products from the histograms of their reviews
    assert (
        review_histograms.shape[1] == 5
    ), f"""
        Expected array of review histograms to have shape (num_samples, 5),
        found {review_histograms.shape} instead.
        """
    # Asserting that all elements of review_histograms are whole numbers (as they are review counts)
    np.testing.assert_array_equal(review_histograms, review_histograms.astype("int"))
    review_sums = np.sum(review_histograms * np.arange(1, 6).reshape(1, 5), axis=1)
    review_counts = np.sum(review_histograms, axis=1)
    histogram_means = review_sums / review_counts
    return histogram_means
