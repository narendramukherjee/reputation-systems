from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set plotting parameters
sns.set(style="white", context="talk", font_scale=2.5)
sns.set_color_codes(palette="colorblind")
sns.set_style("ticks", {"axes.linewidth": 2.0})
plt.ion()

HISTOGRAM_ARTIFACT_PATH = Path("./artifacts/ICWSM")
TIMESERIES_ARTIFACT_PATH = Path("./artifacts/hyperparameter_tuning/cnn_tuning")
HERDING_ARTIFACT_PATH = Path("./artifacts/hyperparameter_tuning/inference_with_herding")


def main() -> None:
    # Load up the correlation coefficients of the observed review histograms with the posterior predictive
    histogram_ppc = np.load(HISTOGRAM_ARTIFACT_PATH / "ppc_correlations.npy")
    timeseries_ppc = np.load(TIMESERIES_ARTIFACT_PATH / "ppc_correlations.npy")
    herding_ppc = np.load(HERDING_ARTIFACT_PATH / "ppc_correlations.npy")

    # Plot the correlation of the mean ppc histogram and the hpd_0 and hpd_1 ppc histogram from the 3
    # methods separately in 3 figures
    titles = ["Pearson r of lower HPD of the posterior predictive"
              + f"\n of review histograms with observed histograms for {histogram_ppc.shape[0]} products",
              "Pearson r of mean of the posterior predictive"
              + f"\n of review histograms with observed histograms for {histogram_ppc.shape[0]} products",
              "Pearson r of upper HPD of the posterior predictive"
              + f"\n of review histograms with observed histograms for {histogram_ppc.shape[0]} products"]
    for i in range(3):
        plt.figure()
        plt.hist(histogram_ppc[:, i], alpha=0.5, label="Inference using histograms")
        plt.hist(timeseries_ppc[:, i], alpha=0.5, label="Inference using timeseries")
        plt.hist(herding_ppc[:, i], alpha=0.5, label="Inference using timeseries, simulation includes herding")
        plt.legend(fontsize=20.0)
        plt.title(titles[i], fontsize=24.0)


if __name__ == '__main__':
    main()
