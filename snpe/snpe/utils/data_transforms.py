import numpy as np
import torch


def pad_timeseries_for_cnn(simulations: np.ndarray, device: str) -> torch.Tensor:
    # Time series data comes in as np array of type object
    # Each entry in the array is a time series of reviews of unequal length
    # We pad each series to the length of the longest time series
    # Padding value is the last observation in each time series that needs to be padded
    # Each observation in the time series of reviews is expected to be 5-D
    simulations = [
        torch.from_numpy(simulation.astype(np.dtype("float32"))).type(torch.FloatTensor).to(device) for simulation in simulations
    ]
    lens = [len(simulation) for simulation in simulations]
    max_len = max(lens)

    padded_simulations = simulations[0].data.new_empty((len(simulations), max_len, 5))
    for i, simulation in enumerate(simulations):
        padded_simulations[i, : lens[i], :] = simulation
        if lens[i] < max_len:
            padded_simulations[i, lens[i] :, :] = simulation[-1, :].repeat((max_len - lens[i], 1))
    # 1D CNNs expect input data to have dim (batch X channels X length), so we permute the padded
    # simulations accordingly
    padded_simulations = padded_simulations.permute(0, 2, 1)

    return padded_simulations
