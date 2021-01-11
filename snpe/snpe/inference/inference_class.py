import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import sbi
import sbi.inference as sbi_inference
import sklearn
import torch
from snpe.simulations import simulator_class


class BaseInference:
    def __init__(self, parameter_prior: torch.distributions.Distribution, device: str = "cpu"):
        self.parameter_prior = parameter_prior
        self.device = device
        if self.device == "cpu":
            torch.set_num_threads(mp.cpu_count())

    def load_simulator(self, dirname: Path, simulator_type: str = "double_rho") -> None:
        # The parameters used to initialize the simulator object don't matter here
        # as they will be overridden by those of the loaded simulator
        params = {"review_prior": np.ones(5), "tendency_to_rate": 0.05}
        self.simulator_type = simulator_type
        if self.simulator_type == "single_rho":
            simulator = simulator_class.SingleRhoSimulator(params)
        elif self.simulator_type == "double_rho":
            simulator = simulator_class.DoubleRhoSimulator(params)
        else:
            raise ValueError(f"simulator_type has to be one of single_rho or double rho, found {simulator_type}")

        simulator.load_simulator(dirname)
        self.simulator = simulator

    def infer_snpe_posterior(self, batch_size: int = 50) -> None:
        # Convert the simulations and parameters to pytorch tensors to use with sbi
        simulations = torch.from_numpy(self.simulator.simulations).type(torch.FloatTensor)
        parameters = torch.from_numpy(self.simulator.simulation_parameters["rho"]).type(torch.FloatTensor)

        inference = sbi_inference.SNPE(self.parameter_prior, device=self.device, show_progress_bars=True)
        density_estimator = inference.append_simulations(parameters, simulations).train(
            training_batch_size=batch_size, show_train_summary=True
        )
        self.posterior = inference.build_posterior(density_estimator)

    def get_posterior_samples(self, observations: np.array, num_samples: int = 5_000) -> np.array:
        # Check if array of observations is 2-D and has 5 dimensions (ratings go from 1 to 5)
        observations = sklearn.utils.check_array(observations, ensure_min_features=5)
        if self.simulator_type == "single_rho":
            num_parameters = 1
        else:
            num_parameters = 2
        posterior_samples = np.empty((num_samples, observations.shape[0], num_parameters), dtype=np.float64)

        for row in range(observations.shape[0]):
            posterior_samples[:, row, :] = self.posterior.sample(
                (num_samples,), x=torch.tensor(observations[row, :]).type(torch.FloatTensor), show_progress_bars=False
            ).numpy()
        return posterior_samples

    def save_inference(self, dirname: Path) -> None:
        inference_dict = {
            "simulator_type": self.simulator_type,
            "parameter_prior": self.parameter_prior,
            "device": self.device,
            "posterior": self.posterior,
        }
        with open(dirname / (self.__class__.__name__ + f"_{self.simulator_type}.pkl"), "wb") as f:
            pickle.dump(inference_dict, f)

    def load_inference(self, dirname: Path) -> None:
        with open(dirname / (self.__class__.__name__ + f"_{self.simulator_type}.pkl"), "rb") as f:
            inference_dict = pickle.load(f)
        for key in inference_dict:
            setattr(self, key, inference_dict[key])
