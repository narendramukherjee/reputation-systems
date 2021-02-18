import multiprocessing as mp
import pickle

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import sbi
import sbi.inference as sbi_inference
import sklearn
import torch

from snpe.simulations import simulator_class
from snpe.utils.data_transforms import pad_timeseries_for_cnn
from snpe.utils.embedding_nets import get_cnn_1d


class BaseInference:
    def __init__(self, parameter_prior: torch.distributions.Distribution, device: str = "cpu"):
        self.parameter_prior = parameter_prior
        self.device = device
        if self.device == "cpu":
            torch.set_num_threads(mp.cpu_count())
            print(f"\t Device set to {self.device}, using torch num threads={torch.get_num_threads()}")

    def load_simulator(
        self, dirname: Path, simulator_type: str = "double_rho", simulation_type: str = "timeseries"
    ) -> None:
        # The parameters used to initialize the simulator object don't matter here
        # as they will be overridden by those of the loaded simulator
        params = {"review_prior": np.ones(5), "tendency_to_rate": 0.05, "simulation_type": simulation_type}
        self.simulator_type = simulator_type
        self.simulation_type = simulation_type
        if self.simulator_type == "single_rho":
            simulator = simulator_class.SingleRhoSimulator(params)
        elif self.simulator_type == "double_rho":
            simulator = simulator_class.DoubleRhoSimulator(params)
        else:
            raise ValueError(f"simulator_type has to be one of single_rho or double rho, found {self.simulator_type}")

        simulator.load_simulator(dirname)
        self.simulator = simulator

    def infer_snpe_posterior(
        self,
        embedding_net_creator: Optional[Callable],
        embedding_net_conf: Optional[Dict],
        simulation_transform: Optional[Callable],
        model: str,
        batch_size: int,
        learning_rate: float,
        hidden_features: int,
        num_transforms: int,
    ) -> None:
        # Just need to define some posterior to prevent mypy errors
        self.posterior = None  # type: sbi_inference.posteriors.direct_posterior.DirectPosterior
        raise NotImplementedError

    def get_posterior_samples(self, observations: np.array, num_samples: int = 5_000) -> np.array:
        raise NotImplementedError

    def save_inference(self, dirname: Path) -> None:
        inference_dict = {
            "simulator_type": self.simulator_type,
            "simulation_type": self.simulation_type,
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


class HistogramInference(BaseInference):
    def __init__(self, parameter_prior: torch.distributions.Distribution, device: str = "cpu"):
        super(HistogramInference, self).__init__(parameter_prior, device)

    def infer_snpe_posterior(
        self,
        embedding_net_creator: Optional[Callable] = None,
        embedding_net_conf: Optional[Dict] = None,
        simulation_transform: Optional[Callable] = None,
        model: str = "maf",
        batch_size: int = 50,
        learning_rate: float = 5e-4,
        hidden_features: int = 50,
        num_transforms: int = 5,
    ) -> None:
        # Convert the simulations and parameters to pytorch tensors to use with sbi
        if simulation_transform is not None:
            simulations = simulation_transform(self.simulator.simulations)
        else:
            simulations = torch.from_numpy(self.simulator.simulations).type(torch.FloatTensor)
        parameters = torch.from_numpy(self.simulator.simulation_parameters["rho"]).type(torch.FloatTensor)

        # Get the embedding net for the simulations
        if embedding_net_creator is not None:
            assert (
                embedding_net_conf is not None
            ), f"""embedding_net_conf dict not provided even though
               embedding_net_creator function {embedding_net_creator} provided"""
            embedding_net = embedding_net_creator(simulations[:5], **embedding_net_conf)
        else:
            embedding_net = torch.nn.Identity()
        print(f"Embedding net created: \n {embedding_net}")

        posterior_net = sbi.utils.posterior_nn(
            model=model, embedding_net=embedding_net, hidden_features=hidden_features, num_transforms=num_transforms
        )

        inference = sbi_inference.SNPE(
            prior=self.parameter_prior, density_estimator=posterior_net, device=self.device, show_progress_bars=True
        )
        density_estimator = inference.append_simulations(parameters, simulations).train(
            training_batch_size=batch_size, learning_rate=learning_rate, show_train_summary=True
        )
        # Get the training related metrics
        self.best_validation_log_prob = inference._summary["best_validation_log_probs"][-1]
        self.training_epochs = inference._summary["epochs"][-1]
        # Build the posterior from the density estimator
        self.posterior = inference.build_posterior(density_estimator)

    def get_posterior_samples(self, observations: np.ndarray, num_samples: int = 5_000) -> np.ndarray:
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


class TimeSeriesInference(HistogramInference):
    def __init__(self, parameter_prior: torch.distributions.Distribution, device: str = "cpu"):
        super(TimeSeriesInference, self).__init__(parameter_prior, device)

    def infer_snpe_posterior(
        self,
        embedding_net_creator: Optional[Callable] = get_cnn_1d,
        embedding_net_conf: Optional[Dict] = {},
        simulation_transform: Optional[Callable] = pad_timeseries_for_cnn,
        model: str = "maf",
        batch_size: int = 50,
        learning_rate: float = 5e-4,
        hidden_features: int = 50,
        num_transforms: int = 5,
    ) -> None:
        super(TimeSeriesInference, self).infer_snpe_posterior(
            embedding_net_creator,
            embedding_net_conf,
            simulation_transform,
            model,
            batch_size,
            learning_rate,
            hidden_features,
            num_transforms,
        )

    def get_posterior_samples(self, observations: np.ndarray, num_samples: int = 5_000) -> np.ndarray:
        pass
