import itertools
import multiprocessing as mp
import pickle

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import sbi
import sbi.inference as sbi_inference
import sklearn
import torch

from snpe.simulations import marketplace_simulator_class, simulator_class
from snpe.utils.data_transforms import pad_timeseries_for_cnn
from snpe.utils.embedding_nets import get_cnn_1d


class BaseInference:
    def __init__(self, parameter_prior: torch.distributions.Distribution, device: str = "cpu"):
        self.parameter_prior = parameter_prior
        assert device in ["cpu", "cuda"], f"Device needs to be cpu or cuda, unknown device {device} provided"
        self.device = device
        if self.device == "cpu":
            torch.set_num_threads(mp.cpu_count())
            print(f"\t Device set to {self.device}, using torch num threads={torch.get_num_threads()}")
        # Attribute that stores the length of the padded timeseries simulations
        self.padded_simulation_length = None  # type: Optional[int]

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
        elif self.simulator_type == "single_herding":
            # Add the additional parameters that the herding simulator needs to initialize
            # Again the values aren't important as they will be overridden by those of the loaded simulator
            params.update({"previous_rating_measure": "mean", "min_reviews_for_herding": 5})
            simulator = simulator_class.HerdingSimulator(params)
        elif self.simulator_type == "double_herding":
            # 3 additional parameter for the double herding simulator, will be overridden by the loaded simulator
            params.update(
                {
                    "previous_rating_measure": "mean",
                    "min_reviews_for_herding": 5,
                    "herding_differentiating_measure": "mean",
                }
            )
            simulator = simulator_class.DoubleHerdingSimulator(params)
        elif self.simulator_type == "marketplace":
            # The additional parametes for the marketplace simulator, will be overridden by the loaded simulator
            params.update(
                {
                    "previous_rating_measure": "mean",
                    "min_reviews_for_herding": 5,
                    "herding_differentiating_measure": "mean",
                    "num_products": 1400,
                    "num_total_marketplace_reviews": 140_000,
                    "consideration_set_size": 5,
                }
            )
            simulator = marketplace_simulator_class.MarketplaceSimulator(params)
        else:
            raise ValueError(
                f"""
                simulator_type has to be one of single_rho, double_rho, single_herding, double_herding or marketplace
                found {self.simulator_type} instead
                """
            )

        simulator.load_simulator(dirname)
        # For marketplace simulations, we need the extra step of unravelling the simulations from shape
        # (num_marketplaces X num_products, ) to shape (num_simulations, )
        if self.simulator_type == "marketplace":
            # https://mathieularose.com/how-not-to-flatten-a-list-of-lists-in-python
            simulator.simulations = np.array(
                list(itertools.chain.from_iterable(simulator.simulations)), dtype=object
            )
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

    def get_posterior_samples(self, observations: np.ndarray, num_samples: int = 5_000) -> np.ndarray:
        raise NotImplementedError

    def save_inference(self, dirname: Path) -> None:
        inference_dict = {
            "simulator_type": self.simulator_type,
            "simulation_type": self.simulation_type,
            "parameter_prior": self.parameter_prior,
            "device": self.device,
            "padded_simulation_length": self.padded_simulation_length,
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
        # Add the length of the padded simulations (if timeseries) for later use
        if self.simulation_type == "timeseries":
            self.padded_simulation_length = int(simulations.size()[-1])
        # Get the simulation parameters
        if self.simulator_type in ("single_rho", "double_rho"):
            parameters = torch.from_numpy(self.simulator.simulation_parameters["rho"]).type(torch.FloatTensor)
        elif self.simulator_type in ("single_herding", "marketplace"):
            parameters = np.hstack(
                (self.simulator.simulation_parameters["rho"], self.simulator.simulation_parameters["h_p"][:, None])
            )
            np.testing.assert_array_equal(parameters[:, :2], self.simulator.simulation_parameters["rho"])
            np.testing.assert_array_equal(parameters[:, 2], self.simulator.simulation_parameters["h_p"])
            parameters = torch.from_numpy(parameters).type(torch.FloatTensor)
        elif self.simulator_type == "double_herding":
            parameters = np.hstack(
                (self.simulator.simulation_parameters["rho"], self.simulator.simulation_parameters["h_p"])
            )
            np.testing.assert_array_equal(parameters[:, :2], self.simulator.simulation_parameters["rho"])
            np.testing.assert_array_equal(parameters[:, 2:], self.simulator.simulation_parameters["h_p"])
            parameters = torch.from_numpy(parameters).type(torch.FloatTensor)
        else:
            raise ValueError(
                f"""
                simulator_type has to be one of single_rho, double_rho, single_herding, double_herding or marketplace
                found {self.simulator_type} instead
                """
            )

        # Get the embedding net for the simulations
        if embedding_net_creator is not None:
            assert (
                embedding_net_conf is not None
            ), f"""embedding_net_conf dict not provided even though
               embedding_net_creator function {embedding_net_creator} provided"""
            # We pass a subset of simulations to the embedding net creator function
            # This is needed in the time series case to deduce the dimensionality of the first linear layer
            # in the embedding net
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
        # In case model training was done on gpu, remember to move the neural net to cpu before
        # building the posterior or getting metrics
        if self.device == "cuda":
            inference._neural_net.to(device="cpu")
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
        elif self.simulator_type == "double_rho":
            num_parameters = 2
        elif self.simulator_type in ("single_herding", "marketplace"):
            num_parameters = 3
        elif self.simulator_type == "double_herding":
            num_parameters = 4
        else:
            raise ValueError(
                f"""
                simulator_type has to be one of single_rho, double_rho, single_herding or double_herding
                found {self.simulator_type} instead
                """
            )
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
        assert hasattr(
            self, "simulator"
        ), f"""
        Simulator needs to be loaded before posterior samples can be obtained in the timeseries case.
        Run <inference object>.load_simulator first
        """
        assert (
            self.padded_simulation_length is not None
        ), f"""
        In the timeseries case, simulations are padded and padded_simulation_length should not be None.
        Run inference again
        """
        # Concatenate the input observations with the simulator's array of simulations so that they can be passed
        # together to the padding function. We will then pull out the padded observations from this joint array
        concat_simulations = np.concatenate((self.simulator.simulations, observations), axis=0)
        padded_concat_simulations = pad_timeseries_for_cnn(concat_simulations, device=self.device)
        assert (
            self.padded_simulation_length == padded_concat_simulations.size()[-1]
        ), f"""
        During inference, simulations were padded to length {self.padded_simulation_length}. While getting posterior
        samples, they were padded to {padded_concat_simulations.size()[-1]}. Cut observation length to the same length
        as the maximum used during inference, i.e, {self.padded_simulation_length}
        """
        # Pull out the observations from the concatenated and padded array. They are the last few in that array
        observations = padded_concat_simulations[-observations.shape[0] :, :, :]
        if self.simulator_type == "single_rho":
            num_parameters = 1
        elif self.simulator_type == "double_rho":
            num_parameters = 2
        elif self.simulator_type in ("single_herding", "marketplace"):
            num_parameters = 3
        elif self.simulator_type == "double_herding":
            num_parameters = 4
        else:
            raise ValueError(
                f"""
                simulator_type has to be one of single_rho, double_rho, single_herding or double_herding
                found {self.simulator_type} instead
                """
            )
        posterior_samples = np.empty((num_samples, observations.size()[0], num_parameters), dtype=np.float64)

        for row in range(observations.size()[0]):
            if self.device == "cuda":
                posterior_samples[:, row, :] = self.posterior.sample(
                    (num_samples,), x=observations[row, :, :][None, :, :], show_progress_bars=False
                ).cpu().numpy()
            else:
                posterior_samples[:, row, :] = self.posterior.sample(
                    (num_samples,), x=observations[row, :, :][None, :, :], show_progress_bars=False
                ).numpy()
        return posterior_samples
