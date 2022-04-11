import multiprocessing as mp
import os
import pickle

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from snpe.utils.embedding_nets import RatingPredictorModel
from snpe.utils.functions import nn_converged
from snpe.utils.statistics import dirichlet_kl_divergence
from torch.utils.data import DataLoader, Dataset, TensorDataset, sampler

from . import ARTIFACT_PATH, STARSPACE_PARAMS


class EmbeddingRatingPredictor:
    def __init__(self, predict_fractions: float = False):
        torch.set_num_threads(mp.cpu_count())
        print(f"\t Device set to cpu, using torch num threads={torch.get_num_threads()}")
        self.artifact_path = ARTIFACT_PATH
        # Assert that starspace training has been done before embedding->rating predictor is used
        assert os.path.exists(
            self.artifact_path / "productspace"
        ), f"""
        Found no starspace output in {self.artifact_path}, train starspace model first
        """
        self.model = RatingPredictorModel(
            predict_fractions=predict_fractions, prod_embedding_dim=STARSPACE_PARAMS["dim"]
        )
        print(f"Using the dense network: \n {self.model.net}")

    def process_input_data(self) -> pd.DataFrame:
        # Load the product embeddings
        product_embeddings = pd.read_csv(self.artifact_path / "productspace.tsv", sep="\t", header=None)
        # Load the histograms of ratings of all products
        review_hist = pd.read_csv(self.artifact_path / "rating_histogram_all.txt", sep="\t")

        # Ensure that no products with 0 ratings exist
        assert np.all(
            review_hist.iloc[:, 1:].sum(axis=1) > 0
        ), f"Products with 0 ratings exist, check the DF of review histograms"

        # Clean up the DF of product embeddings
        # We want 2 columns in the clean DF, one with productid and the other with the embeddings
        productids = np.array(product_embeddings[0].str.split("_").apply(lambda x: x[1]))
        prod_df = pd.DataFrame(
            {
                "productid": productids.astype("int"),
                "embedding": [np.array(product_embeddings.iloc[i, 1:]) for i in range(len(productids))],
            }
        )
        # Now merge this clean DF with the review histograms - the productid in the histogram DF is asin
        merged_df = pd.merge(prod_df, review_hist, how="inner", left_on="productid", right_on="asin")
        print(f"Merged product embeddings with review histograms and produced merged DF of shape: {merged_df.shape}")

        return merged_df

    def create_training_data(
        self, input_df: pd.DataFrame, validation_frac: float = 0.1, batch_size: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        # Create a torch tensor from the histogram of ratings. 0 values are a problem with KL divergence calculation
        # So we add 1 to all the star ratings to prevent that problem
        ratings = np.array(input_df.loc[:, ["1", "2", "3", "4", "5"]].astype("int")) + 1
        ratings = torch.from_numpy(ratings).type(torch.FloatTensor)
        # Create a torch tensor with product embeddings too
        embeddings = torch.from_numpy(np.array([np.array(e).astype("float") for e in input_df.embedding])).type(
            torch.FloatTensor
        )

        # Create the dataset
        dataset = TensorDataset(embeddings, ratings)
        train_indices, val_indices = self.train_valid_split(
            num_total_examples=embeddings.size()[0], validation_frac=validation_frac
        )
        print(f"Train set size: {train_indices.size()}, Validation set size: {val_indices.size()}")
        train_loader, val_loader = self.create_dataloaders(dataset, train_indices, val_indices, batch_size)
        return ratings, embeddings, train_loader, val_loader, train_indices, val_indices

    def create_dataloaders(
        self, dataset: TensorDataset, train_indices: torch.Tensor, val_indices: torch.Tensor, batch_size: int, **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        # Sometimes you might want to add additional features to the dataloaders, this can be done through **kwargs
        train_loader_kwargs = {
            "batch_size": min(batch_size, len(train_indices)),
            "drop_last": True,
            "sampler": sampler.SubsetRandomSampler(train_indices),
        }
        train_loader_kwargs = dict(train_loader_kwargs, **kwargs) if kwargs is not None else train_loader_kwargs
        val_loader_kwargs = {
            "batch_size": min(batch_size, len(val_indices)),
            "shuffle": False,
            "drop_last": True,
            "sampler": sampler.SubsetRandomSampler(val_indices),
        }
        val_loader_kwargs = dict(val_loader_kwargs, **kwargs) if kwargs is not None else val_loader_kwargs
        # Now make the train and val dataloaders
        train_loader = DataLoader(dataset, **train_loader_kwargs)
        val_loader = DataLoader(dataset, **val_loader_kwargs)

        return train_loader, val_loader

    def train_valid_split(self, num_total_examples: int, validation_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
        num_training_examples = int((1 - validation_frac) * num_total_examples)

        permuted_indices = torch.randperm(num_total_examples)
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )
        return train_indices, val_indices

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_indices: torch.Tensor,
        val_indices: torch.Tensor,
        num_epochs: int = 500,
        print_per_num_epochs: int = 50,
        convergence_num_epochs: int = 50,
    ) -> None:
        criterion = dirichlet_kl_divergence
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Run the training loop
        for epoch in range(num_epochs):
            # Set the model in training mode for gradient evaluation
            self.model.train()
            # Set current loss value
            train_loss = 0.0
            # Iterate over the DataLoader for training data
            for i, (inputs, targets) in enumerate(train_loader):
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                outputs = self.model(inputs)
                # Compute loss
                loss = criterion(outputs, targets).mean()
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()
                # Add to the total of the training loss
                train_loss += loss.item() * len(inputs)
            # Once all training batches have been run, get the mean training loss
            train_loss /= len(train_indices)

            # Set the model in evaluation mode so that gradients are not evaluated
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_loader):
                    preds = self.model(inputs)
                    loss = criterion(preds, targets).mean()
                    val_loss += loss.item() * len(inputs)
                val_loss /= len(val_indices)

            if epoch % print_per_num_epochs == 0:
                print(f"Train Loss after epoch: {epoch}: {train_loss}")
                print(f"Validation loss after epoch: {epoch}: {val_loss}")
            if nn_converged(epoch, convergence_num_epochs, val_loss, self.model):
                print(f"Stopping after epoch {epoch} as validation loss was not improving further")
                break

        # Process is complete.
        print("Training process has finished.")
        print(f"Best loss: {self.model.best_validation_loss}")

    def baseline_comparison(self, ratings: torch.Tensor, embeddings: torch.Tensor, val_indices: torch.Tensor) -> None:
        # Finally calculate and print the comparison with a baseline model
        # The baseline model here is just predicting the average number of each star rating, averaged across the
        # validation dataset
        baseline_pred = ratings[val_indices].mean(axis=0)[None, :]
        baseline_loss = dirichlet_kl_divergence(baseline_pred, ratings[val_indices]).mean()
        print(f"Baseline loss: {baseline_loss} and NN model loss: {self.model.best_validation_loss}")
        percent_improvement = 100 * (baseline_loss - self.model.best_validation_loss) / baseline_loss
        print(f"Loss improved by {percent_improvement} percent due to NN model")

    def save(self) -> None:
        model_dict = {
            "artifact_path": self.artifact_path,
            "model": self.model
        }
        with open(self.artifact_path / (self.__class__.__name__ + ".pkl"), "wb") as f:
            pickle.dump(model_dict, f)

    def load(self) -> None:
        with open(self.artifact_path / (self.__class__.__name__ + ".pkl"), "rb") as f:
            model_dict = pickle.load(f)
        for key in model_dict:
            setattr(self, key, model_dict[key])
