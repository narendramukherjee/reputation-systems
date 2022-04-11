import os
import pickle

from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture

from . import ARTIFACT_PATH, STARSPACE_PARAMS


class EmbeddingDensityGMM:
    def __init__(self, n_components: int = 10, n_init: int = 5) -> None:
        self.artifact_path = ARTIFACT_PATH
        # Assert that starspace training has been done before embedding->rating predictor is used
        assert os.path.exists(
            self.artifact_path / "productspace"
        ), f"""
        Found no starspace output in {self.artifact_path}, train starspace model first
        """
        self.n_features = STARSPACE_PARAMS["dim"]
        self.n_components = n_components
        self.n_init = n_init
        # 2 separate models, one for products and the other for users
        self.product_model = GaussianMixture(
            n_components=self.n_components, n_init=self.n_init, max_iter=500, random_state=42, verbose=2, verbose_interval=20
        )
        self.user_model = GaussianMixture(
            n_components=self.n_components, n_init=self.n_init, max_iter=500, random_state=42, verbose=2, verbose_interval=20
        )

    def process_input_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # Load the product and user embeddings
        product_embeddings = pd.read_csv(self.artifact_path / "productspace.tsv", sep="\t", header=None)
        user_embeddings = pd.read_csv(self.artifact_path / "userspace.tsv", sep="\t", header=None)
        product_embeddings = np.array(product_embeddings.iloc[:, 1:])
        # The file of user embeddings does not have an index (productid) like the product embeddings
        # So all columns of that file are embedding dimensions
        user_embeddings = np.array(user_embeddings.iloc[:, :])
        assert (
            product_embeddings.shape[-1] == self.n_features
        ), f"Expected {self.n_features} dims in product embeddings, found shape: {product_embeddings.shape}"
        assert (
            user_embeddings.shape[-1] == self.n_features
        ), f"Expected {self.n_features} dims in user embeddings, found shape: {user_embeddings.shape}"
        print(f"Product embeddings of shape: {product_embeddings.shape}")
        print(f"User embeddings of shape: {user_embeddings.shape}")

        return product_embeddings, user_embeddings

    def fit(self, product_embeddings: np.ndarray, user_embeddings: np.ndarray) -> None:
        print(f"Fitting product model: {self.product_model}")
        product_train_idx, product_test_idx = self.train_test_split(product_embeddings.shape[0])
        self.product_model.fit(product_embeddings[product_train_idx])

        print(f"Fitting user model: {self.user_model}")
        user_train_idx, user_test_idx = self.train_test_split(user_embeddings.shape[0])
        self.user_model.fit(user_embeddings[user_train_idx])

        product_baseline_score, user_baseline_score = self.baseline_comparison(
            product_embeddings, user_embeddings, product_train_idx, product_test_idx, user_train_idx, user_test_idx
        )
        product_model_score = self.product_model.score(product_embeddings[product_test_idx])
        user_model_score = self.user_model.score(user_embeddings[user_test_idx])
        print(f"""
            Report for product embedding GMM density estimator:
            Model score: {product_model_score}
            Basline score: {product_baseline_score}
            Model improvement over baseline:
            {100 * (product_model_score - product_baseline_score) / product_baseline_score} percent
        """)
        print(f"""
            Report for user embedding GMM density estimator:
            Model score: {user_model_score}
            Basline score: {user_baseline_score}
            Model improvement over baseline:
            {100 * (user_model_score - user_baseline_score) / user_baseline_score} percent
        """)

    def train_test_split(self, num_samples: int, test_set_frac: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.permutation(num_samples).astype("int")
        num_train_samples = int((1 - test_set_frac) * num_samples)
        train_idx = idx[:num_train_samples]
        test_idx = idx[num_train_samples:]
        print(f"{len(train_idx)} samples in train set, {len(test_idx)} samples in test set")
        return train_idx, test_idx

    def baseline_comparison(
        self,
        product_embeddings: np.ndarray,
        user_embeddings: np.ndarray,
        product_train_idx: np.ndarray,
        product_test_idx: np.ndarray,
        user_train_idx: np.ndarray,
        user_test_idx: np.ndarray,
    ) -> Tuple[float, float]:
        # Here we fit a baseline model to product and user embeddings which contains only 1 mixture component
        print("Training baseline models")
        product_baseline = GaussianMixture(n_components=1, random_state=42, verbose=2, verbose_interval=20)
        product_baseline.fit(product_embeddings[product_train_idx])
        user_baseline = GaussianMixture(n_components=1, random_state=42, verbose=2, verbose_interval=20)
        user_baseline.fit(user_embeddings[user_train_idx])
        product_baseline_score = product_baseline.score(product_embeddings[product_test_idx])
        user_baseline_score = user_baseline.score(user_embeddings[user_test_idx])

        return product_baseline_score, user_baseline_score

    def save(self) -> None:
        model_dict = {
            "artifact_path": self.artifact_path,
            "product_model": self.product_model,
            "user_model": self.user_model
        }
        with open(self.artifact_path / (self.__class__.__name__ + ".pkl"), "wb") as f:
            pickle.dump(model_dict, f)

    def load(self) -> None:
        with open(self.artifact_path / (self.__class__.__name__ + ".pkl"), "rb") as f:
            model_dict = pickle.load(f)
        for key in model_dict:
            setattr(self, key, model_dict[key])
