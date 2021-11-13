import multiprocessing as mp
import subprocess
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from snpe.utils.functions import terminal_execute

from . import ARTIFACT_PATH, BINARY_PATH, STARSPACE_PARAMS


class StarSpaceEmbedder:
    def __init__(self):
        self.starspace_binary = BINARY_PATH / "starspace"
        self.starspace_output = ARTIFACT_PATH / "productspace"
        self.artifact_path = ARTIFACT_PATH
        self.starspace_params = STARSPACE_PARAMS
        self.starspace_params["thread"] = mp.cpu_count()

    def process_input_data(self, test_set_frac: float = 0.2) -> None:
        # Load the input data file
        input_data = pd.read_csv(self.artifact_path / "pageview_summaries.txt", sep="\t")
        print(f"Loaded input data of shape {input_data.shape}")
        expected_input_data_shape = (3204851, 13)
        assert (
            input_data.shape == expected_input_data_shape
        ), f"""
            Unexpected shape {input_data.shape} for input data. Has the data file changed?
            """
        assert (
            input_data.ProductsSeen.isnull().mean() == 0.0
        ), f"""
            Null values found in ProductsSeen, expected no null values in these strings
            """

        # To work with starspace, need to pull the strings of browsed products into space separated strings of
        # products like "product_1 product_2 product_3" and so on
        pageview_strings = (
            pd.Series(input_data["ProductsSeen"].unique())
            .str.split(",")
            .apply(lambda x: ["product_" + str(token) for token in x])
            .str.join(" ")
        )
        # Create a train and a test set from the pageview strings
        pageview_train, pageview_test = train_test_split(pageview_strings, test_size=test_set_frac)
        print(f"Train set size: {pageview_train.shape}")
        print(f"Test set size: {pageview_test.shape}")
        # Save the train and test sets
        print("Saving train and test datasets")
        self.train_path = ARTIFACT_PATH / "pageview_train.txt"
        self.test_path = ARTIFACT_PATH / "pageview_test.txt"
        pageview_train.to_csv(self.train_path, index=False, header=False, encoding="utf-8")
        pageview_test.to_csv(self.test_path, index=False, header=False, encoding="utf-8")

    def train_starspace(self) -> None:
        # Create the starspace training command
        command = [f"{str(self.starspace_binary.resolve())}",
        "train",
        f"-trainFile", f"{str(self.train_path.resolve())}",
        f"-validationFile", f"{str(self.test_path.resolve())}",
        f"-label", "product",
        f"-model", f"{str(self.starspace_output.resolve())}"]
        for key, val in self.starspace_params.items():
            command += [f"-{key}", f"{val}"]

        # Run the starspace command
        print(f"StarSpace command to be run: \n {' '.join(command)}")
        #child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #print(child.stdout.read())
        for path in terminal_execute(command):
            print(path, end="")
