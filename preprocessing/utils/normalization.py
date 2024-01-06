from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import numpy.typing as npt

from custom_types import TrainTestSet, Dataset


class Normalization(ABC):
    def __init__(self, train_test_set: TrainTestSet) -> None:
        super().__init__()

        self.train_test_set: TrainTestSet = train_test_set
        self.metadata: Dict[str, npt.NDArray]

    @abstractmethod
    def set_metadata(self) -> None:
        pass

    @abstractmethod
    def normalize_single_set(self, single_set: Dataset) -> Dataset:
        pass

    def normalize(self) -> TrainTestSet:
        self.set_metadata()
        self.train_test_set["train"] = self.normalize_single_set(
            self.train_test_set["train"]
        )
        self.train_test_set["test"] = self.normalize_single_set(
            self.train_test_set["test"]
        )

        return self.train_test_set


class ComplexSphericalNormalization(Normalization):
    """
    Bounds each feature of the complex vector to a closed unit ball of the complex plane
    """

    def set_metadata(self) -> None:
        self.metadata = {
            "max_magnitude": np.max(
                np.absolute(self.train_test_set["train"]["data"]), axis=0
            )
        }

    def normalize_single_set(self, single_set: Dataset) -> Dataset:
        single_set["data"] /= self.metadata["max_magnitude"]
        return single_set


class ZScoreNormalization(Normalization):
    def set_metadata(self) -> None:
        self.metadata = {
            "means": np.mean(self.train_test_set["train"]["data"], axis=0),
            "deviations": np.std(self.train_test_set["train"]["data"], axis=0),
        }

    def normalize_single_set(self, single_set: Dataset) -> Dataset:
        single_set["data"] = (
            single_set["data"] - self.metadata["means"]
        ) / self.metadata["deviations"]

        return single_set


class ImageNormalization(Normalization):
    """
    Per channel normalization for CNN SAR input
    """

    def set_metadata(self) -> None:
        self.metadata = {
            "max_magnitude": np.max(self.train_test_set["train"]["data"][..., 0]),
            "max_phase": np.max(self.train_test_set["train"]["data"][..., 1]),
        }

    def normalize_single_set(self, single_set: Dataset) -> Dataset:
        single_set["data"][..., 0] /= self.metadata["max_magnitude"]
        single_set["data"][..., 1] /= self.metadata["max_phase"]

        return single_set
