from typing import Dict
import numpy as np
import numpy.typing as npt
from custom_types import Dataset, SingleSet


class Normalize:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.metadata: Dict[str, npt.NDArray[np.float32]]

    def set_metadata(self) -> None:
        self.metadata = {"maxs": np.max(self.dataset["test"]["data"][..., 0])}

    def normalize_single_set(self, single_set: SingleSet) -> SingleSet:
        single_set["data"][..., 0] /= self.metadata["maxs"]
        single_set["data"][..., 1] /= 2 * np.pi

        return single_set

    def normalize_dataset(self) -> Dataset:
        self.set_metadata()

        for key in self.dataset:
            self.dataset[key] = self.normalize_single_set(self.dataset[key])

        return self.dataset
