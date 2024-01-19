from abc import ABC, abstractmethod
import os
import pickle as pkl
from typing import Any, Dict, List
from random import shuffle

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from custom_types import Datapoint, SingleSet, Dataset, TFDataset, CSingleSet, CDataset
from global_configs import DATASET_MAKER, DATA_PREPROCESSING
from dataset_maker.Normalize import Normalize


class DatasetMaker(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.config = DATASET_MAKER
        self.label_config: Dict[int, int]

        self.train_datapoints: List[Datapoint]
        self.test_datapoints: List[Datapoint]
        self.validation_datapoints: List[Datapoint]

        self.dataset: Dataset

    def load_train_datapoints(self) -> None:
        with open(
            os.path.join(
                DATA_PREPROCESSING["save_dir"],
                self.config["train_set_datapoints"] + ".pkl",
            ),
            "rb",
        ) as _f:
            self.train_datapoints = pkl.load(_f)

    def load_test_datapoints(self) -> None:
        with open(
            os.path.join(
                DATA_PREPROCESSING["save_dir"],
                self.config["test_set_datapoints"] + ".pkl",
            ),
            "rb",
        ) as _f:
            self.test_datapoints = pkl.load(_f)

    def __shuffle_datapoints(self, datapoints: List[Datapoint]) -> List[Datapoint]:
        shuffle(datapoints)

        return datapoints

    def shuffle_data(self) -> None:
        self.train_datapoints = self.__shuffle_datapoints(self.train_datapoints)
        self.test_datapoints = self.__shuffle_datapoints(self.test_datapoints)

    def split_off_validation(self) -> None:
        split_index = int(
            len(self.train_datapoints) * (1 - self.config["validation_split"])
        )

        self.validation_datapoints = self.train_datapoints[split_index:]
        self.train_datapoints = self.train_datapoints[:split_index]

    def __filter_labels_for_datapoints(
        self, datapoints: List[Datapoint]
    ) -> List[Datapoint]:
        return [
            datapoint
            for datapoint in datapoints
            if datapoint["label"] not in self.config["filter_labels"]
        ]

    def filter_labels(self) -> None:
        self.train_datapoints = self.__filter_labels_for_datapoints(
            self.train_datapoints
        )
        self.validation_datapoints = self.__filter_labels_for_datapoints(
            self.validation_datapoints
        )
        self.test_datapoints = self.__filter_labels_for_datapoints(self.test_datapoints)

    def __get_labels_from_datapoints(
        self, datapoints: List[Datapoint]
    ) -> npt.NDArray[np.int32]:
        return np.array([datapoint["label"] for datapoint in datapoints])

    def make_label_config(self) -> None:
        all_labels = np.concatenate(
            [
                self.__get_labels_from_datapoints(self.train_datapoints),
                self.__get_labels_from_datapoints(self.validation_datapoints),
                self.__get_labels_from_datapoints(self.test_datapoints),
            ]
        )

        self.label_config = {
            old_lab: new_lab for new_lab, old_lab in enumerate(np.unique(all_labels))
        }

    def __remap_datapoints_labels(self, datapoints: List[Datapoint]) -> List[Datapoint]:
        for datapoint in datapoints:
            datapoint["label"] = self.label_config[datapoint["label"]]

        return datapoints

    def remap_labels(self) -> None:
        self.train_datapoints = self.__remap_datapoints_labels(self.train_datapoints)
        self.validation_datapoints = self.__remap_datapoints_labels(
            self.validation_datapoints
        )
        self.test_datapoints = self.__remap_datapoints_labels(self.test_datapoints)

    def __single_set_from_datapoints(self, datapoints: List[Datapoint]) -> SingleSet:
        return {
            "data": np.stack([datapoint["image"] for datapoint in datapoints]),
            "labels": np.stack([datapoint["label"] for datapoint in datapoints]),
        }

    def set_dataset(self) -> None:
        self.dataset = {
            "train": self.__single_set_from_datapoints(self.train_datapoints),
            "test": self.__single_set_from_datapoints(self.test_datapoints),
            "validation": self.__single_set_from_datapoints(self.validation_datapoints),
        }

    def normalize_dataset(self) -> None:
        self.dataset = Normalize(self.dataset).normalize_dataset()

    def __one_hot_encode_single_set(self, single: SingleSet) -> SingleSet:
        labels = single["labels"].reshape(-1)
        single["labels"] = np.eye(len(self.label_config))[labels]

        return single

    def one_hot_encode_labels(self) -> None:
        self.dataset["train"] = self.__one_hot_encode_single_set(self.dataset["train"])
        self.dataset["validation"] = self.__one_hot_encode_single_set(
            self.dataset["validation"]
        )
        self.dataset["test"] = self.__one_hot_encode_single_set(self.dataset["test"])

    def create_dataset(self) -> None:
        self.load_train_datapoints()
        self.load_test_datapoints()
        self.shuffle_data()
        self.split_off_validation()
        self.filter_labels()
        self.make_label_config()
        self.remap_labels()
        self.set_dataset()
        self.normalize_dataset()
        self.one_hot_encode_labels()

    def dataset_to_tf(self) -> TFDataset:
        return {
            "train_data": tf.constant(self.dataset["train"]["data"]),
            "test_data": tf.constant(self.dataset["test"]["data"]),
            "validation_data": tf.constant(self.dataset["validation"]["data"]),
            "train_labels": tf.constant(self.dataset["train"]["labels"]),
            "test_labels": tf.constant(self.dataset["test"]["labels"]),
            "validation_labels": tf.constant(self.dataset["validation"]["labels"]),
            "label_config": self.label_config,
        }

    @abstractmethod
    def make_neural_network_input(self) -> None:
        pass

    def save_neural_network_input(self, name: str, dataset: Any) -> None:
        with open(os.path.join(self.config["save_dir"], name + ".pkl"), "wb") as _f:
            pkl.dump(dataset, _f)


class CNNDatasetMaker(DatasetMaker):
    def make_neural_network_input(self) -> TFDataset:
        return self.dataset_to_tf()


class DNNDatasetMaker(DatasetMaker):
    def __flatten_single_set(self, single: SingleSet) -> SingleSet:
        single["data"] = single["data"].reshape(single["data"].shape[0], -1)

        return single

    def flatten_dataset(self) -> None:
        self.dataset["train"] = self.__flatten_single_set(self.dataset["train"])
        self.dataset["test"] = self.__flatten_single_set(self.dataset["test"])
        self.dataset["validation"] = self.__flatten_single_set(
            self.dataset["validation"]
        )

    def make_neural_network_input(self) -> TFDataset:
        self.flatten_dataset()

        return self.dataset_to_tf()


class CVNNDatasetMaker(DatasetMaker):
    def create_dataset(self) -> None:
        self.load_train_datapoints()
        self.load_test_datapoints()
        self.shuffle_data()
        self.split_off_validation()
        self.filter_labels()
        self.make_label_config()
        self.remap_labels()
        self.set_dataset()
        self.normalize_dataset()

    def __flatten_single_set(self, single: SingleSet) -> SingleSet:
        shape = single["data"].shape
        single["data"] = single["data"].reshape(shape[0], -1, shape[-1])

        return single

    def flatten_dataset(self) -> None:
        self.dataset["train"] = self.__flatten_single_set(self.dataset["train"])
        self.dataset["test"] = self.__flatten_single_set(self.dataset["test"])
        self.dataset["validation"] = self.__flatten_single_set(
            self.dataset["validation"]
        )

    def __single_set_to_complex(self, single: SingleSet) -> CSingleSet:
        return {
            "data": single["data"][..., 0] * np.exp(1.0j * single["data"][..., 1]),
            "labels": single["labels"],
        }

    def dataset_to_complex(self) -> CDataset:
        return {
            "test": self.__single_set_to_complex(self.dataset["test"]),
            "train": self.__single_set_to_complex(self.dataset["train"]),
            "validation": self.__single_set_to_complex(self.dataset["validation"]),
            "label_config": self.label_config,
        }

    def make_neural_network_input(self) -> CDataset:
        self.flatten_dataset()

        cds = self.dataset_to_complex()

        cds["train"]["data"] = cds["train"]["data"].astype(np.complex128)
        cds["test"]["data"] = cds["test"]["data"].astype(np.complex128)
        cds["validation"]["data"] = cds["validation"]["data"].astype(np.complex128)

        return cds
