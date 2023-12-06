import numpy as np
import tensorflow as tf
from custom_types import RealMatrix, TrainTestSet
from preprocessing.utils.normalization import (
    normalize_complex_matrix,
    normalize_real_matrix,
    normalize_real_tensor,
)


def normalize_set(dataset: TrainTestSet) -> TrainTestSet:
    # since I can't know ahead of time what normalization to use, I have to do this
    if (
        dataset["dtype"] == np.complex64
    ):  # should be complex64 since it's loaded as float32
        normalized = normalize_complex_matrix(
            np.concatenate([dataset["train"]["data"], dataset["test"]["data"]], axis=0)
        )
    elif len(dataset["shape"]) == 2:
        normalized = normalize_real_matrix(
            np.concatenate([dataset["train"]["data"], dataset["test"]["data"]], axis=0)
        )
    else:
        normalized = normalize_real_tensor(
            np.concatenate([dataset["train"]["data"], dataset["test"]["data"]], axis=0)
        )

    train_len = dataset["train"]["data"].shape[0]
    dataset["train"]["data"] = normalized[:train_len]
    dataset["test"]["data"] = normalized[train_len:]

    return dataset


def add_label_config(dataset: TrainTestSet) -> TrainTestSet:
    dataset["label_config"] = dict(
        zip(
            unique := np.unique(
                np.concatenate(
                    [dataset["train"]["labels"], dataset["test"]["labels"]], axis=0
                )
            ),
            np.arange(unique.shape[0]),
        )
    )

    return dataset


def remap_labels(dataset: TrainTestSet) -> TrainTestSet:
    new_labels = np.vectorize(dataset["label_config"].get)(
        np.concatenate([dataset["train"]["labels"], dataset["test"]["labels"]], axis=0)
    )

    train_len = dataset["train"]["labels"].shape[0]
    dataset["train"]["labels"] = new_labels[:train_len]
    dataset["test"]["labels"] = new_labels[train_len:]

    return dataset


def __one_hot_encode_label_vector(labels: RealMatrix) -> RealMatrix:
    ohe = np.zeros((labels.size, np.unique(labels).size))
    ohe[np.arange(labels.size), labels] = 1

    return ohe


def __one_hot_encode(dataset: TrainTestSet) -> TrainTestSet:
    new_labels = __one_hot_encode_label_vector(
        np.concatenate([dataset["train"]["labels"], dataset["test"]["labels"]], axis=0)
    )

    train_len = dataset["train"]["labels"].shape[0]
    dataset["train"]["labels"] = new_labels[:train_len]
    dataset["test"]["labels"] = new_labels[train_len:]

    return dataset


def __numpy_to_tensor(data: RealMatrix) -> tf.Tensor:
    return tf.constant(data)


def __make_tensors(dataset: TrainTestSet) -> TrainTestSet:
    dataset["train"]["data"] = __numpy_to_tensor(dataset["train"]["data"])
    dataset["train"]["labels"] = __numpy_to_tensor(dataset["train"]["labels"])
    dataset["test"]["data"] = __numpy_to_tensor(dataset["test"]["data"])
    dataset["test"]["labels"] = __numpy_to_tensor(dataset["test"]["labels"])

    return dataset


def make_tf_train_test_set(dataset: TrainTestSet) -> TrainTestSet:
    """
    Make TrainTestSet viable for tensorflow use
    """
    return __make_tensors(__one_hot_encode(dataset))
