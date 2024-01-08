import numpy as np
import tensorflow as tf
from custom_types import RealMatrix, TrainTestSet
from preprocessing.utils.normalization import (
    ComplexSphericalNormalization,
    ZScoreNormalization,
    ImageNormalization,
)


def normalize_set(dataset: TrainTestSet) -> TrainTestSet:
    # since I can't know ahead of time what normalization to use, I have to do this
    if (
        dataset["dtype"] == np.complex64
    ):  # should be complex64 since it's loaded as float32
        dataset = ComplexSphericalNormalization(dataset).normalize()
    elif len(dataset["shape"]) == 2:
        dataset = ZScoreNormalization(dataset).normalize()
    else:
        dataset = ImageNormalization(dataset).normalize()

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
    remapper = np.vectorize(dataset["label_config"].get)

    dataset["train"]["labels"] = remapper(dataset["train"]["labels"])
    dataset["test"]["labels"] = remapper(dataset["test"]["labels"])

    return dataset


def __one_hot_encode_label_vector(labels: RealMatrix, nb_of_classes: int) -> RealMatrix:
    labels = labels.reshape(-1)
    return np.eye(nb_of_classes)[labels]


def __one_hot_encode(dataset: TrainTestSet) -> TrainTestSet:
    dataset["train"]["labels"] = __one_hot_encode_label_vector(
        dataset["train"]["labels"], len(dataset["label_config"])
    )
    dataset["test"]["labels"] = __one_hot_encode_label_vector(
        dataset["test"]["labels"], len(dataset["label_config"])
    )

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
