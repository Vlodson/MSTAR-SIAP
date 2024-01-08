import os
import pickle as pkl
from typing import List

from custom_types import DataPoint, SARImage, Dataset
from preprocessing.utils.datapoint_manipulation import (
    csar_image_to_datapoint,
    filter_unknown_labels,
    sar_image_to_cnn_datapoint,
    sar_image_to_dnn_datapoint,
)
from preprocessing.utils.image_manipulation import (
    flatten_image,
    resize_images,
    sar_image_to_csar_image,
)
from preprocessing.utils.datapoint_manipulation import (
    combine_datapoints,
)


def cvnn_preprocessing(images: List[SARImage]) -> Dataset:
    """
    Preprocesses passed sar images for the complex valued neural network

    Returns:
        - data in a 2D matrix of complex row vectors
        - labels in a column vector of integers
    """
    resized_images = resize_images(images)

    datapoints: List[DataPoint] = []
    for image in resized_images:
        csar = sar_image_to_csar_image(image)
        csar["cimage"] = flatten_image(csar["cimage"])
        datapoints.append(csar_image_to_datapoint(csar))

    return combine_datapoints(filter_unknown_labels(datapoints))


def dnn_preprocessing(images: List[SARImage]) -> Dataset:
    """
    Preprocesses passed sar images for the deep neural network

    Returns:
        - data in a tensor of row vectors
        - labels in a vector of categories
    """
    resized_images = resize_images(images)

    dataset = combine_datapoints(
        filter_unknown_labels(
            [sar_image_to_dnn_datapoint(image) for image in resized_images]
        )
    )

    return dataset


def cnn_preprocessing(images: List[SARImage]) -> Dataset:
    """
    Preprocesses passed sar images for the convolutional neural network

    Returns:
        - data in a tensor of 2 channel images, first one being the magnitude and the second one being phase
        - labels in a vector of categories
    """
    resized_images = resize_images(images)

    dataset = combine_datapoints(
        filter_unknown_labels(
            [sar_image_to_cnn_datapoint(image) for image in resized_images]
        )
    )

    return dataset


def save_image_set(dataset: Dataset, name: str) -> None:
    with open(os.path.join("preprocessed_images", name + ".pkl"), "wb") as _f:
        pkl.dump(dataset, _f)


def load_image_set(name: str) -> Dataset:
    with open(os.path.join("preprocessed_images", name + ".pkl"), "rb") as _f:
        dataset = pkl.load(_f)

    return dataset
