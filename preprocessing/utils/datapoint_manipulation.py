from typing import List
import numpy as np
from custom_types import CSARImage, DataPoint, Dataset, SARImage
from preprocessing.labels_dict import LABELS
from preprocessing.utils.image_manipulation import flatten_image


def __encode_label(label: str) -> int:
    for lab, code in LABELS.items():
        if lab in label:
            return code

    return -1


def csar_image_to_datapoint(image: CSARImage) -> DataPoint:
    return {
        "data": image["cimage"],
        "label": __encode_label(image["header"]["target"]),
    }


def sar_image_to_dnn_datapoint(image: SARImage) -> DataPoint:
    return {
        "data": np.concatenate(
            [flatten_image(image["magnitude"]), flatten_image(image["phase"])],
        ),
        "label": __encode_label(image["header"]["target"]),
    }


def sar_image_to_cnn_datapoint(image: SARImage) -> DataPoint:
    return {
        "data": np.stack([image["magnitude"], image["phase"]], axis=2),
        "label": __encode_label(image["header"]["target"]),
    }


def filter_unknown_labels(datapoints: List[DataPoint]) -> List[DataPoint]:
    return [datapoint for datapoint in datapoints if datapoint["label"] != -1]


def combine_datapoints(datapoints: List[DataPoint]) -> Dataset:
    return {
        "data": np.stack([datapoint["data"] for datapoint in datapoints], axis=0),
        "labels": np.stack([datapoint["label"] for datapoint in datapoints], axis=0),
    }
