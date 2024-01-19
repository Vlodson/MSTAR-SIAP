from typing import Dict, TypedDict
import numpy.typing as npt
import numpy as np
from tensorflow import Tensor


class Header(TypedDict):
    filename: str
    cols: int
    rows: int
    target: str
    angle: float  # SAR angle at which image was taken
    azimuth: float  # angle of object on ground
    classification: str  # class after inference


class Image(TypedDict):
    """
    Typed dict with keys for magnitude and phase seperately
    """

    header: Header
    magnitude: npt.NDArray[np.float32]
    phase: npt.NDArray[np.float32]


class Datapoint(TypedDict):
    image: npt.NDArray[np.float32]
    label: int


class SingleSet(TypedDict):
    data: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int32]


class Dataset(TypedDict):
    train: SingleSet
    test: SingleSet
    validation: SingleSet


class CSingleSet(TypedDict):
    data: npt.NDArray[np.complex64]
    labels: npt.NDArray[np.int32]


class CDataset(TypedDict):
    train: CSingleSet
    test: CSingleSet
    validation: CSingleSet
    label_config: Dict[int, int]


class TFDataset(TypedDict):
    train_data: Tensor
    test_data: Tensor
    validation_data: Tensor
    train_labels: Tensor
    test_labels: Tensor
    validation_labels: Tensor
    label_config: Dict[int, int]
