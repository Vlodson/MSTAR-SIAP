from typing import List, TypedDict, TypeAlias, Dict, Tuple
import numpy.typing as npt
import numpy as np


RealMatrix: TypeAlias = npt.NDArray[np.float32]
ComplexMatrix: TypeAlias = npt.NDArray[np.complex64]


class Header(TypedDict):
    filename: str
    cols: int
    rows: int
    target: str
    angle: str  # angle at which image was taken
    classification: str  # class after inference


class SARImage(TypedDict):
    """
    Typed dict with keys for magnitude and phase seperately
    """

    header: Header
    magnitude: RealMatrix
    phase: RealMatrix


class CSARImage(TypedDict):
    """
    Typed dict with key for complex valued image
    """

    header: Header
    cimage: ComplexMatrix


class DataPoint(TypedDict):
    label: int
    data: RealMatrix | ComplexMatrix


class Dataset(TypedDict):
    data: RealMatrix | ComplexMatrix
    labels: RealMatrix


class TrainTestSet(TypedDict):
    train: Dataset
    test: Dataset
    label_config: Dict[
        int, int
    ]  # maps labels from LABELS dict to new labels in train test set
    dtype: type
    shape: Tuple[int]


class LoadDir(TypedDict):
    load_path: str
    save_path: str


class ImageLoaderConfig(TypedDict):
    directories: List[LoadDir]


class ImageReaderConfig(TypedDict):
    directories: List[str]


class IOConfig(TypedDict):
    image_save_dir: str


class DatasetLoaderConfig(TypedDict):
    train_image_sets: List[str]
    test_image_sets: List[str]
