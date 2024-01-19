import os
from typing import List
import pickle as pkl
import numpy as np
import numpy.typing as npt
from scipy.ndimage import rotate
import tqdm
from custom_types import Image, Datapoint
from io_utils.loader import load_images_from_config
from global_configs import DATA_PREPROCESSING


def rotate_image(
    image: npt.NDArray[np.float32], angle: float
) -> npt.NDArray[np.float32]:
    # angle in degrees
    return rotate(image, -angle)


def crop_center(
    image: npt.NDArray[np.float32], new_height: int, new_width: int
) -> npt.NDArray[np.float32]:
    center = np.array(image.shape) // 2
    x = center[1] - new_width // 2
    y = center[0] - new_height // 2

    return image[y : y + new_height, x : x + new_width]


def preprocess_image(image: Image, new_height: int, new_width: int) -> Image:
    for key in ["magnitude", "phase"]:
        image[key] = crop_center(
            rotate_image(image[key], image["header"]["azimuth"]), new_height, new_width
        )
    image["header"]["rows"] = new_height
    image["header"]["cols"] = new_width

    return image


def map_label(label: str) -> int:
    try:
        key = [k for k in DATA_PREPROCESSING["label_mapper"] if k in label][0]
    except IndexError:
        return -1

    return DATA_PREPROCESSING["label_mapper"][key]


def datapoint_from_image(image: Image) -> Datapoint:
    return {
        "image": np.stack([image["magnitude"], image["phase"]], axis=-1),
        "label": map_label(image["header"]["target"]),
    }


def datapoints_from_config() -> List[Datapoint]:
    images = load_images_from_config()

    return [
        datapoint_from_image(
            preprocess_image(
                img,
                DATA_PREPROCESSING["new_shape"]["height"],
                DATA_PREPROCESSING["new_shape"]["width"],
            )
        )
        for img in tqdm.tqdm(images)
    ]


def save_datapoints(datapoints: List[Datapoint], name: str) -> None:
    with open(os.path.join(DATA_PREPROCESSING["save_dir"], name + ".pkl"), "wb") as _f:
        pkl.dump(datapoints, _f)
