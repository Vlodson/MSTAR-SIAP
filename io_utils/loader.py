import os
import pickle as pkl
from typing import List
import tqdm
from custom_types import Image
from global_configs import LOADER


def load_image_pkl(path: str) -> Image:
    with open(path, "rb") as _f:
        img = pkl.load(_f)

    return img


def load_images_from_config() -> List[Image]:
    return [
        load_image_pkl(os.path.join(_dir, fname))
        for _dir in tqdm.tqdm(LOADER["directories"])
        for fname in tqdm.tqdm(os.listdir(_dir))
    ]
