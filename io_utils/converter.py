import os
from typing import List
import tqdm

from io_utils.sar_reader import read_sar_image, save_sar_image_as_pkl
from io_utils.pkl_reader import read_sar_image_from_pkl
from custom_types import SARImage
from configs.config_loader import IMAGE_LOADER, IMAGE_READER


def convert_sar_to_pkl() -> None:
    for load_dir in (dir_bar := tqdm.tqdm(IMAGE_LOADER["directories"])):
        dir_bar.set_postfix({"Directory": load_dir["load_path"]})

        for fname in (f_bar := tqdm.tqdm(next(os.walk(load_dir["load_path"]))[-1])):
            f_bar.set_postfix({"File": fname})

            if ".JPG" in fname or ".HTM" in fname or "ERROR" in fname:
                continue

            fp = os.path.join(load_dir["load_path"], fname)
            image = read_sar_image(fp)

            if image is None:  # skip corrupted data
                continue

            if not os.path.isdir(load_dir["save_path"]):
                os.makedirs(load_dir["save_path"])

            save_fp = os.path.join(
                load_dir["save_path"], image["header"]["filename"] + ".pkl"
            )
            save_sar_image_as_pkl(save_fp, image)


def convert_pkl_to_sar() -> List[SARImage]:
    images = []

    for read_dir in (dir_bar := tqdm.tqdm(IMAGE_READER["directories"])):
        dir_bar.set_postfix({"Directory": read_dir})

        for fname in (f_bar := tqdm.tqdm(next(os.walk(read_dir))[-1])):
            f_bar.set_postfix({"File": fname})

            fp = os.path.join(read_dir, fname)
            images.append(read_sar_image_from_pkl(fp))

    return images
