import os
from typing import Dict
import yaml


def __load_yaml(path: str) -> Dict:
    with open(path, "r+", encoding="utf-8") as _f:
        cfg = yaml.safe_load(_f)

    return cfg


ROOT = "./"
CONFIG_DIR = os.path.join(ROOT, "configs")

READER = __load_yaml(os.path.join(CONFIG_DIR, "reader.yaml"))
LOADER = __load_yaml(os.path.join(CONFIG_DIR, "loader.yaml"))
DATA_PREPROCESSING = __load_yaml(os.path.join(CONFIG_DIR, "image_preprocessing.yaml"))
DATASET_MAKER = __load_yaml(os.path.join(CONFIG_DIR, "dataset_maker.yaml"))
