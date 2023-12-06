from typing import TypedDict
import yaml

from custom_types import (
    IOConfig,
    ImageLoaderConfig,
    ImageReaderConfig,
    DatasetLoaderConfig,
)


def __load_yaml_file(yaml_file: str) -> TypedDict:
    with open(yaml_file, "r+", encoding="utf-8") as _f:
        data = yaml.safe_load(_f)

    return data


IMAGE_LOADER: ImageLoaderConfig = __load_yaml_file("./configs/image_loader.yaml")
IMAGE_READER: ImageReaderConfig = __load_yaml_file("./configs/image_reader.yaml")
IO: IOConfig = __load_yaml_file("./configs/io.yaml")
DATASET_LOADER: DatasetLoaderConfig = __load_yaml_file("./configs/dataset_loader.yaml")
