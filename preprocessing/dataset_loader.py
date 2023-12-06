import os
import pickle as pkl

from configs.config_loader import DATASET_LOADER
from custom_types import TrainTestSet
from preprocessing.image_preprocessing import load_image_set
from preprocessing.utils.dataset_manipulation import (
    combine_set_lists,
    shuffle_dataset,
    validate_set_cohesion,
)
from preprocessing.utils.train_test_manipulation import (
    add_label_config,
    normalize_set,
    remap_labels,
)


def load_train_test_set() -> TrainTestSet:
    train_sets = [load_image_set(name) for name in DATASET_LOADER["train_image_sets"]]
    test_sets = [load_image_set(name) for name in DATASET_LOADER["test_image_sets"]]

    _ = validate_set_cohesion(train_sets, "train")
    _ = validate_set_cohesion(test_sets, "test")
    shape, dtype = validate_set_cohesion(train_sets + test_sets, "train-test split")

    return remap_labels(
        add_label_config(
            normalize_set(
                {
                    "train": shuffle_dataset(combine_set_lists(train_sets)),
                    "test": shuffle_dataset(combine_set_lists(test_sets)),
                    "label_config": {},
                    "shape": shape,
                    "dtype": dtype,
                }
            )
        )
    )


def save_train_test_set(dataset: TrainTestSet, name: str) -> None:
    with open(os.path.join("train_test_sets", name + ".pkl"), "wb") as _f:
        pkl.dump(dataset, _f)


def open_train_test_set(name: str) -> TrainTestSet:
    with open(os.path.join("train_test_sets", name + ".pkl"), "rb") as _f:
        dataset: TrainTestSet = pkl.load(_f)

    return dataset
