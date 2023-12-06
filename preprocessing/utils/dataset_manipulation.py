from typing import List, Tuple
import numpy as np

from custom_types import Dataset


def shuffle_dataset(dataset: Dataset) -> Dataset:
    np.random.seed(42)

    indexes = np.arange(dataset["data"].shape[0])
    np.random.shuffle(indexes)

    return {"data": dataset["data"][indexes], "labels": dataset["labels"][indexes]}


def __validate_shapes(sets: List[Dataset]) -> Tuple[bool, Tuple[int]]:
    return (
        all(ds["data"].shape[1:] == sets[0]["data"].shape[1:] for ds in sets),
        (-1, *sets[0]["data"].shape[1:]),
    )


def __validate_types(sets: List[Dataset]) -> Tuple[bool, type]:
    return (
        all(ds["data"].dtype == sets[0]["data"].dtype for ds in sets),
        sets[0]["data"].dtype,
    )


def validate_set_cohesion(
    sets: List[Dataset], set_type: str
) -> Tuple[Tuple[int], type]:
    validity, shape = __validate_shapes(sets)
    assert validity, f"Shapes missmatch for given {set_type} set"

    validity, dtype = __validate_types(sets)
    assert validity, f"Data types missmatch for given {set_type} set"

    return shape, dtype


def combine_set_lists(sets: List[Dataset]) -> Dataset:
    return {
        "data": np.concatenate([ds["data"] for ds in sets], axis=0),
        "labels": np.concatenate([ds["labels"] for ds in sets], axis=0),
    }
