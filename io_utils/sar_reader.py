from io import BufferedReader
from typing import Dict, Optional, Tuple
import pickle as pkl
import numpy as np
import numpy.typing as npt

from custom_types import Header, SARImage


def __parse_phoenix_header(file: BufferedReader) -> Dict[str, str]:
    header = {}
    for line in file:
        line = line.decode("utf-8")
        line = line.strip()

        if not line:
            continue

        if "PhoenixHeaderVer" in line:
            continue

        if "EndofPhoenixHeader" in line:
            break

        key, value = line.split("=")
        header[key.strip()] = value.strip()

    return header


def __parse_header(phoenix_header: Dict[str, str]) -> Header:
    return {
        "filename": phoenix_header["Filename"],
        "cols": int(phoenix_header["NumberOfColumns"]),
        "rows": int(phoenix_header["NumberOfRows"]),
        "target": phoenix_header["TargetType"]
        if "TargetType" in phoenix_header
        else "clutter",
        "angle": phoenix_header["DesiredDepression"],
        "classification": phoenix_header["Classification"],
    }


def __parse_data(
    file: BufferedReader, header: Header
) -> Tuple[npt.NDArray[np.float32] | None, npt.NDArray[np.float32] | None]:
    try:
        data = (
            np.fromfile(file, dtype=">f4")
            .reshape(-1, header["rows"], header["cols"])
            .astype(np.float32)
        )
    except ValueError:  # corrupted data can't be reshaped
        return None, None

    # return magnitude, phase
    return data[0], data[1]


def read_sar_image(path: str) -> Optional[SARImage]:
    with open(path, "rb") as _f:
        header = __parse_header(__parse_phoenix_header(_f))
        magnitude, phase = __parse_data(_f, header)

    if magnitude is None and phase is None:
        return None

    return {"header": header, "magnitude": magnitude, "phase": phase}


def save_sar_image_as_pkl(path: str, image: SARImage) -> None:
    with open(path, "wb") as _f:
        pkl.dump(image, _f)
