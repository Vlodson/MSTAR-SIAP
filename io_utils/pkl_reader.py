import pickle as pkl
from custom_types import SARImage


def read_sar_image_from_pkl(path: str) -> SARImage:
    with open(path, "rb") as _f:
        image: SARImage = pkl.load(_f)

    return image
