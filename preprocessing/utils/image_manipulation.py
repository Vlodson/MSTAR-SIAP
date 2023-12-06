from typing import List, Tuple
import cv2
import numpy as np

from custom_types import CSARImage, ComplexMatrix, RealMatrix, SARImage


def __resize_image(image: RealMatrix, new_size: Tuple[int, int]) -> RealMatrix:
    return cv2.resize(src=image, dsize=new_size)


def resize_images(
    sar_images: List[SARImage], size: Tuple[int, int] = (100, 100)
) -> List[SARImage]:
    for image in sar_images:
        image["magnitude"] = __resize_image(image["magnitude"], size)
        image["phase"] = __resize_image(image["phase"], size)
        image["header"]["rows"] = size[0]
        image["header"]["cols"] = size[1]

    return sar_images


def sar_image_to_csar_image(sar_image: SARImage) -> CSARImage:
    return {
        "header": sar_image["header"],
        "cimage": sar_image["magnitude"] * np.exp(1.0j * sar_image["phase"]),
    }


def flatten_image(image: RealMatrix | ComplexMatrix) -> RealMatrix | ComplexMatrix:
    return image.ravel()
