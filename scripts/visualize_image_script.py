import numpy as np
from custom_types import CSARImage, SARImage
from io_utils.converter import convert_pkl_to_sar
from visualization.sar_image_visu import (
    visualize_sar_image,
    visualize_real_imag_cimage_parts,
    visualize_cimage_plane,
    visualize_cimage_quiver,
)


def main():
    images = convert_pkl_to_sar()
    image: SARImage = images[0]
    cimage: CSARImage = {
        "header": image["header"],
        "cimage": image["magnitude"] * np.exp(1.0j * image["phase"]),
    }

    visualize_sar_image(image)
    visualize_real_imag_cimage_parts(cimage)
    visualize_cimage_plane(cimage)
    visualize_cimage_quiver(cimage)


if __name__ == "__main__":
    main()
