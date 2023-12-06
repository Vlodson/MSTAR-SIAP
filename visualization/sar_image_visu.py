import os
import matplotlib.pyplot as plt
import numpy as np

from custom_types import CSARImage, Header, SARImage
from configs.config_loader import IO


def __make_image_title(header: Header) -> str:
    return f"{header['filename']} - {header['target']}"


def visualize_sar_image(image: SARImage, save: bool = False) -> None:
    title = __make_image_title(image["header"])

    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)

    # magnitude
    ax[0].imshow(image["magnitude"], "gray")
    ax[0].title.set_text("Magnitude")
    ax[0].invert_yaxis()
    ax[0].xaxis.tick_top()

    # phase
    ax[1].imshow(image["phase"], "hsv")
    ax[1].title.set_text("Phase")
    ax[1].invert_yaxis()
    ax[1].xaxis.tick_top()

    plt.show()

    if save:
        plt.savefig(os.path.join(IO["image_save_dir"], title + "_mag_phase_plot.jpeg"))


def visualize_real_imag_cimage_parts(cimage: CSARImage, save: bool = False) -> None:
    title = __make_image_title(cimage["header"])

    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)

    # real
    ax[0].imshow(cimage["cimage"].real, "gray")
    ax[0].title.set_text("Real Axis")

    # imaginary
    ax[1].imshow(cimage["cimage"].imag, "gray")
    ax[1].title.set_text("Imaginary Axis")

    plt.show()

    if save:
        plt.savefig(os.path.join(IO["image_save_dir"], title + "_real_imag_plot.jpeg"))


def visualize_cimage_plane(cimage: CSARImage, save: bool = False) -> None:
    title = __make_image_title(cimage["header"])

    plt.scatter(
        cimage["cimage"].real, cimage["cimage"].imag, c=np.absolute(cimage["cimage"])
    )
    plt.title(title)
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")

    plt.show()

    if save:
        plt.savefig(os.path.join(IO["image_save_dir"], title + "_cplane_plot.jpeg"))


def visualize_cimage_quiver(cimage: CSARImage, save: bool = False) -> None:
    title = __make_image_title(cimage["header"])

    plt.quiver(
        cimage["cimage"].real,
        cimage["cimage"].imag,
        np.absolute(cimage["cimage"]),
        np.angle(cimage["cimage"]),
        np.absolute(cimage["cimage"]),
    )
    plt.title(title)

    plt.show()

    if save:
        plt.savefig(os.path.join(IO["image_save_dir"], title + "_quiver_plot.jpeg"))
