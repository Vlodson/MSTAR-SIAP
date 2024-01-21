import matplotlib.pyplot as plt
import numpy as np

from custom_types import Header, Image


def __make_image_title(header: Header) -> str:
    return f"{header['filename']} - {header['target']}"


def plot_magnitude_phase(image: Image) -> None:
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


def plot_complex_components(image: Image) -> None:
    title = __make_image_title(image["header"])

    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)

    complex_image = image["magnitude"] * np.exp(1.0j * image["phase"])

    # real
    ax[0].imshow(complex_image.real, "gray")
    ax[0].title.set_text("Real Axis")

    # imaginary
    ax[1].imshow(complex_image.imag, "gray")
    ax[1].title.set_text("Imaginary Axis")

    plt.show()


def plot_complex_plane(image: Image) -> None:
    title = __make_image_title(image["header"])

    complex_image = image["magnitude"] * np.exp(1.0j * image["phase"])

    plt.scatter(complex_image.real, complex_image.imag, c=np.absolute(complex_image))
    plt.title(title)
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")

    plt.show()


def quiver_plot_complex_plane(image: Image) -> None:
    title = __make_image_title(image["header"])

    complex_image = image["magnitude"] * np.exp(1.0j * image["phase"])

    plt.quiver(
        complex_image.real,
        complex_image.imag,
        np.absolute(complex_image),
        np.angle(complex_image),
        np.absolute(complex_image),
    )
    plt.title(title)

    plt.show()
