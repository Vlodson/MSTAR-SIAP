from visualizations.sar_image_visu import plot_magnitude_phase, plot_complex_plane
from io_utils.loader import load_images_from_config


def main():
    images = load_images_from_config()

    for image in images:
        plot_magnitude_phase(image)
        plot_complex_plane(image)


if __name__ == "__main__":
    main()
