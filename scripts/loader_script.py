from io_utils.loader import load_images_from_config


def main():
    imgs = load_images_from_config()
    print(len(imgs))


if __name__ == "__main__":
    main()
