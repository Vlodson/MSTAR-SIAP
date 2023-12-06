from io_utils.converter import convert_pkl_to_sar


def main():
    images = convert_pkl_to_sar()
    print(len(images))


if __name__ == "__main__":
    main()
