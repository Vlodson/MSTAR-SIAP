from preprocessing.image_preprocessing import datapoints_from_config, save_datapoints


def main():
    datapoints = datapoints_from_config()
    save_datapoints(datapoints, "test")


if __name__ == "__main__":
    main()
