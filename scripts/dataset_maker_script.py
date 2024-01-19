from dataset_maker.dataset_makers import CNNDatasetMaker


def main():
    dsm = CNNDatasetMaker()
    dsm.create_dataset()
    tf_ds = dsm.make_neural_network_input()

    print(tf_ds["train_data"].shape)


if __name__ == "__main__":
    main()
