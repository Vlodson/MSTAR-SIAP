from dataset_maker.dataset_makers import CVNNDatasetMaker


def main():
    dsm = CVNNDatasetMaker()
    dsm.create_dataset()
    ds = dsm.make_neural_network_input()

    dsm.save_neural_network_input("cvnn_dataset", ds)


if __name__ == "__main__":
    main()
