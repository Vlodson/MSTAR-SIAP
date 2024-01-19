from neural_networks.dnn import make_model, compile_model
from visualizations.model_history import plot_model_history
from dataset_maker.dataset_makers import DNNDatasetMaker


def main():
    dsm = DNNDatasetMaker()
    dsm.create_dataset()
    ds = dsm.make_neural_network_input()

    model = compile_model(make_model(len(dsm.label_config)))

    hist = model.fit(
        x=ds["train_data"],
        y=ds["train_labels"],
        batch_size=128,
        epochs=100,
        verbose="auto",
        validation_data=(ds["validation_data"], ds["validation_labels"]),
        shuffle=True,
        use_multiprocessing=True,
    )

    plot_model_history(hist)


if __name__ == "__main__":
    main()
