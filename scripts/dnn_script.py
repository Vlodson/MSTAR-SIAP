from preprocessing.dataset_loader import open_train_test_set
from preprocessing.utils.train_test_manipulation import make_tf_train_test_set
from neural_networks.deep_neural_network import make_model, compile_model


def main():
    ds = open_train_test_set("public_chip_dnn")
    ds = make_tf_train_test_set(ds)

    model = compile_model(make_model(out_units=len(ds["label_config"])))

    model.fit(
        x=ds["train"]["data"],
        y=ds["train"]["labels"],
        batch_size=128,
        epochs=100,
        verbose="auto",
        validation_split=0.2,
        shuffle=False,
    )


if __name__ == "__main__":
    main()
