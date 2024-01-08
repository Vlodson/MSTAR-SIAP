import tensorflow as tf

from preprocessing.dataset_loader import open_train_test_set
from preprocessing.utils.train_test_manipulation import make_tf_train_test_set
from neural_networks.conv_neural_network import make_model, compile_model
from visualization.model_history_visu import plot_model_history


def main():
    ds = open_train_test_set("public_chip_cnn")
    ds = make_tf_train_test_set(ds)

    ds["train"]["data"] = tf.expand_dims(ds["train"]["data"][..., 0], axis=3)
    ds["test"]["data"] = tf.expand_dims(ds["test"]["data"][..., 0], axis=3)

    model = compile_model(make_model(out_units=len(ds["label_config"])))

    hist = model.fit(
        x=ds["train"]["data"],
        y=ds["train"]["labels"],
        batch_size=64,
        epochs=100,
        verbose="auto",
        validation_split=0.2,
        shuffle=True,
    )

    plot_model_history(hist)


if __name__ == "__main__":
    main()
