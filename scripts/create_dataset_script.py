from preprocessing.dataset_loader import (
    load_train_test_set,
    save_train_test_set,
    open_train_test_set,
)


def main():
    ds = load_train_test_set()
    print(
        ds["shape"],
        ds["train"]["data"].shape,
        ds["train"]["data"].max(),
        ds["train"]["data"].min(),
    )

    save_train_test_set(ds, "public_chip_cnn")

    ds = open_train_test_set("public_chip_cnn")
    print(ds["shape"])


if __name__ == "__main__":
    main()
