import numpy as np
from io_utils.converter import convert_pkl_to_sar
from preprocessing.image_preprocessing import (
    cvnn_preprocessing,
    dnn_preprocessing,
    cnn_preprocessing,
    save_image_set,
    load_image_set,
)


def main():
    images = convert_pkl_to_sar()
    save_image_set(dnn_preprocessing(images), "pblc_trn_t72_bmp_btr_dnn")
    ds = load_image_set("pblc_trn_t72_bmp_btr_dnn")
    print(np.unique(ds["labels"]))
    print(ds["data"].dtype)
    print(type(ds["data"]), type(ds["labels"]))
    print(ds["data"].shape)


if __name__ == "__main__":
    main()
