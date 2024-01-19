import os
import tqdm
from global_configs import READER
from io_utils.reader import read_sar_image, save_sar_image_as_pkl


def main():
    for load_save_pair in tqdm.tqdm(READER["directories"]):
        load_dir, save_dir = load_save_pair["load_dir"], load_save_pair["save_dir"]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for fname in tqdm.tqdm(os.listdir(load_dir)):
            img = read_sar_image(os.path.join(load_dir, fname))

            if img is not None:
                save_sar_image_as_pkl(
                    os.path.join(save_dir, img["header"]["filename"]), img
                )


if __name__ == "__main__":
    main()
