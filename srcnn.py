import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile as tiff
from keras.callbacks import ModelCheckpoint

from model import get_model

from util import clean_mkdir, load_run_data


def run(data_path, model_weights_path, output_path):
    output_path = Path(output_path)
    model = get_model(model_weights_path)
    x = load_run_data(data_path)
    out_array = model.predict(x)
    for index in range(out_array.shape[0]):
        num, rows, cols, channels = out_array.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    if out_array[index][i][j][k] > 1.0:
                        out_array[index][i][j][k] = 1.0

        out_img = Image.fromarray(np.uint8(out_array[index] * 255))
        #a = np.uint16(out_array[0] * 65025)
        out_img.save(str(output_path / "{}.jpg".format(index)))
        #tiff.imwrite(str(output_path / "{}.jpg".format(index)), a, photometric='rgb')
        #print(type(a))
        #print(a.dtype)
        #print(a)
        #out_img.save(str(output_path / "{}.jpg".format(index)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate/run SRCNN models")
    parser.add_argument(
        "--action",
        type=str,
        default="test",
        help="Train or test the model.",
        choices={"train", "test", "run"},
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Filepath of a saved model to use for eval or inference or"
        + "filepath where to save a newly trained model.",
    )
    parser.add_argument(
        "--output_path", type=str, help="Filepath to output results from run action"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2) #Remember to change those back
    parser.add_argument(
        "--data_path",
        type=str,
        help="Filepath to data directory. Image data should exist at <data_path>/images",
        default="data",
    )
    params = parser.parse_args()
    if params.action == "train":
        train(params.data_path, params.model_path , params.epochs, params.batch_size)
    elif params.action == "test":
        test(params.data_path, params.model_path)
    elif params.action == "run":
        run(params.data_path, params.model_path, params.output_path)
