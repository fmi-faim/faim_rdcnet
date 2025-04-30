import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch_rdc_net.generic_model import RDCNet
from tifffile import imread, imwrite


def main(input_dir: Path, output_dir: Path, model_checkpoint: Path):
    model = RDCNet.load_from_checkpoint(model_checkpoint)

    input_files = input_dir.glob("*.tif")
    for input_file in input_files:
        img = imread(input_file).astype(np.float32)[:, ::2, ::2]
        if img.max() > 1:
            img = np.clip(img / np.quantile(img, 0.999), 0, 1)

        shape = img.shape
        pad_z = (0, (shape[0] // model.hparams.down_sampling_factor[0] + 1) * model.hparams.down_sampling_factor[0] - shape[0])
        pad_y = (0, (shape[1] // model.hparams.down_sampling_factor[1] + 1) * model.hparams.down_sampling_factor[1] - shape[1])
        pad_x = (0, (shape[2] // model.hparams.down_sampling_factor[2] + 1) * model.hparams.down_sampling_factor[2] - shape[2])
        img = np.pad(img, (pad_z, pad_y, pad_x), mode="reflect")

        pred = model.predict_instances(torch.from_numpy(img[np.newaxis, np.newaxis]))[0, 0, :shape[0], :shape[1], :shape[2]]

        output_file = output_dir / (input_file.stem + "_SEG.tif")
        imwrite(output_file, pred.astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory containing .tif files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for segmented .tif files")
    parser.add_argument("--model_checkpoint", type=Path, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    main(
        Path(args.input_dir),
        Path(args.output_dir),
        Path(args.model_checkpoint)
    )