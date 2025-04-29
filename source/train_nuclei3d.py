import argparse
import os
import sys
import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from faim_ipa.utils import get_git_root
from pytorch_rdc_net.generic_model import RDCNet

sys.path.append(str(get_git_root()))

from source.data.Nuclei3D import Nuclei3D
from source.trainer_config import TrainerConfig


def main(
    trainer_config: TrainerConfig,
):
    torch.set_float32_matmul_precision("high")
    dm = Nuclei3D(get_git_root() / "raw_data" / "nuclei-3d")
    model = RDCNet(
        dim=3,
        in_channels=1,
        down_sampling_factor=(1, 8, 8),
        down_sampling_channels=32,
        spatial_dropout_p=0.1,
        channels_per_group=64,
        n_groups=8,
        dilation_rates=[1, 2, 4],
        steps=5,
        instance_size=(5.0, 40.0, 40.0),
    )

    output_dir = (
        get_git_root()
        / "processed_data"
        / f"{datetime.datetime.now().isoformat()}_{Path(os.getcwd()).name}"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=trainer_config.max_epochs,
        precision=trainer_config.precision,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="f1",
                mode="max",
                save_top_k=1,
                filename="rdcnet-{epoch:02d}-{f1:.2f}",
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    v_num = trainer.logger.version

    results = trainer.test(model, dataloaders=dm.train_dataloader(), ckpt_path="best")

    with open(output_dir / f"{v_num}-train_results.yaml", "w") as f:
        yaml.safe_dump(results, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer_config",
        type=str,
        default="trainer_config.yaml",
    )

    args = parser.parse_args()

    trainer_config = TrainerConfig.load(Path(args.trainer_config))

    main(
        trainer_config=trainer_config,
    )
