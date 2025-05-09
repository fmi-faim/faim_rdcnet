import argparse
import datetime
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from faim_ipa.utils import get_git_root
from pytorch_rdc_net.generic_model import RDCNet

sys.path.append(str(get_git_root()))

from source.data.Organoid3D import Organoid3D
from source.rdcnet_config import RDCNetConfig
from source.trainer_config import TrainerConfig


def main(
    rdcnet_config: RDCNetConfig,
    trainer_config: TrainerConfig,
):
    torch.set_float32_matmul_precision("high")
    dm = Organoid3D(get_git_root() / "raw_data" / "organoid-3d")
    model = RDCNet(
            dim=rdcnet_config.dim,
            in_channels=rdcnet_config.in_channels,
            down_sampling_factor=rdcnet_config.down_sampling_factor,
            down_sampling_channels=rdcnet_config.down_sampling_channels,
            spatial_dropout_p=rdcnet_config.spatial_dropout_p,
            channels_per_group=rdcnet_config.channels_per_group,
            n_groups=rdcnet_config.n_groups,
            dilation_rates=rdcnet_config.dilation_rates,
            steps=rdcnet_config.steps,
            instance_size=rdcnet_config.instance_size,
        )

    output_dir = (
        get_git_root()
        / "processed_data"
        / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(os.getcwd()).name}"
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

    results = trainer.test(model, dataloaders=dm.val_dataloader(), ckpt_path="best")

    with open(output_dir / f"{v_num}-val_results.yaml", "w") as f:
        yaml.safe_dump(results, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rdcnet_config",
        type=str,
        default="rdcnet_config.yaml",
    )
    parser.add_argument(
        "--trainer_config",
        type=str,
        default="trainer_config.yaml",
    )

    args = parser.parse_args()

    rdcnet_config = RDCNetConfig.load(Path(args.rdcnet_config))
    trainer_config = TrainerConfig.load(Path(args.trainer_config))

    main(
        rdcnet_config=rdcnet_config,
        trainer_config=trainer_config,
    )