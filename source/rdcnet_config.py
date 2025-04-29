from pathlib import Path
from typing import Optional, Union

import questionary
from faim_ipa.utils import IPAConfig


class RDCNetConfig(IPAConfig):
    model_path: Optional[Path] = None
    dim: int = 2
    in_channels: int = 1
    down_sampling_factor: Union[tuple[int, int], tuple[int, int, int]] = (6, 6)
    down_sampling_channels: int = 8
    spatial_dropout_p: float = 0.1
    channels_per_group: int = 32
    n_groups: int = 4
    dilation_rates: list[int] = [1, 2, 4, 8, 16]
    steps: int = 6
    lr: float = 0.001
    instance_size: Union[tuple[float, float], tuple[float, float, float]] = (20.0, 20.0)

    def config_name(self) -> str:
        return "rdcnet_config.yaml"

    def prompt(self):
        fine_tune = questionary.confirm("Fine tune?").ask()
        if fine_tune:
            self.model_path = questionary.path(
                "Model path",
                default=str(self.model_path),
            ).ask()
        else:
            self.dim = int(
                questionary.text(
                    "dim",
                    default=str(self.dim),
                ).ask()
            )
            self.in_channels = int(
                questionary.text(
                    "in_channels",
                    default=str(self.in_channels),
                ).ask()
            )
            self.down_sampling_factor = tuple(
                int(x)
                for x in questionary.text(
                    "down_sampling_factor",
                    default=",".join(map(str, self.down_sampling_factor)),
                )
                .ask()
                .split(",")
            )
            self.down_sampling_channels = int(
                questionary.text(
                    "down_sampling_channels",
                    default=str(self.down_sampling_channels),
                ).ask()
            )
            self.spatial_dropout_p = float(
                questionary.text(
                    "spatial_dropout_p",
                    default=str(self.spatial_dropout_p),
                ).ask()
            )
            self.channels_per_group = int(
                questionary.text(
                    "channels_per_group",
                    default=str(self.channels_per_group),
                ).ask()
            )
            self.n_groups = int(
                questionary.text(
                    "n_groups",
                    default=str(self.n_groups),
                ).ask()
            )
            self.dilation_rates = list(
                map(
                    int,
                    questionary.text(
                        "dilation_rates",
                        default=",".join(map(str, self.dilation_rates)),
                    )
                    .ask()
                    .split(","),
                )
            )
            self.steps = int(
                questionary.text(
                    "steps",
                    default=str(self.steps),
                ).ask()
            )
            self.lr = float(
                questionary.text(
                    "lr",
                    default=str(self.lr),
                ).ask()
            )
            self.instance_size = tuple(
                float(x)
                for x in questionary.text(
                    "instance_size",
                    default=",".join(map(str, self.instance_size)),
                )
                .ask()
                .split(",")
            )

        self.save()


if __name__ == "__main__":
    config = RDCNetConfig()
    config.prompt()
