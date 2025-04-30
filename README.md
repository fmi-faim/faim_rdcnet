# FAIM RDC-Net

This repo contains a few training and evaluation scripts for instance segmentation with [PyTorch-RDC-Net](https://github.com/fmi-faim/pytorch-rdc-net).

## Commands
The commands to train on the MoNuSeg dataset. For other datasets, the commands are similar, but the dataset name and paths need to be changed. Please check the `pixi.toml` file under the `[task]` section.

### Build configs

```commandline
WD=runs/MoNuSeg-baseline pixi run build_rdcnet_config
WD=runs/MoNuSeg-baseline pixi run build_trainer_config
```

The configs will be saved in the `runs/MoNuSeg-baseline` directory. If the `MoNuSeg-baseline` directory does not exist, it will be created.

### Train
#### SLURM
```commandline
WD=runs/MoNuSeg-baseline ACCOUNT=your_cluster_account pixi run submit_train_MoNuSeg
```

#### Local
```commandline
WD=runs/MoNuSeg-baseline pixi run train_MoNuSeg
```

---
This project was generated with the [faim-ipa-project](https://fmi-faim.github.io/ipa-project-template/) copier template.
