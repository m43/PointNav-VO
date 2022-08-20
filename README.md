# Robustness of Embodied Point Navigation Agents: VO2021 agent

[Frano Rajic](https://m43.github.io/)

[`Project Website`](https://m43.github.io/projects/embodied-ai-robustness/) | [`Paper`](https://www.youtube.com/watch?v=dQw4w9WgXcQ) | [`Code [UCU Mlab]`](https://github.com/m43/ucu-mlab) | [**`>> Code [VO2021] <<`**](https://github.com/m43/vo2021)

This repository contains the evaluation code for reproducing the benchmark results for the VO2021 agent. The codebase of the agent is taken from [Xiaoming-Zhao/PointNav-VO](https://github.com/Xiaoming-Zhao/PointNav-VO).

## Set-up

Start by cloning the repository:
```bash
git clone https://github.com/m43/vo2021.git
cd vo2021
```

With the repository cloned, we recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment using the provided environment setup script, adapted for our local machine setup. Depending on your local machine, you might want to remove the `module purge` and `module load ...` lines in the script, since we used the to prepare the cluster machines we worked with. The script might take a long time to run as `habitat-sim` must be built. The script will print out verbose logs about what bash command was run (`set -o xtrace`) and will stop if an error is encountered (`set -e`). To run the environment setup script:
```bash
bash environment_setup.sh
```

The environment setup script will also automatically download the Gibson dataset split data (a bunch of compressed `.json` files defining the splits of the Gibson dataset) with `gdown`. However, the Gibson dataset (~10GB) itself needs to be downloaded separately. To download (and link) the Gibson dataset, you could do the following:
```bash
# Optionally: create (or move) to a folder where you usually store datasets
mkdir -p /home/frano/data
cd /home/frano/data

# Sign agreement and download gibson_habitat_trainval.zip: https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform
# wget link/to/gibson_habitat_trainval.zip
unzip gibson_habitat_trainval.zip

cd /get/back/to/the/cloned/code/repo # cd -
mkdir -p ./dataset
mkdir -p ./data

# Link the Gibson dataset correctly
ln -s /home/frano/data ./dataset/Gibson
ln -s /home/frano/data data/scene_datasets

# Verify that everything was linked correctly:
tree -L 1 /home/frano/data/gibson/
# /home/frano/data/gibson/
# ├── Ackermanville.glb
# ├── Ackermanville.navmesh
# ├── Adairsville.glb
# ├── Adairsville.navmesh
# ├── Adrian.glb
# ├── Adrian.navmesh
# ├── Airport.glb
# ├── Airport.navmesh
# ├── Albertville.glb
# ...
# └── Yscloskey.navmesh
tree dataset/ -L 5
# dataset/
# ├── Gibson -> /home/frano/data
# └── habitat_datasets
#     └── pointnav
#         └── gibson
#             ├── gibson_quality_ratings.csv
#             └── v2
#                 ├── train
#                 ├── val
#                 └── val_mini
tree -L 2 data dataset
# data
# └── scene_datasets -> /home/frano/data
# dataset/
# ├── Gibson -> /home/frano/data/
# └── habitat_datasets
#     └── pointnav
```

Finally, download the pretrained checkpoints at [this link](https://drive.google.com/drive/folders/1HG_d-PydxBBiDSnqG_GXAuG78Iq3uGdr?usp=sharing) from the original author (as described in [Xiaoming-Zhao/PointNav-VO](https://github.com/Xiaoming-Zhao/PointNav-VO)). Put them under `pretrained_ckpts` with the following structure:
```bash
gdown --folder 1HG_d-PydxBBiDSnqG_GXAuG78Iq3uGdr --output pretrained_ckpts

tree pretrained_ckpts
# pretrained_ckpts
# ├── rl
# │   ├── no_tune
# │   │   └── rl_no_tune.pth
# │   └── tune_vo
# │       └── rl_tune_vo.pth
# └── vo
#     ├── act_forward.pth
#     └── act_left_right_inv_joint.pth
```

## Results reproduction

Activate the created environment:
```bash
# module purge
# module load gcc/8.4.0-cuda cuda/10.1
conda activate vo2021
```

To reproduce the Color Jitter visual corruption results on the validation subset (row 13 of Table 1 of the paper), run the command below. Unlike for other agents, we did not control all sources of randomness of the VO agent and the numbers will differ slightly from the reported ones.
```
python -m pointnav_vo.run --task-type rl --noise 1 --exp-config configs/rl/ddppo_pointnav.yaml --run-type eval --n-gpu 1 --cur-time 123 --video_log_interval 200 --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.GPU.yaml --agent_name vo --dataset_split val --seed 72 --color_jitter
```

This run configuration can be found in `slurm/sbatch_8/8-01.sh`. For other run configurations, consult the set of SLURM scripts in `slurm/sbatch_8` and `slurm/sbatch_11`. Alternatively, consult the `eval.sh` script to see how all the possible corruption settings can be run.

## Citing
If you find our work useful, please consider citing:
```BibTeX
[WIP]: Will be added once the Proceedings of the ECCV 2022 Workshops are published.
```
