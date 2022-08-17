#!/usr/bin/env bash

set -e
set -o xtrace
PWD_START=`pwd`

module purge
module load gcc/8.4.0-cuda
module load python/3.7.7
module load cuda/10.1
module load cmake

conda env create -f environment.yml -n vo2021
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vo2021

conda install pip -y
conda install -y pytorch=1.6.0 torchvision=0.7.0 -c pytorch
cd ..

# ----------------------------------------------------------------------------
# install habitat-sim
# ----------------------------------------------------------------------------

rm -rf habitat-sim-vo
git clone https://github.com/facebookresearch/habitat-sim.git habitat-sim-vo
cd habitat-sim-vo
git checkout 020041d75eaf3c70378a9ed0774b5c67b9d3ce99

conda uninstall numpy -y
pip install -r requirements.txt --prefix="$CONDA_PREFIX"

module purge
module load gcc/8.4.0
module load cuda/10.1
module load cmake
python setup.py install --headless --prefix="$CONDA_PREFIX"

cd ..

# ----------------------------------------------------------------------------
# install habitat-lab
# ----------------------------------------------------------------------------

rm -rf habitat-lab-vo
git clone https://github.com/facebookresearch/habitat-lab.git habitat-lab-vo
cd habitat-lab-vo
git checkout d0db1b55be57abbacc5563dca2ca14654c545552 # challenge-2021

# install both habitat and habitat_baselines
pip install -r requirements.txt --prefix="$CONDA_PREFIX"
pip install -r habitat_baselines/rl/requirements.txt --prefix="$CONDA_PREFIX"
pip install -r habitat_baselines/rl/ddppo/requirements.txt --prefix="$CONDA_PREFIX"
#pip install -r habitat_baselines/il/requirements.txt --prefix="$CONDA_PREFIX"
python setup.py develop --all --prefix="$CONDA_PREFIX"

cd ..

# ----------------------------------------------------------------------------
#   go back home
# ----------------------------------------------------------------------------
cd $PWD_START

pip install torch==1.6.0 torchvision==0.7.0
pip install scikit-image wand pandas torchvision opencv-python==3.4.2.17 numpy==1.17.3 numba Pillow==8.4.0

# ----------------------------------------------------------------------------
#   download the dataset for Gibson PointNav
# ----------------------------------------------------------------------------
pip install gdown --prefix="$CONDA_PREFIX"

gdown https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip
mkdir -p dataset/habitat_datasets/pointnav/gibson/v2
unzip pointnav_gibson_v2.zip -d dataset/habitat_datasets/pointnav/gibson/v2
rm pointnav_gibson_v2.zip
gdown https://drive.google.com/uc?id=15_vh9rZgNhk_B8RFWZqmcW5JRdNQKM2G --output dataset/habitat_datasets/pointnav/gibson/gibson_quality_ratings.csv

# ----------------------------------------------------------------------------
#   Linking
# ----------------------------------------------------------------------------

NOCOLOR='\033[[0m'
RED='\033[[0;31m'
echo -e "\n${RED}NOTE:${NOCOLOR} use command 'ln -s <path to scene datasets> ${PWD}/dataset/Gibson' to link the simulation scenes.\n"
[ -f "./dataset/Gibson" ] && echo "Link found"

# How I did the linking:
# $ tree -L 1 /mnt/terra/data/gibson/
# /mnt/terra/data/gibson/
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
#
# $ ln -s /mnt/terra/data ./dataset/Gibson

# How it looks like after linking:
# $ tree dataset/ -L 5
# dataset/
# ├── Gibson -> /mnt/terra/data
# └── habitat_datasets
#     └── pointnav
#         └── gibson
#             ├── gibson_quality_ratings.csv
#             └── v2
#                 ├── train
#                 ├── val
#                 └── val_mini


# ----------------------------------------------------------------------------
#   Evaluating the agent to reproduce the results
# ----------------------------------------------------------------------------

# Download checkpoints
NOCOLOR='\033[[0m'
RED='\033[[0;31m'
echo -e "\n${RED}NOTE:${NOCOLOR} Pretrained checkpoints should be downloaded by hand from https://drive.google.com/drive/folders/1HG_d-PydxBBiDSnqG_GXAuG78Iq3uGdr "
# https://drive.google.com/drive/folders/1HG_d-PydxBBiDSnqG_GXAuG78Iq3uGdr


## Run VO2021 (make sure you use the correctly setup conda/venv/python environment)
# cd /scratch/izar/rajic/vo
# module purge
# module load gcc/8.4.0-cuda cuda/10.1
# conda activate vo2021

# [ -d "./pretrained_ckpts" ] && echo "Directory ./pretrained_ckpts found."
# export CUDA_LAUNCH_BLOCKING=1 && \

# export POINTNAV_VO_ROOT=`pwd` && \
# export NUMBA_THREADING_LAYER=workqueue && \
# export NUMBA_NUM_THREADS=1 && \
# python ${POINTNAV_VO_ROOT}/launch.py \
#     --repo-path ${POINTNAV_VO_ROOT} \
#     --n_gpus 1 \
#     --task-type rl \
#     --noise 1 \
#     --run-type eval \
#     --addr 127.0.1.1 \
#     --port 8338