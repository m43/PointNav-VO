#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random

import habitat
import numba
import numpy as np
import torch
from habitat import get_config
from habitat_baselines.config.default import get_config as get_baseline_config

from challenge_2020.challenge2021_agent import PointNavAgent
from corruptions.gaussian_noise_model_torch import GaussianNoiseModelTorch
from corruptions.parser import get_corruptions_parser, apply_corruptions_to_config, get_runid_and_logfolder
from corruptions.my_benchmark import MyChallenge


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main():
    _ = GaussianNoiseModelTorch()
    parser = get_corruptions_parser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    args = parser.parse_args()
    print(args)

    if args.challenge_config_file:
        config_paths = args.challenge_config_file
    else:
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    task_config = get_config(config_paths)
    apply_corruptions_to_config(args, task_config)
    args.run_id, args.log_folder = get_runid_and_logfolder(args, task_config)

    ddppo_config = get_baseline_config(
        "configs/rl/ddppo_pointnav.yaml", ["BASE_TASK_CONFIG_PATH", config_paths]
    ).clone()

    ddppo_config.defrost()
    ddppo_config.TORCH_GPU_ID = 0
    ddppo_config.SEED = args.seed
    ddppo_config.TASK_CONFIG.SEED = args.seed
    ddppo_config.TASK_CONFIG.SIMULATOR.SEED = args.seed
    ddppo_config.TASK_CONFIG = task_config
    # ddppo_config.PTH_GPU_ID = 0
    # ddppo_config.RANDOM_SEED = task_config.RANDOM_SEED
    ddppo_config.freeze()

    random.seed(args.seed)
    np.random.seed(args.seed)
    _seed_numba(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    agent = PointNavAgent(ddppo_config, args.evaluation)

    if args.evaluation == "local":
        challenge = MyChallenge(task_config, eval_remote=False, **args.__dict__)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent, args.num_episodes)


if __name__ == "__main__":
    main()
