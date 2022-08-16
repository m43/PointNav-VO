# EXPORT CONFIG BEFORE ANYTHING
# export CHALLENGE_CONFIG_FILE=config_files/challenge_pointnav2021.local.rgbd.CPU.yaml
# export CHALLENGE_CONFIG_FILE=config_files/challenge_pointnav2021.local.rgbd.GPU.yaml

############################
###     RANDOM AGENT     ###
############################

# HABITAT DEFAULT
python random_agent.py --evaluation local \
--agent_name random_agent --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--num_episodes 2

# HABITAT DEFAULT + COLOR JITTER
python random_agent.py --evaluation local \
--agent_name random_agent --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--num_episodes 2

##############################
###     DDPPO Baseline     ###
##############################

# CLEAN
python -u ddppo_agents.py --input-type rgbd --evaluation local --model-path saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth \
--agent_name ddppo --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--habitat_rgb_noise_intensity 0.0 --pyrobot_noise_multiplier 0.0 \
--num_episodes 4

# HABITAT DEFAULT
python -u ddppo_agents.py --input-type rgbd --evaluation local --model-path saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth \
--agent_name ddppo --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--num_episodes 4

# HABITAT DEFAULT + COLOR JITTER
python -u ddppo_agents.py --input-type rgbd --evaluation local --model-path saved/ddppo_pointnav_habitat2021_challenge_baseline_v1.pth \
--agent_name ddppo --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--color_jitter \
--num_episodes 2

##################
###     VO     ###
##################

## CLEAN
#python -m challenge_2020.challenge2021_corruptions --evaluation local \
#--agent_name vo --dataset_split val_mini --challenge_config_file configs/challenge_pointnav2021.local.rgbd.CPU.yaml \
#--habitat_rgb_noise_intensity 0.0 --pyrobot_noise_multiplier 0.0 \
#--num_episodes 4
#
## HABITAT DEFAULT
#python -m challenge_2020.challenge2021_corruptions --evaluation local \
#--agent_name vo --dataset_split val_mini --challenge_config_file configs/challenge_pointnav2021.local.rgbd.GPU.yaml \
#--num_episodes 1
#
## HABITAT DEFAULT + COLOR JITTER
#python -m challenge_2020.challenge2021_corruptions --evaluation local \
#--agent_name vo --dataset_split val_mini --challenge_config_file configs/challenge_pointnav2021.local.rgbd.CPU.yaml \
#--color_jitter \
#--num_episodes 2

# HABITAT DEFAULT
python -m pointnav_vo.run --task-type rl --noise 1 --exp-config configs/rl/ddppo_pointnav.yaml --run-type eval --n-gpu 1 --cur-time 123 \
--agent_name vo --dataset_split val_mini --challenge_config_file configs/challenge_pointnav2021.local.rgbd.GPU.yaml \
--num_episodes 10 --video_log_interval 1

# HABITAT DEFAULT + COLOR JITTER
python -m pointnav_vo.run --task-type rl --noise 1 --exp-config configs/rl/ddppo_pointnav.yaml --run-type eval --n-gpu 1 --cur-time 123 \
--agent_name vo --dataset_split val_mini --challenge_config_file configs/challenge_pointnav2021.local.rgbd.GPU.yaml \
--color_jitter \
--num_episodes 10 --video_log_interval 1

# HABITAT DEFAULT + SPECKLE NOISE s5
python -m pointnav_vo.run --task-type rl --noise 1 --exp-config configs/rl/ddppo_pointnav.yaml --run-type eval --n-gpu 1 --cur-time 123 \
--agent_name vo --dataset_split train --challenge_config_file configs/challenge_pointnav2021.local.rgbd.GPU.yaml \
--visual_corruption Speckle_Noise --visual_severity 5 \
--num_episodes 10 --num_episode_sample 1000

# HABITAT DEFAULT + DEPTH NOISE
python -m pointnav_vo.run --task-type rl --noise 1 --exp-config configs/rl/ddppo_pointnav.yaml --run-type eval --n-gpu 1 --cur-time 123 \
--agent_name vo --dataset_split val --challenge_config_file configs/challenge_pointnav2021.local.rgbd.GPU.yaml \
--depth_noise_multiplier 2.0 \
--num_episodes 10


######################
###     RUSLAN     ###
######################

# HABITAT DEFAULT
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--num_episodes 3 --log_folder logs/round1 --video_log_interval 1
# HABITAT DEFAULT
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--habitat_rgb_noise_intensity 0.1 \
--num_episodes 4 --num_episode_sample 4

# HABITAT VISION CLEAN
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--habitat_rgb_noise_intensity 0 \
--num_episodes 2


## VISUAL CORRUPTIONS
# Defocus_Blur Lighting Speckle_Noise Spatter Motion_Blur

# HABITAT DEFAULT + DEFOCUS BLUR
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--visual_corruption Defocus_Blur --visual_severity 5 \
--num_episodes 2
# HABITAT DEFAULT + LOW LIGHTING
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--visual_corruption Lighting --visual_severity 5 \
--num_episodes 2
# HABITAT DEFAULT + SPECKLE NOISE
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--visual_corruption Speckle_Noise --visual_severity 5 \
--num_episodes 2
# HABITAT DEFAULT + SPATTER
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--visual_corruption Spatter --visual_severity 5 \
--num_episodes 2
# HABITAT DEFAULT + MOTION BLUR
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--visual_corruption Motion_Blur --visual_severity 5 \
--num_episodes 2

# HABITAT DEFAULT + DEPTH NOISE
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--depth_noise_multiplier 2.0 \
--num_episodes 2


# HABITAT DEFAULT + HFOV
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
-hfov 50 \
--num_episodes 2

# HABITAT DEFAULT + RANDOM CROP
# python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
# --agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
# --random_crop --crop_width 50 --crop_height 50

# HABITAT DEFAULT + COLOR JITTER
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--color_jitter \
--num_episodes 2


## DYNAMICS CORRUPTIONS

# pyrobot_robot_spec in [LoCoBot, LoCoBot-Lite]

# HABITAT DEFAULT
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_robot_spec LoCoBot \
--num_episodes 2
# HABITAT DEFAULT + pyrobot_robot=LoCoBot-Lite
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_robot_spec LoCoBot-Lite \
--num_episodes 2


# pyrobot_controller_spec in [ILQR, Proportional, Movebase]

# HABITAT DEFAULT + pyrobot_controller=Movebase
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_controller_spec "Movebase" \
--num_episodes 2
# HABITAT DEFAULT
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_controller_spec Proportional \
--num_episodes 2
# HABITAT DEFAULT + pyrobot_controller=ILQR
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_controller_spec "ILQR" \
--num_episodes 2


# HABITAT DEFAULT + pyrobot_noise_multiplier=0.0
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_noise_multiplier 0.0 \
--num_episodes 2
# HABITAT DEFAULT
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_noise_multiplier 0.5 \
--num_episodes 2
# HABITAT DEFAULT + pyrobot_noise_multiplier=1.0
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_noise_multiplier 1.0 \
--num_episodes 2
# HABITAT DEFAULT + pyrobot_noise_multiplier=2.0
python agent.py --agent-type PPOAgentV2 --input-type depth --evaluation local --ddppo-checkpoint-path saved/pointnav2021_gt_loc_depth_ckpt.345.pth --ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml --vo-config-path saved/config.yaml --vo-checkpoint-path saved/best_checkpoint_064e.pt --pth-gpu-id 0 --rotation-regularization-on --vertical-flip-on  \
--agent_name ruslan --dataset_split val_mini --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.CPU.yaml \
--pyrobot_noise_multiplier 2.0 \
--num_episodes 2

