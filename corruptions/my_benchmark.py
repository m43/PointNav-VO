import os
from collections import defaultdict
from typing import Dict, Optional, List

import PIL.Image
import imageio
import numpy as np
import torch
import tqdm
from habitat.core.agent import Agent
from habitat.core.env import Env
from habitat.core.logging import logger
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations import maps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torchvision import transforms

from corruptions import rgb_sensor_degradations


class MyBenchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
            self, task_config, log_folder, video_log_interval, eval_remote: bool = False, **kwargs
    ) -> None:
        print(task_config)
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=task_config)

        self.log_folder = log_folder
        self.video_log_interval = video_log_interval

        def f(x, k, default):
            return x[k] if k in x else default

        self._random_crop: Optional[bool] = f(kwargs, "random_crop", False)
        self._crop_height: Optional[int] = f(kwargs, "crop_height", None)
        self._crop_width: Optional[int] = f(kwargs, "crop_width", None)
        self._jitter: Optional[bool] = f(kwargs, "color_jitter", False)
        self._tshift: Optional[bool] = f(kwargs, "random_shift", False)
        self._daug_mode: Optional[bool] = f(kwargs, "data_augmentation_mode", False)

        # Parse visual corruption details
        # Provided inputs are
        # - a list of corruptions
        # - a list of severties
        visual_corruption = f(kwargs, "visual_corruption", None)
        visual_severity = f(kwargs, "visual_severity", None)
        if visual_corruption is not None and visual_severity > 0:  # This works
            self._corruptions = [visual_corruption.replace("_", " ")]
            self._severities = [visual_severity]
        else:
            self._corruptions = visual_corruption
            self._severities = visual_severity

        logger.info("Applied corruptions are ")
        logger.info(self._corruptions)
        logger.info(self._severities)

        logger.info(f"Random Crop state {self._random_crop}")
        logger.info(f"Color Jitter state {self._jitter}")
        logger.info(f"Random Translate {self._tshift}")

        # Whether to rotate the input observation or not
        # self._sep_rotate: bool = f(kwargs, "sep_rotate", False)
        self._rotate: bool = f(kwargs, "rotate", False)

        # Data augmentation options
        self._random_cropper = (
            None
            if not self._random_crop
            else transforms.RandomCrop((self._crop_height, self._crop_width))
        )

        self._color_jitter = (
            None if not self._jitter else transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
        )

        self._random_translate = (
            None
            if not self._tshift
            else transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
        )

        self.to_pil = transforms.ToPILImage()  # assumes mode="RGB" for 3 channels

    def remote_evaluate(
            self, agent: "Agent", num_episodes: Optional[int] = None
    ):
        raise NotImplementedError("No soup for you")

    def local_evaluate(
            self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)
        metrics_to_log = ["distance_to_goal", "success", "spl", "softspl"]
        # print(f"self._env.task._config={self._env.task._config}")
        # print(f"self._env.sim.habitat_config={self._env.sim.habitat_config}")
        # print(f"self._env._config={self._env._config}")
        top_down_map_measure = TopDownMap(self._env.sim, self._env.task._config.TOP_DOWN_MAP)

        viridis_cmap = plt.get_cmap("viridis")
        normalizer = Normalize(vmin=None, vmax=None)
        # normalizer = Normalize(
        #     vmin=self._env._config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH,
        #     vmax=self._env._config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        # )

        # csv_log_line_buffer = []
        csv_log_file = os.path.join(self.log_folder, "episode_metrics.csv")
        with open(csv_log_file, "a") as csv_file:
            csv_file.write(','.join(["episode"] + metrics_to_log))
            csv_file.write("\n")

        pbar = tqdm.tqdm(total=num_episodes, )
        count_episodes = 0
        while count_episodes < num_episodes:
            log_video_for_episode = (count_episodes == num_episodes - 1) or (
                    count_episodes % self.video_log_interval) == 0

            agent.reset()
            observations = self._env.reset()
            observations["rgb"] = self.corrupt_rgb_observation(observations["rgb"])

            if log_video_for_episode:
                top_down_map_measure.reset_metric(self._env.current_episode, self._env.task)

            logger.info("Agent stepping around inside environment.")
            images = []
            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)
                observations["rgb"] = self.corrupt_rgb_observation(observations["rgb"])

                if log_video_for_episode:
                    rgb = observations["rgb"]
                    if type(rgb) == torch.Tensor:
                        rgb = rgb.cpu().numpy().astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)

                    depth = observations["depth"]
                    if type(depth) == torch.Tensor:
                        depth = depth.cpu().numpy().squeeze(axis=-1)
                    else:
                        depth = depth.squeeze(axis=-1)
                    # depth = (depth * 255 / 10)
                    depth = normalizer(depth)
                    depth = viridis_cmap(depth, alpha=None, bytes=True)[:, :, :3]  # exclude 'a' from 'rgba'

                    top_down_map_measure.update_metric(self._env.current_episode, action)
                    top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                        top_down_map_measure.get_metric(),  # self._env.get_metrics()["top_down_map"],
                        rgb.shape[0]
                    )
                    output_im = np.concatenate((rgb, depth, top_down_map), axis=1)
                    images.append(output_im)

            logger.info(f"Episode {count_episodes} finished")
            if log_video_for_episode:
                self.images_to_video(images, self.log_folder, f"{count_episodes:04d}")
            metrics = self._env.get_metrics()
            print(f"metrics for count_episodes={count_episodes}")
            csv_log_line = f"{count_episodes}"
            for m in metrics_to_log:
                v = metrics[m]
                agg_metrics[m] += v
                logger.info("{}: {}".format(m, v))
                csv_log_line += f",{v}"
            # csv_log_line_buffer.append(csv_log_line)
            with open(csv_log_file, "a") as csv_file:
                csv_file.write(f"{csv_log_line}\n")
                # for csv_log_line in csv_log_line_buffer:
                #     csv_file.write(f"{csv_log_line}\n")
                # csv_log_line_buffer = []

            pbar.update()
            count_episodes += 1

        avg_metrics = {k: agg_metrics[k] / count_episodes for k in metrics_to_log}

        return avg_metrics

    @staticmethod
    def images_to_video(
            images: List[np.ndarray],
            output_dir: str,
            video_name: str,
            fps: int = 5,
            quality: Optional[float] = 5,
            **kwargs,
    ):

        r"""Calls imageio to run FFMPEG on a list of images. For more info on
        parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
        Args:
            images: The list of images. Images should be HxWx3 in RGB order.
            output_dir: The folder to put the video in.
            video_name: The name for the video.
            fps: Frames per second for the video. Not all values work with FFMPEG,
                use at your own risk.
            quality: Default is 5. Uses variable bit rate. Highest quality is 10,
                lowest is 0.  Set to None to prevent variable bitrate flags to
                FFMPEG so you can manually specify them using output_params
                instead. Specifying a fixed bitrate using ‘bitrate’ disables
                this parameter.
        """
        assert 0 <= quality <= 10
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
        writer = imageio.get_writer(
            os.path.join(output_dir, video_name),
            fps=fps,
            quality=quality,
            **kwargs,
        )
        logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
        for im in images:
            writer.append_data(im)
        writer.close()

    def evaluate(
            self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)

    def corrupt_rgb_observation(self, frame):
        # Work with numpy
        if type(frame) == torch.Tensor:
            im = frame.cpu().numpy().astype(np.uint8)
        else:
            im = np.array(frame)

        # Apply a sequence of corruptions to the RGB frames
        if self._corruptions is not None:
            im = rgb_sensor_degradations.apply_corruption_sequence(
                im, self._corruptions, self._severities
            )

        # Random translation
        if self._tshift:
            if isinstance(im, np.ndarray):
                im = self.to_pil(im)
            im = self._random_translate(im)

        # Random Crop Image
        if self._random_crop:
            if isinstance(im, np.ndarray):
                im = self.to_pil(im)
            im = self._random_cropper(im)

        # Color Jitter
        if self._jitter:
            if isinstance(im, np.ndarray):
                im = self.to_pil(im)
            im = self._color_jitter(im)

        # if self._rotate:
        #     rot_im = copy.deepcopy(im)
        #
        # if self._rotate:
        #     if not isinstance(rot_im, np.ndarray):
        #         rot_im = np.array(im)
        #     rot_im, rot_label = rgb_sensor_degradations.rotate_single(rot_im)

        if isinstance(im, PIL.Image.Image):
            im = np.array(im)

        # Return the same type
        if type(frame) == torch.Tensor:
            return torch.tensor(im, dtype=frame.dtype).to(frame.device)
        else:
            return im if isinstance(im, np.ndarray) else np.array(im)


class MyChallenge(MyBenchmark):
    def __init__(self, task_config, log_folder, video_log_interval, eval_remote=False, **kwargs):
        super().__init__(task_config, log_folder, video_log_interval, eval_remote=eval_remote, **kwargs)
        self._env.seed(task_config.RANDOM_SEED)

    def submit(self, agent, num_episodes=None):
        metrics = super().evaluate(agent, num_episodes=num_episodes)
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))
