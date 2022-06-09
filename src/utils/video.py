import os
from base64 import b64encode

import gym
from gym.wrappers import (
    FilterObservation,
    FlattenObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)
from IPython.display import HTML

from src.utils.config import get_config
from src.utils.path import define_log_dir

cfg = get_config()
log_dir = define_log_dir()


def create_environment():
    """
    To DO : dict observationsではない環境への対応
    ex. Ant-v2ではエラーが出る
    """
    env = gym.make(cfg["Env"])
    env = FilterObservation(env, ["observation", "desired_goal"])
    env = FlattenObservation(env)
    env = RecordVideo(
        env,
        video_folder=os.path.join(log_dir, "videos"),
        episode_trigger=lambda x: x % int(cfg["Video"]["Interval"]) == 0,
    )
    env = RecordEpisodeStatistics(env)
    return env


def display_video(episode):
    path = os.path.join(
        log_dir, "videos", "rl-video-episode-{}.mp4".format(str(int(episode)))
    )
    video_file = open(path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(
        f"<video width=600 controls><source src='{video_url}'></video>"
    )
