import os
from base64 import b64encode

from gym.wrappers import RecordVideo
from IPython.display import HTML

from src.utils.load_config import get_config
from src.utils.path import define_log_dir

cfg = get_config()
log_dir = define_log_dir()


def record_video(env):
    """
    To DO : dict observationsではない環境への対応
    ex. Ant-v2ではエラーが出る
    """
    if cfg["Video"]["is_Record"]:
        env = RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            episode_trigger=lambda x: x % int(cfg["Video"]["Interval"]) == 0,
        )
        return env

    else:
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
