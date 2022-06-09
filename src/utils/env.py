import gym
from gym.wrappers import (
    FilterObservation,
    FlattenObservation,
    RecordEpisodeStatistics,
)

from src.utils.load_config import get_config
from src.utils.video import record_video

cfg = get_config()


def create_env():
    env = gym.make(cfg["Base"]["Env"])
    env = env_preprocess(env)
    env = record_video(env)
    env = record_log(env)
    return env


def env_preprocess(env: gym.wrappers) -> gym.wrappers:
    if cfg["Train"]["Env_preprocess"] == "dict":
        env = FilterObservation(env, ["observation", "desired_goal"])
        env = FlattenObservation(env)
        return env

    if cfg["Train"]["Env_preprocess"] == None:
        return env


def record_log(env: gym.wrappers) -> gym.wrappers:
    if cfg["Train"]["is_Record_log"]:
        return RecordEpisodeStatistics(env)
    else:
        return env
