import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

def make_env(env_id, render_mode="rgb_array"):
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env