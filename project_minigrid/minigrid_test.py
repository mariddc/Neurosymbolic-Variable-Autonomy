import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import imageio
import os

# --- 1. Choose environment ---
env_id = "MiniGrid-LavaCrossingS9N1-v0"

# --- 2. Training environment (no GUI, pixel-based) ---
def make_env(render_mode="rgb_array"):
    env = gym.make(env_id, render_mode=render_mode)  # no GUI while training
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

vec_env = make_vec_env(make_env, n_envs=4)

# --- 3. Train PPO agent ---
model = PPO("CnnPolicy", vec_env, verbose=1)
model.learn(total_timesteps=20_000)

# --- 4. Test and record one episode ---

test_env = make_env()  # for frame capture
obs, info = test_env.reset()

frames = []
for step in range(300):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = test_env.step(action)
    frame = test_env.render()  # returns RGB array
    frames.append(frame)
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()

# --- 5. Save to video (GIF + MP4) ---
os.makedirs("videos", exist_ok=True)

gif_path = "videos/ppo_lavacrossing.gif"
mp4_path = "videos/ppo_lavacrossing.mp4"

print(f"Saving animation to {gif_path} and {mp4_path}...")

imageio.mimsave(gif_path, frames, fps=10)
imageio.mimsave(mp4_path, frames, fps=10, quality=8)

print("✅ Done! You can open the files from:")
print(os.path.abspath("videos/"))
