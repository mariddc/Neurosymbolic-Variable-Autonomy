import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from src.wrappers.SafetyGymSB3Wrapper import SafetyGymSB3Wrapper

def main():
    env = safety_gymnasium.make(
        "SafetyPointGoal2-v0",
        render_mode="rgb_array",
        camera_name="topdown"
    )
    env = SafetyGymSB3Wrapper(env)

    env = DummyVecEnv([lambda: env])

    video_folder = "./videos/"
    video_length = 1000
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix="ppo_topdown_hd"
    )

    try:
        model = PPO.load("./runs/models/ppo_safety_pointgoal2", env=env)
        print('successfully loaded model!')
    except Exception as e:
        print(f"error loading model: {e}")

    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if dones.any():
            break

    env.close()


if __name__ == "__main__":
    main()