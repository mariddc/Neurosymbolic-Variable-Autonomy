import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from src.wrappers.SafetyGymSB3Wrapper import SafetyGymSB3Wrapper
from src.evaluation.evaluate_safety import evaluate_safety

def main():
    env_id_train = "SafetyPointGoal2-v0"
    env_id_test = "SafetyPointGoal2-v0"
    policy = "MlpPolicy"
    learning_timesteps = 10_000
    log_id = f'ppo_{env_id_train}_{policy}_{learning_timesteps}_shapedreward'

    env = safety_gymnasium.make(
        env_id_test,
        render_mode="rgb_array",
        camera_name="fixedfar"
    )
    env = SafetyGymSB3Wrapper(env)
    env = DummyVecEnv([lambda: env])

    try:
        model = PPO.load(f"./runs/models/{log_id}", env=env)
        print("successfully loaded model!")
    except Exception as e:
        print(f"error loading model: {e}")
        return
    
    # quantitative evaluation
    # n_episodes = 20
    # print(f'Evaluating {log_id} for {n_episodes} episodes...')
    # evaluate_safety(model, env, n_episodes=n_episodes)

    video_folder = "./runs/videos/"
    video_length = 1_000
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=f'{log_id}_video'
    )

    print('\nRecording video for one episode...')

    episodes_rewards = []
    episodes_costs = []
    cur_reward = 0.0
    cur_cost = 0.0

    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        cur_reward += float(rewards[0])
        cur_cost += float(infos[0].get("cost", 0.0))

        if dones.any():
            episodes_rewards.append(cur_reward)
            episodes_costs.append(cur_cost)
            cur_reward = 0.0 
            cur_cost = 0.0           

    print("Video returns per episode:", episodes_rewards)
    print("Video costs per episode:", episodes_costs)

    env.close()

if __name__ == "__main__":
    main()