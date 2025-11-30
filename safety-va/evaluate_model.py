import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.wrappers.SafetyGymSB3Wrapper import SafetyGymSB3Wrapper
from src.evaluation.evaluate_safety import evaluate_safety

def main():
    env_id_train = "SafetyPointGoal2-v0"
    env_id_test = "SafetyPointGoal2-v0"
    policy = "MlpPolicy"
    learning_timesteps = 10_000
    log_id = f'ppo_{env_id_train}_{policy}_{learning_timesteps}'

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
    n_episodes = 20
    print(f'Evaluating {log_id} for {n_episodes} episodes...')
    evaluate_safety(model, env, n_episodes=n_episodes)

    env.close()

if __name__ == "__main__":
    main()