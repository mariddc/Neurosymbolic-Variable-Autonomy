import safety_gymnasium
from stable_baselines3 import PPO
from src.wrappers.SafetyGymSB3Wrapper import SafetyGymSB3Wrapper
    
def main():
    env = safety_gymnasium.make("SafetyPointGoal2-v0")

    env = SafetyGymSB3Wrapper(env)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./runs/logs/ppo_logs/",
    )

    # Train model
    print("Training...")
    model.learn(total_timesteps=1_000)
    save_dir = './runs/models/ppo_safety_pointgoal2'
    model.save(save_dir)
    print(f'Model saved at: {save_dir}')

    env.close()


if __name__ == "__main__":
    main()