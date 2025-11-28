import safety_gymnasium
from stable_baselines3 import PPO
from src.wrappers.SafetyGymSB3Wrapper import SafetyGymSB3Wrapper
    
def main():
    env_id = "SafetyPointGoal2-v0"
    policy = "MlpPolicy"
    learning_timesteps = 100
    log_id = f'ppo_{env_id}_{policy}_{learning_timesteps}'

    env = safety_gymnasium.make(env_id)
    env = SafetyGymSB3Wrapper(env)

    # Create PPO model
    model = PPO(
        policy,
        env,
        verbose=0,
        tensorboard_log="./runs/logs/",
    )

    # Train model
    print("Training...")
    model.learn(total_timesteps=learning_timesteps, tb_log_name=log_id)
    save_dir = f'./runs/models/{log_id}'
    model.save(save_dir)
    print(f'Model saved at: {save_dir}')

    env.close()


if __name__ == "__main__":
    main()