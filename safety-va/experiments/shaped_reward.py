import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.wrappers import SafetyGymSB3Wrapper, CostPenaltyWrapper
from src.evaluation import evaluate_safety


def make_env(env_id, penalty_coef=1.0):
    env = safety_gymnasium.make(env_id)
    env = SafetyGymSB3Wrapper(env)
    env = CostPenaltyWrapper(env, penalty_coef=penalty_coef)
    env = DummyVecEnv([lambda: env])
    return env


def train_model_PPO(policy, env, learning_timesteps, log_id, save=False, verbose=False):
    # create, train, save model
    model = PPO(policy, env, verbose=verbose, tensorboard_log="./runs/logs/")

    if verbose:
        print("Training...")

    model.learn(
        total_timesteps=learning_timesteps,
        tb_log_name=log_id,
        progress_bar=verbose
    )

    if save:
        save_dir = f'./runs/models/{log_id}'
        model.save(save_dir)
        print(f'Model saved at: {save_dir}')

    return model


def build_result_dict(lamb, rewards, lengths, costs):
    return {
        'lambda': lamb,
        'mean reward': np.mean(rewards),
        'std reward': np.std(rewards),
        'mean cost': np.mean(costs),
        'std cost': np.std(costs),
        'mean length': np.mean(lengths)
    }


def make_evaluation_figure(results_dict, model_name, n_eps, learning_timesteps):
    fig, ax = plt.subplots(figsize=(8, 6))

    for d in results_dict:
        lam = d['lambda']
        x = d['mean cost']
        sx = d['std cost']
        y = d['mean reward']
        sy = d['std reward']

        ax.scatter(x, y, color='C0')  # point

        ell = Ellipse(  # std ellipse
            (x, y),
            width=2 * sx,
            height=2 * sy,
            facecolor='C0',
            edgecolor='C0',
            alpha=0.15
        )
        ax.add_patch(ell)

        ax.text(x, y, f"λ={lam}", fontsize=9, ha='center', va='bottom')  # label

    ax.set_xlabel("Mean Cost")
    ax.set_ylabel("Mean Reward")
    ax.set_title(
        f"Shaped reward metrics with standard deviation\n"
        f"Model={model_name}; n_episodes:={n_eps}; timesteps={learning_timesteps}"
    )
    ax.grid(True)
    fig.tight_layout()

    return fig


def save_figure(fig, filepath):
    fig.savefig(filepath, dpi=300)
    print(f"Saved figure to: {filepath}")


def main():
    t0 = time.time()

    env_id = "SafetyPointGoal2-v0"
    policy = "MlpPolicy"
    learning_timesteps = 10_000
    log_id = f'ppo_{env_id}_{policy}_{learning_timesteps}_shapedreward'

    n_eval_episodes = 30

    results = []

    # lambdas = np.linspace(0.0, 2.0, 10)
    lambdas = np.array([0.0, .05, .1, .2, .5, 1.0, 1.5, 2.0])

    for lamb in lambdas:
        log_id_lamb = log_id + f'{lamb}'
        env = make_env(env_id, penalty_coef=lamb)
        model = train_model_PPO(policy, env, learning_timesteps, log_id_lamb)

        # evaluate model
        print(f'Evaluating {log_id_lamb} for {n_eval_episodes} episodes...')
        ep_rewards, ep_lengths, ep_costs = evaluate_safety(
            model, env, n_episodes=n_eval_episodes, verbose=False
        )

        res_dict = build_result_dict(lamb, ep_rewards, ep_lengths, ep_costs)

        # save metrics
        results.append(res_dict)

    # plot metrics
    result_fig = make_evaluation_figure(
        results, 'PPO ' + policy, n_eval_episodes, learning_timesteps
    )
    save_figure(result_fig, f'./runs/figures/{log_id}')

    t1 = time.time()
    total = t1 - t0
    print(f"\nTotal runtime: {total:.2f} seconds ({total/60:.2f} minutes)")

    return


if __name__ == "__main__":
    main()
