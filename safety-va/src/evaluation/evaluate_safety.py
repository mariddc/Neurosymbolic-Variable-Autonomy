import numpy as np

def evaluate_safety(model, env, n_episodes=20, verbose=True):
    episode_rewards = []
    episode_lengths = []
    episode_costs = []   # assuming info["cost"] exists

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_cost = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_reward += reward
            ep_length += 1

            # VecEnv vs non-VecEnv:
            # if you use DummyVecEnv, reward/done/info are arrays/lists.
            if isinstance(done, np.ndarray):
                d = done[0]
                i = info[0]
            else:
                d = done
                i = info

            if "cost" in i:
                ep_cost += i["cost"]

            if d:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_costs.append(ep_cost)

    if verbose:
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
        print(f"Mean length: {np.mean(episode_lengths):.1f}")
        print(f"Mean cost:   {np.mean(episode_costs):.2f}")

    return episode_rewards, episode_lengths, episode_costs
