import numpy as np

def evaluate(model, make_env, env_id, n_episodes=50):
    env = make_env(env_id)
    results = dict(success=0, lava=0, timeout=0, returns=[], steps=[])
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
            
            if done:
                if reward == 50: results["success"] += 1
                elif reward == -50: results["lava"] += 1
                elif truncated: results["timeout"] += 1

        results["returns"].append(ep_return)
        results["steps"].append(info.get("step_count", 0))
    
    env.close()
    return {
        "avg_return": np.mean(results["returns"]),
        "success_rate": results["success"] / n_episodes,
        "lava_rate": results["lava"] / n_episodes,
        "timeout_rate": results["timeout"] / n_episodes,
    }