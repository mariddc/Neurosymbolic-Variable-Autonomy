from gymnasium import Wrapper

class SafetyGymSB3Wrapper(Wrapper):
    """
    Wraps Safety-Gymnasium envs to be compatible with Stable-Baselines3.
    Safety-Gymnasium's step() returns: obs, reward, cost, terminated, truncated, info
    SB3 expects: obs, reward, terminated, truncated, info
    """
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info["cost"] = cost 
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)