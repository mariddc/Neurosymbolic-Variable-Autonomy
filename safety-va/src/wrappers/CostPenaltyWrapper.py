import gymnasium as gym
from typing import Any, Dict, Tuple


class CostPenaltyWrapper(gym.Wrapper):
    """
    Wrapper that subtracts a penalty proportional to info['cost'] from the reward.

    r' = r - penalty_coef * cost

    Assumes the wrapped env's step() returns:
        obs, reward, done, info
    and puts the cost in info['cost'] (float or int).
    """

    def __init__(self, env: gym.Env, penalty_coef: float = 1.0, cost_key: str = "cost"):
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.cost_key = cost_key

    def step(
        self, action
    ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Read cost from info (default 0.0 if not present)
        cost = float(info.get(self.cost_key, 0.0))

        shaped_reward = reward - self.penalty_coef * cost

        # Optionally keep everything for logging / debugging
        info.setdefault("original_reward", reward)
        info.setdefault("shaped_reward", shaped_reward)
        info.setdefault("cost", cost)

        return obs, shaped_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
