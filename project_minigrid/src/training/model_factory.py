from stable_baselines3 import PPO, A2C, DQN

class ModelFactory:
    """
    Factory for creating and managing RL models (SB3-compatible).
    
    Usage:
        factory = ModelFactory(algorithm="PPO", policy="CnnPolicy", env=vec_env)
        model = factory.build_model(
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=512,
            ent_coef=0.01,
            total_timesteps=200_000,
        )
        factory.save("../runs/models/ppo_model.zip")
    """

    _ALGO_MAP = {
        "PPO": PPO,
        "A2C": A2C,
        "DQN": DQN,
    }

    def __init__(self, algorithm: str, policy: str, env, verbose: int = 1):
        self.algorithm = algorithm
        self.policy = policy
        self.env = env
        self.verbose = verbose
        self.model = None

    def build_model(self, **kwargs):
        """
        Create and return the model with given hyperparameters.
        """
        AlgoClass = self._ALGO_MAP[self.algorithm]
        self.model = AlgoClass(self.policy, self.env, verbose=self.verbose, **kwargs)
        return self.model

    def train(self, total_timesteps: int, **kwargs):
        """
        Train the model for a given number of timesteps.
        """
    
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return self.model

    def save(self, path: str):
        """
        Save trained model.
        """

        self.model.save(path)

    def load(self, path: str):
        """
        Load a previously trained model into the factory.
        """
        AlgoClass = self._ALGO_MAP[self.algorithm]
        self.model = AlgoClass.load(path)
        return self.model