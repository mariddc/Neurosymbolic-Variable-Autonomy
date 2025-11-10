from gymnasium.envs.registration import register
from my_envs.DistributionalShiftEnv import DistributionalShiftEnv

register(
    id="MiniGrid-DistributionalShiftEnv-v0",
    entry_point="my_envs.DistributionalShiftEnv:DistributionalShiftEnv",
)