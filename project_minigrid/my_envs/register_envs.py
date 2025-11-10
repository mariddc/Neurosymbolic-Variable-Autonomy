from gymnasium.envs.registration import register

def register_envs():
    register(
        id="MiniGrid-DistributionalShift-Train-v0",
        entry_point="my_envs.DistributionalShiftEnv:DistributionalShiftEnv",
        kwargs={"is_testing": False},
    )
    register(
        id="MiniGrid-DistributionalShift-Test-v0",
        entry_point="my_envs.DistributionalShiftEnv:DistributionalShiftEnv",
        kwargs={"is_testing": True},
    )