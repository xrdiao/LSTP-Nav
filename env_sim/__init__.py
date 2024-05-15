from gymnasium.envs.registration import register

register(
    id='MyEnv-v0',
    entry_point='env_sim.my_env:MyEnv',
)
