from gym.envs.registration import register

register(
    id='MyEnv-v0',
    entry_point='env_sim.env_sim:MyEnv',
)