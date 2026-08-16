try:
    from gymnasium.envs.registration import register
except ImportError:  # pragma: no cover
    from gym.envs.registration import register
from .my_env import MyEnv


register(
    id='MyEnv-v0',
    entry_point='env_sim.my_env:MyEnv',
)
