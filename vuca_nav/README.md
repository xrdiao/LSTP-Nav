# VUCA Nav Reproduction

This folder contains a paper-oriented reproduction of:

`Robot Mapless Navigation in VUCA Environments via Deep Reinforcement Learning`

Adaptation notes for the current repository:

- It uses the existing `env_sim.my_env.MyEnv` without modifying `my_env.py`.
- The ego robot uses the paper's 45-action discrete action set.
- Observations follow the paper's robot state, human observable state, and lidar-map split.
- The reward implements the paper's hazardous-area and discomfort-area penalties.
- The value network is adapted as a Stable-Baselines3 feature extractor with:
  - human-wise spatial map encoding
  - GRU-based interaction reasoning
  - attention pooling over humans
  - lidar MLP encoder

Files:

- `config.py`: experiment and model hyperparameters
- `env.py`: wrapper around `env_sim.my_env.MyEnv`
- `observation.py`: paper-style state construction
- `reward.py`: hazardous-area reward
- `model.py`: COA feature extractor
- `train.py`: DQN training entry
- `evaluate.py`: evaluation entry

Run examples:

```bash
python -m vuca_nav.train
python -m vuca_nav.evaluate
```
