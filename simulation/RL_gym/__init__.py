from gym.envs.registration import register

register(
    id='Pygame-v0',
    entry_point='RL_gym.envs:CustomEnv',
    max_episode_steps=2000,
)
