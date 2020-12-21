import gym
import numpy as np

from marathon_envs.envs import MarathonEnvs

from timeit import default_timer as timer
from datetime import timedelta
import os

env_names = [
    'Hopper-v0', 
    # 'Walker2d-v0', 
    # 'Ant-v0', 
    # 'MarathonMan-v0', 
    # 'MarathonManSparse-v0'
    ]
for env_name in env_names:
    print ('-------', env_name, '-------')
    env = MarathonEnvs(env_name, 1)

    obs = env.reset()
    episode_score = 0.
    episode_steps = 0
    episodes = 0
    while episodes < 5:
        # action, _states = model.predict(obs)
        action = [env.action_space.sample() for _ in range(env.number_agents)]
        obs, rewards, dones, info = env.step(action)
        episode_score += rewards
        episode_steps += 1
        env.render()
        if dones:
            print ('episode_score', episode_score, 'episode_steps', episode_steps)
            episode_score = 0.
            episode_steps = 0
            episodes += 1
    env.close()    