import cv2
import numpy as np
import torch
from collections import deque
from marathon_envs.envs import MarathonEnvs

# GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0']
MARATHON_ENVS = ['Hopper-v0', 'Walker2d-v0', 'Ant-v0', 'MarathonMan-v0', 'MarathonManSparse-v0']

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
  observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
  observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
  return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
  images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
  preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
  return images.unsqueeze(dim=0)  # Add batch dimension

port_offset=0
free_ports=[]
class WrappedMarathonEnv():
  def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, n_envs=1):
    # domain, task = env.split('-')
    self.symbolic = symbolic
    assert (symbolic), "symbolic should be True, all marathon envs are symbolic"
    # self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
    
    global port_offset, free_ports
    if free_ports:
      self._port = free_ports.pop()
    else:
      self._port = port_offset
      port_offset += 1
    
    self._env = MarathonEnvs(env, n_envs, self._port)
    if not symbolic:
      self._env = pixels.Wrapper(self._env)
    self.max_episode_length = max_episode_length
    # self.action_repeat = action_repeat
    self.action_repeat = 1
    # if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
    #   print('Using action repeat %d; recommended action repeat for domain is %d' % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    observations = self._env.reset()
    if self.symbolic:
      # observations = np.concatenate(observations)
      observations = torch.tensor(observations)
    else:
      observations = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
    return observations

  def step(self, action):
    action = action.detach().numpy()
    # action = np.reshape(action, (1,-1))
    # reward = [0 for _ in range(self.number_agents)]
    # for k in range(self.action_repeat):
    #   observations, r, d, info = self._env.step(action)
    #   reward += r
    #   self.t += 1  # Increment internal timer
    #   done = d.any() or self.t == self.max_episode_length
    #   if done:
    #     break
    observations, rewards, dones, info = self._env.step(action)
    if self.symbolic:
      # observations = np.concatenate(observations)
      observations = torch.tensor(observations)
    else:
      observations = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
    return observations, rewards, dones

  def render(self):
    cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
    self._env.close()
    global port_offset, free_ports
    free_ports.append(self._port)

  @property
  def observation_size(self):
    return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property 
  def number_agents(self):
    return self._env.number_agents

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    sample = [self._env.action_space.sample() for _ in range(self._env.number_agents)]
    sample = np.array(sample)
    sample = torch.from_numpy(sample)
    return sample

class GymEnv():
  def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    import gym
    self.symbolic = symbolic
    self._env = gym.make(env)
    self._env.seed(seed)
    self.max_episode_length = max_episode_length
    self.action_repeat = action_repeat
    self.bit_depth = bit_depth

  def reset(self):
    self.t = 0  # Reset internal timer
    state = self._env.reset()
    if self.symbolic:
      return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
  
  def step(self, action):
    action = action.detach().numpy()
    reward = 0
    for k in range(self.action_repeat):
      state, reward_k, done, _ = self._env.step(action)
      reward += reward_k
      self.t += 1  # Increment internal timer
      done = done or self.t == self.max_episode_length
      if done:
        break
    if self.symbolic:
      observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
    return observation, reward, done

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property 
  def number_agents(self):
    return 1

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return torch.from_numpy(self._env.action_space.sample())


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, n_envs=1):
  if env in GYM_ENVS:
    if n_envs==1:
      return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
    else: 
      return EnvBatcher(env, (env, symbolic, seed, max_episode_length, action_repeat, bit_depth), {}, n_envs)

  elif env in MARATHON_ENVS:
    return WrappedMarathonEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, n_envs)

# Wrapper for batching environments together
class EnvBatcher():
  def __init__(self, env_class, env_args, env_kwargs, n):
    self.n = n
    self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
    self.dones = [True] * n

  # Resets every environment and returns observation
  def reset(self):
    observations = [env.reset() for env in self.envs]
    self.dones = [False] * self.n
    return torch.cat(observations)

 # Steps/resets every environment and returns (observation, reward, done)
  def step(self, actions):
    done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]  # Done mask to blank out observations and zero rewards for previously terminated environments
    observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
    dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
    self.dones = dones
    observations, rewards, dones = torch.cat(observations), torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
    observations[done_mask] = 0
    rewards[done_mask] = 0
    return observations, rewards, dones

  def close(self):
    [env.close() for env in self.envs]
