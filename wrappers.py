import gym
from gym import spaces
import numpy as np

class GoalWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.threshold = 0.05
        self._max_episode_steps = 200
        
    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward, is_success = self.compute_goal_metrics(obs)
        info['is_success'] = is_success
        if not done:
            done = is_success
        return obs, reward, done, info
                
    def compute_goal_metrics(self, obs):
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        is_success = dist < 0.05
        reward = is_success - 1
        return reward, is_success

    def compute_reward(self, obs):
        reward, _, _ = self.compute_goal_metrics(obs)
        return reward
        
    def reset(self):
        obs = self.env.reset()
        return obs 


class ImageGoalDMCWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, r, done, info = env.step(action)
        reward = 0 if r > 0 else - 1
        if reward == 0:
            done = True
        return dict(observation=obs,
                    achieved_goal=obs,
                    desired_goal=self.desired_goal)

    def reset(self):
        self.desired_goal = self.env.reset()
        obs = self.env.reset()
        return dict(observation=obs,
                    achieved_goal=obs,
                    desired_goal=self.desired_goal)


class LatentGoalWrapper(gym.Wrapper):

    def __init__(self, env,encoder):
        gym.Wrapper.__init__(self, env)
        self.encoder = encoder
        self.use_cuda = torch.cuda.is_available()

    def step(self, action):
        obs_dict, r, done, info = env.step(action)
        obs_dict = self.compute_latents(obs_dict)
        reward = self.compute_reward(obs_dict)
        if reward == 0:
            done = True
        return dict(observation=obs,
                    achieved_goal=obs,
                    desired_goal=self.desired_goal)

    def reset(self):
        obs_dict = self.reset()
        obs_dict = self.compute_latents(obs_dict)
    
    def compute_reward(self, obs_dict):
        ag = obs_dict['latent_achieved_goal']
        dg = obs_dict['latent_desired_goal']
        dist = np.linalg.norm(ag - dg)
        r = 0 if dist < 0.05 else -1

    def compute_latents(self, obs_dict):
        obs, ag, dg = obs_dict['observation'],obs_dict['achieved_goal'],obs_dict['desired_goal']
        obs = torch.tensor(obs)
        ag = torch.tensor(ag)
        dg = torch.tensor(dg)
        if self.use_cuda:
            obs = obs.cuda()
            ag = ag.cuda()
            dg = dg.cuda()
        
        if len(ag.shape) == 3:
            obs = obs.unsqueeze(0)
            ag = ag.unsqueeze(0)
            dg = dg.unsqueeze(0)
            # encode 
            stack = torch.cat((obs,ag,dg),dim=0)
            encoded_latents = self.encoder (stack)
        obs_latent = encoded_latents[0]
        ag_latent = encoded_latents[1]
        dg_latent = encoded_latents[2]

        obs_dict.update(dict(latent_observation=obs_latent,
                             latent_achieved_goal=ag_latent,
                             latent_desired_goal=dg_latent))
        return obs_dict 


class FlatImageGoalDMCWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, r, done, info = env.step(action)
        dist = self.dist_to_goal()
        reward = 0 if dist < 0.1 else - 1
        if reward == 0:
            done = True
        return np.concatenate((obs, self.desired_goal), axis=0)

    def reset(self):
        self.desired_goal = self.env.reset()
        self.goal_xpos = self.env._env.physics.named.data.geom_xpos['finger']
        obs = self.env.reset()
        return np.concatenate((obs, self.desired_goal), axis=0)
    
    @property
    def current_xpos(self):
        return self.env._env.physics.named.data.geom_xpos['finger']

    def dist_to_goal(self):
        return np.lingalg.norm(self.goal_xpos - self.current_xpos)

class FlattenGoalWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = 50

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward, is_success = self.compute_goal_metrics(obs)
        info['is_success'] = is_success
        if not done:
            done = is_success
        return obs, reward, done, info

    def compute_goal_metrics(self, obs):
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        is_success = dist < 0.05
        reward = is_success - 1
        return reward, is_success

    def compute_reward(self, obs):
        reward, _, _ = self.compute_goal_metrics(obs)
        return reward

    def reset(self):
        obs = self.env.reset()
        return obs

class MtnCarWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        obs = self.reset()

        self._max_episode_steps = 200
        
        self.observation_space = gym.spaces.Dict(
            desired_goal=spaces.Box(-np.inf, np.inf,
                                    shape=(1,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf,
                                     shape=(1,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf,
                                   shape=obs['observation'].shape, dtype='float32'),
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = dict(
            observation=obs, desired_goal=self.desired_goal, achieved_goal=obs[0])
        return obs_dict, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.desired_goal = 0.05
        obs_dict = dict(observation=obs,desired_goal=self.desired_goal,achieved_goal=obs[0])
        return obs_dict


class MinigridCoordGoalWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        obs = self.reset()
        obs_dim = 3
        obs_min = 0
        obs_max = self.width - 1
        dir_max = 3

        self._max_episode_steps = 50

        self.observation_space = gym.spaces.Dict(
            desired_goal=spaces.Box(obs_min, obs_max,
                                    shape=(obs_dim-1,), dtype='float32'),
            achieved_goal=spaces.Box(obs_min, obs_max,
                                     shape=(obs_dim-1,), dtype='float32'),
            observation=spaces.Box(obs_min, obs_max,
                                   shape=(obs_dim,), dtype='float32'),
        )

    def step(self, action):
        self.steps +=1
        obs_max = self.width - 1
        dir_max = 3
        obs, reward, done, info = self.env.step(action)
        pos = np.array(self.env.agent_pos) / obs_max
        dir_ = np.array(self.env.agent_dir) / dir_max
        obs = np.append(pos, dir_).astype(np.float32)
        achieved_goal = pos
        obs_dict = dict(
            observation=obs, desired_goal=self.desired_goal, achieved_goal=achieved_goal)
        reward = reward - 1
        info['is_success'] = reward > -1.0
        return obs_dict, reward, done, info

    def reset(self, **kwargs):
        self.steps = 0
        obs_max = self.width - 1
        dir_max = 3
        self.desired_goal = np.array([7, 7]).astype(np.float32) / obs_max
        self.env.reset()
        pos = np.array(self.env.agent_pos) / obs_max
        dir_ = np.array(self.env.agent_dir) / dir_max
        obs = np.append(pos, dir_).astype(np.float32)
        achieved_goal = pos
        obs_dict = dict(
            observation=obs, desired_goal=self.desired_goal, achieved_goal=achieved_goal)
        return obs_dict


class MinigridImageGoalWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        obs = self.reset()
        obs_dim = 3
        obs_min = 0
        obs_max = self.width - 1
        dir_max = 3

        self._max_episode_steps = 50

        self.observation_space = gym.spaces.Dict(
            desired_goal=spaces.Box(obs_min, obs_max,
                                    shape=(obs_dim-1,), dtype='float32'),
            achieved_goal=spaces.Box(obs_min, obs_max,
                                     shape=(obs_dim-1,), dtype='float32'),
            observation=spaces.Box(obs_min, obs_max,
                                   shape=(obs_dim,), dtype='float32'),
        )

    def step(self, action):
        self.steps += 1
        obs_max = self.width - 1
        dir_max = 3
        obs, reward, done, info = self.env.step(action)
        pos = np.array(self.env.agent_pos) / obs_max
        dir_ = np.array(self.env.agent_dir) / dir_max
        obs = np.append(pos, dir_).astype(np.float32)
        achieved_goal = pos
        obs_dict = dict(
            observation=obs, desired_goal=self.desired_goal, achieved_goal=achieved_goal)
        reward = reward - 1
        info['is_success'] = reward > -1.0
        return obs_dict, reward, done, info

    def reset(self,**kwargs):
        self.steps = 0
        obs_max = self.width - 1
        dir_max = 3
        self.desired_goal = np.array([14, 14]).astype(np.float32) / obs_max
        self.env.reset()
        pos = np.array(self.env.agent_pos) / obs_max
        dir_ = np.array(self.env.agent_dir) / dir_max
        obs = np.append(pos, dir_).astype(np.float32)
        achieved_goal = pos
        obs_dict = dict(
            observation=obs, desired_goal=self.desired_goal, achieved_goal=achieved_goal)
        return obs_dict
