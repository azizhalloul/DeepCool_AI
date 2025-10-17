import gym
from gym import spaces
import numpy as np


class DataCenterEnv(gym.Env):
    def __init__(self, df, max_delta=3.0):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.t = 0
        self.max_t = len(self.df) - 1
        self.max_delta = float(max_delta)
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_delta, high=self.max_delta, shape=(1,), dtype=np.float32)


    def step(self, action):
        action = float(np.clip(action, -self.max_delta, self.max_delta))
        self.df.loc[self.t, 'cooling_setpoint_C'] += action
        sp = float(self.df.loc[self.t, 'cooling_setpoint_C'])
        energy_cost = max(0.0, 25.0 - sp) * (1.0 + float(self.df.loc[self.t, 'it_load']))
        self.t += 1
        done = self.t >= self.max_t
        obs_row = self.df.loc[self.t]
        obs = np.array([obs_row['ambient_C'], obs_row['it_load'], obs_row['rack_inlet_C'], obs_row['cooling_setpoint_C']], dtype=np.float32)
        reward = -energy_cost
        if obs_row['rack_inlet_C'] > 30.0:
            reward -= 200.0
        info = {'energy_cost': energy_cost}
        return obs, reward, done, info

    def reset(self, start_idx=0, seed=None, options=None):
        self.t = int(start_idx)
        obs_row = self.df.loc[self.t]
        obs = np.array([
            obs_row['ambient_C'],
            obs_row['it_load'],
            obs_row['rack_inlet_C'],
            obs_row['cooling_setpoint_C']
        ], dtype=np.float32)
        info = {}  # you can add extra info here if needed
        return obs, info


def render(self, mode='human'):
    pass