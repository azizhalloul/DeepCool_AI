import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class DataCenterEnvLSTM(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, lstm_model=None, forecast_len=15, max_delta=3.0):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.t = 0
        self.max_t = len(self.df) - 1
        self.max_delta = float(max_delta)

        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_delta, high=self.max_delta, shape=(1,), dtype=np.float32)

        self.lstm_model = lstm_model
        self.forecast_len = forecast_len

    def step(self, action):
        action = float(np.clip(action, -self.max_delta, self.max_delta))
        self.df.loc[self.t, 'cooling_setpoint_C'] += action

        sp = float(self.df.loc[self.t, 'cooling_setpoint_C'])
        energy_cost = max(0.0, 25.0 - sp) * (1.0 + float(self.df.loc[self.t, 'it_load']))

        self.t += 1
        terminated = self.t >= self.max_t
        truncated = False  # you can implement truncation logic if needed

        obs_row = self.df.loc[self.t]
        obs = np.array([
            obs_row['ambient_C'],
            obs_row['it_load'],
            obs_row['rack_inlet_C'],
            obs_row['cooling_setpoint_C']
        ], dtype=np.float32)

        reward = -energy_cost
        if obs_row['rack_inlet_C'] > 30.0:
            reward -= 200.0

        info = {"energy_cost": energy_cost}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.t = 0
        obs_row = self.df.loc[self.t]
        obs = np.array([
            obs_row['ambient_C'],
            obs_row['it_load'],
            obs_row['rack_inlet_C'],
            obs_row['cooling_setpoint_C']
        ], dtype=np.float32)
        info = {}
        return obs, info

    def render(self, mode="human"):
        obs_row = self.df.loc[self.t]
        print(f"Time {self.t}: Rack Temp={obs_row['rack_inlet_C']:.2f}C | Cooling Setpoint={obs_row['cooling_setpoint_C']:.2f}C")
