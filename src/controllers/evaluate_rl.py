import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from src.env.datacenter_env_rl_lstm import DataCenterEnvLSTM
from src.models.model import LSTMForecaster

# Load dataset
df = pd.read_csv("data/raw/run1.csv")

# Load LSTM
lstm_model = LSTMForecaster(n_features=4)
lstm_model.load_state_dict(torch.load("experiments/model_final.pt"))
lstm_model.eval()
print("✅ Loaded LSTM model.")

# Load RL agent
env = DataCenterEnvLSTM(df, lstm_model=lstm_model, forecast_len=15)
vec_env = VecMonitor(DummyVecEnv([lambda: env]))
model = PPO.load("experiments/ppo_lstm_datacenter", env=vec_env)

# Reset environment
obs = vec_env.reset()
obs = obs[0]  # take the first env from vectorized batch

done = False
rack_temps = []
setpoints = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step([action])  # pass as list for VecEnv
    obs = obs[0]
    rack_temps.append(obs[2])
    setpoints.append(obs[3])

# Plot
plt.figure(figsize=(10,5))
plt.plot(rack_temps, label="Rack Temp (C)")
plt.plot(setpoints, label="Cooling Setpoint (C)")
plt.axhline(30, color='r', linestyle='--', label="Overheat threshold")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature / Setpoint (C)")
plt.legend()
plt.title("Hybrid RL+LSTM Cooling Performance")
plt.savefig("experiments/rl_lstm_forecast_plot.png")
plt.show()

# Metrics
overheating_events = sum(t>30 for t in rack_temps)
avg_setpoint = np.mean(setpoints)
print(f"Overheating events: {overheating_events}")
print(f"Average cooling setpoint: {avg_setpoint:.2f}")
