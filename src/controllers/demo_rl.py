# demo_rl.py
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd
import torch
from stable_baselines3 import PPO
from src.env.datacenter_env_rl_lstm import DataCenterEnvLSTM
from src.models.model import LSTMForecaster
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/raw/run1.csv")

# -----------------------------
# Load LSTM model
# -----------------------------
lstm_model = LSTMForecaster(n_features=4)
lstm_model.load_state_dict(torch.load("experiments/model_final.pt", map_location="cpu"))
lstm_model.eval()
print("✅ Loaded LSTM model.")

# -----------------------------
# Load RL agent
# -----------------------------
env = DummyVecEnv([lambda: DataCenterEnvLSTM(df, lstm_model=lstm_model, forecast_len=15)])
model = PPO.load("experiments/ppo_lstm_datacenter", env=env)
print("✅ Loaded RL agent.")

# -----------------------------
# Run simulation and record frames
# -----------------------------
obs = env.reset()
frames = []
rack_temps = []
setpoints = []

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    obs_val = obs[0] if isinstance(obs, np.ndarray) and obs.ndim == 2 else obs
    rack_temps.append(obs_val[2])
    setpoints.append(obs_val[3])

    # Plot current step
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(rack_temps, label="Rack Temp (C)")
    ax.plot(setpoints, label="Cooling Setpoint (C)")
    ax.axhline(30, color='r', linestyle='--', label="Overheat threshold")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Temperature / Setpoint (C)")
    ax.set_title("Hybrid RL+LSTM Cooling Performance")
    ax.legend()

    # Convert figure to image array (new way)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = image.reshape((height, width, 4))[:, :, :3]  # remove alpha channel
    frames.append(image)
    plt.close(fig)

# -----------------------------
# Save GIF
# -----------------------------
imageio.mimsave("experiments/demo_rl.gif", frames, fps=10)
print("✅ Demo GIF saved: experiments/demo_rl.gif")

# -----------------------------
# Metrics
# -----------------------------
overheating_events = sum(t > 30 for t in rack_temps)
avg_setpoint = np.mean(setpoints)
print(f"Overheating events: {overheating_events}")
print(f"Average cooling setpoint: {avg_setpoint:.2f}")
