import torch
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from src.env.datacenter_env_rl_lstm import DataCenterEnvLSTM
from src.models.model import LSTMForecaster

# Load dataset
df = pd.read_csv("data/raw/run1.csv")

# Load LSTM model
lstm_model = LSTMForecaster(n_features=4)
lstm_model.load_state_dict(torch.load("experiments/model_final.pt"))
lstm_model.eval()
print("✅ Loaded existing LSTM model.")

# Create RL environment
env = DataCenterEnvLSTM(df, lstm_model=lstm_model, forecast_len=15)
env = VecMonitor(DummyVecEnv([lambda: env]))

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
model.save("experiments/ppo_lstm_datacenter")
print("✅ Hybrid RL+LSTM model trained and saved.")
