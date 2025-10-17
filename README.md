# DeepCool_AI 🚀

**Hybrid RL + LSTM AI for Energy-Efficient Data Center Cooling**

---

## Overview

DeepCool_AI is an advanced AI system designed to **optimize data center cooling** using a combination of **Long Short-Term Memory (LSTM) forecasting** and **Reinforcement Learning (RL)**.  
The AI predicts rack inlet temperatures and dynamically adjusts cooling setpoints, significantly **reducing energy consumption** while maintaining safe operational limits.

> 💡 Key Impact: Reduced energy waste and prevented overheating events in real-time.

---

## Features

- **LSTM Forecaster**: Predicts future rack inlet temperatures from historical data.
- **RL Agent (PPO)**: Learns an optimal cooling strategy to minimize energy cost.
- **Hybrid System**: Combines prediction + control for efficient and safe cooling.
- **Visualization**: Generates plots, GIFs, and videos to monitor performance.
- **Metrics**: Tracks overheating events and average cooling setpoints.

---



## Demo

Below is a demo of the AI in action:

[RL+LSTM Cooling Demo] (https://drive.google.com/uc?export=download&id=1JAkam7YOUR_FILE_ID)

- ✅ Overheating events: 15  
- ✅ Average cooling setpoint: 19.96°C  
- The AI dynamically adjusts cooling to prevent overheating while minimizing energy cost.

---

## Project Structure

The project is organized as follows:

- **data/raw/** – Raw input CSV datasets.
- **experiments/** – Generated plots, trained models, and demo GIFs/MP4s.
- **assets/** – Demo visuals including GIFs and small MP4 videos.
- **src/** – Main source code.
  - **controllers/** – Scripts for RL training, evaluation, and demo.
  - **data/** – Dataset loading and simulation utilities.
  - **env/** – Gym environments for RL.
  - **models/** – LSTM forecasting models.
- **requirements.txt** – Python dependencies.
- **README.md** – Project documentation.

