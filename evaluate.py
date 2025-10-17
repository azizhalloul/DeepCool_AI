import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.data.dataset import load_csv
from src.models.lstm import LSTMForecaster
from src.models.tcn import TCNForecaster
from src.controllers.rule_based import rule_controller

def load_model(path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(path, map_location=device)
    model_type = data.get('model_type', 'lstm')
    state = data['state_dict']
    if model_type == 'lstm':
        model = LSTMForecaster(n_features=4, hidden_size=128, num_layers=2, pred_len=15)
    else:
        model = TCNForecaster(num_inputs=4, num_channels=[64,64], pred_len=15)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def evaluate_pipeline(run_csv, model_path=None, plot=True):
    df = load_csv(run_csv)
    input_len = 60
    pred_len = 15
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_path:
        model = load_model(model_path, device=device)
    current_setpoint = float(df.loc[0, 'cooling_setpoint_C'])
    energy_costs = []
    overheating_events = 0
    for t in range(0, len(df) - input_len - pred_len):
        window = df.loc[t: t + input_len - 1]
        x = window[['ambient_C','it_load','rack_inlet_C','cooling_setpoint_C']].values.astype('float32')[None,...]
        if model is not None:
            with torch.no_grad():
                inp = torch.tensor(x).to(device)
                pred = model(inp).cpu().numpy().ravel()
        else:
            last_inlet = x[0,-1,2]
            pred = np.ones(pred_len) * last_inlet
        new_setpoint = rule_controller(pred, current_setpoint)
        energy = max(0.0, 25.0 - new_setpoint) * (1.0 + float(df.loc[t, 'it_load']))
        energy_costs.append(energy)
        actual_future_inlet = float(df.loc[t + input_len, 'rack_inlet_C'])
        if actual_future_inlet > 30.0:
            overheating_events += 1
        current_setpoint = new_setpoint
    avg_energy = float(np.mean(energy_costs))
    print('Avg energy cost (proxy):', avg_energy)
    print('Overheating events:', overheating_events)
    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(df['timestamp'], df['rack_inlet_C'], label='rack_inlet_C')
        plt.xlabel('time'); plt.ylabel('°C'); plt.legend(); plt.tight_layout()
        os.makedirs('experiments', exist_ok=True)
        plt.savefig('experiments/forecast_plot.png')
        print('Saved experiments/forecast_plot.png')
    return {'avg_energy': avg_energy, 'overheats': overheating_events}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    evaluate_pipeline(args.run, model_path=args.model)
