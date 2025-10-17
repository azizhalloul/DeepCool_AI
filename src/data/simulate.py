import numpy as np
import pandas as pd
from datetime import datetime
import argparse



def generate_synthetic_run(length_minutes=24*60, seed=0, start_ts=None):
    """Generate synthetic telemetry DataFrame with columns:
    timestamp, ambient_C, it_load, rack_inlet_C, cooling_setpoint_C
    """
    np.random.seed(seed)
    if start_ts is None:
        start_ts = datetime.now()
    t = pd.date_range(start=start_ts, periods=length_minutes, freq='T')

    ambient = 22 + 3 * np.sin(np.linspace(0, 2*np.pi, length_minutes) - 0.2) + 0.5*np.random.randn(length_minutes)
    load = 0.5 + 0.35*np.sin(np.linspace(0, 8*np.pi, length_minutes))
    load += 0.08*np.sin(np.linspace(0, 40*np.pi, length_minutes))
    load += 0.04*np.random.randn(length_minutes)
    load = np.clip(load, 0, 1)

    setpoint = 19 + 1.5*np.sin(np.linspace(0, 1*np.pi, length_minutes))
    inlet = np.zeros(length_minutes)
    inlet[0] = ambient[0] + 4.5 * load[0]
    for i in range(1, length_minutes):
        inertia = 0.985
        heat_gain = 6.0 * load[i]
        cooling_effect = max(0.0, (setpoint[i-1] - ambient[i]) * 0.1)
        inlet[i] = inertia*inlet[i-1] + (1-inertia)*(ambient[i] + heat_gain - cooling_effect) + 0.15*np.random.randn()

    df = pd.DataFrame({
        'timestamp': t,
        'ambient_C': ambient,
        'it_load': load,
        'rack_inlet_C': inlet,
        'cooling_setpoint_C': setpoint,
    })
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/raw/run1.csv')
    parser.add_argument('--minutes', type=int, default=1440)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = generate_synthetic_run(length_minutes=args.minutes, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f'Saved {args.out} ({args.minutes} minutes)')
