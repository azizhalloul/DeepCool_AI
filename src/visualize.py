import matplotlib.pyplot as plt

def plot_time_series(df, out='experiments/ts_plot.png'):
    plt.figure(figsize=(12,5))
    plt.plot(df['timestamp'], df['rack_inlet_C'], label='rack_inlet_C')
    plt.plot(df['timestamp'], df['ambient_C'], label='ambient_C')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print('Saved', out)
