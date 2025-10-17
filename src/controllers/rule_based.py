import numpy as np

def rule_controller(predicted_inlet, current_setpoint, threshold=27.0, min_setpoint=15.0, max_setpoint=24.0):
    if hasattr(predicted_inlet, 'max'):
        peak = float(np.max(predicted_inlet))
    else:
        peak = float(predicted_inlet)
    new_setpoint = float(current_setpoint)
    if peak > threshold:
        new_setpoint = max(min_setpoint, new_setpoint - 1.8)
    elif peak < (threshold - 1.5):
        new_setpoint = min(max_setpoint, new_setpoint + 0.4)
    new_setpoint = float(np.clip(new_setpoint, min_setpoint, max_setpoint))
    return new_setpoint
