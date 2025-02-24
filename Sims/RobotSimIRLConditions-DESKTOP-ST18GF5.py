import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define simulation parameters
simTime = 120  # seconds
dt = 0.01  # time step

# Function to ensure folders exist
def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Define the PID simulation function
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0
    
    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        self.previous_error = error
        return output, error

def simulate_pid(kp, ki, kd, simulation_time=simTime, dt=0.01):
    t = np.arange(0, simulation_time, dt)
    position = np.zeros_like(t)
    speed = np.zeros_like(t)
    errors = np.zeros_like(t)
    controller = PIDController(kp, ki, kd)
    
    for i in range(1, len(t)):
        output, error = controller.compute(1.0, position[i-1], dt)
        speed[i] = output
        position[i] = position[i-1] + speed[i] * dt
        errors[i] = error
    
    return t, position, speed, errors

# Function to save results and generate graphs
def analyze_pid_variation(param_name, param_values, base_kp=0.5, base_ki=0.5, base_kd=0.5):
    folder = f"Results/{param_name}_variation"
    ensure_folder(folder)
    results = []
    
    for value in param_values:
        kp, ki, kd = base_kp, base_ki, base_kd
        if param_name == "Kp":
            kp = value
        elif param_name == "Ki":
            ki = value
        elif param_name == "Kd":
            kd = value
        
        t, position, speed, errors = simulate_pid(kp, ki, kd)
        overshoot = max(0, (np.max(position) - 1.0) * 100)
        steady_state_error = abs(1.0 - position[-1])
        settling_time = next((t[i] for i in range(len(t)) if np.all(np.abs(position[i:] - 1.0) <= 0.005)), float('inf'))
        
        results.append({
            "Parameter Value": value,
            "Overshoot (%)": overshoot,
            "Settling Time (s)": settling_time,
            "Steady-State Error": steady_state_error
        })
        
        plt.figure(figsize=(10, 5))
        plt.plot(t, position, label=f"{param_name}={value}")
        plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Setpoint')
        plt.axhline(y=1.005, color='red', linestyle=':', linewidth=1, label='Tolerance Band')
        plt.axhline(y=0.995, color='red', linestyle=':', linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.title(f"Response for {param_name}={value}")
        plt.legend()
        plt.grid()
        plt.savefig(f"{folder}/{param_name}_{value}.png")
        plt.close()
    
    df = pd.DataFrame(results)
    df.to_csv(f"{folder}/{param_name}_analysis.csv", index=False)

# Running the analysis
kp_values = np.linspace(0.1, 1.0, 5)
print('done')
ki_values = np.linspace(0.1, 1.0, 5)
print('done')
kd_values = np.linspace(0.1, 1.0, 5)
print('done')

analyze_pid_variation("Kp", kp_values)
print('done')
analyze_pid_variation("Ki", ki_values)
print('done')
analyze_pid_variation("Kd", kd_values)
print('done')

print("Simulation and analysis complete. Check the 'Results' folder.")
