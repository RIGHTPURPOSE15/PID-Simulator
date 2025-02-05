import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import config as cg

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()
        
        # Fixed robot constraints
        self.max_speed = 1.0  # m/s
        self.max_acceleration = 0.5  # m/s²
        self.min_turning_radius = 0.1  # m
        
    def reset(self):
        self.previous_error = 0
        self.integral = 0
        self.last_output = 0
        
    def apply_robot_constraints(self, output, dt):
        # Apply acceleration limits
        max_output_change = self.max_acceleration * dt
        output_change = output - self.last_output
        output_change = np.clip(output_change, -max_output_change, max_output_change)
        output = self.last_output + output_change
        
        # Apply speed limits
        output = np.clip(output, -self.max_speed, self.max_speed)
        
        self.last_output = output
        return output
        
    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        
        # Basic PID computation
        output = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        
        # Apply robot's physical constraints
        output = self.apply_robot_constraints(output, dt)
        
        self.previous_error = error
        return output, error

def calculate_settling_time(t, position, setpoint, tolerance=0.005):
    """
    Calculate settling time as the first time at which the output stays within ±tolerance*setpoint for the remainder of the simulation.
    """
    steady_state_band = tolerance * setpoint
    for i in range(len(position)):
        # Check if for all future times, the error remains within the band
        if np.all(np.abs(position[i:] - setpoint) <= steady_state_band):
            return t[i]
    return float('inf')
    
    # Find the last time the signal leaves the band
    for i in range(len(settled_indices)-1):
        if settled_indices[i+1] - settled_indices[i] > 1:
            continue
    
    # Return the time when the signal finally enters the band
    return t[settled_indices[0]]

def simulate_pid(kp, ki, kd, simulation_time=120, dt=0.01):
    t = np.arange(0, simulation_time, dt)
    position = np.zeros_like(t)
    speed = np.zeros_like(t)
    errors = np.zeros_like(t)
    
    controller = PIDController(kp=kp, ki=ki, kd=kd)
    
    # Simulation loop
    for i in range(1, len(t)):
        output, error = controller.compute(1.0, position[i-1], dt)  # Step input of 1.0
        speed[i] = output
        position[i] = position[i-1] + speed[i] * dt
        errors[i] = error
    
    return t, position, speed, errors

config1 = cg.config(0.1, 1, 0.2)

pid_configs = config1.Kp()

plt.figure(figsize=(15, 7))

# Plot tolerance bands
plt.axhline(y=1.02, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±0.5% Tolerance')
plt.axhline(y=0.98, color='red', linestyle=':', linewidth=1, alpha=0.5)

for config in pid_configs:
    t, position, speed, errors = simulate_pid(**config)
    label = "Kp=" + str(config['kp']) + ", Ki=" + str(config['ki']) + ", Kd=" + str(config['kd'])
    plt.plot(t, position, label=label, linewidth=1.25)

# Add setpoint line
plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Setpoint')

plt.title('Position Response with Different PID Parameters', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Position', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'Sims\pid_tuning_comparison.png')
plt.close()

# Calculate performance metrics including settling time
results = []
for config in pid_configs:
    t, position, speed, errors = simulate_pid(**config)
    
    settling_time = calculate_settling_time(t, position, setpoint=1.0)
    
    results.append({
        'Kp': config['kp'],
        'Ki': config['ki'],
        'Kd': config['kd'],
        'Rise Time (s)': t[np.where(position >= 0.9)[0][0]] if any(position >= 0.9) else 'N/A',
        'Settling Time (s)': settling_time if settling_time != float('inf') else 'N/A',
        'Overshoot (%)': max(0, (np.max(position) - 1.0) * 100),
        'Steady-State Error': abs(1.0 - position[-1])
    })

results_df = pd.DataFrame(results)
print("\
PID Tuning Performance Metrics:")
print(results_df)