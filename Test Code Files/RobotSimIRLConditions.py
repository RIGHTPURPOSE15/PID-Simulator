import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import config as cg

simTime = 120
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()
        
        # Fixed robot constraints
        self.max_speed = 1.0       # m/s
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

# -------------------------------
# Performance Metric Functions
# -------------------------------
def calculate_settling_time(t, position, setpoint, tolerance=0.005):
    """
    Calculate settling time as the first time at which the output stays within ±tolerance*setpoint 
    for the remainder of the simulation.
    """
    steady_state_band = tolerance * setpoint
    for i in range(len(position)):
        if np.all(np.abs(position[i:] - setpoint) <= steady_state_band):
            return t[i]
    return float('inf')

# -------------------------------
# Simulation Function
# -------------------------------
def simulate_pid(kp, ki, kd, simulation_time=simTime, dt=0.01,
                 wheel_diameter=0.1, track_width=0.5):
    """
    Simulate the closed-loop response of the system controlled by a PID controller.
    Also computes the left and right wheel speeds.
    
    Parameters:
      kp, ki, kd: PID gains.
      simulation_time: Total time of simulation (s).
      dt: Time step (s).
      wheel_diameter: Diameter of the wheels (m).
      track_width: Distance between the wheels (m).
    
    Returns:
      t: Time array.
      position: Position of the robot over time.
      speed: Overall speed command over time.
      errors: Error signal over time.
      left_wheel_speed, right_wheel_speed: Linear speeds of the left and right wheels.
      left_wheel_angular_speed, right_wheel_angular_speed: Angular speeds (rad/s) of the wheels.
    """
    t = np.arange(0, simulation_time, dt)
    position = np.zeros_like(t)
    speed = np.zeros_like(t)
    errors = np.zeros_like(t)
    
    controller = PIDController(kp=kp, ki=ki, kd=kd)
    
    # Simulation loop
    for i in range(1, len(t)):
        # Using a step input of 1.0 as the setpoint.
        output, error = controller.compute(1.0, position[i-1], dt)
        speed[i] = output
        position[i] = position[i-1] + speed[i] * dt
        errors[i] = error

    # For a differential drive robot moving in a straight line, we assume:
    #   left wheel speed = right wheel speed = overall speed.
    # However, you can modify this section to include turning dynamics.
    left_wheel_speed = speed.copy()
    right_wheel_speed = speed.copy()
    
    # Compute wheel angular speeds (rad/s) from linear speed and wheel radius.
    wheel_radius = wheel_diameter / 2.0
    left_wheel_angular_speed = left_wheel_speed / wheel_radius
    right_wheel_angular_speed = right_wheel_speed / wheel_radius

    return (t, position, speed, errors, 
            left_wheel_speed, right_wheel_speed, 
            left_wheel_angular_speed, right_wheel_angular_speed)

# -------------------------------
# MAIN SCRIPT
# -------------------------------

# Create a configuration instance.
# (Here, range 0.1 to 1 with increment 0.2 is used for demonstration.)
config1 = cg.config(0.5, 0.5, 0.4)

# You can choose which set of parameters to vary.
# For example, to vary Kp:
pid_configs = config1.Kp()

# Alternatively, you can generate all combinations using the PID method:
# pid_configs = config1.PID(kp_range=(0.1,1,0.3), ki_range=(0.1,1,0.3), kd_range=(0.1,1,0.3))

# Create a figure with two subplots: 
#   Top: position response.
#   Bottom: wheel speeds.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot tolerance bands and setpoint in the position response subplot.
ax1.axhline(y=1.02, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±0.5% Tolerance')
ax1.axhline(y=0.98, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Setpoint')

# Colors for plotting (one per configuration)
colors = plt.cm.viridis(np.linspace(0, 1, len(pid_configs)))

# Loop through each PID configuration.
for idx, config in enumerate(pid_configs):
    (t, position, speed, errors,
     left_speed, right_speed,
     left_ang_speed, right_ang_speed) = simulate_pid(**config, simulation_time=simTime)  # shorter sim for clarity

    label = f"Kp={config['kp']}, Ki={config['ki']}, Kd={config['kd']}"
    ax1.plot(t, position, label=label, color=colors[idx], linewidth=1.5)
    
    # Plot wheel speeds in the bottom subplot.
    # (Since our simulation is for straight–line motion, left and right speeds are the same.
    #  However, they are plotted separately for demonstration.)
    ax2.plot(t, left_speed, linestyle='--', color=colors[idx],
             label=f"Left wheel ({label})")
    ax2.plot(t, right_speed, linestyle='-.', color=colors[idx],
             label=f"Right wheel ({label})")

# Customize the position subplot.
ax1.set_title('Position Response with Different PID Parameters', fontsize=16)
ax1.set_ylabel('Position (m)', fontsize=14)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend(fontsize=10)

# Customize the wheel speed subplot.
ax2.set_title('Wheel Linear Speeds', fontsize=16)
ax2.set_xlabel('Time (s)', fontsize=14)
ax2.set_ylabel('Speed (m/s)', fontsize=14)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend(fontsize=10, ncol=2)

plt.tight_layout()
plt.savefig(r'Sims\pid_tuning_with_wheel_speeds.png')
plt.close()

# -------------------------------
# Performance Metrics Calculation
# -------------------------------
results = []
for config in pid_configs:
    t, position, speed, errors, *_ = simulate_pid(**config, simulation_time=30)
    # Determine rise time: first time the position reaches 90% of setpoint.
    rise_time = t[np.where(position >= 0.9)[0][0]] if np.any(position >= 0.9) else 'N/A'
    settling_time = calculate_settling_time(t, position, setpoint=1.0)
    settling_time = settling_time if settling_time != float('inf') else 'N/A'
    
    results.append({
        'Kp': config['kp'],
        'Ki': config['ki'],
        'Kd': config['kd'],
        'Rise Time (s)': rise_time,
        'Settling Time (s)': settling_time,
        'Overshoot (%)': max(0, (np.max(position) - 1.0) * 100),
        'Steady-State Error': abs(1.0 - position[-1])
    })

results_df = pd.DataFrame(results)
print("\nPID Tuning Performance Metrics:")
print(results_df)