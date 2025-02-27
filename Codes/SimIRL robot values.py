import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Simulation parameters
st = 120             # Total simulation time in seconds
dt = 0.01            # Time step in seconds
max_track_width = 1.0  # Physical track limits: robot must remain within [-1.0, 1.0]
setpoint = 0.0       # Desired center line position (robot should center on 0.0)

# Ensure results folder exists
RESULTS_FOLDER = 'Sim_Results'
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Settling time calculation:
# Requires that the error remains within ±tolerance continuously for window_duration seconds.
def calculate_settling_time(time_array, position_array, setpoint, tolerance=0.05, window_duration=3.0):
    window_samples = int(window_duration / dt)
    for i in range(len(position_array) - window_samples):
        if np.all(np.abs(position_array[i:i+window_samples] - setpoint) <= tolerance):
            return time_array[i]
    return time_array[-1]  # if never settled, return full simulation time

# Full PID Controller class with noise, delay, and actuator saturation.
class PIDController:
    def __init__(self, kp, ki, kd, friction=0.010, noise_level=0.001, delay=0.001, max_acceleration=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.friction = friction       # friction in the control (could be used as feed-forward)
        self.noise_level = noise_level # sensor noise level
        self.delay = delay             # control delay
        self.max_acceleration = max_acceleration  # actuator saturation
        self.reset()

    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        self.delay_steps = int(self.delay / dt) if dt > 0 else 0
        self.delay_buffer = [0.0 for _ in range(self.delay_steps)]

    def apply_constraints(self, output, dt):
        # Limit the change in output based on maximum acceleration.
        max_change = self.max_acceleration * dt
        output_change = output - self.last_output
        output_change = np.clip(output_change, -max_change, max_change)
        output = self.last_output + output_change
        output = np.clip(output, -self.max_acceleration, self.max_acceleration)
        self.last_output = output
        return output

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        self.previous_error = error

        # Basic PID computation
        raw_output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Apply constraints (saturation, maximum change)
        constrained_output = self.apply_constraints(raw_output, dt)
        
        # Subtract friction (feed-forward effect) and add noise
        output_with_friction = constrained_output - self.friction * measured_value
        noise = np.random.normal(0, self.noise_level)
        output_noisy = output_with_friction + noise
        
        # Simulate control delay using a buffer
        self.delay_buffer.append(output_noisy)
        delayed_output = self.delay_buffer.pop(0) if self.delay_buffer else output_noisy
        
        return delayed_output

# Simulation function that uses the full PIDController and additional physics
def simulate_pid(kp, ki, kd, sim_time=st, dt=0.01, setpoint=setpoint, max_track_width=max_track_width):
    n_steps = int(sim_time / dt)
    time_vals = np.linspace(0, sim_time, n_steps)
    
    # Start with an offset (e.g., robot starts 0.5 m off-center)
    position = 0.5  
    velocity = 0.0
    
    # For sensor filtering (low-pass filter parameters)
    filtered_position = position
    filter_alpha = 0.1  # weight for new measurement
    
    # Instantiate the PID controller with nonlinear dynamics considerations
    pid = PIDController(kp, ki, kd, friction=0.05, noise_level=0.005, delay=0.001, max_acceleration=1.0)
    pid.reset()
    
    positions = []
    
    # Nonlinear friction parameters
    viscous_friction = 0.1   # proportional to velocity
    coulomb_friction = 0.05  # constant opposing force
    
    for i in range(n_steps):
        # Sensor filtering: update filtered position from the true position
        filtered_position = filter_alpha * position + (1 - filter_alpha) * filtered_position
        
        # Get control command from PID controller based on the filtered sensor reading.
        control = pid.compute(setpoint, filtered_position, dt)
        
        # Use the control as an acceleration command (already saturated within PID)
        acceleration = control
        
        # Compute friction force: opposes the direction of velocity.
        if velocity != 0:
            friction_force = viscous_friction * velocity + coulomb_friction * np.sign(velocity)
        else:
            friction_force = 0.0
        
        # Net acceleration after subtracting friction
        net_acceleration = acceleration - friction_force
        
        # Update velocity and then position
        velocity += net_acceleration * dt
        new_position = position + velocity * dt
        
        # Reflective boundary conditions: if the new position exceeds the track limits, reflect it.
        if new_position > max_track_width:
            new_position = max_track_width - (new_position - max_track_width)
            velocity = -velocity * 0.5  # damp the velocity on reflection
        elif new_position < -max_track_width:
            new_position = -max_track_width - (new_position + max_track_width)
            velocity = -velocity * 0.5
        
        position = new_position
        positions.append(position)
    
    positions = np.array(positions)
    settling_time = calculate_settling_time(time_vals, positions, setpoint, tolerance=0.05, window_duration=3.0)
    return time_vals, positions, settling_time

# Analysis function: Vary one PID parameter while keeping the others fixed.
# Each parameter is varied over 10 data points around a working point that yields settling times less than the maximum.
def analyze_pid_variation(param_name, values, sim_time=st, dt=0.01, setpoint=setpoint):
    results = []
    for val in values:
        if param_name == 'Kp':
            kp, ki, kd = val, 0.5, 0.5
        elif param_name == 'Ki':
            kp, ki, kd = 1.0, val, 0.5
        elif param_name == 'Kd':
            kp, ki, kd = 1.0, 0.5, val
        
        time_vals, pos_vals, settle_time = simulate_pid(kp, ki, kd, sim_time, dt, setpoint, max_track_width)
        results.append({'Parameter': param_name, 'Value': val, 'Settling_Time': settle_time})
        
        plt.figure(figsize=(8, 4))
        plt.plot(time_vals, pos_vals, label='Position')
        plt.axhline(y=setpoint, color='green', linestyle='--', label='Setpoint')
        plt.axhline(y=setpoint + 0.05, color='red', linestyle=':', label='Tolerance')
        plt.axhline(y=setpoint - 0.05, color='red', linestyle=':')
        plt.ylim(-max_track_width * 1.5, max_track_width * 1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title(f'PID Response for {param_name} = {val:.2f}')
        plt.legend()
        plt.grid(True)
        filename = os.path.join(RESULTS_FOLDER, f'{param_name}_response_{val:.2f}.png')
        plt.savefig(filename)
        plt.close()
    
    df = pd.DataFrame(results)
    csv_filename = os.path.join(RESULTS_FOLDER, f'{param_name}_analysis.csv')
    df.to_csv(csv_filename, index=False)
    print(f'Saved CSV analysis for {param_name} in {csv_filename}')

# Define ideal parameter ranges (10 data points each) around the working point.
if __name__ == '__main__':
    kp_values = np.linspace(0.8, 2.2, 15)  # e.g., from 0.8 to 1.2
    ki_values = np.linspace(0.1, 0.7, 7)  # e.g., from 0.4 to 0.6
    kd_values = np.linspace(0.3, 3.0, 28)  # e.g., from 0.4 to 1.2

    print('Running analysis for Kp...')
    analyze_pid_variation('Kp', kp_values)
    print('Running analysis for Ki...')
    analyze_pid_variation('Ki', ki_values)
    print('Running analysis for Kd...')
    analyze_pid_variation('Kd', kd_values)

    print('Simulation and CSV analysis complete. Check the "Real_Results" folder for output files.')

# ------------------------------------------------------------------
# Explanation for Gradually Increasing Error Peaks:
#
# In some PID constant combinations, you may observe that the error peaks
# (i.e., overshoots) gradually increase over time. This behavior can occur due to:
#
# 1. **Integrator Windup:** If the integral term accumulates a large error during periods
#    when the actuator is saturated or when the system is experiencing a disturbance,
#    it can lead to progressively larger corrective actions once the error changes sign.
#
# 2. **Insufficient Derivative Damping:** If the derivative term is too small relative
#    to the proportional and integral terms, it may fail to adequately counteract the
#    overshoot caused by the other terms, resulting in oscillations that grow over time.
#
# The above simulation uses sensor filtering, delay, actuator saturation, and a nonlinear
# friction model. Tuning the PID parameters (as in the "ideal" ranges above) can help ensure
# that the settling time is less than the simulation duration and that error peaks do not
# progressively increase.
# ------------------------------------------------------------------
