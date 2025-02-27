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

# We will use 10 data points in the same range for Kp, Ki, and Kd.
def grid_search_pid(sim_time=st, dt=dt, setpoint=setpoint, max_track_width=max_track_width):
    # Define a "working" range that ensures the system settles.
    kp_values = np.linspace(0.8, 2.2, 15)
    ki_values = np.linspace(0.1, 0.7, 7)
    kd_values = np.linspace(0.3, 3.0, 28)
    
    results = []
    # Optionally, you can create a subfolder for grid search plots.
    grid_folder = os.path.join(RESULTS_FOLDER, 'grid_search')
    if not os.path.exists(grid_folder):
        os.makedirs(grid_folder)
    
    for kp in kp_values:
        for ki in ki_values:
            for kd in kd_values:
                time_vals, pos_vals, settle_time = simulate_pid(kp, ki, kd, sim_time, dt, setpoint, max_track_width)
                # Only record combinations that settle (settling_time < sim_time)
                results.append({
                    'Kp': kp,
                    'Ki': ki,
                    'Kd': kd,
                    'Settling_Time': settle_time
                })
                
                # Save plot for each combination if desired.
                # (You could comment out the plot generation if there are too many.)
                plt.figure(figsize=(8, 4))
                plt.plot(time_vals, pos_vals, label='Position')
                plt.axhline(y=setpoint, color='green', linestyle='--', label='Setpoint')
                plt.axhline(y=setpoint + 0.05, color='red', linestyle=':', label='Tolerance')
                plt.axhline(y=setpoint - 0.05, color='red', linestyle=':')
                plt.ylim(-max_track_width * 1.5, max_track_width * 1.5)
                plt.xlabel('Time (s)')
                plt.ylabel('Position (m)')
                plt.title(f'PID Response: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}\nSettling Time = {settle_time:.2f}s')
                plt.legend()
                plt.grid(True)
                filename = os.path.join(grid_folder, f'PID_Kp{kp:.2f}_Ki{ki:.2f}_Kd{kd:.2f}.png')
                plt.savefig(filename)
                plt.close()
    
    df = pd.DataFrame(results)
    csv_filename = os.path.join(RESULTS_FOLDER, 'grid_search_pid_analysis.csv')
    df.to_csv(csv_filename, index=False)
    print(f'Saved grid search CSV analysis in {csv_filename}')

if __name__ == '__main__':
    # Run grid search for PID parameters.
    grid_search_pid()
    
    print('Grid search simulation and CSV analysis complete. Check the "Real_Results" folder for output files.')
