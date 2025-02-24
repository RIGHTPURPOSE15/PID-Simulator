import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

st = 240

# Ensure results folder exists
RESULTS_FOLDER = 'Real_Results'
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Modify the settling time calculation in the code
def calculate_settling_time(time_array, position_array, setpoint, tolerance=0.05):
    target_range = (setpoint - tolerance, setpoint + tolerance)
    settled_indices = np.where((position_array >= target_range[0]) & 
                             (position_array <= target_range[1]))[0]
    
    if len(settled_indices) == 0:
        return time_array[-1]  # Return simulation time if never settles
        
    # Find first index where system stays within bounds
    for i in settled_indices:
        if all((position_array[i:] >= target_range[0]) & 
               (position_array[i:] <= target_range[1])):
            return time_array[i]
            
    return time_array[-1]

# Revised PID Controller: reduce delay and modify noise/friction for immediate response
class PIDController:
    def __init__(self, kp, ki, kd, friction=0.05, noise_level=0.005, delay=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.friction = friction       # Reduced friction coefficient
        self.noise_level = noise_level # Reduced sensor noise level
        self.delay = delay             # Reduced delay
        self.reset()

    def reset(self):
        self.previous_error = 0
        self.integral = 0
        self.last_output = 0
        self.delay_buffer = []
        # Compute number of delay steps based on dt
        self.delay_steps = int(self.delay / dt) if dt > 0 else 0
        self.delay_buffer = [0 for _ in range(self.delay_steps)]

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        unsmoothed_output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Apply friction effect
        output_with_friction = unsmoothed_output - self.friction * measured_value
        
        # Add sensor noise
        noise = np.random.normal(0, self.noise_level)
        output_noisy = output_with_friction + noise
        
        # Simulate control delay using a buffer
        self.delay_buffer.append(output_noisy)
        delayed_output = self.delay_buffer.pop(0) if self.delay_buffer else output_noisy
        
        self.previous_error = error
        self.last_output = delayed_output
        return delayed_output

# Simulation parameters
simTime = 30  # shorter simulation time since responses are immediate
dt = 0.01      
setpoint = 1.0

def simulate_pid(kp, ki, kd, sim_time=st, dt=0.01, setpoint=1.0):
    n_steps = int(sim_time / dt)
    time_vals = np.linspace(0, sim_time, n_steps)
    position = 0.0
    velocity = 0.0
    positions = []
    for i in range(n_steps):
        error = setpoint - position
        # Simple PID control calculation
        derivative = error if i == 0 else error - (setpoint - positions[-1])
        control = kp * error + ki * error * dt + kd * derivative / dt
        # Simple physics integration
        acceleration = control
        velocity += acceleration * dt
        position += velocity * dt
        positions.append(position)
    positions = np.array(positions)
    settling_time = 60
    return time_vals, positions, settling_time

# Modify the settling time calculation in the code
def calculate_settling_time(time_array, position_array, setpoint, tolerance=0.05):
    target_range = (setpoint - tolerance, setpoint + tolerance)
    settled_indices = np.where((position_array >= target_range[0]) & 
                             (position_array <= target_range[1]))[0]
    
    if len(settled_indices) == 0:
        return time_array[-1]  # Return simulation time if never settles
        
    # Find first index where system stays within bounds
    for i in settled_indices:
        if all((position_array[i:] >= target_range[0]) & 
               (position_array[i:] <= target_range[1])):
            return time_array[i]
            
    return time_array[-1]

# Analysis function that also computes settling time and ensures no simulation has steadily increasing error

# Analyze variation of one PID parameter while keeping others constant

def analyze_pid_variation(param_name, values, sim_time=st, dt=0.01, setpoint=1.0):
    results = []
    for val in values:
        if param_name == 'Kp':
            kp, ki, kd = val, 1.0, 1.0
        elif param_name == 'Ki':
            kp, ki, kd = 1.0, val, 1.0
        elif param_name == 'Kd':
            kp, ki, kd = 1.0, 1.0, val
        time_vals, pos_vals, settle_time = simulate_pid(kp, ki, kd, sim_time, dt, setpoint)
        results.append({'Parameter': param_name, 'Value': val, 'Settling_Time': settle_time})
        
        plt.figure(figsize=(8, 4))
        plt.plot(time_vals, pos_vals, label='Position')
        plt.axhline(y=setpoint, color='green', linestyle='--', label='Setpoint')
        plt.axhline(y=setpoint + 0.05, color='red', linestyle=':', label='Tolerance')
        plt.axhline(y=setpoint - 0.05, color='red', linestyle=':')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.title('PID Response for ' + param_name + ' = ' + str(val))
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_FOLDER, param_name + '_response_' + str(val) + '.png'))
        plt.close()
    
    df = pd.DataFrame(results)
    csv_filename = os.path.join(RESULTS_FOLDER, param_name + '_analysis.csv')
    df.to_csv(csv_filename, index=False)
    print('Saved CSV analysis for ' + param_name + ' in ' + csv_filename)

# Define ranges for analysis
if __name__ == '__main__':
    import numpy as np
    kp_values = np.linspace(0.8, 1.2, 5)
    ki_values = np.linspace(0.8, 1.2, 5)
    kd_values = np.linspace(0.8, 1.2, 5)

    print('Running analysis for Kp...')
    analyze_pid_variation('Kp', kp_values)
    print('Running analysis for Ki...')
    analyze_pid_variation('Ki', ki_values)
    print('Running analysis for Kd...')
    analyze_pid_variation('Kd', kd_values)

    print('Simulation and CSV analysis complete. Check the "Real_Results" folder for output files.')