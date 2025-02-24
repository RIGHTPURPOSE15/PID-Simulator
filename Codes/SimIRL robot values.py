# Simplified line following robot simulation without physical considerations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class LineFollowingRobot:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()
        
    def reset(self):
        self.previous_error = 0
        self.integral = 0
        
    def compute(self, target_position, current_position, dt):
        error = target_position - current_position
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

def simulate_line_following(controller, tolerance=0.005):
    dt = 0.01
    sim_time = 10  # Reduced simulation time
    n_steps = int(sim_time / dt)
    
    position = 0.0  # Start position (deviation from line)
    target = 1.0    # Target position (line position)
    
    positions = []
    times = []
    settling_time = None
    
    for i in range(n_steps):
        current_time = i * dt
        
        # Simple position update based on controller output
        control_signal = controller.compute(target, position, dt)
        position += control_signal * dt  # Direct integration without physics
        
        positions.append(position)
        times.append(current_time)
        
        # Check for settling time - must stay within tolerance
        if settling_time is None and abs(target - position) < tolerance:
            if i > 50 and all([abs(target - pos) < tolerance for pos in positions[-50:]]):
                settling_time = current_time
                
    if settling_time is None or settling_time == 0:
        settling_time = sim_time  # Default to simulation time if no settling detected
        
    return np.array(times), np.array(positions), settling_time

# Test different controller parameters
def analyze_controller(param_name, param_values):
    results = []
    
    for value in param_values:
        kp = 1.0
        ki = 0.5
        kd = 0.2
        
        if param_name == 'Kp':
            kp = value
        elif param_name == 'Ki':
            ki = value
        elif param_name == 'Kd':
            kd = value
            
        controller = LineFollowingRobot(kp, ki, kd)
        t_arr, pos_arr, settling_time = simulate_line_following(controller)
        
        results.append({
            'Parameter': param_name,
            'Value': value,
            'Settling Time': settling_time,
            'Final Error': abs(1.0 - pos_arr[-1])
        })
        
        plt.figure(figsize=(10, 5))
        plt.plot(t_arr, pos_arr, label=f'Response ({param_name}={value:.2f})')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Target')
        plt.axhline(y=1.005, color='g', linestyle=':', label='Tolerance')
        plt.axhline(y=0.995, color='g', linestyle=':')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.title(f'Line Following Response - {param_name}={value:.2f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'line_following_{param_name}_{value:.2f}.png')
        plt.close()
    
    df = pd.DataFrame(results)
    print(f"\
Results for {param_name}:")
    print(df)
    return df

# Test ranges
kp_values = np.linspace(0.5, 2.0, 4)
ki_values = np.linspace(0.2, 1.0, 4)
kd_values = np.linspace(0.1, 0.5, 4)

print("Testing Kp variations:")
kp_results = analyze_controller('Kp', kp_values)

print("\
Testing Ki variations:")
ki_results = analyze_controller('Ki', ki_values)

print("\
Testing Kd variations:")
kd_results = analyze_controller('Kd', kd_values)

print("done")