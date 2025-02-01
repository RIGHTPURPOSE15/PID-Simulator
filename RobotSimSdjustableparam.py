import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import pandas as pd

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.reset()
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0
        self.last_time = None
    
    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        
        if self.output_limits is not None:
            output = np.clip(output, *self.output_limits)
            if output == self.output_limits[0] or output == self.output_limits[1]:
                self.integral -= error * dt
        
        self.previous_error = error
        return output, error

def calculate_settling_time(time, signal, setpoint, tolerance=0.02):
    error_band = tolerance * abs(setpoint) if setpoint != 0 else tolerance
    within_tolerance = np.abs(signal - setpoint) <= error_band
    indices = np.where(within_tolerance)[0]
    
    if len(indices) == 0:
        return time[-1]
    
    sequences = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    
    for seq in reversed(sequences):
        if seq[-1] == len(signal) - 1:
            return time[seq[0]]
    
    return time[-1]

def create_simulation(kp, ki=0.1, kd=0.1, dt=0.05, t_max=30.0):
    t = np.arange(0, t_max, dt)
    pid = PIDController(kp=kp, ki=ki, kd=kd, output_limits=(-0.5, 0.5))
    
    position = np.zeros_like(t)
    velocity = np.zeros_like(t)
    errors = np.zeros_like(t)
    setpoint = 0.0
    initial_position = 1.0
    position[0] = initial_position
    
    for i in range(1, len(t)):
        control_signal, error = pid.compute(setpoint, position[i-1], dt)
        velocity[i] = velocity[i-1] + control_signal * dt
        position[i] = position[i-1] + velocity[i] * dt
        errors[i] = error
    
    speed = np.abs(velocity)  # Calculate speed as absolute value of velocity
    
    return t, position, speed, errors

def create_animation(kp, save_path_gif, save_path_png):
    t, position, speed, errors = create_simulation(kp)
    
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    plt.tight_layout(pad=3.0)
    
    # Calculate performance metrics
    overshoot = np.max(np.abs(position))
    steady_state_error = np.abs(position[-1])
    settling_time = calculate_settling_time(t, position, 0)
    
    def animate(frame):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Position plot
        ax1.plot(t[:frame], position[:frame], 'b-', label='Position')
        ax1.plot(t, [0] * len(t), 'r--', label='Setpoint')
        ax1.set_xlim(0, 30)
        ax1.set_ylim(-2, 2)
        ax1.set_title(f'Position Response (Kp={kp})')
        ax1.grid(True)
        ax1.legend()
        
        # Speed plot
        ax2.plot(t[:frame], speed[:frame], 'm-', label='Speed')
        ax2.set_xlim(0, 30)
        ax2.set_ylim(0, np.max(speed) * 1.1)
        ax2.set_title('Speed')
        ax2.grid(True)
        ax2.legend()
        
        # Error plot
        ax3.plot(t[:frame], errors[:frame], 'g-', label='Error')
        ax3.set_xlim(0, 30)
        ax3.set_ylim(-2, 2)
        ax3.set_title('Error')
        ax3.grid(True)
        ax3.legend()
        
        # Add performance metrics text
        plt.figtext(0.02, 0.02, f'Overshoot: {overshoot:.3f}\
Steady-state Error: {steady_state_error:.3f}\
Settling Time: {settling_time:.3f}s',
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(t), interval=50)
    
    # Save as GIF
    if save_path_gif:
        os.makedirs(os.path.dirname(save_path_gif), exist_ok=True)
        anim.save(save_path_gif, writer=PillowWriter(fps=20))
    
    # Save final frame as PNG
    if save_path_png:
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        animate(len(t)-1)  # Draw final frame
        plt.savefig(save_path_png)
    
    plt.close()

# Create animations for different Kp values
kp_values = [0.5, 1.0, 2.0]
results = []

for kp in kp_values:
    gif_path = f'Sims/Kp/simulation_Kp_{kp}.gif'
    png_path = f'Sims/Kp/simulation_Kp_{kp}_final.png'
    create_animation(kp, gif_path, png_path)
    
    # Calculate metrics for results table
    t, position, speed, errors = create_simulation(kp)
    results.append({
        'Kp': kp,
        'Overshoot': np.max(np.abs(position)),
        'Steady-State Error': np.abs(position[-1]),
        'Settling Time': calculate_settling_time(t, position, 0)
    })

# Display results
results_df = pd.DataFrame(results)
print("\
Performance Metrics:")
print(results_df)