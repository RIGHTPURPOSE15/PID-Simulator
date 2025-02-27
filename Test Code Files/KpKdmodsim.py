import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def simulate_and_save(Kp, Kd, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    
    # Data setup
    t = np.linspace(0, 10, 200)  # Increased points for better gradient calculation
    y_target = np.sin(t)
    y_robot = np.zeros_like(t)
    prev_error = 0
    
    line_target, = ax1.plot(t, y_target, 'r--', label='Target')
    line_robot, = ax1.plot(t, y_robot, 'b-', label='Robot')
    error_line, = ax2.plot(t, np.zeros_like(t), 'g-', label='Error')
    
    ax1.set_title(f'PD Control (Kp={Kp}, Kd={Kd})')
    ax1.legend()
    ax2.set_title('Error')
    ax2.legend()
    
    def update(frame):
        if frame > 0:
            error = y_target[frame-1] - y_robot[frame-1]
            error_derivative = (error - prev_error) / (t[1] - t[0])
            y_robot[frame] = y_robot[frame-1] + (Kp * error + Kd * error_derivative) * (t[1] - t[0])
        
        line_robot.set_ydata(y_robot)
        error_line.set_ydata(y_target - y_robot)
        return line_robot, error_line
    
    ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
    ani.save(filename, writer='pillow')
    plt.close()

# Run simulations with different parameters
parameters = [(0.5, 0.1), (1.0, 0.2), (2.0, 0.05)]
for Kp, Kd in parameters:
    simulate_and_save(Kp, Kd, f'pd_control_Kp{Kp}_Kd{Kd}.gif')

print("Simulations completed. What parameters would you like to try next?")