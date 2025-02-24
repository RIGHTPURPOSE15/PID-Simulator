import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.animation as animation
import pygame
import os
from PIL import Image, ImageSequence

# PID Controller class
class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        
    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative
        
        # Save error for next iteration
        self.prev_error = error
        
        return P + I + D

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.3)

# Setup the first subplot for robot animation
ax1.set_xlim(0, 30)
ax1.set_ylim(-5, 5)
ax1.set_title('Robot Line Following')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
target_line = ax1.axhline(y=0, color='r', linestyle='--', label='Target Line')
robot_point, = ax1.plot([], [], 'bo', markersize=15, label='Robot')
ax1.grid(True)
ax1.legend()

# Setup the second subplot for error plotting
ax2.set_xlim(0, 30)
ax2.set_ylim(-5, 5)
ax2.set_title('Error Over Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('Error')
target_line = ax2.axhline(y=0, color='r', linestyle='--', label='Target Line')
error_line, = ax2.plot([], [], 'g-', label='Error')
ax2.grid(True)
ax2.legend()

# Initialize PID controller
pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.1)

# Initialize data lists
t_data = []
error_data = []
robot_x = 0
robot_y = 5  # Start with some offset

def animate(frame):
    global robot_x, robot_y
    
    # Update time
    t = frame * 0.1
    t_data.append(t)
    
    # Compute control signal
    control_signal = pid.compute(0, robot_y, 0.1)
    
    # Update robot position
    robot_x = t
    robot_y += control_signal * 0.1
    
    # Store error
    error = 0 - robot_y
    error_data.append(error)
    
    # Update plots
    robot_point.set_data([robot_x], [robot_y])
    error_line.set_data(t_data, error_data)
    
    return robot_point, error_line

# Create animation
anim = FuncAnimation(fig, animate, frames=300, interval=24, blit=True)

# Save animation as GIF
anim.save('Sims/robot_animation.gif', writer='pillow')

print("Animation saved as robot_animation.gif")

# Initialize Pygame
pygame.init()

# Load the GIF
gif = Image.open('Sims/robot_animation.gif')

# Get the dimensions of the GIF
width, height = gif.size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Robot Line Following Simulation')

# Convert GIF frames to Pygame surfaces
frames = []
for frame in ImageSequence.Iterator(gif):
    frame_surface = pygame.image.fromstring(frame.convert('RGBA').tobytes(), frame.size, 'RGBA')
    frames.append(frame_surface)

# Animation loop
clock = pygame.time.Clock()
running = True
frame_index = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Display current frame
    screen.blit(frames[frame_index], (0, 0))
    pygame.display.flip()
    
    # Move to next frame
    frame_index = (frame_index + 1) % len(frames)
    clock.tick(5)  # Control animation speed

pygame.quit()

print("Pygame window closed")

plt.close()