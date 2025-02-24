# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Import additional libraries for animation
from matplotlib.animation import FuncAnimation

# Define the PID Controller class
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
        
        # Return control signal
        return P + I + D

print("PID Controller class defined successfully")

# Define the system model (second-order system)
class SystemModel:
    def __init__(self, mass=1.0, damping=0.5, spring_constant=2.0):
        self.mass = mass
        self.damping = damping
        self.spring_constant = spring_constant
        self.position = 0
        self.velocity = 0
        
    def update(self, force, dt):
        # Calculate acceleration using F = ma
        acceleration = (force - self.damping * self.velocity - 
                       self.spring_constant * self.position) / self.mass
        
        # Update velocity and position using simple Euler integration
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        return self.position

print("System Model class defined successfully")

# Simulate the PID controller with the system model

def simulate_pid(Kp, Ki, Kd, setpoint, simulation_time, dt):
    # Initialize PID controller and system model
    pid = PIDController(Kp, Ki, Kd)
    system = SystemModel()
    
    # Time vector
    time = np.arange(0, simulation_time, dt)
    
    # Initialize arrays to store results
    positions = []
    control_signals = []
    
    # Simulation loop
    for t in time:
        # Compute control signal
        control_signal = pid.compute(setpoint, system.position, dt)
        
        # Update system with control signal
        position = system.update(control_signal, dt)
        
        # Store results
        positions.append(position)
        control_signals.append(control_signal)
    
    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot position
    plt.subplot(2, 1, 1)
    plt.plot(time, positions, label="Position")
    plt.axhline(setpoint, color="r", linestyle="--", label="Setpoint")
    plt.title("System Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()

    # Plot control signal
    plt.subplot(2, 1, 2)
    plt.plot(time, control_signals, label="Control Signal")
    plt.title("Control Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Signal")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the simulation with default parameters
simulate_pid(Kp=2.0, Ki=1.0, Kd=0.5, setpoint=1.0, simulation_time=10.0, dt=0.01)


# Define the Robot class
class TwoWheeledRobot:
    def __init__(self, wheel_base=0.5):
        self.x = 0  # X position
        self.y = 0  # Y position
        self.theta = 0  # Orientation (radians)
        self.wheel_base = wheel_base  # Distance between wheels

    def update(self, v, omega, dt):
        # Update robot's position and orientation using kinematics
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

    def get_position(self):
        return self.x, self.y, self.theta

# Define the line-following simulation

def simulate_line_following(Kp, Ki, Kd, setpoint, simulation_time, dt):
    # Initialize PID controller and robot
    pid = PIDController(Kp, Ki, Kd)
    robot = TwoWheeledRobot()

    # Time vector
    time = np.arange(0, simulation_time, dt)

    # Initialize arrays to store results
    x_positions = []
    y_positions = []

    # Simulation loop
    for t in time:
        # Compute control signal (steering angle)
        error = setpoint - robot.y  # Error is the deviation from the line (y = setpoint)
        omega = pid.compute(0, error, dt)  # Control signal for angular velocity

        # Update robot with constant forward velocity and computed angular velocity
        robot.update(v=1.0, omega=omega, dt=dt)

        # Store results
        x, y, _ = robot.get_position()
        x_positions.append(x)
        y_positions.append(y)

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, max(x_positions) + 1)
    ax.set_ylim(-2, 2)
    ax.axhline(setpoint, color="r", linestyle="--", label="Line")
    robot_dot, = ax.plot([], [], 'bo', label="Robot")

    def update(frame):
        robot_dot.set_data(x_positions[frame], y_positions[frame])
        return robot_dot,

    ani = FuncAnimation(fig, update, frames=len(x_positions), interval=dt*1000, blit=True)
    plt.legend()
    plt.title("Two-Wheeled Robot Line Following")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

# Run the line-following simulation
simulate_line_following(Kp=2.0, Ki=0.5, Kd=0.1, setpoint=0.0, simulation_time=10.0, dt=0.01)