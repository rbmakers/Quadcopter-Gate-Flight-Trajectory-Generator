"""
Multi-Quadcopter Visualizer: Simulates and visualizes multiple quadcopters.
PATCHED VERSION - Fixed PID controller yaw tracking and position gains
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import sys
import os

# Add rotorpy path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multirotor import BatchedMultirotor, BatchedMultirotorParams
from crazyflie_params import quad_params

class ViewControls:
    """Handles interactive view controls."""
    def __init__(self, ax):
        self.ax = ax
        self.elev = 25
        self.azim = 45
        self.auto_rotate = False
        self.rotation_speed = 0.5
        
    def on_key(self, event):
        if event.key == 'up':
            self.elev = min(self.elev + 5, 90)
        elif event.key == 'down':
            self.elev = max(self.elev - 5, -90)
        elif event.key == 'left':
            self.azim = (self.azim - 5) % 360
        elif event.key == 'right':
            self.azim = (self.azim + 5) % 360
        elif event.key == 'r':
            self.auto_rotate = not self.auto_rotate
        elif event.key == '1':
            self.elev, self.azim = 25, 45
        elif event.key == '2':
            self.elev, self.azim = 90, 45
        elif event.key == '3':
            self.elev, self.azim = 0, 0
        elif event.key == '4':
            self.elev, self.azim = 0, 90
        elif event.key == '5':
            self.elev, self.azim = 35.264, 45
    
    def update(self):
        if self.auto_rotate:
            self.azim = (self.azim + self.rotation_speed) % 360
        self.ax.view_init(elev=self.elev, azim=self.azim)

class PIDController:
    """Classical PID controller for attitude and thrust.
    PATCHED: Fixed yaw tracking and position gain initialization."""
    
    def __init__(self, mass, 
                 kp_att=np.array([10.0, 10.0, 10.0]), 
                 ki_att=np.array([0.1, 0.1, 0.1]), 
                 kd_att=np.array([0.5, 0.5, 0.5]),
                 kp_thrust=15.0, ki_thrust=0.1, kd_thrust=5.0,
                 kp_pos=10.0, kd_pos=5.0):  # ADDED position gains
        self.mass = mass
        self.g = 9.81
        
        # Gains
        self.kp_att, self.ki_att, self.kd_att = kp_att, ki_att, kd_att
        self.kp_thrust, self.ki_thrust, self.kd_thrust = kp_thrust, ki_thrust, kd_thrust
        self.kp_pos, self.kd_pos = kp_pos, kd_pos  # ADDED: Store position gains
        
        # Error integration
        self.int_att_err = np.zeros(3)
        self.int_thrust_err = 0.0
        self.prev_att_err = np.zeros(3)
        self.prev_thrust_err = 0.0
        
        # Timestep (will be set in compute_control)
        self.dt = 0.01

    def compute_control(self, state, des_pos, des_vel, des_acc, des_yaw):
        """
        PATCHED PID Control: Fixed yaw tracking (was frozen at 0.0)
        Position -> Attitude -> Moment with proper yaw following
        """
        dt = self.dt if hasattr(self, 'dt') else 0.01
        
        # Ensure inputs are numpy arrays (fixes Torch compatibility)
        curr_x = state['x'].detach().cpu().numpy() if isinstance(state['x'], torch.Tensor) else state['x']
        curr_v = state['v'].detach().cpu().numpy() if isinstance(state['v'], torch.Tensor) else state['v']
        curr_q = state['q'].detach().cpu().numpy() if isinstance(state['q'], torch.Tensor) else state['q']
    
        # 1. Position Control -> acc_cmd
        pos_err = des_pos - curr_x
        vel_err = des_vel - curr_v
        
        # Use stored position gains
        acc_cmd = des_acc + self.kp_pos * pos_err + self.kd_pos * vel_err
    
        # 2. PATCHED: Use actual desired yaw instead of fixed 0.0
        # Small angle approximation for desired Roll and Pitch
        des_roll = (acc_cmd[0] * np.sin(des_yaw) - acc_cmd[1] * np.cos(des_yaw)) / self.g
        des_pitch = (acc_cmd[0] * np.cos(des_yaw) + acc_cmd[1] * np.sin(des_yaw)) / self.g
        
        # Limit tilts to 30 degrees to prevent flip-over crashes
        des_roll = np.clip(des_roll, -0.5, 0.5)
        des_pitch = np.clip(des_pitch, -0.5, 0.5)
        
        des_euler = np.array([des_roll, des_pitch, des_yaw])  # PATCHED: Use des_yaw
    
        # 3. Thrust Calculation with Tilt Compensation
        mass = 0.03
        thrust_vertical = mass * (acc_cmd[2] + self.g)
        thrust = thrust_vertical / (np.cos(des_roll) * np.cos(des_pitch))
    
        # 4. Attitude Control -> Moment
        try:
            curr_euler = R.from_quat(curr_q).as_euler('xyz')
        except ValueError: 
            curr_euler = np.zeros(3)
    
        att_err = des_euler - curr_euler
        att_err[2] = (att_err[2] + np.pi) % (2 * np.pi) - np.pi  # Yaw Wrap
    
        # PID Logic
        self.int_att_err += att_err * dt
        self.int_att_err = np.clip(self.int_att_err, -0.5, 0.5)  # Anti-windup
        
        d_att_err = (att_err - self.prev_att_err) / dt
        
        # Apply gains
        moment = self.kp_att * att_err + self.ki_att * self.int_att_err + self.kd_att * d_att_err
        
        self.prev_att_err = att_err
    
        return {'cmd_thrust': thrust, 'cmd_moment': moment}

class SE3Controller:
    """Simple SE3 controller for trajectory tracking."""
    
    def __init__(self, mass, kp_pos=10.0, kd_pos=5.0, kp_att=3000.0, kd_att=360.0):
        self.mass = mass
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.g = 9.81
        
    def compute_control(self, state, des_pos, des_vel, des_acc, des_yaw):
        """Compute control commands."""
        # Position error
        pos_err = des_pos - state['x']
        vel_err = des_vel - state['v']
        
        # Desired acceleration
        acc_cmd = des_acc + self.kp_pos * pos_err + self.kd_pos * vel_err
        
        # Desired force
        F_des = self.mass * (acc_cmd + np.array([0, 0, self.g]))
        
        # Current rotation
        R_curr = R.from_quat(state['q']).as_matrix()
        b3 = R_curr @ np.array([0, 0, 1])
        
        # Thrust
        thrust = np.dot(F_des, b3)
        
        # Desired orientation
        b3_des = F_des / np.linalg.norm(F_des)
        c1_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0])
        b2_des = np.cross(b3_des, c1_des)
        b2_des = b2_des / np.linalg.norm(b2_des)
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack([b1_des, b2_des, b3_des])
        
        # Orientation error
        S_err = 0.5 * (R_des.T @ R_curr - R_curr.T @ R_des)
        att_err = np.array([-S_err[1, 2], S_err[0, 2], -S_err[0, 1]])
        
        # Moment
        I = np.array([[quad_params['Ixx'], 0, 0],
                      [0, quad_params['Iyy'], 0],
                      [0, 0, quad_params['Izz']]])
        moment = -self.kp_att * att_err - self.kd_att * state['w']
        
        return {'cmd_thrust': thrust, 'cmd_moment': moment}

class MultiQuadVisualizer:
    """Visualizes multiple quadcopters flying through gates."""
    
    def __init__(self, gates, trajectories, dt=0.01, device='cuda', trajectory_points=None, quad_size=0.15, 
             ctrl_type='se3', ctrl_args={}):
        
        self.gates = gates
        self.trajectories = trajectories
        self.dt = dt
        self.quad_size = quad_size
        # Ensure device is a torch.device object
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.trajectory_points = trajectory_points
        self.num_quads = len(trajectories)
        
        print(f"Initializing visualizer with device: {self.device}")
        
        # Setup simulation
        self.ctrl_type = ctrl_type
        self.ctrl_args = ctrl_args
        self._setup_simulation()
        
        # Setup visualization
        self._setup_visualization()
        
        # Tracking
        self.current_step = 0
        self.trajectories_completed = 0
        self.quad_trails = [[] for _ in range(self.num_quads)]
        
    def _setup_simulation(self):
        """Initialize quadcopter dynamics."""
        print(f"Setting up simulation on device: {self.device}")
        
        # Create a test tensor to get the actual device format
        test_tensor = torch.zeros(1, device=self.device)
        actual_device = test_tensor.device
        print(f"Actual device from tensor: {actual_device}")
        
        # Use the actual device consistently
        self.device = actual_device
        
        # Create batched parameters with the actual device
        params_list = [quad_params for _ in range(self.num_quads)]
        self.batched_params = BatchedMultirotorParams(
            params_list, self.num_quads, self.device
        )
        
        # Initial states - ensure all tensors are on correct device
        initial_states = {
            'x': torch.zeros(self.num_quads, 3, dtype=torch.float64, device=self.device),
            'v': torch.zeros(self.num_quads, 3, dtype=torch.float64, device=self.device),
            'q': torch.zeros(self.num_quads, 4, dtype=torch.float64, device=self.device),
            'w': torch.zeros(self.num_quads, 3, dtype=torch.float64, device=self.device),
            'wind': torch.zeros(self.num_quads, 3, dtype=torch.float64, device=self.device),
            'rotor_speeds': torch.ones(self.num_quads, 4, dtype=torch.float64, device=self.device) * 1788.53
        }
        
        # Set initial positions from trajectories
        for i in range(self.num_quads):
            pos_np = self.trajectories[i]['pos'][0]
            initial_states['x'][i] = torch.tensor(pos_np, dtype=torch.float64, device=self.device)
            initial_states['q'][i, 3] = 1.0  # Set w component of quaternion
        
        print(f"All states on device: {initial_states['x'].device}")
        print(f"Batched params device: {self.batched_params.device}")
        
        self.batched_quad = BatchedMultirotor(
            self.batched_params, self.num_quads, initial_states,
            self.device, control_abstraction='cmd_ctbm', aero=True, integrator='rk4'
        )
        
        self.state = initial_states
        
        # Setup controllers with PATCHED PID
        if self.ctrl_type == 'pid':
            self.controllers = [PIDController(quad_params['mass'], **self.ctrl_args) for _ in range(self.num_quads)]
        else:
            self.controllers = [SE3Controller(quad_params['mass'], **self.ctrl_args) for _ in range(self.num_quads)]
        
        print(f"✓ Simulation setup complete")
        
    def _setup_visualization(self):
        """Setup matplotlib visualization."""
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        
        self.view_ctrl = ViewControls(self.ax)
        self.fig.canvas.mpl_connect('key_press_event', self.view_ctrl.on_key)
        
        # Quad colors
        self.quad_colors = plt.cm.rainbow(np.linspace(0, 1, self.num_quads))
        
    def draw_gates(self):
        """Draw all gates."""
        for i, gate in enumerate(self.gates):
            wv = gate.get_world_vertices()
            self.ax.plot(wv[:, 0], wv[:, 1], wv[:, 2], 
                        color=gate.color, linewidth=3, alpha=0.7)
            
            # Gate label
            self.ax.text(gate.pos[0], gate.pos[1], gate.pos[2] + 2,
                        f"G{i+1}", fontsize=10, weight='bold')
    
    def draw_quadcopter(self, pos, quat, color, size=0.15):
        # Check if quaternion is valid (norm should be 1.0)
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6 or np.isnan(quat_norm):
            return       
         
        """Draw a single quadcopter."""
        rot_matrix = R.from_quat(quat).as_matrix()
        arm_length = size
        
        # Arms
        arms = [
            np.array([[0, 0, 0], [arm_length, arm_length, 0]]),
            np.array([[0, 0, 0], [-arm_length, arm_length, 0]]),
            np.array([[0, 0, 0], [-arm_length, -arm_length, 0]]),
            np.array([[0, 0, 0], [arm_length, -arm_length, 0]]),
        ]
        
        for arm in arms:
            rotated_arm = (rot_matrix @ arm.T).T + pos
            self.ax.plot(rotated_arm[:, 0], rotated_arm[:, 1], rotated_arm[:, 2],
                        color=color, linewidth=2, alpha=0.9)
        
        # Body
        body_size = size * 0.1
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * body_size
        rotated_vertices = (rot_matrix @ vertices.T).T + pos
        
        faces = [
            [rotated_vertices[j] for j in [0, 1, 2, 3]],
            [rotated_vertices[j] for j in [4, 5, 6, 7]],
            [rotated_vertices[j] for j in [0, 1, 5, 4]],
            [rotated_vertices[j] for j in [2, 3, 7, 6]],
            [rotated_vertices[j] for j in [0, 3, 7, 4]],
            [rotated_vertices[j] for j in [1, 2, 6, 5]],
        ]
        
        poly3d = Poly3DCollection(faces, alpha=0.8, facecolor=color,
                                 edgecolor='black', linewidth=0.5)
        self.ax.add_collection3d(poly3d)
    
    def simulate_step(self):
        """Simulate one timestep."""
        # Get desired states for each quad
        controls = {}
        controls['cmd_thrust'] = torch.zeros(self.num_quads, 1, dtype=torch.float64, device=self.device)
        controls['cmd_moment'] = torch.zeros(self.num_quads, 3, 1, dtype=torch.float64, device=self.device)
        
        active_quads = []
        
        for i in range(self.num_quads):
            traj = self.trajectories[i]
            
            if self.current_step < len(traj['pos']):
                active_quads.append(i)
                
                # Desired state
                des_pos = traj['pos'][self.current_step]
                des_vel = traj['vel'][self.current_step]
                des_acc = traj['acc'][self.current_step]
                des_yaw = traj['yaw'][self.current_step]
                
                # Get current state
                state_i = {
                    'x': self.state['x'][i].cpu().numpy(),
                    'v': self.state['v'][i].cpu().numpy(),
                    'q': self.state['q'][i].cpu().numpy(),
                    'w': self.state['w'][i].cpu().numpy()
                }
                
                # Compute control
                try:
                    ctrl = self.controllers[i].compute_control(
                        state_i, des_pos, des_vel, des_acc, des_yaw
                    )
                    
                    controls['cmd_thrust'][i, 0] = torch.tensor(ctrl['cmd_thrust'], dtype=torch.float64, device=self.device)
                    moment_tensor = torch.tensor(ctrl['cmd_moment'], dtype=torch.float64, device=self.device)
                    controls['cmd_moment'][i, :, 0] = moment_tensor
                        
                except Exception as e:
                    print(f"Warning: Control computation failed for quad {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    controls['cmd_thrust'][i, 0] = self.batched_params.mass[i].squeeze() * 9.81
                    controls['cmd_moment'][i, :, 0] = 0.0
            else:
                # Inactive quad - set to hover
                controls['cmd_thrust'][i, 0] = self.batched_params.mass[i].squeeze() * 9.81
                controls['cmd_moment'][i, :, 0] = 0.0
        
        # Step dynamics for active quads
        if len(active_quads) > 0:
            try:
                if self.current_step == 0:
                    print(f"Control shapes:")
                    print(f"  cmd_thrust: {controls['cmd_thrust'].shape}")
                    print(f"  cmd_moment: {controls['cmd_moment'].shape}")
                    print(f"  Active quads: {active_quads}")
                
                self.state = self.batched_quad.step(self.state, controls, self.dt, active_quads)
            except Exception as e:
                print(f"Warning: Dynamics step failed: {e}")
                print(f"Control shapes: thrust={controls['cmd_thrust'].shape}, moment={controls['cmd_moment'].shape}")
                print(f"Active quads: {active_quads}")
                import traceback
                traceback.print_exc()
        
        self.current_step += 1
        
        # Reset if all quads finished
        if self.current_step >= max([len(traj['pos']) for traj in self.trajectories]):
            self.current_step = 0
            self.trajectories_completed += 1
            self.quad_trails = [[] for _ in range(self.num_quads)]
            
            # Reset state
            for i in range(self.num_quads):
                pos_np = self.trajectories[i]['pos'][0]
                self.state['x'][i] = torch.tensor(pos_np, dtype=torch.float64, device=self.device)
                self.state['v'][i] = torch.zeros(3, dtype=torch.float64, device=self.device)
                self.state['q'][i] = torch.tensor([0, 0, 0, 1], dtype=torch.float64, device=self.device)
                self.state['w'][i] = torch.zeros(3, dtype=torch.float64, device=self.device)
            
            print(f"Trajectory cycle {self.trajectories_completed} complete")
    
    def animate(self, frame):
        """Animation function."""
        self.ax.clear()
        
        # Simulate
        self.simulate_step()
        
        # Draw gates
        self.draw_gates()
        
        # Draw trajectory points if available
        if self.trajectory_points:
            pts = np.array(self.trajectory_points)
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        'k--', linewidth=1, alpha=0.3, label='Reference Path')
        
        # Draw quadcopters and trails
        for i in range(self.num_quads):
            pos = self.state['x'][i].cpu().numpy()
            quat = self.state['q'][i].cpu().numpy()
            
            # Update trail
            if self.current_step < len(self.trajectories[i]['pos']):
                self.quad_trails[i].append(pos.copy())
                if len(self.quad_trails[i]) > 200:
                    self.quad_trails[i].pop(0)
            
            # Draw trail
            if len(self.quad_trails[i]) > 1:
                trail = np.array(self.quad_trails[i])
                self.ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                            color=self.quad_colors[i], linewidth=1.5, alpha=0.6)
            
            # Draw quad
            self.draw_quadcopter(pos, quat, self.quad_colors[i], size=self.quad_size)
        
        # Ground plane
        grid_range = 50
        x_grid = np.linspace(-grid_range, grid_range, 10)
        y_grid = np.linspace(-grid_range, grid_range, 10)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.2)
        
        # Set limits
        self.ax.set_xlim([-grid_range, grid_range])
        self.ax.set_ylim([-grid_range, grid_range])
        self.ax.set_zlim([0, 20])
        
        self.ax.set_xlabel('X (m)', fontsize=10, weight='bold')
        self.ax.set_ylabel('Y (m)', fontsize=10, weight='bold')
        self.ax.set_zlabel('Z (m)', fontsize=10, weight='bold')
        
        title = f"Multi-Quadcopter Gate Flight\n"
        title += f"Cycles: {self.trajectories_completed} | "
        title += f"Time: {self.current_step * self.dt:.1f}s | "
        title += f"Quads: {self.num_quads}"
        self.ax.set_title(title, fontsize=12, weight='bold')
        
        self.view_ctrl.update()
        self.ax.set_box_aspect([1, 1, 0.5])
    
    def run(self):
        """Run the visualization."""
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, interval=int(self.dt * 1000),
            cache_frame_data=False, blit=False
        )
        plt.tight_layout()
        plt.show()
