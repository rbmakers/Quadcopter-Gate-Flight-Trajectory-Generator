"""
Trajectory Generator: Creates smooth reference trajectories through gates.
PATCHED VERSION - Added approach/exit points and relaxed spline smoothing
"""

import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev

class TrajectoryGenerator:
    """Generates smooth, flyable trajectories through gates."""
    
    def __init__(self, gates, start_position=None, circular_layout=False):
        self.gates = gates
        self.start_position = start_position
        self.circular_layout = circular_layout
        
    def generate_least_flyable_trajectory(self):
        """
        PATCHED: Generate waypoints with approach and exit points for each gate.
        This ensures the quad approaches gates from the correct angle.
        """
        if len(self.gates) == 0:
            return []
        
        waypoints = []
        
        # Start position
        if self.start_position is not None:
            start_pos = self.start_position.copy()
        else:
            first_gate = self.gates[0]
            approach_vec = np.array([np.cos(first_gate.yaw), np.sin(first_gate.yaw), 0])
            start_pos = first_gate.pos - approach_vec * 5.0
            start_pos[2] = first_gate.pos[2]
        
        waypoints.append(start_pos)
        
        # PATCHED: Add approach, gate center, and exit points for each gate
        approach_distance = 3.0  # meters before/after gate
        
        for i, gate in enumerate(self.gates):
            # Gate normal direction (direction quad should fly through)
            normal_vec = np.array([np.cos(gate.yaw), np.sin(gate.yaw), 0])
            
            # Approach point (before gate)
            approach_pt = gate.pos - normal_vec * approach_distance
            waypoints.append(approach_pt)
            
            # Gate center (must pass through here)
            waypoints.append(gate.pos.copy())
            
            # Exit point (after gate) 
            exit_pt = gate.pos + normal_vec * approach_distance
            waypoints.append(exit_pt)
        
        # End position
        if self.circular_layout:
            # For circular layout, smoothly return to start
            last_gate = self.gates[-1]
            first_gate = self.gates[0]
            
            # Vector pointing from last gate toward first gate
            vec_to_first = first_gate.pos - last_gate.pos
            vec_to_first_normalized = vec_to_first / np.linalg.norm(vec_to_first)
            
            # Intermediate point to smooth the turn back
            intermediate = last_gate.pos + vec_to_first_normalized * 5.0
            waypoints.append(intermediate)
            
            # Return close to start for loop closure
            waypoints.append(start_pos.copy())
        else:
            # For sequential layout, continue past last gate
            last_gate = self.gates[-1]
            exit_vec = np.array([np.cos(last_gate.yaw), np.sin(last_gate.yaw), 0])
            end_pos = last_gate.pos + exit_vec * 5.0
            end_pos[2] = last_gate.pos[2]
            waypoints.append(end_pos)
        
        return waypoints

    def generate_dubins_trajectory(self, min_radius=3.0):
        """
        Creates a rounded path by inserting entry/exit points for every corner.
        """
        # Get the standard "sharp" waypoints (Start -> Gate1 -> Gate2 -> ... -> End)
        base_waypoints = self.generate_least_flyable_trajectory()
        if len(base_waypoints) < 3:
            return base_waypoints

        dubins_waypoints = [base_waypoints[0]]  # Start at the first point

        for i in range(1, len(base_waypoints) - 1):
            p_prev = base_waypoints[i-1]
            p_curr = base_waypoints[i]
            p_next = base_waypoints[i+1]

            # 1. Calculate incoming and outgoing vectors
            v_in = p_curr - p_prev
            v_out = p_next - p_curr
            
            d_in = np.linalg.norm(v_in)
            d_out = np.linalg.norm(v_out)
            
            v_in_unit = v_in / d_in
            v_out_unit = v_out / d_out

            # 2. Calculate the angle of the turn
            cos_theta = np.clip(np.dot(-v_in_unit, v_out_unit), -1.0, 1.0)
            angle = np.arccos(cos_theta) 

            # 3. Calculate how far back from the gate the turn must start
            half_angle = (np.pi - angle) / 2.0
            offset_dist = min_radius * np.tan(half_angle)

            # Safety check: ensure offset doesn't exceed segment lengths
            max_offset = min(d_in, d_out) * 0.4 
            actual_offset = min(offset_dist, max_offset)

            # 4. Create the two "Curve" points
            p_entry = p_curr - v_in_unit * actual_offset
            p_exit = p_curr + v_out_unit * actual_offset

            dubins_waypoints.append(p_entry)
            dubins_waypoints.append(p_exit)

        dubins_waypoints.append(base_waypoints[-1])  # End at last point
        return dubins_waypoints    

    def generate_smooth_trajectories(self, num_trajectories=4, dt=0.01, 
                                        total_time=None, smoothness='medium',
                                        path_mode='shortest', min_radius=3.0):
        """
        Generate smooth trajectories through gates with specific path constraints.
        
        path_mode: 
            'shortest' - Direct lines through gate centers with approach/exit points.
            'dubins'   - Rounded corners to respect min_radius curvature.
            'random'   - Shortest path base with random perpendicular offsets.
        """
        # Step 1: Generate the base waypoints based on mode
        if path_mode == 'dubins':
            waypoints = self.generate_dubins_trajectory(min_radius)
        else:
            waypoints = self.generate_least_flyable_trajectory()
        
        if len(waypoints) < 2:
            print("Warning: Not enough waypoints found to generate a trajectory.")
            return []
        
        # Calculate total trajectory distance
        total_dist = sum([np.linalg.norm(waypoints[i+1] - waypoints[i]) 
                         for i in range(len(waypoints)-1)])
        
        # Auto-calculate flight time if not provided
        if total_time is None:
            avg_speed = 4.0 if self.circular_layout else 3.5
            total_time = max(total_dist / avg_speed, 10.0 if self.circular_layout else 8.0)
        
        # Debugging Output
        print(f"\nTrajectory planning Profile:")
        print(f"  Mode: {path_mode.upper()} | Smoothness: {smoothness}")
        print(f"  Distance: {total_dist:.2f}m | Target Time: {total_time:.2f}s")
        print(f"  Avg Speed: {total_dist/total_time:.2f} m/s")
        print(f"  Input Waypoints: {len(waypoints)}")
        
        trajectories = []
        
        for traj_idx in range(num_trajectories):
            # Step 2: Handle waypoint variation
            if path_mode == 'random':
                # Apply random perpendicular offsets to non-terminal waypoints
                varied_waypoints = []
                for i, wp in enumerate(waypoints):
                    if i == 0 or i == len(waypoints) - 1:
                        varied_waypoints.append(wp.copy())
                    else:
                        # Find closest gate to this waypoint
                        min_dist = float('inf')
                        closest_gate = None
                        for gate in self.gates:
                            dist = np.linalg.norm(wp - gate.pos)
                            if dist < min_dist:
                                min_dist = dist
                                closest_gate = gate
                        
                        if closest_gate is not None:
                            # Create a perpendicular offset vector
                            perp = np.array([-np.sin(closest_gate.yaw), np.cos(closest_gate.yaw), 0])
                            offset = perp * np.random.uniform(-0.4, 0.4)
                            varied_waypoints.append(wp + offset)
                        else:
                            varied_waypoints.append(wp.copy())
            else:
                # For 'shortest' and 'dubins', use the waypoints exactly as calculated
                varied_waypoints = [wp.copy() for wp in waypoints]

            # Step 3: Interpolate the waypoints into a continuous trajectory
            trajectory = self._interpolate_trajectory(
                varied_waypoints, total_time, dt, smoothness, path_mode
            )
            
            if trajectory is not None:
                trajectories.append(trajectory)
            else:
                print(f"Warning: Failed to generate trajectory instance {traj_idx+1}")
        
        return trajectories
    
    def _interpolate_trajectory(self, waypoints, total_time, dt, smoothness, path_mode='shortest'):
        """
        PATCHED: Interpolates waypoints into a smooth continuous trajectory.
        For Dubins paths, s=0 ensures curve stays on calculated radius.
        For shortest path, uses small smoothing factor for gentler motion.
        """
        if len(waypoints) < 3:
            print("Error: Not enough waypoints for interpolation.")
            return None
            
        waypoints = np.array(waypoints)
        num_steps = int(total_time / dt)
        
        # PATCHED: Adaptive smoothing based on path mode
        if path_mode == 'dubins':
            # Dubins requires s=0 to maintain exact curvature
            smoothing_factor = 0
        else:
            # Shortest path uses gentle smoothing for less jerky motion
            smoothing_factor = len(waypoints) * 0.05
        
        # Setup Spline
        try:
            tck, u = splprep([waypoints[:,0], waypoints[:,1], waypoints[:,2]], 
                            s=smoothing_factor, k=3)
        except Exception as e:
            print(f"Spline fitting failed: {e}")
            return None
            
        # Evaluate position at high resolution
        u_new = np.linspace(0, 1.0, num_steps)
        out = splev(u_new, tck)
        pos = np.column_stack(out)
        
        # Calculate Velocities (Numerical derivative of position)
        vel = np.zeros_like(pos)
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
        vel[0] = (pos[1] - pos[0]) / dt
        vel[-1] = (pos[-1] - pos[-2]) / dt
        
        # Calculate Accelerations (Numerical derivative of velocity)
        acc = np.zeros_like(pos)
        acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * dt)
        acc[0] = (vel[1] - vel[0]) / dt
        acc[-1] = (vel[-1] - vel[-2]) / dt

        # Calculate Yaw (heading in direction of travel)
        yaw = np.zeros(len(pos))
        for i in range(len(pos)):
            v_xy = vel[i, :2]
            v_norm = np.linalg.norm(v_xy)
            if v_norm > 0.1:
                yaw[i] = np.arctan2(v_xy[1], v_xy[0])
            elif i > 0:
                yaw[i] = yaw[i-1]
        
        return {
            'pos': pos,
            'vel': vel,
            'acc': acc,
            'yaw': yaw,
            'dt': dt,
            'total_time': total_time
        }
