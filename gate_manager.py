"""
Gate Manager: Generates gates in specified order with flyability constraints.
"""

import numpy as np
import random

class FPVGate:
    def __init__(self, shape, pos, scale, yaw, color):
        self.shape = shape
        self.pos = np.array(pos)
        self.scale = scale
        self.yaw = yaw
        self.color = color
        self.vertices = self._generate_vertices()

    def _generate_vertices(self):
        s = self.scale
        if self.shape == 'circle':
            phi = np.linspace(0, 2 * np.pi, 30)
            return np.vstack((np.zeros_like(phi), s * np.cos(phi), s * np.sin(phi) + s)).T
        elif self.shape == 'square':
            return np.array([[0,-s,0], [0,s,0], [0,s,2*s], [0,-s,2*s], [0,-s,0]])
        elif self.shape == 'rectangle':
            return np.array([[0,-s*1.5,0], [0,s*1.5,0], [0,s*1.5,2*s], [0,-s*1.5,2*s], [0,-s*1.5,0]])
        elif self.shape == 'triangle':
            return np.array([[0,-s,0], [0,s,0], [0,0,2*s], [0,-s,0]])
        elif self.shape == 'pentagon':
            angles = np.linspace(0, 2*np.pi, 6)
            return np.vstack((np.zeros_like(angles), s * np.cos(angles), s * np.sin(angles) + s)).T
        return np.array([[0,-s,0], [0,s,0], [0,s,2*s], [0,-s,2*s], [0,-s,0]])

    def get_world_vertices(self):
        cos_y, sin_y = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
        return (self.vertices @ R.T) + self.pos

class GateManager:
    """Manages gate generation with specified order and constraints."""
    
    SHAPE_MAP = {
        'c': 'circle',
        's': 'square',
        't': 'triangle',
        'p': 'pentagon',
        'r': 'rectangle'
    }
    
    SHAPE_COLORS = {
        'circle': np.array([0.2, 0.8, 0.2]),      # Green
        'square': np.array([0.2, 0.2, 0.8]),      # Blue
        'triangle': np.array([0.8, 0.2, 0.2]),    # Red
        'pentagon': np.array([0.8, 0.2, 0.8]),    # Magenta
        'rectangle': np.array([0.8, 0.6, 0.2])    # Orange
    }
    
    def __init__(self, gate_order='cstpr', 
                 min_distance=8.0,
                 horizontal_distance=20.0,
                 start_distance=20.0,
                 min_area=9.0,
                 max_angle_change=72.0,
                 max_altitude_change=2.0,
                 circular_layout=True,
                 world_box_size=40.0):
        """
        Args:
            gate_order: String specifying gate order (e.g., 'cstpr')
            min_distance: Minimum 3D distance between gates (meters)
            horizontal_distance: Target horizontal distance between gates (meters)
            start_distance: Distance from start position to first gate (meters)
            min_area: Minimum gate area (m^2)
            max_angle_change: Maximum yaw angle change between consecutive gates (degrees)
            max_altitude_change: Maximum altitude change between consecutive gates (meters)
            circular_layout: If True, arrange gates in a circular pattern
            world_box_size: Size of cubic world bounds (meters)
        """
        self.gate_order = gate_order.lower()
        self.min_distance = min_distance
        self.horizontal_distance = horizontal_distance
        self.start_distance = start_distance
        self.min_area = min_area
        self.max_angle_change = np.radians(max_angle_change)
        self.max_altitude_change = max_altitude_change
        self.circular_layout = circular_layout
        self.world_box_size = world_box_size
        self.world_bounds = world_box_size / 2.0  # ±world_bounds
        
        # Parse gate shapes from order string
        self.gate_shapes = [self.SHAPE_MAP.get(c, 'square') for c in self.gate_order]
        
    def generate_gates_circular(self):
        """Generate gates in a circular/polygonal pattern."""
        n_gates = len(self.gate_shapes)
        gates = []
        
        # Calculate angle between gates for a full loop
        angle_step = 2 * np.pi / n_gates
        
        # Calculate radius to fit in world bounds with some margin
        # Radius from center to gates
        max_radius = (self.world_bounds - 5.0)  # 5m margin
        target_radius = min(self.horizontal_distance * 0.8, max_radius)
        
        # Center altitude
        base_altitude = 6.0
        
        print(f"\nGenerating {n_gates} gates in circular layout:")
        print(f"  Circle radius: {target_radius:.1f}m")
        print(f"  Angle between gates: {np.degrees(angle_step):.1f}°")
        print(f"  World bounds: ±{self.world_bounds:.1f}m")
        print(f"  Base altitude: {base_altitude:.1f}m")
        
        for i, shape in enumerate(self.gate_shapes):
            # Angle for this gate (start at 0, go counterclockwise)
            angle = i * angle_step
            
            # Position on circle
            x = target_radius * np.cos(angle)
            y = target_radius * np.sin(angle)
            
            # Add altitude variation
            alt_offset = np.sin(angle * 2) * self.max_altitude_change * 0.5
            z = base_altitude + alt_offset
            
            pos = [x, y, z]
            
            # Gate yaw: face tangent to circle (perpendicular to radius)
            # For counterclockwise motion, gate should face in direction of travel
            yaw = angle + np.pi / 2
            
            scale = np.sqrt(self.min_area)
            color = self.SHAPE_COLORS.get(shape, np.random.rand(3))
            gate = FPVGate(shape, pos, scale, yaw, color)
            gates.append(gate)
            
            print(f"  Gate {i+1} ({shape}): pos=({x:.1f}, {y:.1f}, {z:.1f}), yaw={np.degrees(yaw):.1f}°")
        
        print(f"\nSuccessfully generated all {n_gates} gates in circular pattern\n")
        return gates
    
    def is_flyable(self, gate_a, gate_b):
        """Check if a drone can transition from gate_a to gate_b."""
        vec_ab = gate_b.pos - gate_a.pos
        dist_3d = np.linalg.norm(vec_ab)
        dist_2d = np.linalg.norm(vec_ab[:2])
        
        # Check minimum 3D distance
        if dist_3d < self.min_distance * 0.5:
            return False
        
        # Check altitude change
        if abs(vec_ab[2]) > self.max_altitude_change * 1.5:
            return False
        
        # Check yaw angle change (more lenient)
        if gate_b.yaw is not None and gate_a.yaw is not None:
            angle_diff = np.abs((gate_b.yaw - gate_a.yaw + np.pi) % (2 * np.pi) - np.pi)
            if angle_diff > self.max_angle_change * 1.2:
                return False
        
        return True
    
    def generate_gates_sequential(self, max_attempts=5000):
        """Generate gates in sequential pattern (original method)."""
        gates = []
        attempts = 0
        
        print(f"\nGenerating {len(self.gate_shapes)} gates sequentially:")
        print(f"  Horizontal distance: {self.horizontal_distance}m")
        print(f"  Max altitude change: {self.max_altitude_change}m")
        print(f"  Max angle change: {np.degrees(self.max_angle_change):.1f}°")
        print(f"  World bounds: ±{self.world_bounds:.1f}m")
        
        while len(gates) < len(self.gate_shapes) and attempts < max_attempts:
            shape = self.gate_shapes[len(gates)]
            scale = np.sqrt(self.min_area)
            
            if len(gates) == 0:
                # First gate: center at origin, facing +x
                pos = [0.0, 0.0, 6.0]
                yaw = 0.0
            else:
                last_gate = gates[-1]
                
                # Generate yaw with limited angle change
                max_yaw_change = self.max_angle_change
                yaw_change = random.uniform(-max_yaw_change * 0.8, max_yaw_change * 0.8)
                yaw = last_gate.yaw + yaw_change
                yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
                
                # Generate position
                forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0])
                distance = random.uniform(self.horizontal_distance * 0.85, 
                                        self.horizontal_distance * 1.15)
                
                alt_change = random.uniform(-self.max_altitude_change * 0.7, 
                                          self.max_altitude_change * 0.7)
                
                pos = [
                    last_gate.pos[0] + distance * forward_dir[0],
                    last_gate.pos[1] + distance * forward_dir[1],
                    np.clip(last_gate.pos[2] + alt_change, 3, 12)
                ]
                
                # Check world bounds
                if (abs(pos[0]) > self.world_bounds or 
                    abs(pos[1]) > self.world_bounds):
                    attempts += 1
                    continue
            
            color = self.SHAPE_COLORS.get(shape, np.random.rand(3))
            new_gate = FPVGate(shape, pos, scale, yaw, color)
            
            if len(gates) == 0:
                gates.append(new_gate)
                print(f"  Gate 1 ({shape}): pos=({new_gate.pos[0]:.1f}, {new_gate.pos[1]:.1f}, {new_gate.pos[2]:.1f}), yaw={np.degrees(new_gate.yaw):.1f}°")
            else:
                if self.is_flyable(gates[-1], new_gate):
                    valid = True
                    for prev_gate in gates[:-1]:
                        dist = np.linalg.norm(new_gate.pos - prev_gate.pos)
                        if dist < self.min_distance * 0.5:
                            valid = False
                            break
                    
                    if valid:
                        gates.append(new_gate)
                        vec = new_gate.pos - gates[-2].pos
                        horiz_dist = np.linalg.norm(vec[:2])
                        alt_change = vec[2]
                        angle_change = np.degrees(new_gate.yaw - gates[-2].yaw)
                        print(f"  Gate {len(gates)} ({shape}): pos=({new_gate.pos[0]:.1f}, {new_gate.pos[1]:.1f}, {new_gate.pos[2]:.1f}), yaw={np.degrees(new_gate.yaw):.1f}°")
                        print(f"    -> H_dist={horiz_dist:.1f}m, alt_Δ={alt_change:.1f}m, yaw_Δ={angle_change:.1f}°")
            
            attempts += 1
        
        if len(gates) < len(self.gate_shapes):
            print(f"\nWarning: Only generated {len(gates)}/{len(self.gate_shapes)} gates after {attempts} attempts")
        else:
            print(f"\nSuccessfully generated all {len(gates)} gates in {attempts} attempts\n")
        
        return gates
    
    def generate_gates(self):
        """Generate gates using circular or sequential layout."""
        if self.circular_layout:
            return self.generate_gates_circular()
        else:
            return self.generate_gates_sequential()
    
    def get_start_position(self, gates):
        """Calculate appropriate start position based on first gate."""
        if len(gates) == 0:
            return np.array([0, 0, 5])
        
        first_gate = gates[0]
        
        if self.circular_layout:
            # For circular layout, start outside the circle approaching first gate
            # Calculate direction from circle center to first gate
            direction_to_gate = first_gate.pos[:2] / np.linalg.norm(first_gate.pos[:2])
            
            # Place start position behind first gate (outside the circle)
            start_pos = np.zeros(3)
            start_pos[:2] = first_gate.pos[:2] - direction_to_gate * self.start_distance
            start_pos[2] = first_gate.pos[2]
        else:
            # For sequential layout, use gate normal direction
            approach_vec = np.array([np.cos(first_gate.yaw), np.sin(first_gate.yaw), 0])
            start_pos = first_gate.pos - approach_vec * self.start_distance
            start_pos[2] = first_gate.pos[2]
        
        return start_pos
