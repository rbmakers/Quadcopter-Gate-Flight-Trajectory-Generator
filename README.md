# Multi-Quadcopter Gate Flight Trajectory Generator

A physics-based simulation system for generating and visualizing smooth reference trajectories for quadcopters flying through FPV-style gates. This system supports multiple gate layouts, path planning strategies, and control algorithms with real-time 3D visualization.


https://github.com/user-attachments/assets/5936938b-f715-467f-b37b-a1117731c563


## Features

### 🎯 Gate Generation
- **Multiple gate shapes**: Circle, Square, Triangle, Pentagon, Rectangle
- **Flexible layouts**: 
  - **Circular**: Gates arranged in a loop pattern (recommended)
  - **Sequential**: Gates placed in a straight line
- **Customizable constraints**: Distance, altitude, and angle limits between gates
- **Automatic flyability checking**: Ensures physically feasible trajectories

### 🛤️ Path Planning
- **Shortest Path**: Direct lines through gate centers with approach/exit points
- **Dubins Path**: Rounded corners respecting minimum turn radius constraints
- **Random Variations**: Generate multiple unique paths through the same gates
- **Spline-based smoothing**: Ensures continuous position, velocity, and acceleration

### 🎮 Control Systems
- **SE3 Controller**: Geometric tracking controller (default, recommended)
  - Position tracking with tunable P/D gains
  - Attitude control using rotation matrices
- **PID Controller**: Classical cascade controller
  - Position → Attitude → Moment control
  - Separate gains for roll/pitch/yaw and altitude

### 📊 Simulation & Visualization
- **Multi-quadcopter support**: Simulate multiple drones simultaneously
- **Real-time 3D visualization**: Interactive matplotlib-based rendering
- **Physics-based dynamics**: PyTorch-accelerated rigid body simulation
- **Batched computation**: Efficient GPU/CPU parallelization
- **Interactive camera controls**: Rotate, zoom, and preset views

## Installation

### Prerequisites
```bash
python >= 3.8
numpy
scipy
matplotlib
torch
torchdiffeq
roma  # Quaternion operations
```

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd quadcopter-gate-flight

# Install dependencies
pip install numpy scipy matplotlib torch torchdiffeq roma
```

## Quick Start

### Basic Usage
```bash
# Default: 5 gates (circle, square, triangle, pentagon, rectangle) in circular layout
python main_gate_flight.py

# Simple 2-gate test
python main_gate_flight.py --gate_order cs

# Custom gate sequence with sequential layout
python main_gate_flight.py --gate_order cstpr --sequential_layout
```

### Path Planning Modes
```bash
# Shortest path (direct lines through gates)
python main_gate_flight.py --optimized_path --path_mode shortest

# Dubins path (rounded corners, smoother turns)
python main_gate_flight.py --path_mode dubins --min_turn_radius 3.0

# Random variations (4 different trajectories)
python main_gate_flight.py --varied_path --num_quads 4
```

### Controller Selection
```bash
# SE3 controller (recommended)
python main_gate_flight.py --controller se3 --kp_pos 15.0 --kd_pos 8.0

# PID controller with custom gains
python main_gate_flight.py --controller pid --kp_att_pid 15.0 --ki_att_pid 0.05
```

## Command-Line Options

### Gate Configuration
| Option | Default | Description |
|--------|---------|-------------|
| `--gate_order` | `cstpr` | Gate sequence (c=circle, s=square, t=triangle, p=pentagon, r=rectangle) |
| `--circular_layout` | `True` | Arrange gates in circular pattern |
| `--world_box_size` | `40.0` | Size of simulation world (meters) |

### Distance & Angle Constraints
| Option | Default | Description |
|--------|---------|-------------|
| `--start_distance` | `20.0` | Distance from start to first gate (m) |
| `--horizontal_distance` | `20.0` | Target spacing between gates (m) |
| `--min_distance` | `8.0` | Minimum 3D distance between any gates (m) |
| `--max_angle_change` | `72.0` | Maximum yaw change between gates (degrees) |
| `--max_altitude_change` | `2.0` | Maximum altitude change between gates (m) |

### Trajectory Options
| Option | Default | Description |
|--------|---------|-------------|
| `--path_mode` | `shortest` | Path strategy: `shortest` or `dubins` |
| `--min_turn_radius` | `3.0` | Minimum turn radius for dubins paths (m) |
| `--optimized_path` | `True` | Use shortest path (all quads follow same trajectory) |
| `--varied_path` | - | Generate varied paths (different trajectories per quad) |

### Controller Gains

**SE3 Controller:**
| Option | Default | Description |
|--------|---------|-------------|
| `--controller` | `se3` | Controller type |
| `--kp_pos` | `15.0` | Position P gain |
| `--kd_pos` | `8.0` | Position D gain |
| `--kp_att_se3` | `4000.0` | Attitude P gain |
| `--kd_att_se3` | `400.0` | Attitude D gain |

**PID Controller:**
| Option | Default | Description |
|--------|---------|-------------|
| `--controller` | `pid` | Controller type |
| `--kp_att_pid` | `15.0` | Attitude P gain (range: 1-20) |
| `--ki_att_pid` | `0.05` | Attitude I gain (range: 0-1) |
| `--kd_att_pid` | `0.5` | Attitude D gain (range: 0.1-2) |
| `--kp_thrust` | `15.0` | Altitude P gain |
| `--ki_thrust` | `0.05` | Altitude I gain |
| `--kd_thrust` | `5.0` | Altitude D gain |

### Simulation Parameters
| Option | Default | Description |
|--------|---------|-------------|
| `--num_quads` | `1` (optimized) / `4` (varied) | Number of quadcopters |
| `--quad_size` | `0.5` | Visual size of quadcopters (m) |
| `--dt` | `0.01` | Simulation timestep (seconds) |
| `--device` | `cuda` (if available) | Compute device: `cuda` or `cpu` |
| `--show_trajectory` | - | Show trajectory plot before simulation |

## Usage Examples

### Example 1: Simple Circular Course
```bash
python main_gate_flight.py \
    --gate_order cs \
    --circular_layout \
    --path_mode shortest \
    --show_trajectory
```

### Example 2: Challenging Dubins Path
```bash
python main_gate_flight.py \
    --gate_order cstpr \
    --path_mode dubins \
    --min_turn_radius 2.5 \
    --max_angle_change 90 \
    --controller se3 \
    --kp_pos 18.0
```

### Example 3: Multiple Quadcopters with Variations
```bash
python main_gate_flight.py \
    --gate_order cstpr \
    --varied_path \
    --num_quads 4 \
    --quad_size 0.8 \
    --world_box_size 50
```

### Example 4: PID Controller Tuning
```bash
python main_gate_flight.py \
    --controller pid \
    --kp_att_pid 12.0 \
    --kd_att_pid 0.8 \
    --kp_thrust 18.0 \
    --show_trajectory
```

### Example 5: Large-Scale Course
```bash
python main_gate_flight.py \
    --gate_order cstprcstpr \
    --world_box_size 60 \
    --horizontal_distance 25 \
    --max_angle_change 60 \
    --path_mode dubins \
    --min_turn_radius 4.0
```

## Interactive Controls

During simulation, use these keyboard controls:

| Key | Action |
|-----|--------|
| `Arrow Keys` | Rotate camera view |
| `R` | Toggle auto-rotation |
| `1` | Preset view: Isometric (elev=25°, azim=45°) |
| `2` | Preset view: Top-down (elev=90°) |
| `3` | Preset view: Side (elev=0°, azim=0°) |
| `4` | Preset view: Front (elev=0°, azim=90°) |
| `5` | Preset view: Perspective (elev=35°, azim=45°) |
| `Ctrl+C` | Stop simulation |

## Project Structure

```
├── main_gate_flight.py      # Main entry point and argument parsing
├── gate_manager.py           # Gate generation and placement logic
├── trajectory_gen.py         # Path planning and trajectory smoothing
├── multi_quad_viz.py         # Visualization and controller implementation
├── multirotor.py             # Physics dynamics (batched simulation)
├── crazyflie_params.py       # Quadcopter physical parameters
└── README.md                 # This file
```

## Algorithm Details

### Gate Placement (Circular Layout)
1. Calculate angular spacing: `2π / num_gates`
2. Position gates on circle with configurable radius
3. Orient gates tangent to circle (perpendicular to radius)
4. Add sinusoidal altitude variation for 3D complexity

### Trajectory Generation
1. **Waypoint Selection**: 
   - Start position → Approach points → Gate centers → Exit points → End position
   - Approach/exit points placed 3m before/after each gate along normal direction
   
2. **Path Smoothing**:
   - **Shortest**: Direct spline through waypoints with minimal smoothing
   - **Dubins**: Insert rounded corners at each turn respecting `min_turn_radius`
   
3. **Spline Interpolation**: 
   - 3D cubic B-spline ensures C² continuity
   - Numerical differentiation for velocity and acceleration
   - Yaw computed from velocity direction

### Control Pipeline
```
Reference Trajectory → Controller → Thrust + Moment → Motor Allocation → Dynamics → State Update
```

**SE3**: Position error → Desired acceleration → Desired rotation → Attitude error → Moment

**PID**: Position error → Desired acceleration → Desired angles → Angle error → Moment (with integral anti-windup)

## Performance Notes

- **GPU Acceleration**: Simulation runs significantly faster on CUDA-enabled GPUs
- **Batch Processing**: Multiple quadcopters simulated in parallel with minimal overhead
- **Timestep Selection**: `dt=0.01s` provides good accuracy/speed tradeoff
- **Visualization**: Real-time rendering may slow down with >10 quadcopters

## Troubleshooting

**Issue**: "Failed to generate gates"
- Solution: Increase `--world_box_size` or reduce `--horizontal_distance`

**Issue**: Quadcopter diverges or crashes
- Solution: Reduce gains (`--kp_pos`, `--kp_att`) or increase `--dt`

**Issue**: Path too jerky
- Solution: Use `--path_mode dubins` with higher `--min_turn_radius`

**Issue**: GPU out of memory
- Solution: Use `--device cpu` or reduce `--num_quads`

## References

- **Quadcopter Dynamics Backend**: This project uses [rotorpy](https://github.com/spencerfolk/rotorpy) for the multirotor dynamics simulation (`multirotor.py`)
- **Quadcopter Parameters**: Based on Crazyflie 2.0 nano quadcopter physical parameters
- **SE3 Control**: Lee et al., "Geometric Tracking Control of a Quadrotor UAV on SE(3)"
- **Dubins Paths**: Dubins, L.E., "On Curves of Minimal Length with a Constraint on Average Curvature"

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this code in your research, please cite:
```
@misc{quadcopter_gate_flight,
  author = {rbmakers},
  title = {Multi-Quadcopter Gate Flight Trajectory Generator},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rbmakers}
}
```

## Acknowledgments

Special thanks to the [rotorpy](https://github.com/spencerfolk/rotorpy) project for providing the robust multirotor dynamics simulation framework that powers this system.
