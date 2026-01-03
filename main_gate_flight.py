"""
Main script for generating and visualizing quadcopter reference flights through gates.
PATCHED VERSION - Improved default parameters for better gate tracking

Usage:
    python main_gate_flight.py --gate_order cstpr --num_quads 1
    python main_gate_flight.py --circular_layout --optimized_path
    python main_gate_flight.py --gate_order cs --world_box_size 40 --max_angle_change 72
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from gate_manager import GateManager
from trajectory_gen import TrajectoryGenerator
from multi_quad_viz import MultiQuadVisualizer
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Quadcopter Gate Flight System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Gate configuration
    parser.add_argument('--gate_order', type=str, default='cstpr',
                      help='Gate order: c=circle, s=square, t=triangle, p=pentagon, r=rectangle')
    
    # Layout options
    parser.add_argument('--circular_layout', action='store_true', default=True,
                      help='Arrange gates in circular pattern (recommended)')
    parser.add_argument('--sequential_layout', action='store_false', dest='circular_layout',
                      help='Arrange gates sequentially instead of circular')
    
    # World constraints
    parser.add_argument('--world_box_size', type=float, default=40.0,
                      help='Size of cubic world bounds (meters) - gates confined within ±size/2')
    
    # Distance constraints
    parser.add_argument('--start_distance', type=float, default=20.0,
                      help='Distance from start position to first gate (meters)')
    parser.add_argument('--horizontal_distance', type=float, default=20.0,
                      help='Target horizontal distance between consecutive gates (meters)')
    parser.add_argument('--min_distance', type=float, default=8.0,
                      help='Minimum 3D distance between any two gates (meters)')
    
    # Angular and altitude constraints
    parser.add_argument('--max_angle_change', type=float, default=72.0,
                      help='Maximum yaw angle change between consecutive gates (degrees)')
    parser.add_argument('--max_altitude_change', type=float, default=2.0,
                      help='Maximum altitude change between consecutive gates (meters)')
    
    # Trajectory options
    parser.add_argument('--optimized_path', action='store_true', default=True,
                      help='Use optimized (shortest) path through gates')
    parser.add_argument('--varied_path', action='store_false', dest='optimized_path',
                      help='Add variations to trajectories (not shortest path)')
    
    # Simulation parameters
    parser.add_argument('--num_quads', type=int, default=None,
                      help='Number of quadcopters flying simultaneously (default: 1 for optimized_path, 4 for varied_path)')
    parser.add_argument('--quad_size', type=float, default=0.5,
                      help='Visual size of quadcopters (default: 0.5m, larger=easier to see attitude)')
    parser.add_argument('--dt', type=float, default=0.01,
                      help='Simulation timestep (seconds)')
    parser.add_argument('--show_trajectory', action='store_true',
                      help='Show trajectory visualization before simulation')
    parser.add_argument('--device', type=str, 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run simulation (cuda/cpu)')

    # Path Mode Args
    parser.add_argument('--path_mode', type=str, default='shortest', choices=['shortest', 'dubins'],
                      help='Path strategy: shortest (direct) or dubins (rounded corners)')
    parser.add_argument('--min_turn_radius', type=float, default=3.0,
                      help='Minimum turn radius for dubins path (meters)')

    parser.add_argument('--controller', type=str, default='se3', choices=['se3', 'pid'],
                        help='Type of controller to use')
    
    # PATCHED: Improved default SE3 gains for better tracking
    parser.add_argument('--kp_pos', type=float, default=15.0,
                       help='Position P gain (was 10.0)')
    parser.add_argument('--kd_pos', type=float, default=8.0,
                       help='Position D gain (was 5.0)')
    parser.add_argument('--kp_att_se3', type=float, default=4000.0,
                       help='Attitude P gain for SE3 (was 3000.0)')
    parser.add_argument('--kd_att_se3', type=float, default=400.0,
                       help='Attitude D gain for SE3 (was 360.0)')
    
    # PATCHED: Improved default PID gains
    parser.add_argument('--kp_att_pid', type=float, default=15.0, 
                       help='PID attitude P gain (was 1.0), Range: 1-20')
    parser.add_argument('--ki_att_pid', type=float, default=0.05, 
                       help='PID attitude I gain (was 0.04), Range: 0-1')
    parser.add_argument('--kd_att_pid', type=float, default=0.5, 
                       help='PID attitude D gain (was 0.1), Range: 0.1-2')
    
    parser.add_argument('--kp_thrust', type=float, default=15.0, 
                       help='Proportional gain for altitude (was 10.0)')
    parser.add_argument('--ki_thrust', type=float, default=0.05,
                       help='Integral gain for altitude (was 0.04)')
    parser.add_argument('--kd_thrust', type=float, default=5.0, 
                       help='Derivative gain for altitude (was 2.0)')
    
    return parser.parse_args()

def visualize_gates_and_trajectory(gates, trajectory_points, start_pos, circular_layout):
    """Show the generated gates and trajectory."""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot start position
    ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]], 
              c='green', marker='*', s=400, alpha=0.9, 
              label='Start Position', edgecolors='black', linewidth=2)
    
    # Plot gates
    for i, gate in enumerate(gates):
        wv = gate.get_world_vertices()
        ax.plot(wv[:, 0], wv[:, 1], wv[:, 2], 
               color=gate.color, linewidth=5, alpha=0.85,
               label=f"Gate {i+1}: {gate.shape}")
        
        # Draw gate center
        ax.scatter([gate.pos[0]], [gate.pos[1]], [gate.pos[2]], 
                  c=gate.color, marker='o', s=100, alpha=0.7, 
                  edgecolors='black', linewidth=1)
        
        # Draw gate normal direction
        normal_vec = np.array([np.cos(gate.yaw), np.sin(gate.yaw), 0]) * 4
        ax.quiver(gate.pos[0], gate.pos[1], gate.pos[2],
                 normal_vec[0], normal_vec[1], normal_vec[2],
                 color=gate.color, arrow_length_ratio=0.25, 
                 linewidth=2.5, alpha=0.6)
        
        # Draw path lines between consecutive gates
        if i > 0:
            ax.plot([gates[i-1].pos[0], gate.pos[0]], 
                   [gates[i-1].pos[1], gate.pos[1]], 
                   [gates[i-1].pos[2], gate.pos[2]], 
                   'gray', linestyle='--', alpha=0.3, linewidth=2)
    
    # For circular layout, connect last gate back to first
    if circular_layout and len(gates) > 0:
        ax.plot([gates[-1].pos[0], gates[0].pos[0]], 
               [gates[-1].pos[1], gates[0].pos[1]], 
               [gates[-1].pos[2], gates[0].pos[2]], 
               'gray', linestyle='--', alpha=0.3, linewidth=2, label='Loop closure')
    
    # Plot trajectory
    if trajectory_points is not None and len(trajectory_points) > 0:
        traj = np.array(trajectory_points)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
               'b-', linewidth=3, alpha=0.8, label='Reference Trajectory')
        
        # Mark waypoints
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], 
                  c='blue', marker='o', s=70, alpha=0.6, 
                  edgecolors='darkblue', linewidth=1)
    
    # Ground plane with grid
    bounds = 22
    x_grid = np.linspace(-bounds, bounds, 5)
    y_grid = np.linspace(-bounds, bounds, 5)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.15)
    
    # World box bounds
    ax.set_xlim([-bounds, bounds])
    ax.set_ylim([-bounds, bounds])
    ax.set_zlim([0, 16])
    
    ax.set_xlabel('X (m)', fontsize=12, weight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, weight='bold')
    ax.set_zlabel('Altitude (m)', fontsize=12, weight='bold')
    
    layout_type = 'Circular' if circular_layout else 'Sequential'
    ax.set_title(f'Gate Configuration ({layout_type} Layout) and Reference Trajectory', 
                fontsize=14, weight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set view angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(4)  # Show for 4 seconds
    plt.close()

def print_summary(args, gates, start_pos):
    """Print configuration summary."""
    print("\n" + "="*75)
    print(" " * 20 + "MULTI-QUADCOPTER GATE FLIGHT SYSTEM")
    print("="*75)
    print(f"\n{'Configuration:':<25}")
    print(f"  {'Gate Order:':<25} {args.gate_order} ({len(gates)} gates)")
    print(f"  {'Layout:':<25} {'Circular' if args.circular_layout else 'Sequential'}")
    print(f"  {'World Box Size:':<25} {args.world_box_size}m (±{args.world_box_size/2:.1f}m bounds)")
    print(f"  {'Start Distance:':<25} {args.start_distance}m")
    print(f"  {'Horizontal Distance:':<25} {args.horizontal_distance}m")
    print(f"  {'Min 3D Distance:':<25} {args.min_distance}m")
    print(f"  {'Max Angle Change:':<25} {args.max_angle_change}°")
    print(f"  {'Max Altitude Change:':<25} {args.max_altitude_change}m")
    print(f"\n{'Flight Options:':<25}")
    print(f"  {'Controller:':<25} {args.controller.upper()}")
    print(f"  {'Optimized Path:':<25} {args.optimized_path}")
    print(f"  {'Path Mode:':<25} {args.path_mode}")
    print(f"  {'Num Quadcopters:':<25} {args.num_quads}")
    print(f"  {'Quadcopter Size:':<25} {args.quad_size}m (visual scale)")
    print(f"  {'Device:':<25} {args.device}")
    print(f"  {'Timestep:':<25} {args.dt}s")
    print(f"\n{'Start Position:':<25} ({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f})m")
    
    # PATCHED: Show controller gains
    if args.controller == 'se3':
        print(f"\n{'SE3 Controller Gains:':<25}")
        print(f"  {'kp_pos:':<25} {args.kp_pos}")
        print(f"  {'kd_pos:':<25} {args.kd_pos}")
        print(f"  {'kp_att:':<25} {args.kp_att_se3}")
        print(f"  {'kd_att:':<25} {args.kd_att_se3}")
    else:
        print(f"\n{'PID Controller Gains:':<25}")
        print(f"  {'kp_att:':<25} {args.kp_att_pid}")
        print(f"  {'ki_att:':<25} {args.ki_att_pid}")
        print(f"  {'kd_att:':<25} {args.kd_att_pid}")
        print(f"  {'kp_thrust:':<25} {args.kp_thrust}")
        print(f"  {'kd_thrust:':<25} {args.kd_thrust}")
    
    print("="*75 + "\n")

def main():
    args = parse_args()
    
    # Set default num_quads based on path type if not explicitly specified
    if args.num_quads is None:
        if args.optimized_path:
            args.num_quads = 1
            print("Using 1 quadcopter (optimized path - all trajectories identical)")
        else:
            args.num_quads = 4
            print("Using 4 quadcopters (varied paths - trajectories differ)")
    
    # Step 1: Generate gates
    print("\n[1/3] Generating gate configuration...")
    gate_manager = GateManager(
        gate_order=args.gate_order,
        min_distance=args.min_distance,
        horizontal_distance=args.horizontal_distance,
        start_distance=args.start_distance,
        max_angle_change=args.max_angle_change,
        max_altitude_change=args.max_altitude_change,
        circular_layout=args.circular_layout,
        world_box_size=args.world_box_size
    )
    gates = gate_manager.generate_gates()
    
    if len(gates) < 2:
        print("\n❌ ERROR: Need at least 2 gates to create trajectory!")
        print("Try adjusting constraints:")
        print("  - Use --circular_layout for better gate placement")
        print("  - Increase --world_box_size")
        print("  - Increase --max_angle_change")
        return
    
    # Get start position
    start_pos = gate_manager.get_start_position(gates)
    
    # Print summary
    print_summary(args, gates, start_pos)
    
    # Step 2: Generate reference trajectories
    print("\n[2/3] Generating reference trajectories...")
    traj_gen = TrajectoryGenerator(
        gates=gates, 
        start_position=start_pos,
        circular_layout=args.circular_layout
    )
    trajectory_points = traj_gen.generate_least_flyable_trajectory()
    
    print(f"Trajectory waypoints generated: {len(trajectory_points)}")
    
    # Generate smooth trajectories
    reference_trajectories = traj_gen.generate_smooth_trajectories(
        num_trajectories=args.num_quads,
        dt=args.dt,
        path_mode=args.path_mode,
        min_radius=args.min_turn_radius
    )
    
    if not reference_trajectories:
        print("❌ ERROR: Failed to generate trajectories!")
        return
    
    print(f"\n✅ Generated {len(reference_trajectories)} reference trajectories:")
    for i, traj in enumerate(reference_trajectories):
        print(f"  Trajectory {i+1}: {len(traj['pos'])} timesteps, "
              f"{traj['total_time']:.2f}s duration, "
              f"max speed: {np.max(np.linalg.norm(traj['vel'], axis=1)):.2f} m/s")
    
    # Visualize gates and trajectory
    if args.show_trajectory:
        print("\nDisplaying gate configuration and trajectory plot...")
        visualize_gates_and_trajectory(gates, trajectory_points, start_pos, args.circular_layout)
    
    # Step 3: Start multi-quadcopter visualization
    print("\n[3/3] Starting multi-quadcopter visualization...")
    print("="*75)
    print("Keyboard controls:")
    print("  Arrow keys    : Rotate camera view")
    print("  R             : Toggle auto-rotation")
    print("  1-5           : Preset camera views")
    print("  Ctrl+C        : Stop simulation")
    print("="*75 + "\n")
    
    print("🚁 Launching visualization... (close window or press Ctrl+C to stop)")
    
    # PATCHED: Pass position gains to controller
    ctrl_args = {}
    if args.controller == 'se3':
        ctrl_args = {
            'kp_pos': args.kp_pos, 
            'kd_pos': args.kd_pos, 
            'kp_att': args.kp_att_se3, 
            'kd_att': args.kd_att_se3
        }
    else:
        ctrl_args = {
            'kp_att': np.array([args.kp_att_pid]*3),
            'ki_att': np.array([args.ki_att_pid]*3),
            'kd_att': np.array([args.kd_att_pid]*3),
            'kp_thrust': args.kp_thrust, 
            'ki_thrust': args.ki_thrust, 
            'kd_thrust': args.kd_thrust,
            'kp_pos': args.kp_pos,  # PATCHED: Add position gains
            'kd_pos': args.kd_pos
        }    
        
    visualizer = MultiQuadVisualizer(
        gates=gates,
        trajectories=reference_trajectories,
        dt=args.dt,
        device=args.device,
        trajectory_points=trajectory_points,
        quad_size=args.quad_size,
        ctrl_type=args.controller,
        ctrl_args=ctrl_args
    )
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\n\n⏹  Stopping visualization...")
        print("="*75)
        print("✅ Session complete!")
        print("="*75)

if __name__ == "__main__":
    main()
