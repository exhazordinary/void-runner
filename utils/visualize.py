"""
VOID_RUNNER - Visualization Tools
=================================
Tools for visualizing curiosity, exploration patterns, and training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import torch


# Custom colormap for curiosity (black -> cyan -> magenta)
CURIOSITY_COLORS = ['#0a0a0a', '#00ffff', '#ff00ff', '#ffffff']
curiosity_cmap = LinearSegmentedColormap.from_list('curiosity', CURIOSITY_COLORS)


def plot_training_curves(metrics_path: str, save_path: Optional[str] = None):
    """
    Plot training curves from metrics file.

    Args:
        metrics_path: Path to metrics.json
        save_path: Optional path to save figure
    """
    with open(metrics_path) as f:
        metrics = json.load(f)

    steps = [m['step'] for m in metrics]
    rewards = [m['avg_reward'] for m in metrics]
    intrinsic = [m['avg_intrinsic'] for m in metrics]
    coverage = [m.get('coverage', 0) * 100 for m in metrics]
    goals = [m.get('goals_reached', 0) for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VOID_RUNNER Training Progress', fontsize=16, fontweight='bold')

    # Style
    for ax in axes.flat:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#333')
        ax.spines['top'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['right'].set_color('#333')

    fig.patch.set_facecolor('#1a1a1a')

    # Extrinsic rewards
    axes[0, 0].plot(steps, rewards, color='#00ff88', linewidth=2)
    axes[0, 0].fill_between(steps, rewards, alpha=0.3, color='#00ff88')
    axes[0, 0].set_title('Extrinsic Reward', color='white', fontweight='bold')
    axes[0, 0].set_xlabel('Steps', color='white')
    axes[0, 0].set_ylabel('Avg Reward', color='white')

    # Intrinsic rewards (curiosity)
    axes[0, 1].plot(steps, intrinsic, color='#ff00ff', linewidth=2)
    axes[0, 1].fill_between(steps, intrinsic, alpha=0.3, color='#ff00ff')
    axes[0, 1].set_title('Intrinsic Reward (Curiosity)', color='white', fontweight='bold')
    axes[0, 1].set_xlabel('Steps', color='white')
    axes[0, 1].set_ylabel('Avg Curiosity', color='white')

    # Coverage
    axes[1, 0].plot(steps, coverage, color='#00ffff', linewidth=2)
    axes[1, 0].fill_between(steps, coverage, alpha=0.3, color='#00ffff')
    axes[1, 0].set_title('State Space Coverage', color='white', fontweight='bold')
    axes[1, 0].set_xlabel('Steps', color='white')
    axes[1, 0].set_ylabel('Coverage %', color='white')
    axes[1, 0].set_ylim(0, 100)

    # Goals reached
    axes[1, 1].plot(steps, goals, color='#ffff00', linewidth=2)
    axes[1, 1].fill_between(steps, goals, alpha=0.3, color='#ffff00')
    axes[1, 1].set_title('Goals Reached', color='white', fontweight='bold')
    axes[1, 1].set_xlabel('Steps', color='white')
    axes[1, 1].set_ylabel('Total Goals', color='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a')
        print(f"Saved training curves to {save_path}")

    plt.show()


def create_curiosity_heatmap(
    agent,
    env_size: int = 15,
    resolution: int = 50,
    save_path: Optional[str] = None
):
    """
    Create a heatmap of curiosity across the state space.

    This visualizes what the agent finds "interesting" -
    brighter areas indicate higher prediction error (novelty).

    Args:
        agent: Trained VoidRunnerAgent
        env_size: Size of the environment
        resolution: Grid resolution for heatmap
        save_path: Optional path to save figure
    """
    # Create grid of positions
    x = np.linspace(0, env_size - 1, resolution)
    y = np.linspace(0, env_size - 1, resolution)
    xx, yy = np.meshgrid(x, y)

    # Compute curiosity for each position
    positions = np.stack([xx.flatten(), yy.flatten()], axis=1)
    positions_tensor = torch.FloatTensor(positions).to(agent.device)

    with torch.no_grad():
        curiosity = agent.curiosity.compute_intrinsic_reward(
            positions_tensor, update=False
        ).cpu().numpy()

    curiosity_grid = curiosity.reshape(resolution, resolution)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    im = ax.imshow(
        curiosity_grid,
        cmap=curiosity_cmap,
        origin='lower',
        extent=[0, env_size, 0, env_size]
    )

    ax.set_title('Curiosity Heatmap\n(Brighter = More Novel)',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', color='white')
    ax.set_ylabel('Y Position', color='white')
    ax.tick_params(colors='white')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Curiosity Level')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0a0a0a')
        print(f"Saved curiosity heatmap to {save_path}")

    plt.show()


def visualize_exploration_trajectory(
    positions: List[Tuple[float, float]],
    env_size: int = 15,
    goal_pos: Tuple[int, int] = None,
    save_path: Optional[str] = None
):
    """
    Visualize the exploration trajectory of an agent.

    Args:
        positions: List of (x, y) positions visited
        env_size: Size of the environment
        goal_pos: Position of the goal
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    positions = np.array(positions)

    # Create color gradient for trajectory (time progression)
    colors = plt.cm.plasma(np.linspace(0, 1, len(positions)))

    # Plot trajectory
    for i in range(1, len(positions)):
        ax.plot(
            positions[i-1:i+1, 0],
            positions[i-1:i+1, 1],
            color=colors[i],
            linewidth=1.5,
            alpha=0.7
        )

    # Start and end markers
    ax.scatter(positions[0, 0], positions[0, 1],
               color='#00ff00', s=200, marker='o',
               label='Start', zorder=10, edgecolors='white')
    ax.scatter(positions[-1, 0], positions[-1, 1],
               color='#ff0000', s=200, marker='*',
               label='End', zorder=10, edgecolors='white')

    # Goal marker
    if goal_pos:
        ax.scatter(goal_pos[0], goal_pos[1],
                   color='#ffff00', s=300, marker='D',
                   label='Goal', zorder=10, edgecolors='white')

    ax.set_xlim(-0.5, env_size - 0.5)
    ax.set_ylim(-0.5, env_size - 0.5)
    ax.set_title('Exploration Trajectory\n(Color = Time Progression)',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', color='white')
    ax.set_ylabel('Y Position', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')

    # Grid
    ax.set_xticks(range(env_size))
    ax.set_yticks(range(env_size))
    ax.grid(True, alpha=0.2, color='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0a0a0a')
        print(f"Saved trajectory to {save_path}")

    plt.show()


def create_exploration_animation(
    frames: List[np.ndarray],
    save_path: str = "exploration.gif",
    fps: int = 10
):
    """
    Create an animated GIF of exploration.

    Args:
        frames: List of RGB frames
        save_path: Path to save GIF
        fps: Frames per second
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a0a')

    im = ax.imshow(frames[0])
    ax.axis('off')

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 // fps, blit=True
    )

    ani.save(save_path, writer='pillow', fps=fps)
    print(f"Saved animation to {save_path}")
    plt.close()


def compare_with_without_curiosity(
    metrics_with: str,
    metrics_without: str,
    save_path: Optional[str] = None
):
    """
    Compare training with and without curiosity.

    Args:
        metrics_with: Path to metrics with curiosity
        metrics_without: Path to metrics without curiosity
        save_path: Optional path to save figure
    """
    with open(metrics_with) as f:
        m_with = json.load(f)
    with open(metrics_without) as f:
        m_without = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Curiosity vs No Curiosity', fontsize=16, fontweight='bold')
    fig.patch.set_facecolor('#1a1a1a')

    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')

    # Reward comparison
    steps_w = [m['step'] for m in m_with]
    rewards_w = [m['avg_reward'] for m in m_with]
    steps_wo = [m['step'] for m in m_without]
    rewards_wo = [m['avg_reward'] for m in m_without]

    axes[0].plot(steps_w, rewards_w, color='#00ffff', linewidth=2, label='With Curiosity')
    axes[0].plot(steps_wo, rewards_wo, color='#ff6666', linewidth=2, label='Without Curiosity')
    axes[0].set_title('Reward Comparison', color='white', fontweight='bold')
    axes[0].set_xlabel('Steps', color='white')
    axes[0].set_ylabel('Avg Reward', color='white')
    axes[0].legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')

    # Coverage comparison
    coverage_w = [m.get('coverage', 0) * 100 for m in m_with]
    coverage_wo = [m.get('coverage', 0) * 100 for m in m_without]

    axes[1].plot(steps_w, coverage_w, color='#00ffff', linewidth=2, label='With Curiosity')
    axes[1].plot(steps_wo, coverage_wo, color='#ff6666', linewidth=2, label='Without Curiosity')
    axes[1].set_title('Coverage Comparison', color='white', fontweight='bold')
    axes[1].set_xlabel('Steps', color='white')
    axes[1].set_ylabel('Coverage %', color='white')
    axes[1].legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a')

    plt.show()


if __name__ == "__main__":
    # Demo: Create sample visualization
    print("VOID_RUNNER Visualization Tools")
    print("Usage:")
    print("  from utils.visualize import plot_training_curves, create_curiosity_heatmap")
    print("  plot_training_curves('outputs/metrics.json')")
