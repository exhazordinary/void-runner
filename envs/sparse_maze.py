"""
VOID_RUNNER - Sparse Reward Maze Environment
============================================
A challenging environment where the agent must navigate
to a goal with NO intermediate rewards.

This is the perfect testbed for curiosity-driven exploration:
- Without intrinsic motivation, random agents rarely find the goal
- With curiosity, agents systematically explore until they find it
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class SparseMazeEnv(gym.Env):
    """
    Sparse reward maze environment.

    The agent starts at one corner and must reach the goal at another.
    No rewards are given except for reaching the goal.

    Observation: Agent's (x, y) position
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    Reward: +1 for reaching goal, 0 otherwise
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        size: int = 15,
        walls: bool = True,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.size = size
        self.render_mode = render_mode

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )

        # Movement directions: Up, Right, Down, Left
        self.directions = [
            np.array([0, 1]),   # Up
            np.array([1, 0]),   # Right
            np.array([0, -1]),  # Down
            np.array([-1, 0])   # Left
        ]

        # Generate maze walls
        self.walls = set()
        if walls:
            self._generate_walls()

        # Initialize state
        self.agent_pos = None
        self.goal_pos = np.array([size - 2, size - 2])
        self.visited_states = set()
        self.steps = 0
        self.max_steps = size * size * 2

    def _generate_walls(self):
        """Generate maze walls."""
        # Create some internal walls to make navigation challenging
        for i in range(2, self.size - 2, 3):
            for j in range(self.size - 3):
                if np.random.random() > 0.3:
                    self.walls.add((i, j))
            for j in range(2, self.size):
                if np.random.random() > 0.3:
                    self.walls.add((i + 1, j))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Start at bottom-left corner
        self.agent_pos = np.array([1, 1], dtype=np.float32)
        self.visited_states = set()
        self.visited_states.add(tuple(self.agent_pos))
        self.steps = 0

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        return self.agent_pos.copy()

    def _get_info(self) -> Dict[str, Any]:
        return {
            "unique_states_visited": len(self.visited_states),
            "distance_to_goal": np.linalg.norm(self.agent_pos - self.goal_pos),
            "coverage": len(self.visited_states) / (self.size * self.size)
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1

        # Compute new position
        direction = self.directions[action]
        new_pos = self.agent_pos + direction

        # Check boundaries and walls
        if (0 <= new_pos[0] < self.size and
            0 <= new_pos[1] < self.size and
            tuple(new_pos) not in self.walls):
            self.agent_pos = new_pos.astype(np.float32)

        # Track visited states
        self.visited_states.add(tuple(self.agent_pos))

        # Check if goal reached
        reached_goal = np.array_equal(self.agent_pos.astype(int), self.goal_pos)

        # Sparse reward: only at goal
        reward = 1.0 if reached_goal else 0.0

        # Episode ends at goal or max steps
        terminated = reached_goal
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        # Create grid visualization
        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # Background
        grid[:, :] = [40, 40, 40]

        # Walls
        for wall in self.walls:
            grid[wall[1], wall[0]] = [100, 100, 100]

        # Visited states (curiosity trail)
        for state in self.visited_states:
            x, y = int(state[0]), int(state[1])
            if 0 <= x < self.size and 0 <= y < self.size:
                grid[y, x] = [0, 100, 100]  # Cyan for visited

        # Goal
        grid[self.goal_pos[1], self.goal_pos[0]] = [0, 255, 0]  # Green

        # Agent
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        grid[agent_y, agent_x] = [255, 0, 100]  # Pink

        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.imshow(grid)
            plt.pause(0.01)
            plt.clf()

        return grid


class SparseGridWorldEnv(gym.Env):
    """
    Large sparse grid world - the ultimate test for curiosity.

    A massive grid with a single goal. Without curiosity,
    finding the goal through random exploration is nearly impossible.
    """

    def __init__(self, size: int = 50, render_mode: Optional[str] = None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )

        self.directions = [
            np.array([0, 1]), np.array([1, 0]),
            np.array([0, -1]), np.array([-1, 0])
        ]

        self.agent_pos = None
        self.goal_pos = np.array([size - 1, size - 1])
        self.visited_states = set()
        self.steps = 0
        self.max_steps = size * size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.visited_states = {(0, 0)}
        self.steps = 0
        return self.agent_pos.copy(), {"coverage": 0, "visited": 1}

    def step(self, action):
        self.steps += 1

        new_pos = self.agent_pos + self.directions[action]
        new_pos = np.clip(new_pos, 0, self.size - 1).astype(np.float32)
        self.agent_pos = new_pos

        self.visited_states.add(tuple(self.agent_pos.astype(int)))

        reached_goal = np.array_equal(self.agent_pos.astype(int), self.goal_pos)
        reward = 10.0 if reached_goal else 0.0

        terminated = reached_goal
        truncated = self.steps >= self.max_steps

        info = {
            "coverage": len(self.visited_states) / (self.size * self.size),
            "visited": len(self.visited_states)
        }

        return self.agent_pos.copy(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        grid[:, :] = [20, 20, 30]

        for state in self.visited_states:
            grid[state[1], state[0]] = [0, 80, 80]

        grid[self.goal_pos[1], self.goal_pos[0]] = [0, 255, 0]
        grid[int(self.agent_pos[1]), int(self.agent_pos[0])] = [255, 50, 100]

        return grid


# Register environments
gym.register(
    id="SparseMaze-v0",
    entry_point="envs.sparse_maze:SparseMazeEnv",
    max_episode_steps=500,
)

gym.register(
    id="SparseGrid-v0",
    entry_point="envs.sparse_maze:SparseGridWorldEnv",
    max_episode_steps=5000,
)
