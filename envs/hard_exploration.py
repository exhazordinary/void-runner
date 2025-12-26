"""
VOID_RUNNER - Hard Exploration Environments
============================================
Challenging environments for testing exploration algorithms.

Includes:
1. DeceptiveRewardMaze: Misleading rewards that trap naive agents
2. KeyDoorEnv: Must find key before door (temporal abstraction)
3. StochasticMaze: Random transitions add noise
4. MultiGoalSparse: Multiple goals, very sparse rewards
5. MontezumaLite: Simplified Montezuma-style platformer
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List


class DeceptiveRewardMaze(gym.Env):
    """
    Maze with deceptive local rewards.

    The obvious path (following local rewards) leads to a trap.
    The true goal requires ignoring local rewards and exploring.

    This tests whether curiosity can overcome reward hacking.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size: int = 20, render_mode: Optional[str] = None):
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

        # True goal (hidden, far from deceptive rewards)
        self.true_goal = np.array([size - 2, size - 2])

        # Deceptive reward region (local optimum trap)
        self.trap_center = np.array([size // 2, size // 2])
        self.trap_radius = 3

        self.agent_pos = None
        self.steps = 0
        self.max_steps = size * size * 2
        self.visited = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([1, 1], dtype=np.float32)
        self.steps = 0
        self.visited = {(1, 1)}
        return self.agent_pos.copy(), {}

    def step(self, action: int):
        self.steps += 1

        # Move
        new_pos = self.agent_pos + self.directions[action]
        new_pos = np.clip(new_pos, 0, self.size - 1).astype(np.float32)
        self.agent_pos = new_pos
        self.visited.add(tuple(self.agent_pos.astype(int)))

        # Reward calculation
        reward = 0.0

        # True goal (sparse, hard to find)
        if np.allclose(self.agent_pos, self.true_goal, atol=0.5):
            reward = 10.0
            terminated = True
        else:
            terminated = False

        # Deceptive reward (gradient toward trap)
        dist_to_trap = np.linalg.norm(self.agent_pos - self.trap_center)
        if dist_to_trap < self.trap_radius:
            # In trap zone - small positive reward (the trap!)
            reward += 0.1 * (self.trap_radius - dist_to_trap)

        truncated = self.steps >= self.max_steps

        info = {
            'true_goal_dist': np.linalg.norm(self.agent_pos - self.true_goal),
            'in_trap': dist_to_trap < self.trap_radius,
            'coverage': len(self.visited) / (self.size ** 2)
        }

        return self.agent_pos.copy(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        grid[:, :] = [20, 20, 30]

        # Trap zone (red-ish)
        for x in range(self.size):
            for y in range(self.size):
                if np.linalg.norm([x, y] - self.trap_center) < self.trap_radius:
                    grid[y, x] = [100, 30, 30]

        # True goal (green)
        grid[int(self.true_goal[1]), int(self.true_goal[0])] = [0, 255, 0]

        # Agent (cyan)
        grid[int(self.agent_pos[1]), int(self.agent_pos[0])] = [0, 255, 255]

        return grid


class KeyDoorEnv(gym.Env):
    """
    Key-Door environment: Must find key before opening door.

    Tests temporal abstraction and memory:
    1. Agent must find key (no reward)
    2. Then find door (reward only if key collected)

    Without curiosity, random exploration almost never succeeds.
    """

    def __init__(self, size: int = 15, render_mode: Optional[str] = None):
        super().__init__()
        self.size = size
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        # Observation: [x, y, has_key]
        self.observation_space = spaces.Box(
            low=0, high=max(size - 1, 1), shape=(3,), dtype=np.float32
        )

        self.directions = [
            np.array([0, 1]), np.array([1, 0]),
            np.array([0, -1]), np.array([-1, 0])
        ]

        # Key and door positions
        self.key_pos = np.array([1, size - 2])
        self.door_pos = np.array([size - 2, 1])

        self.agent_pos = None
        self.has_key = False
        self.steps = 0
        self.max_steps = size * size * 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([size // 2, size // 2], dtype=np.float32)
        self.has_key = False
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            float(self.has_key)
        ], dtype=np.float32)

    def step(self, action: int):
        self.steps += 1

        # Move
        new_pos = self.agent_pos + self.directions[action]
        self.agent_pos = np.clip(new_pos, 0, self.size - 1).astype(np.float32)

        reward = 0.0
        terminated = False

        # Check key
        if not self.has_key and np.allclose(self.agent_pos, self.key_pos, atol=0.5):
            self.has_key = True

        # Check door
        if np.allclose(self.agent_pos, self.door_pos, atol=0.5):
            if self.has_key:
                reward = 10.0
                terminated = True
            # No reward without key

        truncated = self.steps >= self.max_steps

        info = {
            'has_key': self.has_key,
            'key_dist': np.linalg.norm(self.agent_pos - self.key_pos),
            'door_dist': np.linalg.norm(self.agent_pos - self.door_pos),
        }

        return self._get_obs(), reward, terminated, truncated, info


class StochasticMaze(gym.Env):
    """
    Maze with stochastic transitions.

    Actions succeed with probability p, otherwise random action.
    This creates noise that prediction-based curiosity must handle.

    Tests robustness of curiosity to environmental stochasticity.
    """

    def __init__(
        self,
        size: int = 15,
        success_prob: float = 0.8,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.size = size
        self.success_prob = success_prob
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )

        self.directions = [
            np.array([0, 1]), np.array([1, 0]),
            np.array([0, -1]), np.array([-1, 0])
        ]

        self.goal_pos = np.array([size - 2, size - 2])
        self.agent_pos = None
        self.steps = 0
        self.max_steps = size * size * 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([1, 1], dtype=np.float32)
        self.steps = 0
        return self.agent_pos.copy(), {}

    def step(self, action: int):
        self.steps += 1

        # Stochastic transition
        if np.random.random() > self.success_prob:
            action = np.random.randint(4)  # Random action

        new_pos = self.agent_pos + self.directions[action]
        self.agent_pos = np.clip(new_pos, 0, self.size - 1).astype(np.float32)

        reached_goal = np.allclose(self.agent_pos, self.goal_pos, atol=0.5)
        reward = 1.0 if reached_goal else 0.0
        terminated = reached_goal
        truncated = self.steps >= self.max_steps

        return self.agent_pos.copy(), reward, terminated, truncated, {}


class MultiGoalSparse(gym.Env):
    """
    Multiple goals scattered in environment, very sparse rewards.

    Agent must discover ANY goal. Tests breadth of exploration.
    """

    def __init__(
        self,
        size: int = 30,
        n_goals: int = 5,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.size = size
        self.n_goals = n_goals
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.float32
        )

        self.directions = [
            np.array([0, 1]), np.array([1, 0]),
            np.array([0, -1]), np.array([-1, 0])
        ]

        # Random goal positions
        self.goals = []
        self.agent_pos = None
        self.collected = set()
        self.steps = 0
        self.max_steps = size * size * 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate random goal positions
        self.goals = []
        while len(self.goals) < self.n_goals:
            pos = np.random.randint(2, self.size - 2, size=2)
            # Ensure goals are spread out
            if all(np.linalg.norm(pos - g) > 5 for g in self.goals):
                self.goals.append(pos)

        self.agent_pos = np.array([self.size // 2, self.size // 2], dtype=np.float32)
        self.collected = set()
        self.steps = 0

        return self.agent_pos.copy(), {}

    def step(self, action: int):
        self.steps += 1

        new_pos = self.agent_pos + self.directions[action]
        self.agent_pos = np.clip(new_pos, 0, self.size - 1).astype(np.float32)

        reward = 0.0
        for i, goal in enumerate(self.goals):
            if i not in self.collected and np.allclose(self.agent_pos, goal, atol=0.5):
                self.collected.add(i)
                reward = 1.0

        terminated = len(self.collected) == self.n_goals
        truncated = self.steps >= self.max_steps

        info = {
            'goals_collected': len(self.collected),
            'goals_remaining': self.n_goals - len(self.collected),
        }

        return self.agent_pos.copy(), reward, terminated, truncated, info


class MontezumaLite(gym.Env):
    """
    Simplified Montezuma's Revenge-style environment.

    Features:
    - Multiple rooms connected by doors
    - Keys unlock doors
    - Ladders for vertical movement
    - Very sparse rewards (only for collecting treasures)

    This is the classic hard exploration benchmark, simplified.
    """

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # 3 rooms: left, center, right
        self.room_size = 10
        self.n_rooms = 3

        self.action_space = spaces.Discrete(6)  # up, right, down, left, climb_up, climb_down
        # Observation: [x, y, room, has_key1, has_key2]
        self.observation_space = spaces.Box(
            low=0, high=max(self.room_size, self.n_rooms), shape=(5,), dtype=np.float32
        )

        # Room layout
        self.doors = {
            (0, 1): np.array([self.room_size - 1, 5]),  # Left to Center
            (1, 2): np.array([self.room_size - 1, 5]),  # Center to Right
        }

        self.keys = {
            0: np.array([5, 2]),  # Key in left room
            1: np.array([5, 8]),  # Key in center room
        }

        self.treasure = np.array([5, 5])  # Treasure in right room

        self.ladders = [
            (1, 3, 3, 7),  # room 1, x=3, y from 3 to 7
        ]

        self.agent_pos = None
        self.agent_room = 0
        self.collected_keys = set()
        self.steps = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([2, 5], dtype=np.float32)
        self.agent_room = 0
        self.collected_keys = set()
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.agent_room,
            1.0 if 0 in self.collected_keys else 0.0,
            1.0 if 1 in self.collected_keys else 0.0,
        ], dtype=np.float32)

    def step(self, action: int):
        self.steps += 1

        # Movement
        if action == 0:  # Up
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.room_size - 1)
        elif action == 1:  # Right
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.room_size - 1)
        elif action == 2:  # Down
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 3:  # Left
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        # Climb actions only work on ladders (simplified: just vertical movement)
        elif action == 4:  # Climb up
            self.agent_pos[1] = min(self.agent_pos[1] + 2, self.room_size - 1)
        elif action == 5:  # Climb down
            self.agent_pos[1] = max(self.agent_pos[1] - 2, 0)

        reward = 0.0
        terminated = False

        # Check key collection
        if self.agent_room in self.keys:
            key_pos = self.keys[self.agent_room]
            if np.allclose(self.agent_pos, key_pos, atol=1.0):
                if self.agent_room not in self.collected_keys:
                    self.collected_keys.add(self.agent_room)

        # Check room transition
        for (from_room, to_room), door_pos in self.doors.items():
            if self.agent_room == from_room:
                if np.allclose(self.agent_pos, door_pos, atol=1.0):
                    # Need key to pass
                    if from_room in self.collected_keys:
                        self.agent_room = to_room
                        self.agent_pos = np.array([1, 5], dtype=np.float32)

        # Check treasure (in room 2)
        if self.agent_room == 2:
            if np.allclose(self.agent_pos, self.treasure, atol=1.0):
                reward = 100.0
                terminated = True

        truncated = self.steps >= self.max_steps

        info = {
            'room': self.agent_room,
            'keys_collected': len(self.collected_keys),
        }

        return self._get_obs(), reward, terminated, truncated, info


# Register environments
gym.register(id="DeceptiveMaze-v0", entry_point="envs.hard_exploration:DeceptiveRewardMaze")
gym.register(id="KeyDoor-v0", entry_point="envs.hard_exploration:KeyDoorEnv")
gym.register(id="StochasticMaze-v0", entry_point="envs.hard_exploration:StochasticMaze")
gym.register(id="MultiGoal-v0", entry_point="envs.hard_exploration:MultiGoalSparse")
gym.register(id="MontezumaLite-v0", entry_point="envs.hard_exploration:MontezumaLite")
