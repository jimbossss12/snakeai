#!/usr/bin/env python3
"""
The Ultimate Advanced Snake AI – Adapted and Optimized Version

Features:
  - Deterministic snake initialization (center, length 3, facing right).
  - Automatic loading of a previously saved best model.
  - Continuous training with a limit on the number of steps per game.
  - Obstacle spawning begins from a configurable episode onward.
  - Real-time parameter adjustments via a PyQt5 GUI.
  - Soft target network updates every episode.
  - A startup system benchmark that “adapts” key parameters (FPS, grid size,
    network complexity, etc.) to the performance of the host system.
  - Enhanced logging and error handling for easier debugging.
  - Remote control via a simple REST API (if run in remote mode).
  
Enjoy your optimized Snake AI!
"""

import sys
import time
import random
import os
import math
import threading
import logging
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Deque, Dict

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# --- PyQt5 and Matplotlib ---
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- Flask for Remote Control ---
from flask import Flask, request, jsonify

# --- Set Up Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Global Locks ---
stats_lock = threading.Lock()
config_lock = threading.Lock()  # For live-updating CONFIG

# --- Default Parameters ---
default_params = {
    "lr": 0.0005,
    "batch_size": 64,
    "memory_capacity": 50000,
    "seed": 42,
    "log_dir": os.path.join(os.path.expanduser("~"), "runs"),
    "gamma": 0.99,
    "tau": 0.005,
    "n_steps": 3,
    "beta_start": 0.4,
    "beta_end": 1.0,
    "grad_clip": 1.0,
    "max_steps": 500,
    "obstacle_start": 1,  # Episode from which obstacles start
    # New obstacle parameters:
    "obstacle_spawn_prob": 0.5,    # Probability for an obstacle to spawn
    "obstacle_base_count": 40,       # Base allowed number of obstacles
    "obstacle_base_lifetime": 15,   # Base lifetime (seconds)
    "obstacle_lifetime_increment": 5,  # Additional lifetime per phase (seconds)
    "obstacle_phase_interval": 1,  # Seconds before phase increases
    "checkpoint_interval": 50,      # Checkpoint every n episodes
}

# --- Global Training Statistics ---
global_stats = {"episodes": [], "loss": [], "q_values": [], "td_errors": []}

# --- Global Configuration Dictionary (live-updated by GUI or remote API) ---
CONFIG = {
    "cell_size": 20,
    "grid_width": 30,
    "grid_height": 30,
    "fps": 10,
    "num_episodes": 1000,  # Not used in continuous training.
    "memory_capacity": default_params["memory_capacity"],
    "batch_size": default_params["batch_size"],
    "gamma": default_params["gamma"],
    "lr": default_params["lr"],
    "tau": default_params["tau"],
    "n_steps": default_params["n_steps"],
    "beta_start": default_params["beta_start"],
    "beta_end": default_params["beta_end"],
    "grad_clip": default_params["grad_clip"],
    "max_steps": default_params["max_steps"],
    "obstacle_start": default_params["obstacle_start"],
    # Obstacle parameters:
    "obstacle_spawn_prob": default_params["obstacle_spawn_prob"],
    "obstacle_base_count": default_params["obstacle_base_count"],
    "obstacle_base_lifetime": default_params["obstacle_base_lifetime"],
    "obstacle_lifetime_increment": default_params["obstacle_lifetime_increment"],
    "obstacle_phase_interval": default_params["obstacle_phase_interval"],
    "checkpoint_interval": default_params["checkpoint_interval"],
    # Also include:
    "seed": default_params["seed"],
    "log_dir": default_params["log_dir"],
    # We'll add "hidden_sizes" for the network; default value below.
    "hidden_sizes": [256, 128],
}

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- System Benchmark Functions ---
def system_benchmark() -> float:
    """
    Run a simple compute-intensive loop to estimate the system's performance.
    Returns the time in seconds to complete the loop.
    """
    start = time.time()
    s = 0.0
    for i in range(10**6):
        s += math.sqrt(i)
    duration = time.time() - start
    return duration

def adapt_to_system() -> Dict[str, any]:
    """
    Based on a simple benchmark, adapt key configuration parameters so that
    the program can run well even on low-power systems.
    """
    bench = system_benchmark()
    logging.info(f"System benchmark: {bench:.3f} seconds for test loop")
    if bench > 0.5:
        # Low-power system detected.
        adapted = {
            "fps": 5,
            "grid_width": 20,
            "grid_height": 20,
            "hidden_sizes": [128, 64],
        }
    else:
        # Otherwise, use default (high-performance) settings.
        adapted = {
            "fps": 10,
            "grid_width": 30,
            "grid_height": 30,
            "hidden_sizes": [256, 128],
        }
    return adapted

# Adapt the configuration based on system performance.
adapted_config = adapt_to_system()
with config_lock:
    CONFIG.update(adapted_config)

# --- Game Constants (update from CONFIG) ---
CELL_SIZE = CONFIG["cell_size"]
GRID_WIDTH = CONFIG["grid_width"]
GRID_HEIGHT = CONFIG["grid_height"]
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = CONFIG["fps"]

# --- Colors ---
BLACK: Tuple[int, int, int] = (0, 0, 0)
WHITE: Tuple[int, int, int] = (255, 255, 255)
GREEN: Tuple[int, int, int] = (0, 255, 0)
RED: Tuple[int, int, int] = (255, 0, 0)
BLUE: Tuple[int, int, int] = (0, 0, 255)

# --- Directions ---
UP: Tuple[int, int] = (0, -1)
RIGHT: Tuple[int, int] = (1, 0)
DOWN: Tuple[int, int] = (0, 1)
LEFT: Tuple[int, int] = (-1, 0)
DIRECTIONS: List[Tuple[int, int]] = [UP, RIGHT, DOWN, LEFT]

# --- Helper Functions ---
def random_grid_position(exclude: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Return a random grid position not in the 'exclude' set."""
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in exclude:
            return pos

def draw_rect(surface: pygame.Surface, color: Tuple[int, int, int], grid_pos: Tuple[int, int]) -> None:
    """Draw a rectangle at the given grid position."""
    rect = pygame.Rect(grid_pos[0] * CELL_SIZE, grid_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, color, rect)

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Compute the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_new_direction(current_dir: Tuple[int, int], action: int) -> Tuple[int, int]:
    """
    Compute a new direction based on the current direction and action.
    Actions: 0 = keep, 1 = turn left, 2 = turn right.
    """
    idx = DIRECTIONS.index(current_dir)
    if action == 0:
        return current_dir
    elif action == 1:
        return DIRECTIONS[(idx - 1) % len(DIRECTIONS)]
    elif action == 2:
        return DIRECTIONS[(idx + 1) % len(DIRECTIONS)]
    return current_dir

# --- Environment Classes ---
class Snake:
    """Snake with deterministic initialization (center, length 3, facing right)."""
    def __init__(self) -> None:
        center = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
        self.body: List[Tuple[int, int]] = [
            center,
            (center[0] - 1, center[1]),
            (center[0] - 2, center[1])
        ]
        self.direction: Tuple[int, int] = RIGHT
        self.grow_flag: bool = False

    def update(self) -> None:
        """Update snake position and manage growth."""
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = ((head_x + dx) % GRID_WIDTH, (head_y + dy) % GRID_HEIGHT)
        self.body.insert(0, new_head)
        if not self.grow_flag:
            self.body.pop()
        else:
            self.grow_flag = False

    def grow(self) -> None:
        """Set flag to grow the snake on the next update."""
        self.grow_flag = True

    def collision_with_self(self) -> bool:
        """Return True if the snake collides with itself."""
        return self.body[0] in self.body[1:]

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the snake on the given surface."""
        for segment in self.body:
            draw_rect(surface, BLUE, segment)

@dataclass
class Obstacle:
    """Obstacle on the grid."""
    position: Tuple[int, int]
    spawn_time: float
    lifetime: float

    def expired(self, current_time: float) -> bool:
        """Return True if the obstacle's lifetime has expired."""
        return (current_time - self.spawn_time) > self.lifetime

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the obstacle."""
        draw_rect(surface, RED, self.position)

class Food:
    """Food that the snake eats."""
    def __init__(self, snake: Snake, obstacles: List[Obstacle]) -> None:
        self.position: Tuple[int, int] = self._new_position(snake, obstacles)

    def _new_position(self, snake: Snake, obstacles: List[Obstacle]) -> Tuple[int, int]:
        """Select a new position not occupied by the snake or obstacles."""
        occupied = set(snake.body)
        for obs in obstacles:
            occupied.add(obs.position)
        return random_grid_position(occupied)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the food."""
        draw_rect(surface, GREEN, self.position)

# --- State Representation ---
def get_danger_distance(head: Tuple[int, int], direction: Tuple[int, int],
                        body: List[Tuple[int, int]], obstacles: List[Obstacle],
                        max_check: int = 10) -> float:
    """
    Look ahead in the given direction and return a normalized distance to danger.
    """
    x, y = head
    dx, dy = direction
    for steps in range(1, max_check + 1):
        x = (x + dx) % GRID_WIDTH
        y = (y + dy) % GRID_HEIGHT
        if (x, y) in body[1:] or any(obs.position == (x, y) for obs in obstacles):
            return steps / max_check
    return 1.0

def get_state(snake: Snake, food: Food, obstacles: List[Obstacle]) -> np.ndarray:
    """
    Create a state representation including:
      - Danger distances (straight, left, right)
      - Normalized food position differences
      - Normalized Manhattan distance to food
      - One-hot encoding of current direction
      - Normalized snake length
      - Normalized distance to the nearest obstacle
    """
    head = snake.body[0]
    direction = snake.direction
    left_dir = (-direction[1], direction[0])
    right_dir = (direction[1], -direction[0])

    danger_straight = get_danger_distance(head, direction, snake.body, obstacles)
    danger_left = get_danger_distance(head, left_dir, snake.body, obstacles)
    danger_right = get_danger_distance(head, right_dir, snake.body, obstacles)

    food_dx = (food.position[0] - head[0]) / GRID_WIDTH
    food_dy = (food.position[1] - head[1]) / GRID_HEIGHT
    norm_manhattan = manhattan_distance(head, food.position) / (GRID_WIDTH + GRID_HEIGHT)

    dir_onehot = [1 if direction == d else 0 for d in DIRECTIONS]
    snake_length = len(snake.body) / (GRID_WIDTH * GRID_HEIGHT)
    obstacle_dist = min([manhattan_distance(head, obs.position) for obs in obstacles] or [1.0]) / (GRID_WIDTH + GRID_HEIGHT)

    state = np.array(
        [danger_straight, danger_left, danger_right, food_dx, food_dy, norm_manhattan, snake_length, obstacle_dist] +
        dir_onehot,
        dtype=np.float32)
    return state

# --- Noisy Linear Layer ---
class NoisyLinear(nn.Module):
    """
    Noisy Linear layer using factorized Gaussian noise.
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
            self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_sigma", None)

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        if bias:
            self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.unsqueeze(1) * epsilon_in.unsqueeze(0))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon if self.bias_mu is not None else None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.bias_mu is not None else None
        return F.linear(x, weight, bias)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

# --- Dueling DQN ---
class DuelingDQN(nn.Module):
    """
    Dueling DQN using Noisy Linear layers and Layer Normalization.
    The network architecture is determined by the 'hidden_sizes' parameter from CONFIG.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Optional[List[int]] = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        self.feature = nn.Sequential(
            NoisyLinear(input_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.LeakyReLU(0.01),
            NoisyLinear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.LeakyReLU(0.01),
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_sizes[1], 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
            NoisyLinear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_sizes[1], 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
            NoisyLinear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# --- Prioritized Replay Memory ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class PrioritizedReplayMemory:
    """
    Prioritized replay memory for storing experiences.
    """
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_prio = 1.0

    def push(self, *args) -> None:
        """Add a new experience to memory."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], np.ndarray, torch.Tensor]:
        if len(self.buffer) == 0:
            return [], np.array([]), torch.tensor([])
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: torch.Tensor) -> None:
        prios = priorities.detach().cpu().numpy()
        np.put(self.priorities, indices, prios, mode='raise')
        self.max_prio = max(self.max_prio, np.max(prios))

    def __len__(self) -> int:
        return len(self.buffer)

# --- DQN Agent ---
class DQNAgent:
    """
    DQN Agent using Dueling DQN with Noisy Nets and multi-step returns.
    Performs soft target network updates.
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.policy_net = DuelingDQN(state_dim, action_dim, hidden_sizes=CONFIG["hidden_sizes"]).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_sizes=CONFIG["hidden_sizes"]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=CONFIG["lr"], amsgrad=True)
        self.memory = PrioritizedReplayMemory(CONFIG["memory_capacity"])

        self.beta = CONFIG["beta_start"]
        self.beta_inc = (CONFIG["beta_end"] - CONFIG["beta_start"]) / CONFIG["num_episodes"]
        self.n_step_buffer: Deque[Tuple] = deque(maxlen=CONFIG["n_steps"])
        self.grad_clip = CONFIG["grad_clip"]

        self.stats: Dict[str, list] = {"loss": [], "q_values": [], "td_errors": []}

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using the policy network with NoisyNet exploration."""
        self.policy_net.reset_noise()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = q_values.max(1)[1].item()
        return action

    def push_n_step(self, state: np.ndarray, action: int, reward: float,
                    next_state: np.ndarray, done: bool) -> None:
        """Push an experience onto the multi-step buffer and compute the multi-step return."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < CONFIG["n_steps"]:
            return
        R = sum([self.n_step_buffer[i][2] * (CONFIG["gamma"] ** i) for i in range(CONFIG["n_steps"])])
        state0, action0, _, _, _ = self.n_step_buffer[0]
        last_next_state, done_last = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        self.memory.push(state0, action0, last_next_state, R, done_last)

    def flush_n_step(self) -> None:
        """Flush any remaining experiences in the multi-step buffer at episode end."""
        while self.n_step_buffer:
            n = len(self.n_step_buffer)
            R = sum([self.n_step_buffer[i][2] * (CONFIG["gamma"] ** i) for i in range(n)])
            state0, action0, _, _, _ = self.n_step_buffer[0]
            last_next_state, done_last = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            self.memory.push(state0, action0, last_next_state, R, done_last)
            self.n_step_buffer.popleft()

    def optimize_model(self) -> Optional[float]:
        if len(self.memory) < CONFIG["batch_size"]:
            return None
        transitions, indices, weights = self.memory.sample(CONFIG["batch_size"], self.beta)
        self.beta = min(CONFIG["beta_end"], self.beta + self.beta_inc)

        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool, device=device)
        weights = weights.to(device)

        current_q = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_actions = self.policy_net(state_batch).max(1)[1]
            next_q = self.target_net(state_batch).gather(1, next_actions.unsqueeze(1))
            target_q = reward_batch + (1 - done_batch.float()) * CONFIG["gamma"] * next_q.squeeze()

        loss = F.huber_loss(current_q.squeeze(), target_q, reduction='none')
        loss = (weights * loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        td_errors = (current_q.squeeze() - target_q).abs()
        self.memory.update_priorities(indices, td_errors + 1e-5)

        self.stats["loss"].append(loss.item())
        self.stats["q_values"].append(current_q.mean().item())
        self.stats["td_errors"].append(td_errors.mean().item())

        return loss.item()

    def update_target(self) -> None:
        """Soft update of the target network."""
        tau = CONFIG["tau"]
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

# --- Utility: Update Optimizer Learning Rate ---
def update_optimizer_lr(optimizer, new_lr):
    """Update the optimizer's learning rate."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# --- Game Class ---
class Game:
    """
    The game environment with custom reward logic, visualization, and obstacle spawning.
    """
    def __init__(self, agent: DQNAgent, writer: SummaryWriter) -> None:
        pygame.init()
        self.agent = agent
        self.writer = writer
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font = pygame.font.SysFont("Arial", 20)
        pygame.display.set_caption("Advanced Snake AI")
        self.clock = pygame.time.Clock()
        self.episode_rewards: List[float] = []

    def run_episode(self, episode: int) -> int:
        """Run one game episode and return the score."""
        snake = Snake()  # Deterministic start
        obstacles: List[Obstacle] = []
        food = Food(snake, obstacles)
        score = 0
        state = get_state(snake, food, obstacles)
        done = False
        episode_start = pygame.time.get_ticks()
        steps = 0
        prev_distance = manhattan_distance(snake.body[0], food.position)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            current_time = (pygame.time.get_ticks() - episode_start) / 1000.0
            if steps >= CONFIG["max_steps"]:
                done = True

            # Spawn obstacles if past obstacle_start episode and after obstacle_phase_interval
            if episode >= CONFIG["obstacle_start"] and current_time > CONFIG["obstacle_phase_interval"]:
                phase = min(int((current_time - CONFIG["obstacle_phase_interval"]) // CONFIG["obstacle_phase_interval"]), 4)
                lifetime = CONFIG["obstacle_base_lifetime"] + CONFIG["obstacle_lifetime_increment"] * phase
                if len(obstacles) < CONFIG["obstacle_base_count"] + phase and random.random() < CONFIG["obstacle_spawn_prob"]:
                    occupied = set(snake.body) | {food.position} | {obs.position for obs in obstacles}
                    new_pos = random_grid_position(occupied)
                    obstacles.append(Obstacle(new_pos, current_time, lifetime))

            action = self.agent.select_action(state)
            snake.direction = get_new_direction(snake.direction, action)
            snake.update()
            steps += 1

            new_distance = manhattan_distance(snake.body[0], food.position)
            shaping_reward = 0.05 * (prev_distance - new_distance)
            prev_distance = new_distance
            reward = -0.2 + shaping_reward

            if snake.collision_with_self() or any(snake.body[0] == obs.position for obs in obstacles):
                reward = -20
                done = True

            if snake.body[0] == food.position:
                snake.grow()
                score += 1
                reward = 10 + len(snake.body) * 0.5
                food = Food(snake, obstacles)
                prev_distance = manhattan_distance(snake.body[0], food.position)

            obstacles = [obs for obs in obstacles if not obs.expired(current_time)]
            next_state = get_state(snake, food, obstacles)
            self.agent.push_n_step(state, action, reward, next_state, done)
            state = next_state

            loss = self.agent.optimize_model()
            if loss is not None and steps % 50 == 0:
                self.writer.add_scalar("Loss/step", loss, episode * 1000 + steps)

            self.screen.fill(BLACK)
            snake.draw(self.screen)
            food.draw(self.screen)
            for obs in obstacles:
                obs.draw(self.screen)
            info_text = f"Episode: {episode}  Score: {score}  Steps: {steps}"
            info = self.font.render(info_text, True, WHITE)
            self.screen.blit(info, (10, 10))
            pygame.display.flip()
            self.clock.tick(FPS)

        self.agent.flush_n_step()
        return score

# --- Checkpoint Functions ---
def save_checkpoint(agent: DQNAgent, episode: int, score: float, path: str) -> None:
    """Save model checkpoint."""
    torch.save({
        "episode": episode,
        "policy_state": agent.policy_net.state_dict(),
        "target_state": agent.target_net.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "memory": agent.memory,
        "stats": agent.stats,
        "score": score
    }, path)

def load_checkpoint(agent: DQNAgent, path: str) -> None:
    if os.path.exists(path):
        # Salli sekä PrioritizedReplayMemory että Transition ladattaviksi
        with torch.serialization.safe_globals([PrioritizedReplayMemory, Transition]):
            checkpoint = torch.load(path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint["policy_state"])
        agent.target_net.load_state_dict(checkpoint["target_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.memory = checkpoint["memory"]
        agent.stats = checkpoint["stats"]
        logging.info(f"Checkpoint loaded: {path}")
    else:
        logging.info("No checkpoint found; using default initialization.")


# --- Training Thread ---
def training_thread():
    """
    The training thread continuously runs episodes, reading the latest CONFIG
    (updated live by the GUI or remote API) so that changes take effect immediately.
    """
    with config_lock:
        initial_config = CONFIG.copy()
    seed_value = int(initial_config.get("seed", 42))
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    logging.info(f"Random seed set to: {seed_value}")

    writer = SummaryWriter(log_dir=initial_config["log_dir"])
    state_dim = 12  # 8 features + 4 one-hot direction
    action_dim = 3
    agent = DQNAgent(state_dim, action_dim)
    checkpoint_dir = os.path.join(initial_config["log_dir"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    load_checkpoint(agent, best_model_path)

    game = Game(agent, writer)
    best_score = -np.inf
    episode = 0

    logging.info("Training started. Close the Pygame window or press Ctrl+C to stop.")
    try:
        while True:
            with config_lock:
                current_config = CONFIG.copy()
            update_optimizer_lr(agent.optimizer, current_config["lr"])
            score = game.run_episode(episode)
            writer.add_scalar("Training/Score", score, episode)
            if agent.stats["loss"]:
                writer.add_scalar("Training/Loss", np.mean(agent.stats["loss"][-100:]), episode)
                writer.add_scalar("Training/Q_Value", np.mean(agent.stats["q_values"][-100:]), episode)
            with stats_lock:
                global_stats["episodes"].append(episode)
                global_stats["loss"].append(np.mean(agent.stats["loss"][-100:]) if agent.stats["loss"] else 0)
                global_stats["q_values"].append(np.mean(agent.stats["q_values"][-100:]) if agent.stats["q_values"] else 0)
                global_stats["td_errors"].append(np.mean(agent.stats["td_errors"][-100:]) if agent.stats["td_errors"] else 0)
            if score > best_score:
                best_score = score
                save_checkpoint(agent, episode, score, os.path.join(checkpoint_dir, "best_model.pth"))
            if episode % current_config["checkpoint_interval"] == 0:
                save_checkpoint(agent, episode, score, os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth"))
            agent.update_target()
            episode += 1
    except Exception as e:
        logging.exception("Exception in training thread:")
    finally:
        writer.close()
        pygame.quit()

# --- Live CONFIG Updater (for GUI and Remote API) ---
def update_live_config(key, value):
    """Update the global CONFIG with a new value."""
    with config_lock:
        CONFIG[key] = value
    logging.info(f"Updated {key} to {value}")

# --- Chart Widget (Matplotlib in PyQt) ---
class ChartWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax_loss = self.figure.add_subplot(311)
        self.ax_q = self.figure.add_subplot(312)
        self.ax_td = self.figure.add_subplot(313)
        self.figure.tight_layout()

    def update_charts(self, stats):
        self.ax_loss.clear()
        self.ax_q.clear()
        self.ax_td.clear()
        self.ax_loss.plot(stats["episodes"], stats["loss"], label="Loss", color="red")
        self.ax_loss.set_title("Loss")
        self.ax_loss.legend()
        self.ax_q.plot(stats["episodes"], stats["q_values"], label="Q Value", color="orange")
        self.ax_q.set_title("Q Value")
        self.ax_q.legend()
        self.ax_td.plot(stats["episodes"], stats["td_errors"], label="TD Error", color="green")
        self.ax_td.set_title("TD Error")
        self.ax_td.legend()
        self.canvas.draw()

# --- PyQt5 GUI Main Window ---
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake AI - Parameter Control")
        self.resize(600, 800)
        self.training_started = False
        self.initUI()
        self.applyDarkStyle()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateCharts)
        self.timer.start(1000)  # Update charts every second

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        formLayout = QtWidgets.QFormLayout()

        # Parameter widgets with live update callbacks:
        self.lrSpin = QtWidgets.QDoubleSpinBox()
        self.lrSpin.setRange(1e-6, 1)
        self.lrSpin.setDecimals(6)
        self.lrSpin.setValue(default_params["lr"])
        self.lrSpin.valueChanged.connect(lambda val: update_live_config("lr", val))
        formLayout.addRow("Learning Rate:", self.lrSpin)

        self.batchSizeSpin = QtWidgets.QSpinBox()
        self.batchSizeSpin.setRange(1, 10000)
        self.batchSizeSpin.setValue(default_params["batch_size"])
        self.batchSizeSpin.valueChanged.connect(lambda val: update_live_config("batch_size", val))
        formLayout.addRow("Batch Size:", self.batchSizeSpin)

        self.memorySpin = QtWidgets.QSpinBox()
        self.memorySpin.setRange(1, 1000000)
        self.memorySpin.setValue(default_params["memory_capacity"])
        self.memorySpin.valueChanged.connect(lambda val: update_live_config("memory_capacity", val))
        formLayout.addRow("Memory Capacity:", self.memorySpin)

        self.seedSpin = QtWidgets.QSpinBox()
        self.seedSpin.setRange(0, 100000)
        self.seedSpin.setValue(default_params["seed"])
        self.seedSpin.valueChanged.connect(lambda val: update_live_config("seed", val))
        formLayout.addRow("Random Seed:", self.seedSpin)

        self.logDirEdit = QtWidgets.QLineEdit(default_params["log_dir"])
        self.logDirEdit.textChanged.connect(lambda text: update_live_config("log_dir", text))
        formLayout.addRow("Log Directory:", self.logDirEdit)

        self.gammaSpin = QtWidgets.QDoubleSpinBox()
        self.gammaSpin.setRange(0.0, 1.0)
        self.gammaSpin.setDecimals(2)
        self.gammaSpin.setValue(default_params["gamma"])
        self.gammaSpin.valueChanged.connect(lambda val: update_live_config("gamma", val))
        formLayout.addRow("Gamma:", self.gammaSpin)

        self.tauSpin = QtWidgets.QDoubleSpinBox()
        self.tauSpin.setRange(0.0, 1.0)
        self.tauSpin.setDecimals(3)
        self.tauSpin.setValue(default_params["tau"])
        self.tauSpin.valueChanged.connect(lambda val: update_live_config("tau", val))
        formLayout.addRow("Tau:", self.tauSpin)

        self.nStepsSpin = QtWidgets.QSpinBox()
        self.nStepsSpin.setRange(1, 10)
        self.nStepsSpin.setValue(default_params["n_steps"])
        self.nStepsSpin.valueChanged.connect(lambda val: update_live_config("n_steps", val))
        formLayout.addRow("N Steps:", self.nStepsSpin)

        self.betaStartSpin = QtWidgets.QDoubleSpinBox()
        self.betaStartSpin.setRange(0.0, 1.0)
        self.betaStartSpin.setDecimals(2)
        self.betaStartSpin.setValue(default_params["beta_start"])
        self.betaStartSpin.valueChanged.connect(lambda val: update_live_config("beta_start", val))
        formLayout.addRow("Beta Start:", self.betaStartSpin)

        self.betaEndSpin = QtWidgets.QDoubleSpinBox()
        self.betaEndSpin.setRange(0.0, 1.0)
        self.betaEndSpin.setDecimals(2)
        self.betaEndSpin.setValue(default_params["beta_end"])
        self.betaEndSpin.valueChanged.connect(lambda val: update_live_config("beta_end", val))
        formLayout.addRow("Beta End:", self.betaEndSpin)

        self.gradClipSpin = QtWidgets.QDoubleSpinBox()
        self.gradClipSpin.setRange(0.0, 100.0)
        self.gradClipSpin.setDecimals(2)
        self.gradClipSpin.setValue(default_params["grad_clip"])
        self.gradClipSpin.valueChanged.connect(lambda val: update_live_config("grad_clip", val))
        formLayout.addRow("Grad Clip:", self.gradClipSpin)

        self.maxStepsSpin = QtWidgets.QSpinBox()
        self.maxStepsSpin.setRange(1, 10000)
        self.maxStepsSpin.setValue(default_params["max_steps"])
        self.maxStepsSpin.valueChanged.connect(lambda val: update_live_config("max_steps", val))
        formLayout.addRow("Max Steps per Game:", self.maxStepsSpin)

        self.obstacleStartSpin = QtWidgets.QSpinBox()
        self.obstacleStartSpin.setRange(1, 10000)
        self.obstacleStartSpin.setValue(default_params["obstacle_start"])
        self.obstacleStartSpin.valueChanged.connect(lambda val: update_live_config("obstacle_start", val))
        formLayout.addRow("Obstacle Start Episode:", self.obstacleStartSpin)

        self.obstacleSpawnProbSpin = QtWidgets.QDoubleSpinBox()
        self.obstacleSpawnProbSpin.setRange(0.0, 1.0)
        self.obstacleSpawnProbSpin.setDecimals(2)
        self.obstacleSpawnProbSpin.setValue(default_params["obstacle_spawn_prob"])
        self.obstacleSpawnProbSpin.valueChanged.connect(lambda val: update_live_config("obstacle_spawn_prob", val))
        formLayout.addRow("Obstacle Spawn Prob:", self.obstacleSpawnProbSpin)

        self.obstacleBaseCountSpin = QtWidgets.QSpinBox()
        self.obstacleBaseCountSpin.setRange(1, 100)
        self.obstacleBaseCountSpin.setValue(default_params["obstacle_base_count"])
        self.obstacleBaseCountSpin.valueChanged.connect(lambda val: update_live_config("obstacle_base_count", val))
        formLayout.addRow("Obstacle Base Count:", self.obstacleBaseCountSpin)

        self.obstacleBaseLifetimeSpin = QtWidgets.QSpinBox()
        self.obstacleBaseLifetimeSpin.setRange(1, 300)
        self.obstacleBaseLifetimeSpin.setValue(default_params["obstacle_base_lifetime"])
        self.obstacleBaseLifetimeSpin.valueChanged.connect(lambda val: update_live_config("obstacle_base_lifetime", val))
        formLayout.addRow("Obstacle Base Lifetime (s):", self.obstacleBaseLifetimeSpin)

        self.obstacleLifetimeIncrementSpin = QtWidgets.QSpinBox()
        self.obstacleLifetimeIncrementSpin.setRange(0, 100)
        self.obstacleLifetimeIncrementSpin.setValue(default_params["obstacle_lifetime_increment"])
        self.obstacleLifetimeIncrementSpin.valueChanged.connect(lambda val: update_live_config("obstacle_lifetime_increment", val))
        formLayout.addRow("Obstacle Lifetime Increment (s):", self.obstacleLifetimeIncrementSpin)

        self.obstaclePhaseIntervalSpin = QtWidgets.QSpinBox()
        self.obstaclePhaseIntervalSpin.setRange(1, 300)
        self.obstaclePhaseIntervalSpin.setValue(default_params["obstacle_phase_interval"])
        self.obstaclePhaseIntervalSpin.valueChanged.connect(lambda val: update_live_config("obstacle_phase_interval", val))
        formLayout.addRow("Obstacle Phase Interval (s):", self.obstaclePhaseIntervalSpin)

        layout.addLayout(formLayout)

        self.startButton = QtWidgets.QPushButton("Start Training")
        self.startButton.clicked.connect(self.startTraining)
        layout.addWidget(self.startButton)

        self.chartWidget = ChartWidget()
        layout.addWidget(self.chartWidget)

        self.setLayout(layout)

    def startTraining(self):
        if not self.training_started:
            self.training_started = True
            training_thread_obj = threading.Thread(target=training_thread, daemon=True)
            training_thread_obj.start()
            QtWidgets.QMessageBox.information(self, "Info", "Training started and will run continuously.\nYou can change parameters in real time.")
            self.startButton.setEnabled(False)
        else:
            QtWidgets.QMessageBox.information(self, "Info", "Training is already running.")

    def updateCharts(self):
        with stats_lock:
            stats_copy = {
                "episodes": list(global_stats["episodes"]),
                "loss": list(global_stats["loss"]),
                "q_values": list(global_stats["q_values"]),
                "td_errors": list(global_stats["td_errors"]),
            }
        self.chartWidget.update_charts(stats_copy)

    def applyDarkStyle(self):
        darkStyle = """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: Arial;
            font-size: 14px;
        }
        QLineEdit, QDoubleSpinBox, QSpinBox {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            padding: 4px;
            border-radius: 4px;
        }
        QPushButton {
            background-color: #007ACC;
            border: none;
            padding: 6px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #005999;
        }
        """
        self.setStyleSheet(darkStyle)

# --- Remote Control API using Flask ---
app = Flask(__name__)

@app.route('/update_config', methods=['POST'])
def update_config():
    """
    Expects JSON with 'key' and 'value' to update the CONFIG.
    """
    data = request.json
    key = data.get('key')
    value = data.get('value')
    if key is None or value is None:
        return jsonify({'error': 'Invalid parameters'}), 400
    update_live_config(key, value)
    return jsonify({'message': f'Updated {key} to {value}'}), 200

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Return the current global statistics in JSON format."""
    with stats_lock:
        stats_copy = {
            "episodes": list(global_stats["episodes"]),
            "loss": list(global_stats["loss"]),
            "q_values": list(global_stats["q_values"]),
            "td_errors": list(global_stats["td_errors"])
        }
    return jsonify(stats_copy), 200

# Global flag to track if training has started via the API.
training_thread_started = False

@app.route('/start_training', methods=['POST'])
def start_training_api():
    """Start the training thread via the API."""
    global training_thread_started
    if not training_thread_started:
        training_thread_obj = threading.Thread(target=training_thread, daemon=True)
        training_thread_obj.start()
        training_thread_started = True
        return jsonify({'message': 'Training started'}), 200
    else:
        return jsonify({'message': 'Training is already running'}), 200

def run_remote_server():
    """Run the Flask remote control server."""
    app.run(host='0.0.0.0', port=5000)

# --- Main GUI Function ---
def main_gui():
    app_qt = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app_qt.exec_())

# --- Main Entry Point ---
if __name__ == "__main__":
    # Tarkistetaan, käynnistetäänkö etätilassa (remote mode) komentorivin parametrilla.
    if '--remote' in sys.argv:
        # Käynnistetään ensin Flask-palvelin etäohjausta varten.
        remote_thread = threading.Thread(target=run_remote_server, daemon=True)
        remote_thread.start()
        logging.info("Remote control server started on port 5000.")
        # Käynnistetään koulutus (ilman graafista käyttöliittymää).
        training_thread()
    else:
        # Käynnistetään GUI-tilassa.
        main_gui()
