from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
import numpy as np

Position = Tuple[int, int]


@dataclass
class EnvConfig:
    rows: int = 24
    cols: int = 24
    max_steps: int = 400

    # Rewards / penalties
    goal_reward: float = 15.0
    trap_penalty: float = -10.0
    coin_reward: float = 2.0
    invalid_move_penalty: float = -1.0
    step_penalty: float = -0.1
    timeout_penalty: float = -5.0


@dataclass
class MapConfig:
    seed: int = 2026
    wall_density: float = 0.22   # 18% walls
    trap_density: float = 0.07   # 5% traps
    num_coins: int = 6           # more “treasure” on the map
    max_attempts: int = 300      # regenerate if no valid path


class GridWorldTreasureEnv:
    """
    24x24 Treasure Hunt environment for tabular Q-learning.

    State = (row, col, coin_mask)
    - coin_mask is a bitmask of collected coins (len(coins) <= 10 recommended)
    """

    ACTIONS = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        map_config: Optional[MapConfig] = None,
        map_snapshot: Optional[Dict] = None,
    ):
        self.cfg = config or EnvConfig()
        self.mcfg = map_config or MapConfig()

        self.start_pos: Position = (0, 0)
        self.goal_pos: Position = (self.cfg.rows - 1, self.cfg.cols - 1)

        self.walls: Set[Position] = set()
        self.traps: Set[Position] = set()
        self.coins: List[Position] = []

        if map_snapshot is not None:
            self._load_map_snapshot(map_snapshot)
        else:
            self._generate_map()

        self.num_coin_states = 2 ** len(self.coins)
        self.n_actions = 4
        self.n_states = self.cfg.rows * self.cfg.cols * self.num_coin_states

        # Runtime
        self.agent_pos: Position = self.start_pos
        self.coin_mask: int = 0
        self.steps: int = 0
        self.total_reward: float = 0.0

    # ---------------- Map generation / loading ----------------
    def _load_map_snapshot(self, snap: Dict) -> None:
        self.start_pos = tuple(snap["start_pos"])
        self.goal_pos = tuple(snap["goal_pos"])
        self.walls = {tuple(x) for x in snap["walls"]}
        self.traps = {tuple(x) for x in snap["traps"]}
        self.coins = [tuple(x) for x in snap["coins"]]

        # sanity
        if self.start_pos in self.walls or self.goal_pos in self.walls:
            raise ValueError("Invalid snapshot: start/goal in walls.")
        if self.start_pos in self.traps or self.goal_pos in self.traps:
            raise ValueError("Invalid snapshot: start/goal in traps.")

    def _generate_map(self) -> None:
        rng = np.random.default_rng(self.mcfg.seed)

        rows, cols = self.cfg.rows, self.cfg.cols
        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        forbidden = {self.start_pos, self.goal_pos}

        n_total = rows * cols
        n_walls = int(n_total * self.mcfg.wall_density)
        n_traps = int(n_total * self.mcfg.trap_density)
        n_coins = int(self.mcfg.num_coins)

        for _ in range(self.mcfg.max_attempts):
            self.walls.clear()
            self.traps.clear()
            self.coins.clear()

            candidates = [p for p in all_cells if p not in forbidden]
            rng.shuffle(candidates)

            # walls
            self.walls = set(candidates[:n_walls])

            remaining = [p for p in candidates[n_walls:] if p not in self.walls]
            rng.shuffle(remaining)

            # traps
            self.traps = set(remaining[:n_traps])

            remaining2 = [p for p in remaining[n_traps:] if p not in self.traps]
            rng.shuffle(remaining2)

            # coins
            self.coins = remaining2[:n_coins]

            # validate safe path from start to goal (avoid walls+traps)
            if self._path_exists(self.start_pos, self.goal_pos, avoid_traps=True):
                # optional: ensure coins are reachable safely
                if all(self._path_exists(self.start_pos, coin, avoid_traps=True) for coin in self.coins):
                    return

        raise RuntimeError("Failed to generate solvable map. Reduce densities or increase max_attempts.")

    def _path_exists(self, src: Position, dst: Position, avoid_traps: bool = True) -> bool:
        blocked = set(self.walls)
        if avoid_traps:
            blocked |= set(self.traps)

        q = deque([src])
        visited = {src}
        while q:
            r, c = q.popleft()
            if (r, c) == dst:
                return True
            for dr, dc in self.ACTIONS.values():
                nr, nc = r + dr, c + dc
                np_ = (nr, nc)
                if 0 <= nr < self.cfg.rows and 0 <= nc < self.cfg.cols and np_ not in visited and np_ not in blocked:
                    visited.add(np_)
                    q.append(np_)
        return False

    # ---------------- Core API ----------------
    def reset(self) -> int:
        self.agent_pos = self.start_pos
        self.coin_mask = 0
        self.steps = 0
        self.total_reward = 0.0
        return self._get_state_index()

    def step(self, action: int):
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}")

        self.steps += 1
        dr, dc = self.ACTIONS[action]
        next_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        reward = 0.0
        done = False
        invalid_move = False
        event = "move"

        # invalid: out of bounds or hit wall
        if (not self._inside_grid(next_pos)) or (next_pos in self.walls):
            invalid_move = True
            reward += self.cfg.invalid_move_penalty
            event = "invalid_move"
        else:
            self.agent_pos = next_pos
            reward += self.cfg.step_penalty

            # coin collection
            coin_idx = self._coin_index_at(self.agent_pos)
            if coin_idx >= 0:
                already = (self.coin_mask >> coin_idx) & 1
                if not already:
                    self.coin_mask |= (1 << coin_idx)
                    reward += self.cfg.coin_reward
                    event = "coin"

            # trap / goal
            if self.agent_pos in self.traps:
                reward = self.cfg.trap_penalty
                done = True
                event = "trap"
            elif self.agent_pos == self.goal_pos:
                reward = self.cfg.goal_reward
                done = True
                event = "goal"

        # timeout
        if (not done) and self.steps >= self.cfg.max_steps:
            reward = self.cfg.timeout_penalty
            done = True
            event = "timeout"

        self.total_reward += reward
        next_state = self._get_state_index()

        info = {
            "position": self.agent_pos,
            "coin_mask": self.coin_mask,
            "steps": self.steps,
            "event": event,
            "invalid_move": invalid_move,
            "action_name": self.ACTION_NAMES[action],
            "total_reward": self.total_reward,
            "success": done and (self.agent_pos == self.goal_pos),
        }
        return next_state, reward, done, info

    # ---------------- State helpers ----------------
    def _inside_grid(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.cfg.rows and 0 <= c < self.cfg.cols

    def _coin_index_at(self, pos: Position) -> int:
        try:
            return self.coins.index(pos)
        except ValueError:
            return -1

    def _get_state_index(self) -> int:
        r, c = self.agent_pos
        m = self.coin_mask
        return (r * self.cfg.cols + c) * self.num_coin_states + m

    def decode_state_index(self, state_idx: int) -> Tuple[int, int, int]:
        pos_index, mask = divmod(state_idx, self.num_coin_states)
        r, c = divmod(pos_index, self.cfg.cols)
        return r, c, mask

    def action_name(self, action: int) -> str:
        return self.ACTION_NAMES.get(action, f"A{action}")

    def is_coin_collected(self, coin_idx: int) -> bool:
        return bool((self.coin_mask >> coin_idx) & 1)

    def get_map_snapshot(self) -> Dict:
        return {
            "rows": self.cfg.rows,
            "cols": self.cfg.cols,
            "start_pos": list(self.start_pos),
            "goal_pos": list(self.goal_pos),
            "walls": [list(p) for p in sorted(self.walls)],
            "traps": [list(p) for p in sorted(self.traps)],
            "coins": [list(p) for p in self.coins],
            "max_steps": self.cfg.max_steps,
            "seed": self.mcfg.seed,
            "wall_density": self.mcfg.wall_density,
            "trap_density": self.mcfg.trap_density,
            "num_coins": self.mcfg.num_coins,
        }