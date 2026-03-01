from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageTk

from environment import GridWorldTreasureEnv, EnvConfig, MapConfig

Position = Tuple[int, int]


class TreasureGameUI:
    TILE_SIZE = 32
    PADDING = 10

    BG_MAIN = "#0f172a"
    BG_PANEL = "#111827"
    GRID_LINE = "#334155"
    TEXT_MAIN = "#e5e7eb"
    TEXT_MUTED = "#94a3b8"
    ACCENT = "#a78bfa"

    KEY_TO_ACTION = {
        "Up": 0, "Down": 1, "Left": 2, "Right": 3,
        "w": 0, "s": 1, "a": 2, "d": 3,
        "W": 0, "S": 1, "A": 2, "D": 3,
    }

    def __init__(self, master: tk.Tk, results_dir: str = "results"):
        self.master = master
        self.master.title("Treasure Hunt RL Demo (24x24)")
        self.master.configure(bg=self.BG_MAIN)

        self.results_dir = Path(results_dir)
        self.map_snapshot = self._try_load_map_snapshot()
        self.env = self._build_env_from_snapshot_or_seed()

        self.q_table = self._load_q_table(self.results_dir / "q_table.npy")

        self.auto_running = False
        self.after_id = None
        self.manual_mode = False

        self.current_state = self.env.reset()
        self.last_reward = 0.0
        self.last_info = {"event": "reset", "action_name": "-", "success": False}

        # trail (for nicer demo)
        self.path_trail: List[Position] = [self.env.agent_pos]
        self.trail_ids: List[int] = []

        # Load tiles (use your clean images directly)
        self.tiles = self._load_tiles(Path("assets/tiles"), self.TILE_SIZE)

        # Pre-assign variants for walls/traps so they look consistent
        self.variant_rng = np.random.default_rng(2026)
        self.wall_variant: Dict[Position, str] = {}
        self.trap_variant: Dict[Position, str] = {}
        for p in self.env.walls:
            self.wall_variant[p] = "wall1" if self.variant_rng.random() < 0.5 else "wall2"
        for p in self.env.traps:
            self.trap_variant[p] = "trap1" if self.variant_rng.random() < 0.5 else "trap2"

        self._build_ui()
        self._bind_keys()

        self._build_grid_items()
        self._update_agent()
        self.update_status()

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- load env / results ----------------
    def _try_load_map_snapshot(self):
        p = self.results_dir / "map_snapshot.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _build_env_from_snapshot_or_seed(self) -> GridWorldTreasureEnv:
        if self.map_snapshot is not None:
            return GridWorldTreasureEnv(
                config=EnvConfig(rows=24, cols=24, max_steps=400),
                map_config=MapConfig(seed=123, wall_density=0.18, trap_density=0.05, num_coins=6),
                map_snapshot=self.map_snapshot,
            )
        # fallback (still deterministic)
        return GridWorldTreasureEnv(
            config=EnvConfig(rows=24, cols=24, max_steps=400),
            map_config=MapConfig(seed=123, wall_density=0.18, trap_density=0.05, num_coins=6),
        )

    def _load_q_table(self, path: Path) -> np.ndarray:
        if not path.exists():
            messagebox.showwarning(
                "Q-table not found",
                f"Cannot find {path}.\nRun train.py first.\nUI will use an all-zero table."
            )
            return np.zeros((self.env.n_states, self.env.n_actions), dtype=np.float32)

        q = np.load(path)
        expected = (self.env.n_states, self.env.n_actions)
        if q.shape != expected:
            messagebox.showwarning(
                "Q-table mismatch",
                f"Expected {expected}, got {q.shape}.\nPlease re-run train.py."
            )
            return np.zeros((self.env.n_states, self.env.n_actions), dtype=np.float32)
        return q.astype(np.float32)

    # ---------------- policy ----------------
    def greedy_action(self, state: int) -> int:
        qv = self.q_table[state]
        max_q = np.max(qv)
        best = np.flatnonzero(np.isclose(qv, max_q))
        return int(np.random.choice(best))

    # ---------------- tiles ----------------
    def _load_tiles(self, tile_dir: Path, tile_size: int) -> Dict[str, ImageTk.PhotoImage]:
        """
        Supports both:
          floor.png / floor_clean.png
          wall_arch_clean.png etc.
        """
        tile_dir.mkdir(parents=True, exist_ok=True)

        def load_one(key: str, candidates: List[str], fallback_rgba=(80, 80, 80, 255)):
            for name in candidates:
                p = tile_dir / name
                if p.exists():
                    im = Image.open(p).convert("RGBA").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    return ImageTk.PhotoImage(im)
            # fallback
            im = Image.new("RGBA", (tile_size, tile_size), fallback_rgba)
            return ImageTk.PhotoImage(im)

        tiles = {
            "floor": load_one("floor", ["floor.png", "floor_clean.png"], (40, 50, 70, 255)),
            "start": load_one("start", ["start.png", "start_clean.png"], (50, 90, 150, 255)),
            "goal": load_one("goal", ["goal.png", "goal_clean.png"], (160, 120, 50, 255)),
            "player": load_one("player", ["player.png", "player_clean.png"], (70, 140, 220, 255)),
            "treasure": load_one("treasure", ["treasure.png", "treasure_clean.png"], (160, 120, 40, 255)),
            "wall1": load_one("wall1", ["wall_arch.png", "wall_arch_clean.png"], (120, 120, 120, 255)),
            "wall2": load_one("wall2", ["wall_bricks.png", "wall_bricks_clean.png"], (110, 110, 110, 255)),
            "trap1": load_one("trap1", ["trap_blade.png", "trap_blade_clean.png"], (180, 60, 60, 255)),
            "trap2": load_one("trap2", ["trap_spikes.png", "trap_spikes_clean.png"], (180, 60, 60, 255)),
        }

        # quick check
        missing = []
        for k, im in tiles.items():
            # can't easily check PhotoImage origin; just warn if directory empty
            pass
        return tiles

    # ---------------- UI ----------------
    def _build_ui(self):
        self.master.grid_columnconfigure(0, weight=0)
        self.master.grid_columnconfigure(1, weight=0)

        canvas_w = self.env.cfg.cols * self.TILE_SIZE + 2 * self.PADDING
        canvas_h = self.env.cfg.rows * self.TILE_SIZE + 2 * self.PADDING

        left = tk.Frame(self.master, bg=self.BG_MAIN)
        left.grid(row=0, column=0, sticky="n", padx=(12, 8), pady=12)

        tk.Label(
            left, text="🗺️  24×24 Dungeon Treasure Hunt",
            bg=self.BG_MAIN, fg=self.TEXT_MAIN, font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(
            left, width=canvas_w, height=canvas_h,
            bg=self.BG_PANEL, highlightthickness=1, highlightbackground=self.GRID_LINE, bd=0
        )
        self.canvas.pack()

        tk.Label(
            left,
            text="Keyboard: ↑↓←→ / WASD | R: Reset | Space: Auto Play / Stop",
            bg=self.BG_MAIN, fg=self.TEXT_MUTED, font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(8, 0))

        right = tk.Frame(self.master, bg=self.BG_MAIN)
        right.grid(row=0, column=1, sticky="n", padx=(8, 12), pady=12)

        self.mode_var = tk.StringVar(value="Mode: Greedy AI Demo")
        self.speed_ms_var = tk.IntVar(value=140)

        self.status_vars = {
            "result": tk.StringVar(),
            "step": tk.StringVar(),
            "pos": tk.StringVar(),
            "action": tk.StringVar(),
            "event": tk.StringVar(),
            "reward": tk.StringVar(),
            "total": tk.StringVar(),
            "coins": tk.StringVar(),
        }

        hud = tk.Frame(right, bg=self.BG_PANEL, highlightthickness=1, highlightbackground=self.GRID_LINE)
        hud.pack(fill="x", pady=(0, 10))

        tk.Label(hud, text="🎮 HUD", bg=self.BG_PANEL, fg=self.TEXT_MAIN,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        tk.Label(hud, textvariable=self.mode_var, bg=self.BG_PANEL, fg=self.ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(0, 8))

        for k in ["result", "step", "pos", "action", "event", "reward", "total", "coins"]:
            tk.Label(hud, textvariable=self.status_vars[k], bg=self.BG_PANEL, fg=self.TEXT_MAIN,
                     font=("Consolas", 10), anchor="w").pack(fill="x", padx=10, pady=1)

        ctrl = tk.Frame(right, bg=self.BG_PANEL, highlightthickness=1, highlightbackground=self.GRID_LINE)
        ctrl.pack(fill="x", pady=(0, 10))

        tk.Label(ctrl, text="⚙️ Controls", bg=self.BG_PANEL, fg=self.TEXT_MAIN,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(8, 6))

        btns = tk.Frame(ctrl, bg=self.BG_PANEL)
        btns.pack(fill="x", padx=10, pady=(0, 8))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        self._btn(btns, "↻ Reset", self.reset_episode).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        self._btn(btns, "⏭ Step (AI)", self.step_once).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        self._btn(btns, "▶ Auto", self.start_auto).grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        self._btn(btns, "⏹ Stop", self.stop_auto).grid(row=1, column=1, sticky="ew", padx=3, pady=3)

        sp = tk.Frame(ctrl, bg=self.BG_PANEL)
        sp.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(sp, text="Auto Speed (ms/step)", bg=self.BG_PANEL, fg=self.TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")
        tk.Scale(
            sp, from_=60, to=600, orient="horizontal", resolution=10, length=240,
            variable=self.speed_ms_var, bg=self.BG_PANEL, fg=self.TEXT_MAIN,
            troughcolor="#1f2937", highlightthickness=0
        ).pack(anchor="w")

        log = tk.Frame(right, bg=self.BG_PANEL, highlightthickness=1, highlightbackground=self.GRID_LINE)
        log.pack(fill="both", expand=True)

        tk.Label(log, text="📜 Event Log", bg=self.BG_PANEL, fg=self.TEXT_MAIN,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(8, 6))

        self.log_text = tk.Text(
            log, height=12, width=42, bg="#1f2937", fg=self.TEXT_MAIN,
            insertbackground=self.TEXT_MAIN, relief="flat", wrap="word",
            font=("Consolas", 9), padx=8, pady=8
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_text.configure(state="disabled")
        self._log("UI ready. Run train.py first. Then Auto/Step. Arrow keys/WASD for manual moves.")

    def _btn(self, parent, text, cmd):
        return tk.Button(
            parent, text=text, command=cmd,
            bg="#374151", fg="#f9fafb", activebackground="#4b5563",
            relief="flat", bd=0, padx=10, pady=8,
            font=("Segoe UI", 9, "bold"), cursor="hand2"
        )

    def _bind_keys(self):
        self.master.bind("<Key>", self.on_key)

    def on_key(self, event: tk.Event):
        k = event.keysym if event.keysym else event.char
        if k in ("r", "R"):
            self.reset_episode()
            return
        if k == "space":
            if self.auto_running:
                self.stop_auto()
            else:
                self.start_auto()
            return
        if k in self.KEY_TO_ACTION:
            self.manual_mode = True
            self.mode_var.set("Mode: Manual Control")
            self.stop_auto()
            self._apply_action(self.KEY_TO_ACTION[k], source="MANUAL")

    # ---------------- rendering ----------------
    def _cell_center(self, pos: Position):
        r, c = pos
        x = self.PADDING + c * self.TILE_SIZE + self.TILE_SIZE // 2
        y = self.PADDING + r * self.TILE_SIZE + self.TILE_SIZE // 2
        return x, y

    def _tile_for_pos(self, pos: Position) -> str:
        if pos in self.env.walls:
            return self.wall_variant.get(pos, "wall1")
        if pos in self.env.traps:
            return self.trap_variant.get(pos, "trap1")
        if pos == self.env.start_pos:
            return "start"
        if pos == self.env.goal_pos:
            return "goal"

        # coins show treasure only if not collected
        if pos in self.env.coins:
            idx = self.env.coins.index(pos)
            return "floor" if self.env.is_coin_collected(idx) else "treasure"

        return "floor"

    def _build_grid_items(self):
        self.canvas.delete("all")
        self.cell_items: Dict[Position, int] = {}

        for r in range(self.env.cfg.rows):
            for c in range(self.env.cfg.cols):
                pos = (r, c)
                key = self._tile_for_pos(pos)
                cx, cy = self._cell_center(pos)
                item = self.canvas.create_image(cx, cy, image=self.tiles[key])
                self.cell_items[pos] = item

        ax, ay = self._cell_center(self.env.agent_pos)
        self.agent_item = self.canvas.create_image(ax, ay, image=self.tiles["player"])

        self._clear_trail()

    def _refresh_dynamic_tiles(self):
        # coins can change from treasure -> floor
        for pos in self.env.coins:
            item = self.cell_items.get(pos)
            if item is None:
                continue
            self.canvas.itemconfig(item, image=self.tiles[self._tile_for_pos(pos)])

    def _update_agent(self):
        ax, ay = self._cell_center(self.env.agent_pos)
        self.canvas.coords(self.agent_item, ax, ay)

    # ---------------- trail ----------------
    def _clear_trail(self):
        for tid in self.trail_ids:
            try:
                self.canvas.delete(tid)
            except Exception:
                pass
        self.trail_ids = []

    def _draw_trail_segment(self):
        if len(self.path_trail) < 2:
            return
        (r1, c1), (r2, c2) = self.path_trail[-2], self.path_trail[-1]
        x1, y1 = self._cell_center((r1, c1))
        x2, y2 = self._cell_center((r2, c2))
        lid = self.canvas.create_line(x1, y1, x2, y2, fill=self.ACCENT, width=2)
        self.trail_ids.append(lid)

    # ---------------- controls ----------------
    def reset_episode(self):
        self.stop_auto()
        self.current_state = self.env.reset()
        self.last_reward = 0.0
        self.last_info = {"event": "reset", "action_name": "-", "success": False}
        self.path_trail = [self.env.agent_pos]
        self._build_grid_items()
        self.update_status()
        self._log("Episode reset.")

    def step_once(self):
        self.manual_mode = False
        self.mode_var.set("Mode: Greedy AI Demo")
        if self.is_terminal():
            self._log("Episode ended. Press Reset.")
            return
        action = self.greedy_action(self.current_state)
        self._apply_action(action, source="AI")

    def start_auto(self):
        self.manual_mode = False
        self.mode_var.set("Mode: Greedy AI Demo")
        if self.auto_running:
            return
        if self.is_terminal():
            self.reset_episode()
        self.auto_running = True
        self._log(f"Auto started ({self.speed_ms_var.get()} ms/step).")
        self._auto_loop()

    def _auto_loop(self):
        if not self.auto_running:
            return
        if self.is_terminal():
            self.auto_running = False
            return
        action = self.greedy_action(self.current_state)
        self._apply_action(action, source="AI")
        if self.auto_running and not self.is_terminal():
            self.after_id = self.master.after(int(self.speed_ms_var.get()), self._auto_loop)

    def stop_auto(self):
        self.auto_running = False
        if self.after_id is not None:
            try:
                self.master.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def is_terminal(self) -> bool:
        return self.last_info.get("event") in {"goal", "trap", "timeout"}

    def _apply_action(self, action: int, source: str):
        next_state, reward, done, info = self.env.step(action)

        self.current_state = next_state
        self.last_reward = reward
        self.last_info = info

        self.path_trail.append(self.env.agent_pos)
        self._draw_trail_segment()

        self._refresh_dynamic_tiles()
        self._update_agent()
        self.update_status()

        self._log(
            f"[{source}] step={self.env.steps:03d} action={info.get('action_name','-'):>5} "
            f"event={info.get('event','-'):>12} reward={reward:+.2f} pos={info.get('position')}"
        )

        if done:
            if info.get("success", False):
                self._log("🎉 SUCCESS: reached GOAL.")
            elif info.get("event") == "trap":
                self._log("💥 FAILED: stepped on TRAP.")
            else:
                self._log("⏰ FAILED: TIMEOUT.")
            self.stop_auto()

    def update_status(self):
        collected = sum(int(self.env.is_coin_collected(i)) for i in range(len(self.env.coins)))
        self.status_vars["result"].set(
            "Status: SUCCESS ✅" if self.last_info.get("event") == "goal" else
            "Status: FAILED ❌" if self.last_info.get("event") in {"trap", "timeout"} else
            ("Status: Running (Manual)" if self.manual_mode else "Status: Running")
        )
        self.status_vars["step"].set(f"Step: {self.env.steps}/{self.env.cfg.max_steps}")
        self.status_vars["pos"].set(f"Position: {self.env.agent_pos}")
        self.status_vars["action"].set(f"Action: {self.last_info.get('action_name','-')}")
        self.status_vars["event"].set(f"Event: {self.last_info.get('event','-')}")
        self.status_vars["reward"].set(f"Reward: {self.last_reward:+.2f}")
        self.status_vars["total"].set(f"Total Reward: {self.env.total_reward:+.2f}")
        self.status_vars["coins"].set(f"Treasure: {collected}/{len(self.env.coins)}")

    def _log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("1.0", text + "\n")
        # trim
        lines = self.log_text.get("1.0", "end-1c").splitlines()
        if len(lines) > 140:
            self.log_text.delete("120.0", "end")
        self.log_text.configure(state="disabled")

    def on_close(self):
        self.stop_auto()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TreasureGameUI(root, results_dir="results")
    root.mainloop()