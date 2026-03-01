from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from environment import GridWorldTreasureEnv, EnvConfig, MapConfig
from q_learning_agent import TabularQLearningAgent


def moving_average(values, window=100):
    if not values:
        return []
    out = []
    for i in range(len(values)):
        s = max(0, i - window + 1)
        chunk = values[s:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def train(
    episodes: int = 8000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.997,
    seed: int = 42,
    map_seed: int = 2026,
):
    env = GridWorldTreasureEnv(
        config=EnvConfig(rows=24, cols=24, max_steps=400),
        map_config=MapConfig(seed=map_seed, wall_density=0.22, trap_density=0.07, num_coins=6),
    )

    agent = TabularQLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    rewards, steps_list, success_flags = [], [], []

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        final_info = {}

        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            ep_steps += 1
            final_info = info

        agent.decay_epsilon_step()

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        success_flags.append(1 if final_info.get("success", False) else 0)

        if ep % 200 == 0:
            recent = success_flags[-200:]
            success_rate = sum(recent) / len(recent)
            avg_reward = sum(rewards[-200:]) / len(rewards[-200:])
            print(
                f"Episode {ep:5d}/{episodes} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"avg_reward(200)={avg_reward:.2f} | "
                f"success_rate(200)={success_rate:.2%}"
            )

    return env, agent, rewards, steps_list, success_flags


def evaluate_greedy(env: GridWorldTreasureEnv, agent: TabularQLearningAgent, episodes: int = 30):
    success = 0
    total_steps = 0.0
    total_reward = 0.0

    for _ in range(episodes):
        state = env.reset()
        done = False
        final_info = {}
        ep_reward = 0.0

        while not done:
            action = agent.greedy_action(state)
            state, reward, done, final_info = env.step(action)
            ep_reward += reward

        total_steps += env.steps
        total_reward += ep_reward
        success += 1 if final_info.get("success", False) else 0

    print(
        f"[Greedy Eval] success={success}/{episodes} ({success/episodes:.2%}) | "
        f"avg_steps={total_steps/episodes:.2f} | "
        f"avg_reward={total_reward/episodes:.2f}"
    )


def save_results(out_dir: str, env: GridWorldTreasureEnv, agent, rewards, steps_list, success_flags, config: dict):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save map snapshot (UI will load this to match training map)
    with (out / "map_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(env.get_map_snapshot(), f, indent=2)

    # Save Q-table
    agent.save(str(out / "q_table.npy"))

    # Save metrics
    with (out / "training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "total_reward", "steps", "success"])
        for i, (r, s, ok) in enumerate(zip(rewards, steps_list, success_flags), start=1):
            w.writerow([i, r, s, ok])

    with (out / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Plots
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, linewidth=1, label="Reward")
    plt.plot(moving_average(rewards, 100), linewidth=2, label="MA(100)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "reward_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps_list, linewidth=1, label="Steps")
    plt.plot(moving_average(steps_list, 100), linewidth=2, label="MA(100)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Steps Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "steps_curve.png", dpi=150)
    plt.close()

    rolling_success = moving_average(success_flags, 200)
    plt.figure(figsize=(8, 4))
    plt.plot(rolling_success, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Rolling Success Rate (200)")
    plt.title("Training Success Rate")
    plt.tight_layout()
    plt.savefig(out / "success_rate_curve.png", dpi=150)
    plt.close()

    print(f"✅ Saved results to: {out.resolve()}")


if __name__ == "__main__":
    config = {
        "episodes": 8000,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.997,
        "seed": 42,
        "map_seed": 123,
    }

    env, agent, rewards, steps_list, success_flags = train(**config)
    evaluate_greedy(env, agent, episodes=30)
    save_results("results", env, agent, rewards, steps_list, success_flags, config)