import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from Cartes_Fred import loop_path
from grillescas2 import grids as grids2
from grillescas3 import grids as grids3

N_ENVS = 8
max_episodes = 1250
reward_schedule = (1, -1, -0.01)
average = 100

fleches = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}


class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_numbers = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                self.episode_numbers.append(self.episode_count)
                self.episode_rewards.append(info["episode"]["r"])
        return True


# ─── Utilities ───────────────────────────────────────────────────────────────

def average_by_blocks(values, block_size=average):
    block_means, block_ends = [], []
    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        if block:
            block_means.append(np.mean(block))
            block_ends.append(start + len(block))
    return block_ends, block_means


def extract_policy(model, desc):
    n_rows, n_cols = len(desc), len(desc[0])
    final_policy = {}
    for s in range(n_rows * n_cols):
        action, _ = model.predict(s, deterministic=True)
        final_policy[s] = int(action)
    return final_policy


def evaluate_goal_rate(model, desc, n_eval_episodes=200):
    eval_env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
        reward_schedule=reward_schedule
    )
    successes = []
    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = truncated = False
        total_reward = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(int(action))
            total_reward += reward
        successes.append(1 if total_reward > 0 else 0)
    eval_env.close()
    return np.mean(successes)


def display_policy_grid(desc, final_policy, title="Politique finale"):
    n_rows, n_cols = len(desc), len(desc[0])
    grid_labels = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            s = r * n_cols + c
            cell = desc[r][c]
            if cell == "H":
                row.append("H")
            elif cell == "G":
                row.append("G")
            elif cell == "S":
                row.append(f"S{fleches[final_policy[s]]}")
            else:
                row.append(fleches[final_policy[s]])
        grid_labels.append(row)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", linewidth=2))
            ax.text(c + 0.5, r + 0.5, grid_labels[r][c], ha="center", va="center", fontsize=18)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_rewards_together(rewards_1, rewards_2, label1, label2, block_size=100):
    ep1, avg1 = average_by_blocks(rewards_1, block_size=block_size)
    ep2, avg2 = average_by_blocks(rewards_2, block_size=block_size)
    plt.figure(figsize=(8, 5))
    plt.plot(ep1, avg1, label=label1)
    plt.plot(ep2, avg2, label=label2)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense cumulative moyenne de l'algorithme A2C pour le cas 1")
    plt.title(f"Récompense cumulative moyenne sur {block_size} épisodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_training_rewards(results, block_size=100, title="Récompense cumulative moyenne"):
    plt.figure(figsize=(10, 6))
    for result in results:
        episodes, avg_rewards = average_by_blocks(result["episode_rewards"], block_size=block_size)
        plt.plot(episodes, avg_rewards, label=result["grid_name"])
    plt.xlabel("Épisode")
    plt.ylabel("Récompense cumulative moyenne")
    plt.title(f"{title} sur {block_size} épisodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_on_grid(desc, grid_name, n_envs=N_ENVS, max_ep=max_episodes,
                  is_slippery=False, success_rate=1.0, block_size=100):
    print(f"\n===== {grid_name} =====")

    train_env = make_vec_env(
        "FrozenLake-v1",
        n_envs=n_envs,
        env_kwargs={
            "desc": desc,
            "is_slippery": is_slippery,
            "success_rate": success_rate,
            "reward_schedule": reward_schedule
        }
    )
    train_env = VecMonitor(train_env)

    reward_callback = EpisodeRewardCallback()
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=max_ep, verbose=1)

    model = A2C(
        "MlpPolicy",
        train_env,
        n_steps=15,
        gamma=0.99,
        ent_coef=0.01,
        use_rms_prop=True,
        normalize_advantage=True,
        verbose=1,
        stats_window_size=10
    )

    model.learn(total_timesteps=int(1e10), callback=[reward_callback, stop_callback])

    final_policy = extract_policy(model, desc)
    goal_rate = evaluate_goal_rate(model, desc, n_eval_episodes=200)
    block_episodes, block_avg_rewards = average_by_blocks(
        reward_callback.episode_rewards, block_size=block_size
    )

    train_env.close()

    return {
        "grid_name": grid_name,
        "desc": desc,
        "model": model,
        "episode_rewards": reward_callback.episode_rewards,
        "block_episodes": block_episodes,
        "block_avg_rewards": block_avg_rewards,
        "goal_rate": goal_rate,
        "final_policy": final_policy
    }

if __name__ == "__main__":


    result_slip = train_on_grid(
        loop_path, "loop - slippery",
        is_slippery=True, success_rate=0.5
    )
    result_noslip = train_on_grid(
        loop_path, "loop - non-slippery",
        is_slippery=False, success_rate=1.0
    )

    display_policy_grid(loop_path, result_slip["final_policy"],   title="A2C - loop slippery")
    display_policy_grid(loop_path, result_noslip["final_policy"], title="A2C - loop non-slippery")

    plot_training_rewards_together(
        result_slip["episode_rewards"],
        result_noslip["episode_rewards"],
        label1="slippery=True, success_rate=1/2",
        label2="slippery=False",
        block_size=average
    )

    results2 = []
    for grid_name, desc in grids2.items():
        result = train_on_grid(desc, grid_name)
        results2.append(result)
        print(f"{grid_name} - goal rate: {result['goal_rate']:.2%}")

    plot_all_training_rewards(results2, block_size=average,
                              title="A2C Cas 2 - récompense cumulative moyenne")
    for result in results2:
        display_policy_grid(result["desc"], result["final_policy"],
                            title=f"Politique finale - {result['grid_name']}")

    results3 = []
    for grid_name, desc in grids3.items():
        result = train_on_grid(desc, grid_name)
        results3.append(result)
        print(f"{grid_name} - goal rate: {result['goal_rate']:.2%}")

    plot_all_training_rewards(results3, block_size=average,
                              title="A2C Cas 3 - récompense cumulative moyenne")
    for result in results3:
        display_policy_grid(result["desc"], result["final_policy"],
                            title=f"Politique finale - {result['grid_name']}")