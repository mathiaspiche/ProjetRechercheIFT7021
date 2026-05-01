import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from grillescas2 import grids
from A2C import reward_schedule, fleches, max_epi, N_ENVS

max_episodes = max_epi

class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


def average_by_blocks(values, block_size=100):
    block_means = []
    block_ends = []

    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        if len(block) > 0:
            block_means.append(np.mean(block))
            block_ends.append(start + len(block))

    return block_ends, block_means


def extract_policy(model, desc):
    n_rows = len(desc)
    n_cols = len(desc[0])
    final_policy = {}

    for s in range(n_rows * n_cols):
        action, _ = model.predict(s, deterministic=True)
        final_policy[s] = int(action)

    return final_policy



def display_policy_grid(desc, final_policy, title="Learned final policy"):
    n_rows = len(desc)
    n_cols = len(desc[0])

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
            rect = plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", linewidth=2)
            ax.add_patch(rect)
            ax.text(
                c + 0.5,
                r + 0.5,
                grid_labels[r][c],
                ha="center",
                va="center",
                fontsize=18
            )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def evaluate_goal_rate(model, desc, n_eval_episodes=200):
    eval_env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
        reward_schedule=reward_schedule
    )

    successes = []

    for _ in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(int(action))
            total_reward += reward

        successes.append(1 if total_reward > 0 else 0)

    eval_env.close()
    return np.mean(successes)


def plot_all_training_rewards(results, block_size=100):
    plt.figure(figsize=(9, 5))

    for result in results:
        episodes, avg_rewards = average_by_blocks(result["episode_rewards"], block_size=block_size)

        for i, avg in enumerate(avg_rewards):
            start_ep = i * block_size + 1
            end_ep = min((i + 1) * block_size, len(result["episode_rewards"]))

        plt.plot(episodes, avg_rewards, marker="o", label=result["grid_name"])

    plt.xlabel("Épisode")
    plt.ylabel("Récompense cumulative moyenne de l'algorithme A2C pour le cas 2.")
    plt.title(f"Récompense cumulative moyenne sur {block_size} épisodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_on_grid(desc, grid_name, N_ENVS=N_ENVS, max_episodes=1250):
    callback_max_episodes = StopTrainingOnMaxEpisodes(
        max_episodes=max_episodes,
        verbose=1
    )
    reward_callback = EpisodeRewardCallback()

    train_env = make_vec_env(
        "FrozenLake-v1",
        N_ENVS=N_ENVS,
        env_kwargs={
            "desc": desc,
            "is_slippery": False,
            "reward_schedule": reward_schedule
        }
    )
    train_env = VecMonitor(train_env)

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

    model.learn(
        total_timesteps=int(1e10),
        callback=[reward_callback, callback_max_episodes]
    )

    final_policy = extract_policy(model, desc)
    goal_rate = evaluate_goal_rate(model, desc, n_eval_episodes=200)

    train_env.close()

    return {
        "grid_name": grid_name,
        "desc": desc,
        "model": model,
        "episode_rewards": reward_callback.episode_rewards,
        "goal_rate": goal_rate,
        "final_policy": final_policy
    }


grilles = {
    "grid_0": grids[0],
    "grid_1": grids[1],
    "grid_2": grids[2]
}

block_size = 100

results = []

for grid_name, desc in grilles.items():
    result = train_on_grid(
        desc=desc,
        grid_name=grid_name,
        N_ENVS=N_ENVS,
        max_episodes=max_episodes
    )
    results.append(result)

plot_all_training_rewards(results, block_size=block_size)

for result in results:
    display_policy_grid(
        result["desc"],
        result["final_policy"],
        title=f"Final policy - {result['grid_name']}"
    )
