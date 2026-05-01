import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from A2C import fleches, reward_schedule, max_epi


desc = ["SHGF", "FHHF", "FHHF", "FFFF"]
desc1=["SFFF","FHFF","FFFF","HFFG"]    # peu de trous
desc2 = ["SFFF","FHFH","FFFH", "HFFG"] # nombre moyen de trous
desc3 =["SHHH","FHHH","FFHH","HFFG"]   # beaucoup de trous
desc4 =["SFFF","FHHF","FHHF","FFFG" ]  # trous centres
desc5= ["SFFH","FHFH","FFFH","HFFG"]   # trous pres du chemin direct
desc6 = ["SFFH","FFFF","HFFH","HFFG"]  # trous sur les cotes

average = 100

def average_by_blocks(values, block_size=average):
    block_means = []
    block_ends = []

    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        if len(block) > 0:
            block_means.append(np.mean(block))
            block_ends.append(start + len(block))

    return block_ends, block_means


def state_to_row_col(s, n_cols):
    return s // n_cols, s % n_cols


def is_terminal_state(desc, s):
    n_cols = len(desc[0])
    r, c = state_to_row_col(s, n_cols)
    return desc[r][c] in ["H", "G"]


def display_policy_grid(desc, final_policy, title="Politique finale"):
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
                c + 0.5, r + 0.5, grid_labels[r][c],
                ha="center", va="center", fontsize=18
            )

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
    plt.ylabel("Récompense cumulative moyenne")
    plt.title(f"Optimized MCTS : récompense cumulative moyenne sur {block_size} épisodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_rewards_triple(rewards_1, rewards_2, rewards_3, label1, label2, label3, title, block_size=100):
    ep1, avg1 = average_by_blocks(rewards_1, block_size=block_size)
    ep2, avg2 = average_by_blocks(rewards_2, block_size=block_size)
    ep3, avg3 = average_by_blocks(rewards_3, block_size=block_size)

    plt.figure(figsize=(8, 5))
    plt.plot(ep1, avg1, label=label1)
    plt.plot(ep2, avg2, label=label2)
    plt.plot(ep3, avg3, label=label3)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense cumulative moyenne")
    plt.title(f"{title} : récompense cumulative moyenne sur {block_size} épisodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


class OptimizedMCTS:
    def __init__(
        self,
        env,
        n_actions,
        gamma=0.99,
        exploration_c=3, # hyperparamètre
    ):
        self.env = env
        self.n_actions = n_actions
        self.gamma = gamma
        self.exploration_c = exploration_c

        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))
        self.N_sa = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))
        self.N_s = defaultdict(float)

    def uct_score(self, state, action):
        if self.N_sa[state][action] == 0:
            return np.inf

        exploitation = self.Q[state][action]
        exploration = self.exploration_c * np.sqrt(
            np.log(self.N_s[state] + 1.0) / self.N_sa[state][action]
        )
        return exploitation + exploration

    def select_action_uct(self, state):
        scores = [self.uct_score(state, a) for a in range(self.n_actions)]
        return int(np.argmax(scores))

    def run_episode(self):
        state, _ = self.env.reset()
        done = False

        trajectory = []
        total_reward = 0.0
        discount = 1.0

        while not done:
            action = self.select_action_uct(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            trajectory.append((state, action))
            total_reward += discount * reward
            discount *= self.gamma

            state = next_state

        terminal_reward = reward

        self.backpropagate(trajectory, terminal_reward)
        return total_reward

    def backpropagate(self, trajectory, terminal_reward):
        G = terminal_reward

        for state, action in reversed(trajectory):
            self.N_s[state] += 1.0
            self.N_sa[state][action] += 1.0

            n = self.N_sa[state][action]
            self.Q[state][action] += (G - self.Q[state][action]) / n

            G = self.gamma * G


    def extract_policy(self, desc):
        n_rows = len(desc)
        n_cols = len(desc[0])
        final_policy = {}

        for s in range(n_rows * n_cols):
            if is_terminal_state(desc, s):
                final_policy[s] = 0
            else:
                final_policy[s] = int(np.argmax(self.Q[s]))

        return final_policy

def train_case(cas, desc, is_slippery, success_rate, reward_schedule):
    print(f"\n===== {cas} =====")

    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=is_slippery,
        success_rate=success_rate,
        reward_schedule=reward_schedule
    )

    mcts = OptimizedMCTS(
        env=env,
        n_actions=env.action_space.n,
        gamma=0.99,
        exploration_c=2,
    )

    episode_rewards = []

    for episode in range(1, max_epi + 1):
        ep_reward = mcts.run_episode()
        episode_rewards.append(ep_reward)

        if episode % average == 0:
            print(
                f"Épisode {episode}/{max_epi} | "
                f"récompense moyenne = {np.mean(episode_rewards[-average:]):.3f}"
            )

    env.close()
    return mcts, episode_rewards


if __name__ == "__main__":
    mcts_slip, rewards_slip = train_case(
        cas="Case 1: is_slippery=True, success_rate=1/2",
        desc=desc,
        is_slippery=True,
        success_rate=1.0 / 2.0,
        reward_schedule=reward_schedule
    )

    policy_slip = mcts_slip.extract_policy(desc)
    display_policy_grid(desc, policy_slip, title="Optimized MCTS - slippery")

    mcts_noslip, rewards_noslip = train_case(
        cas="Case 2: is_slippery=False, success_rate=1",
        desc=desc,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_schedule
    )

    policy_noslip = mcts_noslip.extract_policy(desc)
    display_policy_grid(desc, policy_noslip, title="Optimized MCTS - non-slippery")

    plot_training_rewards_together(
        rewards_slip,
        rewards_noslip,
        label1="slippery=True, success_rate=1/2",
        label2="slippery=False",
        block_size=average
    )

    mcts_d1, rewards_d1 = train_case(
        cas="desc1",
        desc=desc1,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_schedule
    )
    policy_d1 = mcts_d1.extract_policy(desc1)
    display_policy_grid(desc1, policy_d1, title="Optimized MCTS - desc1")

    mcts_d2, rewards_d2 = train_case(
        cas="desc2",
        desc=desc2,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_schedule
    )
    policy_d2 = mcts_d2.extract_policy(desc2)
    display_policy_grid(desc2, policy_d2, title="Optimized MCTS - desc2")

    mcts_d3, rewards_d3 = train_case(
        cas="desc3",
        desc=desc3,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_schedule
    )
    policy_d3 = mcts_d3.extract_policy(desc3)
    display_policy_grid(desc3, policy_d3, title="Optimized MCTS - desc3")

    plot_training_rewards_triple(
        rewards_d1, rewards_d2, rewards_d3,
        label1="desc1", label2="desc2", label3="desc3",
        title="Comparaison desc1 / desc2 / desc3",
        block_size=average
    )

    mcts_d4, rewards_d4 = train_case(
        cas="desc4 - is_slippery=True, success_rate=1/2",
        desc=desc4,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_schedule
    )
    policy_d4 = mcts_d4.extract_policy(desc4) #
    display_policy_grid(desc4, policy_d4, title="Optimized MCTS - desc4")

    mcts_d5, rewards_d5 = train_case(
        cas="desc5 - is_slippery=True, success_rate=1/2",
        desc=desc5,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_schedule
    )
    policy_d5 = mcts_d5.extract_policy(desc5)
    display_policy_grid(desc5, policy_d5, title="Optimized MCTS - desc5")

    mcts_d6, rewards_d6 = train_case(
        cas="desc6 - is_slippery=True, success_rate=1/2",
        desc=desc6,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_schedule
    )
    policy_d6 = mcts_d6.extract_policy(desc6)
    display_policy_grid(desc6, policy_d6, title="Optimized MCTS - desc6")

    plot_training_rewards_triple(
        rewards_d4, rewards_d5, rewards_d6,
        label1="desc4", label2="desc5", label3="desc6",
        title="Comparaison desc4 / desc5 / desc6",
        block_size=average
    )