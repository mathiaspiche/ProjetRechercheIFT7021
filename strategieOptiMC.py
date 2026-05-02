import gymnasium as gym
import numpy as np
from collections import defaultdict
from A2C import (reward_schedule, max_episodes, average,
                 display_policy_grid, plot_training_rewards)
from Cartes_Fred import loop_path
from grillescas2 import grids as grids2
from grillescas3 import grids as grids3


def is_terminal_state(desc, s):
    # Retourne vrai si l'état s correspond à un trou ou à l'état but
    n_cols = len(desc[0])
    r, c = s // n_cols, s % n_cols
    return desc[r][c] in ["H", "G"]


class OptimizedMCTS:
    # Classe définissant un algorithme de Monte Carlo Tree Search optimisé pour
    # l'environnement FrozenLake. Elle maintient une table de valeurs Q et un
    # compteur de visites pour chaque paire (état, action), et utilise la formule
    # UCT lors de la sélection des actions. À chaque épisode, l'algorithme
    # suit une trajectoire dans l'environnement, puis ajuste en commencant par le dernier état
    # les récompenses obtenues pour mettre à jour les valeurs Q de chaque paire (état, action) visitée.
    def __init__(
        self,
        env,
        n_actions,
        gamma=0.99,
        exploration_c=3,
    ):
        self.env = env
        self.n_actions = n_actions
        self.gamma = gamma
        self.exploration_c = exploration_c

        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))
        self.N_sa = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))
        self.N_s = defaultdict(float)

    def uct_score(self, state, action):
        # Méthode prenant en entrée un état et une action et
        # retournant la valeur UCT de cette paire.
        if self.N_sa[state][action] == 0:
            return np.inf

        exploitation = self.Q[state][action]
        exploration = self.exploration_c * np.sqrt(
            np.log(self.N_s[state] + 1.0) / self.N_sa[state][action]
        )
        return exploitation + exploration

    def select_action_uct(self, state):
        # Méthode permettant de sélectionner une action à partir d'un état s
        # selon les valeurs UCT des paires (s,a)
        scores = [self.uct_score(state, a) for a in range(self.n_actions)]
        return int(np.argmax(scores))

    def run_episode(self):
        # Méthode qui effectue une trajectoire dans la grille de jeu et
        # retourne les récompenses obtenues.
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
        # Méthode prenant en entrée une trajectoire effectuée dans l'environnement,
        # et la récompense de l'état terminal. Elle calcule le rendement actualisé en
        # partant du dernier état et met à jour les valeurs Q des paires (s,a)
        G = terminal_reward

        for state, action in reversed(trajectory):
            self.N_s[state] += 1.0
            self.N_sa[state][action] += 1.0

            n = self.N_sa[state][action]
            self.Q[state][action] += (G - self.Q[state][action]) / n

            G = self.gamma * G

    def extract_policy(self, desc):
        # Méthode prenant en entrée un modèle MCTS et permettant d'obtenir la
        # politique résultante des prédictions de la stratégie implémentée
        n_rows = len(desc)
        n_cols = len(desc[0])
        final_policy = {}

        for s in range(n_rows * n_cols):
            if is_terminal_state(desc, s):
                final_policy[s] = 0
            else:
                final_policy[s] = int(np.argmax(self.Q[s]))

        return final_policy


def train_case(desc, is_slippery, success_rate, reward_schedule):
    # Méthode prenant en entrée une configuration de grille, les paramètres is_slippery
    # et success_rate de Gymnasium et une fonction de récompense. Elle soutire la politique
    # obtenue après avoir fait plusieurs itérations de la stratégie et retourne, entre autres, les
    # récompenses obtenues.
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

    for episode in range(1, max_episodes + 1):
        ep_reward = mcts.run_episode()
        episode_rewards.append(ep_reward)

    env.close()
    return mcts, episode_rewards


if __name__ == "__main__":
    # Ici, on effectue les expériences et on imprime les figures désirées
    # Cas 1 : loop slippery vs non-slippery
    mcts_slip, rewards_slip = train_case(
        cas="Case 1: is_slippery=True, success_rate=1/2",
        desc=loop_path,
        is_slippery=True,
        success_rate=1.0 / 2.0,
        reward_schedule=reward_schedule
    )
    policy_slip = mcts_slip.extract_policy(loop_path)
    display_policy_grid(loop_path, policy_slip, title="Optimized MCTS - slippery")

    mcts_noslip, rewards_noslip = train_case(
        cas="Case 2: is_slippery=False, success_rate=1",
        desc=loop_path,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_schedule
    )
    policy_noslip = mcts_noslip.extract_policy(loop_path)
    display_policy_grid(loop_path, policy_noslip, title="Optimized MCTS - non-slippery")

    plot_training_rewards(
        [
            (rewards_slip,   "slippery=True, success_rate=1/2"),
            (rewards_noslip, "slippery=False")
        ],
        block_size=average,
        title="Optimized MCTS Cas 1 - récompense cumulative moyenne"
    )

    # Cas 2 : différentes dispositions de trous
    results2 = []
    for grid_name, desc in grids2.items():
        mcts, episode_rewards = train_case(
            cas=grid_name,
            desc=desc,
            is_slippery=False,
            success_rate=1.0,
            reward_schedule=reward_schedule
        )
        final_policy = mcts.extract_policy(desc)
        results2.append({
            "grid_name": grid_name,
            "desc": desc,
            "episode_rewards": episode_rewards,
            "final_policy": final_policy
        })

    plot_training_rewards(results2, block_size=average,
                          title="Optimized MCTS Cas 2 - récompense cumulative moyenne")
    for result in results2:
        display_policy_grid(result["desc"], result["final_policy"],
                            title=f"Optimized MCTS - {result['grid_name']}")

    # Cas 3 : différentes dispositions de trous
    results3 = []
    for grid_name, desc in grids3.items():
        mcts, episode_rewards = train_case(
            cas=grid_name,
            desc=desc,
            is_slippery=False,
            success_rate=1.0,
            reward_schedule=reward_schedule
        )
        final_policy = mcts.extract_policy(desc)
        results3.append({
            "grid_name": grid_name,
            "desc": desc,
            "episode_rewards": episode_rewards,
            "final_policy": final_policy
        })

    plot_training_rewards(results3, block_size=average,
                          title="Optimized MCTS Cas 3 - récompense cumulative moyenne")
    for result in results3:
        display_policy_grid(result["desc"], result["final_policy"],
                            title=f"Optimized MCTS - {result['grid_name']}")