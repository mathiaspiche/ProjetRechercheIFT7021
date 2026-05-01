import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from Cartes_Fred import loop_path
from grillescas2 import grids as grids2
from grillescas3 import grids as grids3
from Q_VDBE_Soft import Q_VCBE_softmax
from A2C import reward_schedule

n_trainings = 100
n_episodes = 1000
n_episodes_cas23 = 100
max_steps = 100
start_epsilon = 0.99
sigma = 10
alpha = 0.4
temperature = 1
gamma = 0.9


def defined_greedy(env, policy, n_episodes, max_steps):
    wins = 0
    total_steps = 0
    episode_rewards = np.zeros(n_episodes)
    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            action = np.random.choice(env.action_space.n, p=policy[state])
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                if reward > 0:
                    wins += 1
                total_steps += step + 1
                break
        episode_rewards[ep] = ep_reward
    avg_steps = total_steps / n_episodes
    return wins, avg_steps, episode_rewards

print("\nCas 1")

env_cas1 = [
    gym.make('FrozenLake-v1', desc=loop_path, map_name="loop_no_slip",
             is_slippery=False, success_rate=90/100, reward_schedule=reward_schedule),
    gym.make('FrozenLake-v1', desc=loop_path, map_name="loop_50_slip",
             is_slippery=True, success_rate=1.0/2.0, reward_schedule=reward_schedule)
]

superior_policy = np.array([[0,    1,    0,    0   ],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [1,    0,    0,    0   ],
                            [0,    1,    0,    0   ],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0,    0,    0,    1   ],
                            [0,    1,    0,    0   ],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0,    0,    0,    1   ],
                            [0,    0,    1,    0   ],
                            [0,    0,    1,    0   ],
                            [0,    0,    1,    0   ],
                            [0,    0,    0,    1   ]])

n_actions = env_cas1[0].action_space.n
states_number = env_cas1[0].observation_space.n
average_politics_cas1 = np.zeros([len(env_cas1), states_number, n_actions])
average_rewards_cas1 = np.zeros([len(env_cas1) * 2, n_episodes])

for iteration in range(len(env_cas1)):
    rew = np.zeros([n_trainings, n_episodes])
    policies = np.zeros([n_trainings, states_number, n_actions])
    for i in range(n_trainings):
        policies[i, :, :], win, sta, rew[i], cha = Q_VCBE_softmax(
            env_cas1[iteration], n_episodes, max_steps, start_epsilon, sigma, alpha, temperature, gamma)
    average_rewards_cas1[iteration] = np.average(rew, 0)
    average_politics_cas1[iteration] = np.average(policies, 0)

for iteration in range(len(env_cas1)):
    rew = np.zeros([n_trainings, n_episodes])
    for i in range(n_trainings):
        win, sta, rew[i] = defined_greedy(env_cas1[iteration], superior_policy, n_episodes, max_steps)
    average_rewards_cas1[iteration + len(env_cas1)] = np.average(rew, 0)

plt.plot(np.transpose(average_rewards_cas1))
plt.title("Récompenses moyennes par épisode pour le cas d'étude #1 avec l'algorithme Q-VDBE-Softmax", wrap=True)
plt.xlabel("Épisode")
plt.ylabel("Récompenses moyennes (sur 100 itérations)")
plt.legend(["Sans glissement, Q_VDBE", "success_rate = 1/2, Q_VDBE",
            "Sans glissement, greedy avec bonne politique", "success_rate=1/2, greedy avec bonne politique"])
plt.show()

print("\nCas 2")

env_cas2 = [
    gym.make('FrozenLake-v1', desc=grids2["Deux trous sur seize"], map_name="zig_zag",
             is_slippery=False, success_rate=90/100, reward_schedule=reward_schedule),
    gym.make('FrozenLake-v1', desc=grids2["Quatre trous sur seize"], map_name="zig_zag",
             is_slippery=False, success_rate=90/100, reward_schedule=reward_schedule),
    gym.make('FrozenLake-v1', desc=grids2["Neuf trous sur seize"], map_name="zig_zag",
             is_slippery=False, success_rate=90/100, reward_schedule=reward_schedule)
]

n_actions_cas2 = env_cas2[0].action_space.n
states_number_cas2 = env_cas2[0].observation_space.n
average_politics_cas2 = np.zeros([len(env_cas2), states_number_cas2, n_actions_cas2])
average_rewards_cas2 = np.zeros([len(env_cas2), n_episodes_cas23])

for iteration in range(len(env_cas2)):
    print(f"\n  env {iteration + 1}/{len(env_cas2)}")
    rew = np.zeros([n_trainings, n_episodes_cas23])
    policies = np.zeros([n_trainings, states_number_cas2, n_actions_cas2])
    for i in range(n_trainings):
        if (i + 1) % 10 == 0:
            print(f"    training {i + 1}/{n_trainings} | avg reward so far: {np.mean(rew[:i]):.3f}")
        policies[i, :, :], win, sta, rew[i], cha = Q_VCBE_softmax(
            env_cas2[iteration], n_episodes_cas23, max_steps, start_epsilon, sigma, alpha, temperature, gamma)
    average_rewards_cas2[iteration] = np.average(rew, 0)
    average_politics_cas2[iteration] = np.average(policies, 0)
    print(f"  env {iteration + 1} done | mean reward: {np.mean(average_rewards_cas2[iteration]):.3f}")

plt.plot(np.transpose(average_rewards_cas2))
plt.title("Récompenses moyennes par épisode pour le cas d'étude #2 avec l'algorithme Q-VDBE-Softmax", wrap=True)
plt.xlabel("Épisode")
plt.ylabel("Récompenses moyennes (sur 100 itérations)")
plt.legend(["Deux trous sur seize", "Quatre trous sur seize", "Neuf trous sur seize"])
plt.show()


env_cas3 = [
    gym.make('FrozenLake-v1', desc=grids3["Trous centrés"], map_name="Centre",
             is_slippery=False, reward_schedule=reward_schedule),
    gym.make('FrozenLake-v1', desc=grids3["Trous près du chemin direct"], map_name="Pres",
             is_slippery=False, reward_schedule=reward_schedule),
    gym.make('FrozenLake-v1', desc=grids3["Trous sur les côtés"], map_name="Cotes",
             is_slippery=False, reward_schedule=reward_schedule)
]

n_actions_cas3 = env_cas3[0].action_space.n
states_number_cas3 = env_cas3[0].observation_space.n
average_politics_cas3 = np.zeros([len(env_cas3), states_number_cas3, n_actions_cas3])
average_rewards_cas3 = np.zeros([len(env_cas3), n_episodes_cas23])

for iteration in range(len(env_cas3)):
    rew = np.zeros([n_trainings, n_episodes_cas23])
    policies = np.zeros([n_trainings, states_number_cas3, n_actions_cas3])
    for i in range(n_trainings):
        policies[i, :, :], win, sta, rew[i], cha = Q_VCBE_softmax(
            env_cas3[iteration], n_episodes_cas23, max_steps, start_epsilon, sigma, alpha, temperature, gamma)
    average_rewards_cas3[iteration] = np.average(rew, 0)
    average_politics_cas3[iteration] = np.average(policies, 0)

plt.plot(np.transpose(average_rewards_cas3))
plt.title("Récompenses moyennes par épisode pour le cas d'étude #3 avec l'algorithme Q-VDBE-Softmax", wrap=True)
plt.xlabel("Épisode")
plt.ylabel("Récompenses moyennes (sur 100 itérations)")
plt.legend(["Trous centrés", "Trous près du chemin direct", "Trous sur les côtés"])
plt.show()
