import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from grillescas2 import grids
from Q_VDBE_Soft import Q_VDBE_softmax


env_0 = gym.make('FrozenLake-v1', desc=grids["Deux trous sur seize"], map_name="zig_zag", is_slippery=False, success_rate=90/100, reward_schedule=(1, -1, -1/(16**2)) )


env_1 = gym.make('FrozenLake-v1',
                 desc=grids["Quatre trous sur seize"],
                 map_name="zig_zag",
                 is_slippery=False,
                 success_rate=90/100,
                 reward_schedule=(1, -1, -1/(16**2))
                 )
env_2 = gym.make('FrozenLake-v1',
                 desc=grids["Neuf trous sur seize"],
                 map_name="zig_zag",
                 is_slippery=False,
                 success_rate=90/100,
                 reward_schedule=(1, -1, -1/(16**2))
                 )
env = [env_0, env_1, env_2]
n_trainings = 100
average_rewards = np.zeros([len(env), n_trainings])
n_episodes = 100
max_steps = 100
start_epsilon = 0.99
sigma = 10
alpha = 0.4
temperature = 1
gamma = 0.9

n_actions = env[0].action_space.n
states_number = env[0].observation_space.n
average_politics = np.zeros([len(env), states_number, n_actions])

for iteration in range(len(env)):
  rew = np.zeros([n_trainings, n_episodes])
  policies = np.zeros([n_trainings, states_number, n_actions])
  for i in range(n_trainings):
    _, policies[i, :, :], _, _, rew[i], _ = Q_VCBE_softmax(env[iteration], n_episodes, max_steps, start_epsilon, sigma, alpha, temperature, gamma)
  average_rewards[iteration] = np.average(rew, 0)

plt.plot(np.transpose(average_rewards))
plt.title("Récompenses moyennes par épisode pour le cas d'étude #2 avec l'algorithme Q-VDBE-Softmax", wrap=True)
plt.xlabel("Épisode")
plt.ylabel("Récompenses moyennes (sur 100 itérations)")
plt.legend(["Deux trous sur seize", "Quatre trous sur seize", "Neuf trous sur seize"])
plt.show()

print("Average politics: ", average_politics)
