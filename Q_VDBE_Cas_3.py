import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from grillescas3 import grids
from Q_VDBE_Soft import Q_VDBE_softmax


env_3_0 = gym.make('FrozenLake-v1',
                   desc=grids["Trous centrés"],
                   map_name="Centre",
                   is_slippery=False,
                   reward_schedule=(1, -1, -1/(16**2))
                   )

env_3_1 = gym.make('FrozenLake-v1',
                   desc=grids["Trous près du chemin direct"],
                   map_name="Pres",
                   is_slippery=False,
                   reward_schedule=(1, -1, -1/(16**2))
                   )

env_3_2 = gym.make('FrozenLake-v1',
                   desc=grids["Trous sur les côtés"],
                   map_name="Cotes",
                   is_slippery=False,
                   reward_schedule=(1, -1, -1/(16**2))
                   )

env = [env_3_0, env_3_1, env_3_2]
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
    val, policies[i, :, :], win, sta, rew[i], cha = Q_VCBE_softmax(env[iteration],
                                                                   n_episodes,
                                                                   max_steps, 
                                                                   start_epsilon,
                                                                   sigma, alpha, 
                                                                   temperature, 
                                                                   gamma)
  average_rewards[iteration] = np.average(rew, 0)
  average_politics[iteration] = np.average(policies, 0)


plt.plot(np.transpose(average_rewards))
plt.title("Récompenses moyennes par épisode pour le cas d'étude #3 avec l'algorithme Q-VDBE-Softmax", wrap=True)
plt.xlabel("Épisode")
plt.ylabel("Récompenses moyennes (sur 100 itérations)")
plt.legend(["Trous centrés", "Trous près du chemin direct", "Trous sur les côtés"])
plt.show()

print(average_politics)
