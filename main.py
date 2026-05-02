import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from A2C import train_on_grid
from Cartes_Fred import loop_path
from grillescas2 import grids as grids_2
from grillescas3 import grids as grids_3
from Q_VDBE_Soft import Q_VDBE_softmax
from optimized_greedy import superior_policy, superior_policies_1, superior_policies_2, defined_greedy
from collections import defaultdict
from strategieOptiMC import OptimizedMCTS and train_case_mct
import time
from scipy.ndimage import uniform_filter1d
#Initialize hyperparameters
n_trainings = 100
n_episodes = 3000
max_steps = 100
start_epsilon = 0.99
sigma = 10
alpha = 0.4
temperature = 1
gamma = 0.9
reward_schedule = (1, -1, -0.01)
window_size = 10

#Case 1

#Initialize environments
env_0 = gym.make('FrozenLake-v1',
                 desc=loop_path,
                 map_name="loop",
                 is_slippery=False,
                 success_rate=1.0,
                 reward_schedule=reward_schedule
                 )

env_50 = gym.make('FrozenLake-v1',
                  desc=loop_path,
                  map_name="loop_slippery",
                  is_slippery=True,
                  success_rate=1.0/2.0,
                  reward_schedule=reward_schedule
                  )
env_0 = [env_0, env_50]

#Get environment hidden parameters
n_actions = env_0[0].action_space.n
states_number = env_0[0].observation_space.n

#Initialize rewards table
average_rewards = np.zeros([len(env_0)*4, n_episodes])

#Q_VDBE
for iteration in range(len(env_0)):
  rew = np.zeros([n_trainings, n_episodes])
  policies = np.zeros([n_trainings, states_number, n_actions])
  for i in range(n_trainings):
    val, policies[i, :, :], win, sta, rew[i], cha = Q_VCBE_softmax(env_0[iteration],
                                                                   n_episodes,
                                                                   max_steps,
                                                                   start_epsilon,
                                                                   sigma, alpha,
                                                                   temperature,
                                                                   gamma)
  average_rewards[iteration] = uniform_filter1d(np.average(rew, 0), size=window_size)
  

#Defined greedy
for iteration in range(len(env_0)):
  rew = np.zeros([n_trainings, n_episodes])
  policies = np.zeros([n_trainings, states_number, n_actions])
  for i in range(n_trainings):
    win, sta, rew[i] = defined_greedy(env_0[iteration], superior_policy,
                                      n_episodes, max_steps)

  average_rewards[iteration+len(env_0)] = uniform_filter1d(np.average(rew, 0), size=window_size)


#A2C
return_noslip = train_on_grid(
  grid_name="Case 2: is_slippery=False, success_rate=1", desc=loop_path, 
  is_slippery=False, success_rate=1.0, block_size=100)
average_rewards[4] = uniform_filter1d(return_noslip.episode_rewards, size=window_size)

return_slip = train_on_grid(
    grid_name="Case 1: is_slippery=True, success_rate=1/2", desc=loop_path, 
    is_slippery=True, success_rate=1.0 / 2.0, block_size=100)
average_rewards[5] = uniform_filter1d(return_slip.episode_rewards, size=window_size)


#MCT
mcts_noslip, rewards_noslip = train_case_mct(desc=loop_path, is_slippery=False, 
                                             success_rate=1.0, 
                                             reward_schedule=reward_schedule)
average_rewards[6] = uniform_filter1d(rewards_noslip, size=window_size)

mcts_slip, rewards_slip = train_case_mct(desc=loop_path, is_slippery=True, 
                                         success_rate=1.0 / 2.0, 
                                         reward_schedule=reward_schedule)
average_rewards[7] = uniform_filter1d(rewards_slip, size=window_size)

#Plot
plt.plot(np.transpose(average_rewards))
plt.title("Récompenses moyennes par épisode pour le cas d'étude #1", wrap=True)
plt.xlabel("Épisode")
plt.ylabel("Récompenses moyennes sur 100 essais")
plt.legend(["Sans glissement, Q_VDBE",
            "Risque de glissement = 1/2, Q_VDBE",
            "Sans glissement, greedy avec bonne politique",
            "Risque de glissement = 1/2, greedy avec bonne politique",
            "Sans glissement, A2C",
            "Risque de glissement = 1/2, A2C",
            "Sans glissement, MC-tree",
            "Risque de glissement = 1/2, MC-tree",
            ],
           loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

#Cas 2
Création des environnements

env_0 = gym.make('FrozenLake-v1', 
                 desc=grids_2["Deux trous sur seize"], 
                 map_name="2_trous", is_slippery=False, 
                 success_rate=90/100, 
                 reward_schedule=reward_schedule )

env_1 = gym.make('FrozenLake-v1',
                 desc=grids_2["Quatre trous sur seize"],
                 map_name="4_trous",
                 is_slippery=False,
                 success_rate=90/100,
                 reward_schedule=reward_schedule
                 )
env_2 = gym.make('FrozenLake-v1',
                 desc=grids_2["Neuf trous sur seize"],
                 map_name="9_trous",
                 is_slippery=False,
                 success_rate=90/100,
                 reward_schedule=reward_schedule
                 )

env = [env_0, env_1, env_2]
#Reset rewards
average_rewards = np.zeros([len(env)*4, n_episodes])

#Trainings
#Q_VDBE
for iteration in range(len(env)):
  rew = np.zeros([n_trainings, n_episodes])
  print(rew.shape)
  policies = np.zeros([n_trainings, states_number, n_actions])
  for i in range(n_trainings):
    val, policies[i, :, :], win, sta, rew[i], cha = Q_VDBE_softmax(env[iteration],
                                                                   n_episodes,
                                                                   max_steps,
                                                                   start_epsilon,
                                                                   sigma, alpha,
                                                                   temperature,
                                                                   gamma)
  print(np.average(rew, 0).shape)
  average_rewards[iteration] = uniform_filter1d(np.average(rew, 0), size=window_size)
  average_politics[iteration] = np.average(policies, 0)

#Defined Greedy
for iteration in range(len(env)):
  rew = np.zeros([n_trainings, n_episodes])
  policies = np.zeros([n_trainings, states_number, n_actions])
  for i in range(n_trainings):
    win, sta, rew[i] = defined_greedy(env[iteration], 
                                      superior_policies_1[iteration], n_episodes, 
                                      max_steps)
  average_rewards[iteration+len(env)] = uniform_filter1d(np.average(rew, 0), size=window_size)

#A2C
return_2_holes = train_on_grid(grid_name="Case 1: 2 trous", desc=grids_2["Deux trous sur seize"])
average_rewards[len(env)*2] = uniform_filter1d(return_2_holes.episode_rewards, size=window_size)

return_4_holes = train_on_grid(grid_name="Case 2: 4 trous", desc=grids_2["Quatre trous sur seize"])
average_rewards[len(env)*2+1] = uniform_filter1d(return_4_holes.episode_rewards, size=window_size)

return_9_holes = train_on_grid(grid_name="Case 2: 9 trous", desc=grids_2["Neuf trous sur seize"])
average_rewards[len(env)*2+2] = uniform_filter1d(return_9_holes.episode_rewards, size=window_size)

#MCT
mcts_2_t, rewards_2_t = train_case_mct(desc=grids_2["Deux trous sur seize"],
                                       is_slippery=False, success_rate=1.0, 
                                       reward_schedule=reward_schedule)
average_rewards[len(env)*3] = uniform_filter1d(rewards_2_t, size=window_size)

mcts_4_t, rewards_4_t = train_case_mct(desc=grids_2["Quatre trous sur seize"], 
                                       is_slippery=False, success_rate=1.0, 
                                       reward_schedule=reward_schedule)
average_rewards[len(env)*3+1] = uniform_filter1d(rewards_4_t, size=window_size)

mcts_9_t, rewards_9_t = train_case_mct(desc=grids_2["Neuf trous sur seize"], 
                                       is_slippery=False, success_rate=1.0, 
                                       reward_schedule=reward_schedule)
average_rewards[len(env)*3+2] = uniform_filter1d(rewards_9_t, size=window_size)

#Plot
#Colors repeated themselves if we did not define them
color=['black', 'sienna', 'red', 'darkorange', 'gold', 'greenyellow', 'green', 'cyan', 'royalblue', 'blue', 'darkviolet', 'slategray', 'magenta']
for i in range(12):
  plt.plot(average_rewards[i, :], color=color[i] )
plt.title("Récompenses moyennes par épisode pour le cas d'étude #2 avec l'algorithme Q-VDBE-Softmax", wrap=True)
plt.xlabel("Épisode")
plt.ylabel("Récompenses moyennes (sur 100 itérations)")
plt.legend(["Deux trous sur seize, Q_VDBE", "Quatre trous sur seize, Q_VDBE", "Neuf trous sur seize, Q_VDBE",
            "Deux trous sur seize, greedy avec bonne politique", "Quatre trous sur seize, greedy avec bonne politique", "Neuf trous sur seize, greedy avec bonne politique",
            "Deux trous sur seize, A2C", "Quatre trous sur seize, A2C", "Neuf trous sur seize, A2C",
            "Deux trous sur seize, MC-tree", "Quatre trous sur seize, MC-tree", "Neuf trous sur seize, MC-tree"], loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

#Case 3

#Initialize environment
env_3_0 = gym.make('FrozenLake-v1',
                   desc=grids_3["Trous centrés"],
                   map_name="Centre",
                   is_slippery=False,
                   reward_schedule=reward_schedule
                   )

env_3_1 = gym.make('FrozenLake-v1',
                   desc=grids_3["Trous près du chemin direct"],
                   map_name="Pres",
                   is_slippery=False,
                   reward_schedule=reward_schedule
                   )

env_3_2 = gym.make('FrozenLake-v1',
                   desc=grids_3["Trous sur les côtés"],
                   map_name="Cotes",
                   is_slippery=False,
                   reward_schedule=reward_schedule
                   )

env_3 = [env_3_0, env_3_1, env_3_2]

#Initialize rewards table
average_rewards = np.zeros([len(env_0)*4, n_episodes])
#Trainings

#Q_VDBE
for iteration in range(len(env)):
  rew = np.zeros([n_trainings, n_episodes])
  policies = np.zeros([n_trainings, states_number, n_actions])
  for i in range(n_trainings):
    val, policies[i, :, :], win, sta, rew[i], cha = Q_VDBE_softmax(env_3[iteration],
                                                                   n_episodes,
                                                                   max_steps,
                                                                   start_epsilon,
                                                                   sigma, alpha,
                                                                   temperature,
                                                                   gamma)
  average_rewards[iteration] = uniform_filter1d(np.average(rew, 0), size=window_size)
  average_politics[iteration] = np.average(policies, 0)

#Specified greedy
for iteration in range(len(env)):
  rew = np.zeros([n_trainings, n_episodes])
  policies = np.zeros([n_trainings, states_number, n_actions])
  for i in range(n_trainings):
    win, sta, rew[i] = defined_greedy(env_3[iteration], 
                                      superior_policies_2[iteration], 
                                      n_episodes, max_steps)

  average_rewards[iteration+len(env)] = uniform_filter1d(np.average(rew, 0), size=window_size)

#A2C
return_center = train_on_grid(grid_name="Case 1: Trous centrés", desc=grids["Trous centrés"])
average_rewards[len(env)*2] = uniform_filter1d(return_center.episode_rewards, 
                                               size=window_size)

return_path = train_on_grid(grid_name="Case 2: Trous près des chemins directs", 
                            desc=grids["Trous près du chemin direct"])
average_rewards[len(env)*2+1] = uniform_filter1d(return_path.episode_rewards, 
                                                 size=window_size)

return_side = train_on_grid(grid_name="Case 3: Trous sur les côtés", 
                            desc=grids["Trous sur les côtés"])
average_rewards[len(env)*2+2] = uniform_filter1d(return_side.episode_rewards, 
                                                 size=window_size)

#MCT
mcts_center, rewards_center = train_case_mct(desc=grids["Trous centrés"],
                                             is_slippery=False, success_rate=1.0, 
                                             reward_schedule=reward_schedule)
average_rewards[len(env)*3] = uniform_filter1d(rewards_center, size=window_size)

mcts_path, rewards_path = train_case_mct(desc=grids["Trous près du chemin direct"], 
                                         is_slippery=False, success_rate=1.0,
                                         reward_schedule=reward_schedule)
average_rewards[len(env)*3+1] = uniform_filter1d(rewards_path, size=window_size)

mcts_side, rewards_side = train_case_mct(desc=grids["Trous sur les côtés"], 
                                         is_slippery=False, success_rate=1.0, 
                                         reward_schedule=reward_schedule)
average_rewards[len(env)*3+2] = uniform_filter1d(rewards_side, size=window_size)

#Plot
for i in range(12):
  plt.plot(average_rewards[i, :], color=color[i] )
plt.title("Récompenses moyennes par épisode pour le cas d'étude #3", wrap=True)
plt.xlabel("Épisode")
plt.ylabel("Récompenses moyennes (sur 100 itérations)")
plt.legend(["Trous centrés, Q_VDBE", "Trous près des chemins directs, Q_VDBE", "Trous sur les côtés, Q_VDBE",
            "Trous centrés, greedy avec bonne politique", "Trous près des chemins directs, greedy avec bonne politique", "Trous sur les côtés, greedy avec bonne politique",
            "Trous centrés, A2C", "Trous près des chemins directs, A2C", "Trous sur les côtés, A2C",
            "Trous centrés, MC-tree", "Trous près des chemins directs, MC-tree", "Trous sur les côtés, MC-tree"], loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
