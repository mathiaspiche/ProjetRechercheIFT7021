import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from Q_VDBE_Soft import Q_VDBE_softmax
from Cartes_Fred import all

max_eps = 1000
learning_per_step = 100
success_rate_steps = 50
average_wins = np.zeros([success_rate_steps])
for i in range(success_rate_steps):
  env_extreme = gym.make('FrozenLake-v1',
                       desc=loop_path,
                       map_name="zig_zag",
                       is_slippery=True,
                       success_rate=(i/success_rate_steps),
                       reward_schedule=(1, -1, 0)
                       )
  
  wins = np.zeros(learning_per_step)
  stalls = np.zeros(learning_per_step)
  for j in range(learning_per_step):
    _, _, successes, stall = Q_VDBE_softmax(env_extreme, max_eps, 100, 0.99, 0.01, 0.8, 0.5, 0.99)
    wins[j] = successes
    stalls[j] = stall
  average_wins[i] = np.average(wins/max_eps)
  print(np.average(stalls/max_eps))
  print(wins)
  print(average_wins[i])
  print(i, "/", success_rate_steps, " done")
  plt.plot(average_wins)
  plt.show()
