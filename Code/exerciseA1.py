"""
Problem A1
Antonio Remiro Azocar
"""

# import required packages
import numpy as np
import gym
import os

# load the CartPole environment
env = gym.make('CartPole-v0')
#set random seed for environment
env.seed(555)
# maximum episode length
env._max_episode_steps = 300
env.reset() # start process

# parameters
discount_factor = 0.99
step_limit = 20000 # maximum number of steps
number_traj = 3  # number of trajectories

# set random seed for numpy
np.random.seed(555)

# directory where we save data for exercise 3
new_dir = "data"

# if directory does not exist, create
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# trajectories are generated under a uniform random policy
# function uniformrandompolicy
# takes as input the number of generated trajectories 'number_traj'
# returns observations at each step and trajectory results (episode length
# and initial state returns).

def uniformrandompolicy(number_traj):
    dat = np.empty([0, 10])
    # column 1 keeps track of each trajectory step, column 2 stores the reward for each step
    outcomes = np.empty([number_traj, 2])
    # for each episode
    for epi in range(number_traj):
        # reset returns initial observation
        prev_obs = env.reset()
        for step_no in range(step_limit):
            # random action is 0 or 1 w prob 0.5
            action = (np.random.uniform()>0.50)
            # move made corresponding to random action as specified in question
            next_obs, reward, done, info = env.step(action)
            # reward modified to -1 on termination
            if done and step_no != (step_limit-1):
                reward = -1
            # reward modified to zero on non-terminating steps
            elif done and step_no == (step_limit-1):
                reward = 0
            else:
                reward = 0
            # observations for the current state stored
            tmp = np.concatenate([np.reshape(prev_obs, [1,4]), np.reshape(action, [1,1]),
                                   np.reshape(next_obs, [1,4]), np.reshape(reward, [1,1])], axis=1)
            # update observations
            prev_obs = next_obs
            dat = np.vstack((dat, tmp)) # save tmp
            # if gameover end
            if done:
                break
        outcomes[epi, 0] = step_no # keep track of outcome for initial state
        outcomes[epi, 1] = -(np.power(discount_factor, step_no))
    return outcomes, dat

# function call
A, B = uniformrandompolicy(number_traj)
print(A)


