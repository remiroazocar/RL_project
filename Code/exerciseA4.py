"""
Problem A4
Antonio Remiro Azocar
"""

import matplotlib.pyplot as plt
import numpy as np

# import required packages
import numpy as np
import gym
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# load the CartPole environment
env = gym.make('CartPole-v0')
# set random seed for environment
env._max_episode_steps = 300
env.reset() # start process
env.seed(23)
episode_no = 2000
discount_factor = 0.99

# set random seed for numpy
np.random.seed(23)
tf.set_random_seed(23)

# load/saving functionality (for models)
model_dir = "../models/A4" # directory for saving models
# if directory does not exist, create
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
filename = '/A4'

# place holders for neural net
learningrate = tf.placeholder(tf.float32)
action = tf.placeholder(tf.int32, [None, 2])
rew = tf.placeholder(tf.float32, [None, 1])
currentstate = tf.placeholder(tf.float32, [None, 4])
posteriorstate = tf.placeholder(tf.float32, [None, 4])

# parameters for neural net
sizehidden = 100
batchsize = 256
max_epi_length = 300
# learning rates
LRS = [0.02, 0.008, 0.002]
LRS = 0.02
result_container = np.zeros([((episode_no//20)), 2]) # container for performance of different models
# epsilon greedy
greedy_eps = 0.05

# weights
w = {'L1': tf.Variable(tf.truncated_normal([4, sizehidden], 0, 0.05)),
     'L2': tf.Variable(tf.truncated_normal([sizehidden, 2], 0, 0.05))}

# neural nets implemented to represent value function:
# these take as inputs the cartpole game current state and return action for such
# input observation connected to a hidden layer(100) – linear transformation
# + ReLU – followed by a linear layer with one output per action.
def NN_hidden(currentstate):
    hidden = tf.nn.relu(tf.matmul(currentstate, w['L1']))
    out = tf.matmul(hidden, w['L2'])
    return out

### 100 total iterations x no. have settled for 35
TOTAL_RUNS =35
FINAL_CONTAINER = np.zeros([(episode_no//20),2])

# BELLMAN
# done is 0 if game over, 1 otherwise
G_O = rew+1
# predictions
#  current state
Q = tf.reshape(tf.gather_nd(NN_hidden(currentstate), action), [-1,1])
# next state
Qnext = tf.reshape(tf.reduce_max(NN_hidden(posteriorstate)), [-1, 1])
Qnext = tf.multiply(G_O, Qnext) # 0 if G_0 is 0
delt = (rew + (discount_factor * tf.stop_gradient(Qnext))) - Q
# LOSS/TRAINInG DEFINITIONS
loss = tf.multiply(0.50, tf.reduce_mean(tf.square(delt)), name = "loss")
# gradient descent optimisation
train = tf.train.GradientDescentOptimizer(learning_rate=learningrate).minimize(loss)
prediction = tf.argmax((NN_hidden(currentstate)), axis=1)
# variable initialisation
init = tf.global_variables_initializer()
# saving functionality
saver = tf.train.Saver()


# container for bellman loss
# container for means
datdatdat = np.zeros([(episode_no//20),TOTAL_RUNS])
bellman_container = np.zeros([((episode_no//20)),TOTAL_RUNS])

RUN_INDEX =0;


"""
print("Training procedure starting")

while RUN_INDEX < TOTAL_RUNS:
    print("Run ", RUN_INDEX+1)
    with tf.Session() as sess:
        sess.run(init)
        j = 0
        for epi in range(episode_no):
            prev_obs = env.reset()
            k = 0
            while k < max_epi_length-1:
            # for k in range(max_epi_length):
                k = k+1
                # forward pass through NN gives action
                actionb = sess.run(prediction, feed_dict={currentstate: prev_obs.reshape(-1, 4)})
                # greedy policy w epsilon
                if np.random.uniform(0, 1) > greedy_eps:
                    actionb = actionb
                else:
                    actionb = [(np.random.uniform(0, 1) > 0.5) * 1]
                next_obs, reward, done, info = env.step(actionb[0])
                # reward modified to -1 on termination
                if done and k != (max_epi_length - 1):
                    reward = -1
                    env.reset()
                # reward modified to zero on non-terminating steps
                elif done and k == (max_epi_length - 1):
                    reward = 0
                    env.reset()
                else:
                    reward = 0
                # get action, reward, prev observation, next observation
                # create feed dict
                reward = np.reshape(reward, [-1, 1])
                feeddict = {currentstate: prev_obs.reshape(-1, 4), posteriorstate: next_obs.reshape(-1, 4),
                            action: np.reshape([[0, actionb[0]]], [-1, 2]), rew: reward, learningrate: LRS}
                _, bellman_loss = sess.run([train, delt], feed_dict=feeddict) # train NN
                bellman_loss = abs(bellman_loss)
                # Update observations
                prev_obs = next_obs
                if done:
                    env.reset()
                    break

            if epi % 20 == 0:
                dat = np.zeros([20, 1])
                for epi2 in range(20):
                    obs = env.reset()
                    k2 = 0
                    while k2 < max_epi_length-1:
                    # for k2 in range(max_epi_length):
                        k2 = k2+1
                        action2 = prediction.eval({currentstate: obs.reshape(-1, 4)})
                        obs, reward2, donedone, infoinfo = env.step(action2[0])
                        if donedone and k2 != (max_epi_length - 1):
                            reward2 = -1
                            env.reset()
                            break
                        elif donedone and k2 == (max_epi_length - 1):
                            reward2 = 0
                            env.reset()
                            break
                        else:
                            reward2 = 0
                    dat[epi2, 0] = k + 1
                mean = np.mean(dat, axis=0)
                result_container[j, 0] = np.mean(dat, axis=0)
                result_container[j, 1] = bellman_loss
                datdatdat[j, RUN_INDEX] = mean
                bellman_container[j,RUN_INDEX] =bellman_loss
                j = j+1
                print('episode: ', epi, '. average episode length: ', mean, '. bellman loss: ', bellman_loss)
        saver.save(sess, "../models/A4/A4")
        # saved model
        print('Model saved.')
    RUN_INDEX=RUN_INDEX+1
i=0

while i<(episode_no//20):
    FINAL_CONTAINER[i, 0] = np.mean(datdatdat[i,:])
    FINAL_CONTAINER[i, 1] = np.mean(bellman_container[i,:])
    i=i+1


# save results (performance/bellman loss)
filepath = 'A4finalresults'
np.save(filepath, FINAL_CONTAINER)
print(' All results (performance/bellman loss) saved.')
"""

#### TESTING IS ON THE FINAL MODEL
####### FINAL TEST ############
with tf.Session() as sess:
    sess.run(init)
    test = np.empty([0,1])
    print("Restoring model. Test in 100 games (150 max episode length).")
    saver.restore(sess, model_dir+filename)
    print("Model restored.")
    for x in range(100):
        prev_obs_test = env.reset()
        k3 = 0
        while k3<150-1:
            k3 = k3+1
            action_test = sess.run(prediction, feed_dict={currentstate: prev_obs_test.reshape(-1, 4)})
            next_obs_test, reward_test, done_test, indo_test = env.step(action_test[0])
            if done_test and k3 != (150 - 1):
                reward_test = -1
                env.reset()
            # reward modified to zero on non-terminating steps
            elif done_test and k3 == (150 - 1):
                reward_test = 0
                env.reset()
            else:
                reward_test = 0
            # update observation
            prev_obs_test = next_obs_test
            if done_test:
                break
        test = np.vstack([test, k3])
    avgepilengthtest = np.mean(test,axis=0)
    stdepilengthtest = np.std(test,axis=0)
    print('Final results. Average episode length for 100 tot games:')
    print(avgepilengthtest+1)

"""
# load results and make plots
print("Loading results (performance/bellman loss)")
perf = np.load('A4finalresults.npy')
print("Making plots")
xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
# xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
yvalues = perf[:,0]
plt.figure(0)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('average episode length')
plt.savefig('A4_performance_LR_0.02.png', bbox_inches='tight')
# plt.show()

xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
yvalues = perf[:,1]
plt.figure(1)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('Bellman loss')
plt.savefig('A4_bellmanloss_LR_0.02.png', bbox_inches='tight')
# plt.show()
"""
