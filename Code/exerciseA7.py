"""
Problem A7
Antonio Remiro Azocar
"""
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
discount_factor = 0.99  # Discount factor

# set random seed for numpy
np.random.seed(23)
tf.set_random_seed(23)

# load/saving functionality (for models)
model_dir = "../models/A7" # directory for saving models
# if directory does not exist, create
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
filename = '/A7'

# place holders for neural net
learningrate = tf.placeholder(tf.float32)
action = tf.placeholder(tf.int32, [None, 2])
rew = tf.placeholder(tf.float32, [None, 1])
currentstate = tf.placeholder(tf.float32, [None, 4])
posteriorstate = tf.placeholder(tf.float32, [None, 4])
# parameters
sizehidden = 100
batchsize = 256
max_epi_length = 300
# learning rates
LRS = 0.0002
# Create container to store results:
result_container = np.zeros([((episode_no//20)), 2]) # container for performance of different models
# for greedy policy
greedy_eps = 0.05
sizebuffer = 65536 # buffersize

# weights
w = {'L1': tf.Variable(tf.truncated_normal([4, sizehidden], stddev=1.3) / np.sqrt([4, sizehidden]).sum()),
     'L2': tf.Variable(tf.truncated_normal([sizehidden, 2], stddev=1.3) / np.sqrt([sizehidden, 2]).sum())}


# neural net definition
# neural nets implemented to represent value function:
# take as inputs the cartpole game current state and return action for such
# here input observation connected to a hidden layer(100) – linear transformation
# + ReLU – followed by a linear layer with one output per action.
def NN_hidden(currentstate):
    hidden = tf.nn.relu(tf.matmul(currentstate, w['L1']))
    out = tf.matmul(hidden, w['L2'])
    return out

'''
# target weights (in functions now)
targw = {'L1': tf.Variable(tf.truncated_normal([4, sizehidden], stddev=1.3) / np.sqrt([4, sizehidden]).sum()),
         'L2': tf.Variable(tf.truncated_normal([sizehidden, 2], stddev=1.3) / np.sqrt([sizehidden, 2]).sum())}
'''



# BELLMAN EQUATIONS
# G_O is 0 if game over, 1 otherwise
G_O = rew+1
# predictions
# next state
Qnext = tf.reshape(tf.reduce_max(NN_hidden(posteriorstate)), [-1, 1])
Qnext = tf.multiply(G_O, Qnext) # 0 if G_0 is 0
#  current state
Q = tf.reshape(tf.gather_nd(NN_hidden(currentstate), action), [-1,1])
delt = rew + discount_factor * tf.stop_gradient(Qnext) - Q
# LOSS/TRAINING DEFINITIONS
loss = tf.multiply(0.50, tf.reduce_mean(tf.square(delt)), name="loss")
RMSdecay = 0.935
RMSmom = 0.8
# gradient descent optimisation
train = tf.train.RMSPropOptimizer(learning_rate=learningrate, decay = RMSdecay, momentum = RMSmom).minimize(loss)
prediction = tf.argmax((NN_hidden(currentstate)), axis=1)
# variable initialisation
init = tf.global_variables_initializer()
# saving functionality
saver = tf.train.Saver()
# TRAINING
print("Training procedure starting")



"""
import buffer_functions
# start session
i=0
with tf.Session() as sess:
    sess.run(init)
    rep = add2buffer(2560)
    j=0
    for epi in range(episode_no):
        prev_obs = env.reset()
        #  copy each 5
        if epi%5==0
            W1, W2 =sess.run(w['L1'], w['L2'], feed_dict={})
        k = 0
        while k < episode_no- 1:
            k = k+1
            # forward pass through NN gives action
            actionb = sess.run(prediction, feed_dict={currentstate: prev_obs.reshape(-1, 4)})
            # greedy policy w epsilon
            if np.random.uniform(0,1) > greedy_eps:
                actionb = actionb
            else:
                actionb = [(np.random.uniform(0,1) > 0.5) * 1]
            next_obs, reward, done, info = env.step(actionb[0])
            # reward modified to -1 on termination
            if done and k != (max_epi_length - 1):
                reward = -1
                env.reset()
            # reward modified to zero on non-terminating steps
            elif done and k == (max_epi_length - 1):
                reward = 0
                env.reset()
            add2buf([prev_obs, actionb, reward, next_obs, nd*1])
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
        a, b, c, d =  get_batch(replay.sample_buffer(batchsize))
        # if epi % 5 ==0:
           # targw =w
        if epi % 20 == 0:
            dat = np.zeros([100, 1])
            for epi2 in range(100):
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
            result_container[j, i] = np.mean(dat, axis=0)
            result_container[j, i+1] = bellman_loss
            j = j+1
            print('episode: ', epi, '. average episode length: ', mean, '. bellman loss: ', bellman_loss)
    savingpath = saver.save(sess, "../models/A7/A7")
    # saved model
    print('Model saved.')
    i = i+1
"""
with tf.Session() as sess:
    sess.run(init)
    test = np.empty([0,1])
    print("Restoring model. Test in 100 games (max episode length is 150).")
    saver.restore(sess, model_dir+filename)
    print("Model restored.")
    for x in range(100):
        prev_obs_test = env.reset()
        k3 = 0
        while k3<150-1:
            k3 = k3+1
            action_test = sess.run(prediction, feed_dict={currentstate: prev_obs_test.reshape(-1, 4)})
            next_obs_test, reward_test, done_test, info_test = env.step(action_test[0])
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
