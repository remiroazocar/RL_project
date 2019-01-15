import numpy as np
import tensorflow as tf
import gym
import os

# set tensorflow and numpy seeds
np.random.seed(55)
tf.set_random_seed(55)

#parameters
LRS = 0.0001 # the same learning rate is used for Pacman, Boxing whereas for Pong we use 0.0001
# parameters
discount_factor = 0.99
batchsize = 3200 # batch size increased
experience_repbuf_size = 3*10e5 # total size of experience replay buffer (memory)
greedy_eps =1  # this is gradually reduced to 0.1
reward_clipping_ul = 1
reward_clipping_ll = -1
sc= [] # to count steps
rm = []
sbuff = [] # buffer w states

# IF PONG
# env = gym.make('Pong-v3')
# IF MS.PACMAN
# env = gym.make('MsPacman-v3')
# IF BOXING
env = gym.make('Boxing-v3')
env.seed(55)
env._max_episode_steps = 300
env.reset() # start process

# load/saving functionality (for models)
model_dir = "../models/B_Boxing" # directory for saving models
# if directory does not exist, create
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
filename = '/Boxing'

# load/saving functionality (for models)
model_dir = "../models/B_Pacman" # directory for saving models
# if directory does not exist, create
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
filename = '/Pacman'

# load/saving functionality (for models)
model_dir = "../models/B_Pong" # directory for saving models
# if directory does not exist, create
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
filename = '/Pong'

LR = 0.0001
output_size = env.observation_space.shape[0]
stride = 2
const_init = 0.015
episode_length =5000
# rew_ update_every_x_epis = 50
update_batch_every_x_epis =20;
#update_batch_every_x_epis =10;
update_container_every = 50000
result_container = np.zeros([70000, 2])



input = tf.placeholder(tf.float32, [None, 4, 60, 60])
input = tf.transpose(input, [0,2,3,1])
# LAYER1: filter size 6x6, stride 2 16 channels followed by relu
W1 = tf.get_variable("weight_1", shape=[8, 8, 4, 16])
b1 = tf.constant(const_init, shape=[16])
hidden1 = tf.nn.relu(tf.nn.conv2d(input, W1, strides=[1,stride, stride,1], padding='SAME')+b1)
# LAYER 2: filter size 4x4, stride2, 32 channels followed by relu
W2 = tf.get_variable("weight_2", shape=[4, 4, 16, 32])
b2 =tf.constant(const_init, shape=[32])
hidden2 = tf.nn.relu(tf.nn.conv2d(hidden1, W2, strides=[1,stride, stride,1], padding='SAME')+b2)
# LAYER3 - flatten and then fully connected layer od 256 units followed by relu
W3 = tf.get_variable("weight_3", shape=[2048, 256])
b3 = tf.constant(const_init, shape=[256])
hidden3 = tf.nn.relu(tf.matmul(tf.reshape(hidden2, [-1, 2048]), W3) + b3)
# LAYER 4 - linear value (softmax) predicting state-action value
W4 = tf.get_variable("weight_4", shape=[256, output_size])
b4 = tf.constant(const_init, shape=[output_size])
out = tf.matmul(hidden3, W4) + b4


w = [W1, W2, W3, W4, b1,b2 ,b3,b4]

action = tf.placeholder(tf.float32, [None, output_size])
Q = tf.placeholder(tf.float32, [None])
Qsquare = tf.multiply(out, action)
qvals = tf.reduce_sum(Qsquare, reduction_indices=[1])
# delta in worksheet, delta^2 is bellman loss?
# delta = tf.squared_difference(qvals, Q)
# loss term in worksheet
loss = tf.reduce_mean(0.5*tf.squared_difference(qvals, Q))
optimiser = tf.train.RMSPropOptimizer(learning_rate=LR)
train = optimiser.minimize(loss)

stacked_frames = 4

# IMAGE PROCESSING PACKAGES
# from skimage.transform import resize
# from skimage.color import rgb2gray
# variable initialisation
init = tf.global_variables_initializer()
saver =tf.train.Saver()

#TRAINING here
"""
import random
import buffer_functions
with tf.Session() as sess:
    sess.run(init)
    # targets
    targw = sess.run(w)
    runrew = 0 # running reward initialisation
    k = 0 # k keeps track of number of steps
    j=0 # current episode
    prev_obs = env.reset()
    # environment observations reduced in size to 84 x 84 images (in parts i, ii, these
    # are reduced to 28 x28), converted to grey scale
    prev_obs = resize(rgb2gray(prev_obs), (84, 84))[10:70,10:70]
    # stack 4 consecutive frames together to get single agent state (of size 28x28x4)-store as bytes to conserve
    state_tplus1 = np.stack([prev_obs for i in range(stacked_frames)], axis= 0)
    statet = state_tplus1
    for k in range(stacked_frames-1):
        sbuff.append(prev_obs)
    rb = [] # keeps track of in-game scores
    lb = [] # keeps track of loss
    no_loss = 0 # counter of losses
    no_wins = 0 # counter of wins
    c = 0 # another counting index
    targw = sess.run(w)
    agent_steps_limit = 5000000 # the total upper limit of steps is 5 million, although exercise only requires 1m
    # we can explore overfitting
    for agent_step in range(agent_steps_limit):
        # action container
        actionb = np.zeros([range(output_size)])
        if np.random.uniform(0,1) > greedy_eps:
            Qvals = sess.run(Q, feed_dict={X: [statet]})
            id = np.argmax(Qvals)
            actionb[id]=1
        else:
            id = random.randrange(range(output_size))
            actionb[id] = 1
        xx = range(output_size)
        new_obs, reward2, donedone, infoinfo = env.step(xx[id])
        # gradually reduce epsilon as number of steps increases
        if 0 <= k < 50000:
            greedy_eps =1
        if 50000 <= k < 100000:
            greedy_eps = 0.8
        if 100000 <= k < 200000:
            greedy_eps = 0.6
        if 200000 <= k < 300000:
            greedy_eps = 0.5
        if 300000 <= k < 400000:
            greedy_eps = 0.4
        if 400000 <= k < 500000:
            greedy_eps = 0.3
        if k >= 500000:
            greedy_eps = 0.2
        if k >= 800000:
            greedy_eps = 0.1
        # rewards are clipped -> then decide if loss or win
        if np.clip(reward2, reward_clipping_ll, reward_clipping_ul) < 0:
            no_loss=no_loss+1
        elif np.clip(reward2, reward_clipping_ll, reward_clipping_ul) > 0:
            no_wins=no_wins+1
        new_obs = resize(rgb2gray(new_obs), (84, 84))[10:70,10:70];
                     previous_frames = np.array(self.state_buffer)
        previous_frames = np.array(self.state_)
        state_tplus1 = np.empty((stacked_frames, 60, 60))
        state_tplus1[:stacked_frames-1, ...] = np.array(sbuff)
        state_tplus1[stacked_frames-1] = new_obs
        sbuff.popleft()
        sbuff.popleft.append(new_obs)
        tot_reward = tot_reward+np.clip(reward2, reward_clipping_ll, reward_clipping_ul)
        if done:
            if runrew=0:
                runrew = runrew+tot_rewards
            else:
                runrew = (1/100)*tot_reward + (99/100)*runrew
            j=j+1 # add one to episode num
            iii = j%update_batch_every_x_epis
            if iii == 0:
                rb.append(runrew)
        feeddict = {input: [state_tplus1]}
        feeddict = zip_updt(feed_dict,w,targw)
        qvalsvals = sess.run(out, feed_dict=feeddict)
        if done and agent_step != (agent_steps_limit - 1):
            tq=reward2
        else:
            tq=np.max(qvalsvals)*0.99+reward2
        k = k+1
        rm.append(state_t, tq, id)
        state_tplus1 =state_t
        if done and len(rm) <= 5000:
             tot_rewards = 0
             prev_obs = env.reset()
             prev_obs = resize(rgb2gray(prev_obs), (84, 84))[10:70,10:70]
             state_tplus1 = np.stack([prev_obs for zzz in range(stacked_frames)], axis = 0)
             state_t = state_tplus1
        elif len(rm) >= episode_length:  # once episode is undergone
            if k%200==0:
                mb = random.sample(rm, 3200)
                for b in mb:
                    new_stts = b[0]
                    new_targ = b[1]
                newfeeddict = {input:new_stts, Q: new_targ}
                L=  sess.run([loss], feed_dict={input:new_stts, Q: new_targ})
                TL = TL+L  # complete loss over batch
                c =c+1
        if done:

            lb.append(TL/c) if c>0 update losses
            prev_obs = env.reset()
            prev_obs = resize(rgb2gray(prev_obs), (84, 84))[10:70,10:70]
            state_tplus1 = np.stack([prev_obs for zzz in range(stacked_frames)], axis = 0)
            state_t = state_tplus1

# save results (performance/bellman loss)
filepath = 'B34finalresults'
np.save(filepath, result_container)
print(' All results saved.')


# load results
print("Loading results")
perf = np.load('B34finalresults.npy')
print("Saving plots")


# make plots
import matplotlib.pyplot as plt

"""


# remember to change to corresponding environment

# load models
print('Loading models')
sess = tf.Session()

# load pong model
print('Loading Pong model')
new_saver = tf.train.import_meta_graph('../models/B_Pong/pong.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('../models/B_Pong/'))
x = tf.get_collection('x')
q = tf.get_collection('q')
print('Pong model loaded')

# load boxing model
print('Loading Boxing model')
new_saver = tf.train.import_meta_graph('../models/B_Boxing/boxing.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('../models/B_Boxing/'))
x = tf.get_collection('x')
q = tf.get_collection('q')
print('Boxing model loaded')
# load ms pacman model
print('Loading Ms.Pacman model')  # takes longer to load - metadata was stored in much shorter intervals
new_saver = tf.train.import_meta_graph('../models/B_Pacman/pacman.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('../models/B_Pacman/'))
x = tf.get_collection('x')
q = tf.get_collection('q')
print('Ms.Pacman model loaded')

# loading results (didnt saveto collection, saved as .np)
print('Loading full results (stored as numpy file)')
print('Loading Pong results')
pong_results = np.load('../data/pong_results.p')
print('Pong results loaded')
# load ms pacman model
print('Loading Ms.Pacman results')
pacman_results = np.load('../data/pacman_results.p')
print('Ms.Pacman results loaded')
# load boxing model
print('Loading Boxing results')
boxing_results = np.load('../data/boxing_results.p')
print('Boxing results loaded')