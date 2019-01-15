"""
Problem A3ii (hidden)
Antonio Remiro Azocar
"""
# import required packages
import numpy as np
import gym
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# directory where we saved data for exercise 3
new_dir = "../data"

# 2000 trajectories generated in exerciseA2.py are being loaded
print('2000 trajectories generated in exerciseA2.py are being loaded.')
trajsfromA2 = np.load(new_dir+"/forQ3.npy")
print('Data has been loaded.')
datasize = trajsfromA2.shape[0]

# load the CartPole environment
env = gym.make('CartPole-v0')
# set random seed for environment
env.seed(17)
env._max_episode_steps = 300
env.reset() # start process

# parameters
discount_factor = 0.99

# set random seed for numpy
np.random.seed(17)
tf.set_random_seed(17)

# load/saving functionality
model_dir = "../models/A3ii" # directory for saving models
# if directory does not exist, create
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
filename = '/A3ii'

#### NEURAL NET ####

# place holders for neural net
learningrate = tf.placeholder(tf.float32)
action = tf.placeholder(tf.int32, [None, 2])
rew = tf.placeholder(tf.float32, [None, 1])
currentstate = tf.placeholder(tf.float32, [None, 4])
posteriorstate = tf.placeholder(tf.float32, [None, 4])

# parameters for neural net
sizehidden = 100
batchsize = 256
# epoch number
episode_no = 2000
max_epi_length = 300
# learning rates
LRS = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
result_container = np.zeros([(episode_no//20), 12]) # container for performance of different models

# weights
w = {'L1': tf.Variable(tf.truncated_normal([4, sizehidden], 0, 0.05)),
     'L2': tf.Variable(tf.truncated_normal([sizehidden, 2], 0, 0.05))}

# two neural nets implemented to represent value function:
# these take as inputs the cartpole game current state and return action for such
# i) input observation is connected to a linear layer w one output per action
# insert linear NN here.
# ii) input observation connected to a hidden layer(100) – linear transformation
# + ReLU – followed by a linear layer with one output per action.
def NN_hidden(currentstate):
    hidden = tf.nn.relu(tf.matmul(currentstate, w['L1']))
    out = tf.matmul(hidden, w['L2'])
    return out

# BELLMAN
# done is 0 if game over, 1 otherwise
G_O = rew+1
# predictions
# current state
Q = tf.reshape(tf.gather_nd(NN_hidden(currentstate), action),[-1, 1])
# next state
Qnext = tf.multiply(G_O, tf.reshape(tf.reduce_max(NN_hidden(posteriorstate), axis=1), [-1,1]))
#
delt = (rew + (discount_factor * tf.stop_gradient(Qnext))) - Q
# LOSS/TRAINING DEFINITIONS
loss = tf.multiply(0.50, tf.reduce_mean(tf.square(delt)), name ='loss')
# gradient descent optimisation
train = tf.train.GradientDescentOptimizer(learning_rate=learningrate).minimize(loss)
prediction = tf.argmax((NN_hidden(currentstate)), axis=1)
# variable initialisation
init = tf.global_variables_initializer()

# saving functionality
saver = tf.train.Saver()

"""

print("Training procedure starting")

top_score = -10000 # keeps track of best model
i = 0
for lr in LRS:
    print('Training with learning rate', lr)
    w = {'L1': tf.Variable(tf.truncated_normal([4, sizehidden], 0, 0.05)),
         'L2': tf.Variable(tf.truncated_normal([sizehidden, 2], 0, 0.05))}
    with tf.Session() as sess:
        sess.run(init)
        w = {'L1': tf.Variable(tf.truncated_normal([4, sizehidden], 0, 0.05)),
             'L2': tf.Variable(tf.truncated_normal([sizehidden, 2], 0, 0.05))}
        j = 0
        for epi in range(episode_no):
            np.random.shuffle(trajsfromA2)    # shuffle data
            batch_indices = datasize//batchsize
            k = 0
            while k<batch_indices-1:
                k = k+1
                # fetch batch
                li = batchsize*k
                ui = batchsize*(k+1)  # check upper index
                temp_data = trajsfromA2[li:ui,:]
                # compute reward, action, previous observation, new observation for batch
                # feeddict # cols 1-4 are current state, cols 5-8 are next state
                batch_feeddict = {currentstate: temp_data[:, :4], posteriorstate: temp_data[:,5:9],
                                  action: np.c_[np.arange(temp_data.shape[0]), temp_data[:, 4]],
                                  rew: np.reshape(temp_data[:,9], [-1, 1]), learningrate: lr}
                _ = sess.run(train, feed_dict=batch_feeddict) # train NN
            # check every episode (no division by 20)
            if epi % 20 == 0:
                # shrink (time constraints)
                sizetest = trajsfromA2.shape[0]
                datab = trajsfromA2[:sizetest,:]
                feeddict = {currentstate: datab[:, :4], posteriorstate: datab[:, 5:9],
                            action: np.c_[np.arange(datab.shape[0]), datab[:, 4]],
                            rew: np.reshape(datab[:,9], [-1, 1]), learningrate: lr}
                err = loss.eval(feed_dict=feeddict)
                bellman_loss_test = delt.eval(feed_dict=feeddict)
                bellman_loss_test = abs(bellman_loss_test)
                dat = np.zeros([50,1])
                for epi2 in range(50):
                    obs = env.reset()
                    k2 = 0
                    while k2 < max_epi_length-1:
                        k2=k2+1
                        # prediction
                        action2 = prediction.eval({currentstate: obs.reshape(-1, 4)})
                        obs, reward2, donedone, infoinfo = env.step(action2[0])
                        # reward update
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
                    # Store the number of steps of the last game
                    dat[epi2, 0] = k2+1
                avg = np.mean(dat, axis = 0)
                result_container[j, (2*i)] = np.mean(dat, axis=0)

                bellman_loss = 0.5*np.sqrt(err)
                result_container[j, (2 * i) + 1] = bellman_loss
                print('episode: ', epi, '. average episode length: ', avg, '. bellman loss: ', bellman_loss)
                j = j+1
        # process for finding best model -> we only save the best one
        lr_performance = np.mean(result_container[:, 2*i])
        if lr_performance > top_score:
            # we judge each learning rate based on the mean of all the average episode lengths
            print('Best performing learning rate so far is', lr)
            saver.save(sess, "../models/A3ii/A3ii")
            print('Saved the best performing model')
            top_score = lr_performance
    i = i+1

"""

# save results (performance/bellman loss)
filepath = 'A3iifinalresults'
np.save(filepath, result_container)
print(' All results (performance/bellman loss) saved.')

#### TESTING IS ON THE FINAL MODEL
####### FINAL TEST ############
with tf.Session() as sess:
    sess.run(init)
    test = np.empty([0,1])
    print("Restoring model (best performing). Test in 100 games (max episode length set to 150).")
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
                env.reset()
            # update observation
            prev_obs_test = next_obs_test
            if done_test:
                break
        test = np.vstack([test, k3])
    avgepilengthtest = np.mean(test,axis=0)
    stdepilengthtest = np.std(test,axis=0)
    print('Final results for best performing model. Average episode length for 100 tot games:')
    print(avgepilengthtest+1)


"""
# load results and make plots
print("Loading results (performance/bellman loss)")
perf = np.load('A3iifinalresults.npy')

print("Making plots")
xvalues = np.linspace(0,episode_no,num=(episode_no//20))
yvalues = perf[:,0]
plt.figure(0)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('average episode length')
plt.savefig('A3ii_performance_LR1.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
yvalues = perf[:,1]
plt.figure(1)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('Bellman loss')
plt.savefig('A3ii_bellmanloss_LR1.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=(episode_no//20))
yvalues = perf[:,2]
plt.figure(2)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('average episode length')
plt.savefig('A3ii_performance_LR2.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
yvalues = perf[:,3]
plt.figure(3)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('Bellman loss')
plt.savefig('A3ii_bellmanloss_LR2.png', bbox_inches='tight')

print("Making plots")
xvalues = np.linspace(0,episode_no,num=(episode_no//20))
yvalues = perf[:,4]
plt.figure(4)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('average episode length')
plt.savefig('A3ii_performance_LR3.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
yvalues = perf[:,5]
plt.figure(5)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('Bellman loss')
plt.savefig('A3ii_bellmanloss_LR3.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=(episode_no//20))
yvalues = perf[:,6]
plt.figure(6)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('average episode length')
plt.savefig('A3ii_performance_LR4.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
yvalues = perf[:,7]
plt.figure(7)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('Bellman loss')
plt.savefig('A3ii_bellmanloss_LR4.png', bbox_inches='tight')

print("Making plots")
xvalues = np.linspace(0,episode_no,num=(episode_no//20))
yvalues = perf[:,8]
plt.figure(8)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('average episode length')
plt.savefig('A3ii_performance_LR5.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
yvalues = perf[:,9]
plt.figure(9)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('Bellman loss')
plt.savefig('A3ii_bellmanloss_LR5.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=(episode_no//20))
yvalues = perf[:,10]
plt.figure(10)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('average episode length')
plt.savefig('A3ii_performance_LR6.png', bbox_inches='tight')

xvalues = np.linspace(0,episode_no,num=((episode_no//20)))
yvalues = perf[:,11]
plt.figure(11)
plt.plot(xvalues, yvalues)
plt.xlabel('episodes')
plt.ylabel('Bellman loss')
plt.savefig('A3ii_bellmanloss_LR6.png', bbox_inches='tight')
"""