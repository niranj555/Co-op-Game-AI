from ple.games.SpaceInvadersMGame import SpaceInvadersGame
from ple import PLE
import numpy as np
import pygame
import sys
import tensorflow as tf
import random
from collections import deque
import cv2
import argparse
import os.path

#sys.path.append('/usr/local/lib/python2.7/site-packages')

parser  = argparse.ArgumentParser(description="Space Invaders Training or Play!")
parser.add_argument('-t', '--train', action='store', help='train flag')
args    = parser.parse_args()

MODE                = 'train' if args.train else 'play'
SAVED_NETWORKS_DIR  = 'saved_models'
SAVED_AI_NETWORKS_DIR  = 'saved_models_ai'
NETWORKNAME         = args.train
GAME                = 'SpaceInvaders' # name of game
ACTIONS             = 3 # number of possible actions for human player network (go left or right or shoot)
ACTIONS_AI          = 3 # number of possible actions for AI player network (go left or right or shoot)
GAMMA               = 0.95 # discounted factor of future reward
OBSERVE             = 1000 # timesteps to observe before training
EXPLORE             = 3000000 # timesteps to train
FINAL_EPSILON       = 0.0001 # final value of exploration rate
INITIAL_EPSILON     = 0.5  # starting value of exploration rate
REPLAY_MEMORY       = 50000 # number of experiences stored
BATCH               = 32 # size of minibatch

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1   = weight_variable([1600, 512])
    b_fc1   = bias_variable([512])

    W_fc2   = weight_variable([512, ACTIONS])
    b_fc2   = bias_variable([ACTIONS])

    s       = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3      = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1   = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout

def createNetwork_AI():
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1   = weight_variable([1600, 512])
    b_fc1   = bias_variable([512])

    W_fc2   = weight_variable([512, ACTIONS])
    b_fc2   = bias_variable([ACTIONS])

    s       = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    h_conv3      = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1   = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout


def trainNetwork(s, readout, s_ai, readout_ai, sess, sess_ai):
    
    # saving and loading networks
    saver = tf.compat.v1.train.Saver()
    with sess.as_default():
        sess.run(tf.compat.v1.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORKS_DIR)
        print("SavedDir", SAVED_NETWORKS_DIR)
        print("CHECKPOINT ",checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded model:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y = tf.compat.v1.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout_ai, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

    game = SpaceInvadersGame(width=800, height=600)
    p = PLE(game, fps=30, display_screen=True, force_fps=False)
    p.init()
    actions = p.getActionSet()
    # store the previous observations in replay memory
    D = deque()

    x_t = p.getScreenRGB()
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)


    
    #tf.reset_default_graph()

    saver_ai = tf.compat.v1.train.Saver()
    with sess_ai.as_default():
        sess_ai.run(tf.compat.v1.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(SAVED_AI_NETWORKS_DIR)
        print("SavedDir", SAVED_AI_NETWORKS_DIR)
        print("CHECKPOINT ",checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver_ai.restore(sess_ai, checkpoint.model_checkpoint_path)
            print("Successfully loaded ai model:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        with sess.as_default():
            readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        with sess_ai.as_default():
            readout_t_ai = readout_ai.eval(feed_dict={s_ai : [s_t]})[0]
        
        a_t = np.zeros([ACTIONS])
        action_index = np.argmax(readout_t)
        a_t[action_index] = 1
        
        a_t_ai = np.zeros([ACTIONS])
        action_index_ai = 0
        
        if MODE == 'train':
            if random.random() <= epsilon:
                print("-"*10 + "Random Action" + "-"*10)
                action_index_ai = random.randrange(ACTIONS)
                a_t_ai[action_index_ai] = 1
            else:
                action_index_ai = np.argmax(readout_t_ai)
                a_t_ai[action_index_ai] = 1
        else:
            action_index_ai = np.argmax(readout_t_ai)
            a_t_ai[action_index_ai] = 1

        r_t = p.act(actions[np.argmax(a_t_ai)*3 + np.argmax(a_t)])
        terminal_t = p.game_over()

        x_t1_colored = p.getScreenRGB()
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        s_t = s_t1
        t  += 1
        if terminal_t:
            p.reset_game()

        if MODE == 'train':
            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # store the transition in D
            D.append((s_t, a_t_ai, r_t, s_t1, terminal_t))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            if t > OBSERVE:
                minibatch  = random.sample(D, BATCH)
                s_j_batch  = [d[0] for d in minibatch]
                a_batch    = [d[1] for d in minibatch]
                r_batch    = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]
                y_batch    = []

                with sess_ai.as_default():
                    readout_j1_batch = readout_ai.eval(feed_dict = {s_ai : s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                with sess_ai.as_default():
                    train_step.run(feed_dict = {
                        y : y_batch,
                        a : a_batch,
                        s : s_j_batch}
                )

            # save progress every 10000 steps
            if t % 10000 == 0:
                saver_ai.save(sess_ai, NETWORKNAME + '-', global_step = t)

            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE:
                state = "train"
            print(("Assistant MODE: Train. TIMESTEP: %d. STATE: %s. EPSILON: %.3f. ACTION: %d. REWARD: %d. Q[%.3f, %.3f]")
                % (t, state, epsilon, action_index_ai, r_t, readout_t_ai[0], readout_t_ai[1]))
        else:
            print(("Assistant MODE: Play. ACTION: %d. REWARD: %d. Q[%.3f, %.3f]")
                % (action_index_ai, r_t, readout_t_ai[0], readout_t_ai[1]))



def playGame():
    sess = tf.compat.v1.Session()
    sess_ai = tf.compat.v1.Session()
    s, readout = createNetwork()
    s_ai, readout_ai = createNetwork_AI()
    trainNetwork(s, readout, s_ai, readout_ai, sess, sess_ai)

def main():
    playGame()

main()