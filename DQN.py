import numpy as np
import keras.backend.tensorflow_backend as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import gym
import gym_goddard
import random
import matplotlib.pyplot as plt
import pdb

#env = gym.make("MountainCar-v0")
env = gym.make('gym_goddard:Goddard-v0')

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50# 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 10#1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 6  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 1  # Terminal states (end of episodes)
MODEL_NAME = 'GoddardDQN'
MIN_REWARD = 1  # For model save
MEMORY_FRACTION = 0.20
ACTION_SPACE_SIZE = 2
BUCKETS = 20
SAVE = 1
# Environment settings
EPISODES = 12

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 1  # episodes
SHOW_PREVIEW = True

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


if tf.executing_eagerly():
    print('Executing eagerly')

print(f'tensorflow version {tf.__version__}')
print(f'tensorflow.keras version {tf.keras.__version__}')

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass


    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        #self._write_logs(stats, self.step)
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key,value,step=self.step)
                self.writer.flush()

# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def get_discrete_state(self, state):
        #pdb.set_trace()
        discrete_state = (state - env.observation_space.low)/discrete_os_win_size
        discrete_state[2] = discrete_state[2]-1
        #pdb.set_trace()
        current_states = np.stack(tuple(discrete_state.astype(np.int)),axis=0)
        return current_states

    def create_model(self):
        model = Sequential()
        #pdb.set_trace()
        model.add(Dense(8, input_shape=(3,))) #env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        # this converts our 3D feature maps to 1D feature vectors
        #model.add(Dense(64))

        model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        #model.run_eagerly = True
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        #discrete_state = agent.get_discrete_state(env.reset())
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) #Normalize
        #pdb.set_trace()
        discrete_states = agent.get_discrete_state(current_states)
        #pdb.set_trace()
        current_qs_list = self.model.predict(current_states)#(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        #pdb.set_trace() #Normalize
        new_discrete_states = agent.get_discrete_state(new_current_states)
        future_qs_list = self.target_model.predict(new_current_states)#(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (discrete_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(discrete_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        #pdb.set_trace()
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        #Normalize
        # Update target network counter every episode
        #pdb.set_trace()
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        #pdb.set_trace()
        #qs = self.model.predict(state)
        qs = self.model.predict(state.reshape(-1, *state.shape))[0]
        return qs# state.tolist() #.reshape(-1, *state.shape))[0] #Normalize


DISCRETE_OS_SIZE = [BUCKETS] * len(env.observation_space.high) #[20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# epsilon is the likelyhood of random actions, (exploration), this is where we find crazy non-conventional soln's
epsilon = 1
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    discrete_state = agent.get_discrete_state(env.reset())

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            #pdb.set_trace()
            action = [np.argmax(agent.get_qs(discrete_state))]
        else:
            # Get random action
            action = [np.random.randint(0, ACTION_SPACE_SIZE)]

        new_state, reward, done, _ = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward
        #pdb.set_trace()
        new_discrete_state = agent.get_discrete_state(new_state)

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        #pdb.set_trace()
        agent.update_replay_memory((discrete_state, action, reward, new_discrete_state, done))

        agent.train(done, step)

        discrete_state = agent.get_discrete_state(new_state)
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if episode % SAVE == 0:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        epsilon -= epsilon_decay_value
        if epsilon <= 0:
            epsilon = 0
