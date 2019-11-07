"""This model was built using the DeepMind 'Human=level control through deep reinforcement learning' research paper
as a reference. The research paper is available here:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf"""

import sys
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import RMSprop
import numpy as np
import preprocessing

# List of hyper-parameters and constants
DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 3000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 126
TAU = 0.01
# Number of frames to throw into network
NUM_FRAMES = 3


class StreetFighterConvolutionalModel:

    def __init__(self, input_shape, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=8, strides=4, activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=action_space.n, activation='linear'))
        self.model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        self.model.summary()

        self.model_target = Sequential()
        self.model_target.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=8, strides=4, activation='relu'))
        self.model_target.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'))
        self.model_target.add(Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'))
        self.model_target.add(Flatten())
        self.model_target.add(Dense(units=512, activation='relu'))
        self.model_target.add(Dense(units=action_space.n, activation='linear'))
        self.model_target.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        self.model_target.summary()

    def predict(self, data, epsilon):
        q_actions = self.model.predict(data.reshape(1, 200, 256, 3), batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, batch_action, batch_state, batch_reward, batch_done, batch_next_state, obs_num):
        batch_size = batch_state.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in range(batch_size):
            targets[i] = self.model.predict(batch_state[i].reshape(1, 200, 256, NUM_FRAMES), batch_size=1)
            future_action = self.model_target.predict(batch_next_state[i].reshape(1, 200, 256, NUM_FRAMES), batch_size=1)
            targets[i, batch_action[i]] = batch_reward[i]
            if batch_done[i] == False:
                targets[i, batch_action[i]] += DECAY_RATE * np.max(future_action)
            loss = self.model.train_on_batch(batch_state, targets)
            print("loss: ", loss)

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.model_target.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
            self.model_target.set_weights(target_model_weights)
