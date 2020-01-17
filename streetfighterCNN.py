import sys
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
import numpy as np
import gym
import retro
#import h5py

from CNNProcessor import CNNProcessor
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
from preprocessing import preprocess
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'


def main():
    env = retro.make(game=ENV_NAME, state='rom.state', use_restricted_actions=retro.Actions.DISCRETE)
    print(env.observation_space.shape)
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(1,) + (128, 100), data_format='channels_first'))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=2, activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    print(model.output)
    for layer in model.layers:
        print(layer.output_shape)
    for layer in model.layers:
        print(layer.input_shape)
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(processor=CNNProcessor(), model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == "__main__":
    main()