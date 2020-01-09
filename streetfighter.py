import sys
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
#sys.path.append("O:\Oliver\Anaconda\envs\gym\Lib\site-packages")

import retro
#import h5py
from InfoCallback import InfoCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from trainingMetrics import plot_reward

ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'

def main():
    env = retro.make(game=ENV_NAME, state='rom.state', use_restricted_actions=retro.Actions.DISCRETE)
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    # Uncomment the following line to load the model weights from file
    #model.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100000,
                   target_model_update=1e-3, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    training_history = dqn.fit(env, nb_steps=1000000, visualize=True, verbose=2, callbacks=[InfoCallback()], action_repetition=4)
    plot_reward(training_history)

    # Uncomment the following line to overwrite the model weights file after training
    #dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == "__main__":
    main()
