import sys
#sys.path.append("O:\Oliver\Anaconda\envs\gym\Lib\site-packages")
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
import argparse
import retro
#import h5py
from CNNProcessor import CNNProcessor
from InfoCallbackTrain import InfoCallbackTrain
from InfoCallbackTest import InfoCallbackTest
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
import os.path
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from trainingMetrics import plot_reward, plot_wins, STATE_NAME

ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'

def buildModel(weight_path, num_actions):
    model = Sequential()
    # Conv1 32 32 (3) => 30 30 (32)
    # model.add(Conv2D(32, (3, 3), input_shape=X_shape[1:]))
    model.add(Conv2D(32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(1,) + (128, 100), data_format='channels_first'))
    model.add(Activation('relu'))
    # Conv2 30 30 (32) => 28 28 (32)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    # Pool1 28 28 (32) => 14 14 (32)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Conv3 14 14 (32) => 12 12 (64)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # Conv4 12 12 (64) => 6 6 (64)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # Pool2 6 6 (64) => 3 3 (64)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # FC layers 3 3 (64) => 576
    model.add(Flatten())
    # Dense1 576 => 256
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    # Dense2 256 => 10
    model.add(Dense(num_actions))
    model.add(Activation('softmax'))

    # number of steps? and policy used for learning
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    if weight_path != None:
        model.load_weights(weight_path)

    return (model, memory, policy)

def buildAgent(model, memory, policy, num_actions):
    dqn = DQNAgent(processor=CNNProcessor(), model=model, nb_actions=num_actions, memory=memory, 
                    nb_steps_warmup=10000, target_model_update=1e-3, policy=policy, test_policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn

def trainModel(env, dqn, state_path, weight_path):
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2, callbacks=[InfoCallbackTrain()], action_repetition=4)
    dqn.save_weights(weight_path, overwrite=True)

def testModel(env, dqn, num_episodes, state_path):
    dqn.test(env, nb_episodes=num_episodes, visualize=True, callbacks=[InfoCallbackTest()])

def main(mode):

    print("\nState: ", STATE_NAME)
    
    env = retro.make(game=ENV_NAME, state=STATE_NAME, use_restricted_actions=retro.Actions.DISCRETE)
    num_actions = env.action_space.n

    if mode == "train":
        
        WEIGHT_PATH = 'weights/dqn_cnn_{}_weights.h5f'.format(STATE_NAME)

        if os.path.exists(WEIGHT_PATH):
            print("Loading weights from: ", WEIGHT_PATH, '\n')
            (model, memory, policy) = buildModel(WEIGHT_PATH, num_actions)
        else:
            print("No weights found for current state.\n")
            (model, memory, policy) = buildModel(None, num_actions)
           
        dqn = buildAgent(model, memory, policy, num_actions)
        trainModel(env, dqn, STATE_NAME, WEIGHT_PATH)
        plot_wins(mode)

    elif mode == "test":

        WEIGHT_PATH = 'weights/dqn_cnn_{}_weights.h5f'.format(STATE_NAME)

        if os.path.exists(WEIGHT_PATH):
            print("Loading weights from: ", WEIGHT_PATH, '\n')
            (model, memory, policy) = buildModel(WEIGHT_PATH, num_actions)
        else:
            print("No weights found for current state.\n")
            (model, memory, policy) = buildModel(None, num_actions)
           
        dqn = buildAgent(model, memory, policy, num_actions)
        testModel(env, dqn, 100, STATE_NAME)
        plot_wins(mode)

    elif mode == "testscript":
        weightTest()
    else:
        print("No mode specified")

    #WEIGHT_PATH = 'weights/dqn_cnn_ryu4.state_weights.h5f'.format(STATE_NAME)
    
    


def weightTest():

    weight_paths = [
        'dqn_ryu1.state_weights.h5f',
        'dqn_cnn_ryu1.state_weights.h5f',
        'dqn_cnn_ryu4.state_weights.h5f'
    ]

    state_paths = [
        'ryu1.state',
        'ryu1.state',
        'ryu4.state'
    ]

    #for i in range(len(weight_paths)):


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train or test')
    args = parser.parse_args()
    main(args.mode)
