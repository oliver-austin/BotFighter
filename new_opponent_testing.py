import sys
sys.path.append("O:\Oliver\Anaconda\envs\gym\Lib\site-packages")
#sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
import argparse
import retro
#import h5py
from InfoCallbackTest import InfoCallbackTest
from mainSF import buildDQNAgent, buildModel

ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'


def main(difficulty):

    state_paths = [
        'ryu3dhalsim.state',
        'ryu4dhalsim.state',
        'ryu5dhalsim.state',
        'ryu3ehonda.state',
        'ryu4ehonda.state',
        'ryu5ehonda.state'
    ]

    (model, memory, policy) = buildModel('weights/dqn_cnn_ryu4.state_weights.h5f', 126)
    for state_path in state_paths:
        env = retro.make(game=ENV_NAME, state=state_path, use_restricted_actions=retro.Actions.DISCRETE)
        dqn = buildDQNAgent(model, memory, policy, 126)
        dqn.test(env, nb_episodes=100, visualize=True, callbacks=[InfoCallbackTest(state_path)])
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty', help='0, 1, or 2')
    args = parser.parse_args()
    main(args.difficulty)
