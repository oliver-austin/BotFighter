import sys
sys.path.append("O:\Oliver\Anaconda\envs\gym\Lib\site-packages")
#sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
import argparse
import retro
#import h5py
from InfoCallbackShowcase import InfoCallbackShowcase
from mainSF import buildDQNAgent, buildModel

ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'


def main(difficulty):

    state_paths = [
        'ryu1guile.state',
        'ryu4guile.state',
        'ryu8guile.state'
    ]

    (model, memory, policy) = buildModel('weights/dqn_cnn_ryu4.state_weights.h5f', 126)

    state_path = state_paths[int(difficulty)]
    env = retro.make(game=ENV_NAME, state=state_path, use_restricted_actions=retro.Actions.DISCRETE)
    dqn = buildDQNAgent(model, memory, policy, 126)
    dqn.test(env, nb_episodes=100000, visualize=True, callbacks=[InfoCallbackShowcase(state_path)])
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty', help='0, 1, or 2')
    args = parser.parse_args()
    main(args.difficulty)
