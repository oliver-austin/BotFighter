import argparse
from street_fighter_convolutional import StreetFighter

NUM_FRAME = 1000000

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="dqn_training")
parser.add_argument("-r", "--render", default=True, type=bool)
parser.add_argument("-tsl", "--total_step_limit", default=5000000, type=int)
parser.add_argument("-trl", "--total_run_limit", default=None, type=int)
args = parser.parse_args()

game = StreetFighter()
game.train(args.render, args.total_step_limit, args.total_run_limit)