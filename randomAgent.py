import sys
#sys.path.append("O:\Oliver\Anaconda\envs\gym\Lib\site-packages")
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
import argparse
import retro
from trainingMetrics import save_wins


ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'
state = 'rom.state'

def main(nb_episodes):
    env = retro.make(game=ENV_NAME, use_restricted_actions=retro.Actions.DISCRETE)
    obs = env.reset()
    player_win = False
    episode = 0
    while episode < nb_episodes:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if info['matches_won'] == 2:
            player_win = True
        if done or player_win:
            episode = episode + 1
            if player_win == False:
                save_wins(False, "train", state)
            elif player_win == True:
                save_wins(True, "train", state)
                player_win = False
            player_win = False
            obs = env.reset()


            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, help='number of episodes')
    args = parser.parse_args()
    main(args.episodes)