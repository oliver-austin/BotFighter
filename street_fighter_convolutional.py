import sys
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")

import gym
import argparse
import cv2
import numpy as np
import retro
import StreetFighterConvolutionalModel
import ExperienceBuffer
import preprocessing

DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 3000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 6
TAU = 0.01
# Number of frames to throw into network
NUM_FRAMES = 3
FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)
ENV_NAME = 'StreetFighterIISpecialChampionEdition-Genesis'


class StreetFighter():

    def __init__(self):
        game_mode, render, total_step_limit, total_run_limit = self._args()
        self.env = retro.make(game=ENV_NAME, state='rom.state', use_restricted_actions=retro.Actions.DISCRETE)
        self.experience_buffer = ExperienceBuffer.ExperienceBuffer(BUFFER_SIZE)
        self.env.reset()
        self.model = StreetFighterConvolutionalModel.StreetFighterConvolutionalModel(self.env.observation_space.shape, self.env.action_space)
        #self._main_loop(self._game_model(game_mode, env.action_space.n), env, render, total_step_limit, total_run_limit)

        self.prev_frames = []
        state1, reward1, _, _ = self.env.step(0)
        print("STATE1", state1)
        state2 = state1
        state3 = state1

        self.prev_frames = [state1, state2, state3]


    def train(self, render, total_step_limit, total_run_limit):
        run = 0
        total_step = 0
        current_state = preprocessing.preprocess(self.prev_frames)
        #current_state = self.prev_frames
        epsilon = INITIAL_EPSILON
        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1

            step = 0
            score = 0
            obs_num = 0
            while True:
                if total_step >= total_step_limit:
                    print("Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1

                start_state = preprocessing.preprocess(self.prev_frames)
                #start_state = self.prev_frames
                self.prev_frames = []
                predict_movement, predict_q_value = self.model.predict(current_state, epsilon)

                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EPSILON_DECAY

                if render:
                    self.env.render()

                reward, terminal = 0, False
                for i in range(NUM_FRAMES):
                    temp_state, temp_reward, temp_terminal,_ = self.env.step(predict_movement)
                    reward += temp_reward
                    self.prev_frames.append(temp_state)
                    terminal = terminal or temp_terminal

                if terminal:
                    #game_model.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
                    break

                new_state = preprocessing.preprocess(self.prev_frames)
                #new_state = self.prev_frames
                self.experience_buffer.push_back(predict_movement, start_state, reward, terminal, new_state)

                if self.experience_buffer.size > MIN_OBSERVATION:
                    batch_action, batch_state, batch_reward, batch_end, batch_next_state = self.experience_buffer.sample(MINIBATCH_SIZE)
                    self.model.train(batch_action, batch_state, batch_reward, batch_end, batch_next_state, obs_num)
                    self.model.target_train()

                obs_num += 1


    def _args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--mode", default="dqn_training")
        parser.add_argument("-r", "--render", default=True, type=bool)
        parser.add_argument("-tsl", "--total_step_limit", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit", default=None, type=int)
        args = parser.parse_args()
        game_mode = args.mode
        render = args.render
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        return game_mode, render, total_step_limit, total_run_limit
