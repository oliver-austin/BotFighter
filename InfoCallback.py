import tensorflow as tf


class InfoCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.player_win = False

    def on_step_end(self, step, logs=None):
        if (logs['info'])['matches_won'] > (logs['info'])['enemy_matches_won']:
            self.player_win = True
        else:
            self.player_win = False

    def on_episode_end(self, episode, logs=None):
        if self.player_win:
            print("WIN!!!")
        else:
            print("LOSS!!!")
