import tensorflow as tf
from trainingMetrics import save_wins


class InfoCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.player_win = False

    def on_step_end(self, step, logs=None):
        if (logs['info'])['matches_won'] == 2:
            self.player_win = True
        elif (logs['info'])['matches_won'] == 0 and self.player_win is True:
            save_wins(self.player_win, "train")
            self.player_win = False

    def on_episode_end(self, episode, logs=None):
        save_wins(self.player_win, "train")
