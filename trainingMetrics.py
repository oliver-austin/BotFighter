import numpy as np
import matplotlib.pyplot as plt


def plot_reward(training_history):
    session_reward = np.array(training_history.history['episode_reward'])
    session_episodes = np.arange(session_reward.size)

    overall_reward = np.load('reward_history.npy')
    overall_reward = np.concatenate((overall_reward, session_reward))
    np.save('reward_history.npy', session_reward)  # save

    session_regression_line = calculate_regression_line(session_episodes, session_reward)

    plt.scatter(session_episodes, session_reward)
    plt.plot(session_episodes, session_regression_line)
    plt.title('training session results')
    plt.ylabel('episode reward')
    plt.show()

    overall_episodes = np.arange(overall_reward.size)
    overall_regression_line = calculate_regression_line(overall_episodes, overall_reward)

    plt.scatter(overall_episodes, overall_reward)
    plt.plot(overall_episodes, overall_regression_line)
    plt.title('overall training results')
    plt.ylabel('episode reward')
    plt.show()


def calculate_regression_line(episodes, rewards):
    slope = (((np.mean(episodes) * np.mean(rewards)) - np.mean(episodes * rewards)) /
         ((np.mean(episodes) * np.mean(episodes)) - np.mean(episodes * episodes)))
    intercept = np.mean(rewards) - slope * np.mean(episodes)
    regression_line = (slope * episodes) + intercept
    return regression_line
