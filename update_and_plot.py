import sys
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
from IPython.display import HTML
import os.path
import threading
import time

def main():
    update_thread = threading.Thread(target=update)
    #animate_thread = threading.Thread(target=animate_graph)
    update_thread.start()
    animate_graph()
    #animate_thread.start()


def update():
    while True:
        time.sleep(1)
        easy = np.load('showcase/ryu1guile.state_showcase.npy')
        np.save('showcase/ryu1guile.state_showcase.npy', [easy[0]+1, easy[1]])

        normal = np.load('showcase/ryu4guile.state_showcase.npy')
        np.save('showcase/ryu4guile.state_showcase.npy', [normal[0], normal[1]])

        hard = np.load('showcase/ryu8guile.state_showcase.npy')
        np.save('showcase/ryu8guile.state_showcase.npy', [hard[0], hard[1]])


def animate_graph():
    fig = plt.figure()
    n = 1000000000  # Number of frames
    ind = np.array([1, 2, 3])

    plt.xticks(ind, ('Easiest', 'Normal', 'Hardest'))
    plt.ylabel('Wins')
    plt.xlabel('Difficulty')

    anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=n,
                                   interval=100)

    # anim.save('mymovie.mp4',writer=animation.FFMpegWriter(fps=10))
    plt.show()


def animate(frame):
    x = np.array([1, 2, 3])

    easy = np.load('showcase/ryu1guile.state_showcase.npy')
    easy_wins = plt.bar(x[0], easy[0], width=0.4, color='green')
    for i, b in enumerate(easy_wins):
        b.set_height(easy[0])
    easy_losses = plt.bar(x[0], easy[1], width=0.4, color='red', bottom=easy[0])
    for i, b in enumerate(easy_losses):
        b.set_height(easy[1])

    normal = np.load('showcase/ryu4guile.state_showcase.npy')
    normal_wins = plt.bar(x[1], normal[0], width=0.4, color='black')
    for i, b in enumerate(normal_wins):
        b.set_height(normal[0])
    normal_losses = plt.bar(x[1], normal[1], width=0.4, color='blue', bottom=normal[0])
    for i, b in enumerate(normal_losses):
        b.set_height(normal[0])

    hard = np.load('showcase/ryu8guile.state_showcase.npy')
    hard_wins = plt.bar(x[2], hard[0], width=0.4, color='black')
    for i, b in enumerate(hard_wins):
        b.set_height(hard[0])
    hard_losses = plt.bar(x[2], hard[1], width=0.4, color='yellow', bottom=hard[0])
    for i, b in enumerate(hard_losses):
        b.set_height(hard[1])



if __name__ == "__main__":
    main()