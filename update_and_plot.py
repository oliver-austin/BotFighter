import sys
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def main():
    animate_graph()


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
    fig, ax = plt.subplots()
    n = 1000000000  # Number of frames
    ind = np.array([1, 2, 3])

    plt.xticks(ind, ('Easiest', 'Normal', 'Hardest'))
    plt.ylabel('Wins')
    plt.xlabel('Difficulty')
    ann_list = []
    anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=n,
                                   interval=100, fargs=(ann_list,))

    # anim.save('mymovie.mp4',writer=animation.FFMpegWriter(fps=10))
    plt.show()


def animate(frame, ann_list):
    for i, a in enumerate(ann_list):
        a.remove()
    ann_list[:] = []
    x = np.array([1, 2, 3])
    easy = np.load('showcase/ryu1guile.state_showcase.npy')
    easy_wins = plt.bar(x[0], easy[0], width=0.4, color='green')
    for i, b in enumerate(easy_wins):
        b.set_height(easy[0])
    easy_losses = plt.bar(x[0], easy[1], width=0.4, color='red', bottom=easy[0])
    for i, b in enumerate(easy_losses):
        b.set_height(easy[1])

    label = "{:.2f}%".format(easy[0]+easy[1])
    ann = plt.annotate(label,
                 (1, easy[0]+easy[1]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')
    ann_list.append(ann)

    normal = np.load('showcase/ryu4guile.state_showcase.npy')
    normal_wins = plt.bar(x[1], normal[0], width=0.4, color='green')
    for i, b in enumerate(normal_wins):
        b.set_height(normal[0])
    normal_losses = plt.bar(x[1], normal[1], width=0.4, color='red', bottom=normal[0])
    for i, b in enumerate(normal_losses):
        b.set_height(normal[0])

    label = "{:.2f}%".format(normal[0] + normal[1])
    ann = plt.annotate(label,
                       (2, normal[0] + normal[1]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
    ann_list.append(ann)

    hard = np.load('showcase/ryu8guile.state_showcase.npy')
    hard_wins = plt.bar(x[2], hard[0], width=0.4, color='green')
    for i, b in enumerate(hard_wins):
        b.set_height(hard[0])
    hard_losses = plt.bar(x[2], hard[1], width=0.4, color='red', bottom=hard[0])
    for i, b in enumerate(hard_losses):
        b.set_height(hard[1])

    label = "{:.2f}%".format(hard[0]+hard[1])
    ann = plt.annotate(label,
                 (3, hard[0]+hard[1]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')
    ann_list.append(ann)

    max_val = max(easy[0] + easy[1], normal[0] + normal[1], hard[0] + hard[1])
    plt.ylim(0, max_val + 5)


if __name__ == "__main__":
    main()