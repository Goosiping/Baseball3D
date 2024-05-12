import json
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    f = open('./outputs/2.json')
    data = json.load(f)

    xs = np.array(data['x'])
    ys = np.array(data['y'])
    zs = np.array(data['z'])

    x_max = np.max(xs)
    x_min = np.min(xs)
    y_max = np.max(ys)
    y_min = np.min(ys)
    z_max = np.max(zs)
    z_min = np.min(zs)

    c_max = np.max([x_max, y_max, z_max])
    c_min = np.min([x_min, y_min, z_min])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlim([c_min, c_max])
    ax.set_ylim([c_min, c_max])
    ax.set_zlim([c_min, c_max])

    for i in range(340, len(xs)):
        ax.scatter(xs[i], ys[i], zs[i], c='r', marker='o')
        plt.show(block=False)
        plt.pause(0.1)