import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import patches


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def make_circle(points):
    shuffled = list(points)
    circle = (0, 0, 0)
    for i, p in enumerate(shuffled):
        if is_inside_circle(circle, p):
            continue
        circle = (p[0], p[1], 0)
        for j, q in enumerate(shuffled[:i]):
            if is_inside_circle(circle, q):
                continue
            circle = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, euclidean_distance(p, q) / 2)
            for k, r in enumerate(shuffled[:j]):
                if is_inside_circle(circle, r):
                    continue
                circle = make_circumcircle(p, q, r)
    return circle


def is_inside_circle(circle, point):
    center, radius = (circle[0], circle[1]), circle[2]
    return euclidean_distance(center, point) <= radius


def make_circumcircle(p1, p2, p3):
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    ux = ((p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p3[1]) + (p2[0] ** 2 + p2[1] ** 2) * (p3[1] - p1[1]) + (p3[0] ** 2 + p3[1] ** 2) * (p1[1] - p2[1])) / d
    uy = ((p1[0] ** 2 + p1[1] ** 2) * (p3[0] - p2[0]) + (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p3[0]) + (p3[0] ** 2 + p3[1] ** 2) * (p2[0] - p1[0])) / d
    r = euclidean_distance((ux, uy), p1)
    return (ux, uy, r)


if __name__ == '__main__':
    cluster_a = []
    cluster_b = []
    with open("data1.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('	')
            if len(line) != 2:
                print("[ERROR] line has wrong point")
            point = [float(line[0]), float(line[1])]
            cluster_a.append(point)
    with open("data2.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('	')
            if len(line) != 2:
                print("[ERROR] line has wrong point")
            point = [float(line[0]), float(line[1])]
            cluster_b.append(point)

    cluster_a = np.asarray(cluster_a)
    cluster_b = np.asarray(cluster_b)
    cluster_a = cluster_a.T
    cluster_b = cluster_b.T

    circle = make_circle(cluster_b.T)
    an = np.linspace(0, 2 * np.pi, 100)
    plt.fill((circle[2] + 3000) * np.cos(an) + circle[0],
             (circle[2] + 3000) * np.sin(an) + circle[1],
             alpha=0.2,
             facecolor="#BF1E2E",
             edgecolor="#BF1E2E",
             linewidth=1,
             zorder=1)
    plt.axis('equal')
    circle = make_circle(cluster_a.T)
    an = np.linspace(0, 2 * np.pi, 100)
    plt.fill((circle[2] + 3000) * np.cos(an) + circle[0],
             (circle[2] + 3000) * np.sin(an) + circle[1],
             alpha=0.2,
             facecolor="#0D4C6D",
             edgecolor="#0D4C6D",
             linewidth=1,
             zorder=1)
    plt.axis('equal')

    plt.scatter(cluster_a[0], cluster_a[1], marker='o', color="#0D4C6D", edgecolors="#000000", label="cluster 1", alpha=0.6)
    plt.scatter(cluster_b[0], cluster_b[1], marker='o', color="#BF1E2E", edgecolors="#000000", label="cluster 2", alpha=0.6)

    plt.legend(loc="upper left", fontsize='x-large')
    plt.title("K-means Result", fontsize='xx-large', verticalalignment='bottom')
    plt.show()
