import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import patches
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False


g_edge_color = ["#C4323F", "#EF4143", "#033250", "#73BAD6", "#8AB07D", "#FB8402", "#56045A"]
g_color = ["none", "#EF4143", "none", "#73BAD6", "#8AB07D", "#FB8402", "#56045A"]
g_marker = ["^", "o", "s", "v", "o", "*", "8"]
legend=["簇1", "簇2", "簇3", "簇4", "簇5", "簇6", "簇7"]

config_dic = {"title": "K-means Result",
              "xlabel": "x-label",
              "ylabel": "y-label",
              "featrue_id": [0,1]}


def read_legend():
    with open("config.txt", 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            line = line.split(':')
            if len(line) != 2:
                continue
            if str(line[0]) == "featrue_id":
                line[1] = line[1].strip()
                ids = line[1].split(',')
                config_dic[str(line[0])][0] = int(ids[0])
                config_dic[str(line[0])][1] = int(ids[1])
            elif str(line[0]) in config_dic:
                line[1] = line[1].strip()
                config_dic[str(line[0])] = str(line[1])


if __name__ == '__main__':
    read_legend()
    data = np.loadtxt("data.csv", delimiter=",", dtype=np.float64)
    label = np.loadtxt("label.csv", delimiter=",", dtype=np.int32)

    label_num = label.max()

    for i in range(label_num):
        cur_label = i + 1
        mask = (label == cur_label)
        cur_data = data[mask]
        data_num = cur_data.shape[0]
        plt.scatter(cur_data[:,config_dic["featrue_id"][0]],
                    cur_data[:,config_dic["featrue_id"][1]],
                    marker=g_marker[i],
                    color=g_color[i],
                    edgecolors=g_edge_color[i],
                    label=legend[i])

    plt.legend(loc="lower left", fontsize='x-large')
    plt.title(config_dic["title"], fontsize='xx-large', verticalalignment='bottom')
    plt.xlabel(config_dic["xlabel"], fontsize='x-large')
    plt.ylabel(config_dic["ylabel"], fontsize='x-large')
    plt.show()
