import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("n", nargs="*")
args = parser.parse_args()

matplotlib.rc("text", usetex=True)
params = {"text.latex.preamble": [r"\usepackage{bm,amsmath,mathtools,amssymb}"]}
plt.rcParams.update(params)


def load_file(path):
    with open(path) as fd:
        for line in fd:
            tokens = line.split(" ")
            if 'detect' in path:
                _, score, x, y = tokens
            else:
                _, x, y, score = tokens
            score = float(score)
            x = float(x)
            y = float(y)
            yield score, x, y


plt.figure(figsize=(7, len(args.n)))
for j, index in enumerate(args.n):
    progress = []
    for i in range(6):
        points = list(load_file(f"out/mnist/dspn-1/detections/{index}-step{i}.txt"))
        progress.append(points)
    progress.append(list(load_file(f"out/mnist/base-4/groundtruths/{index}.txt")))
    progress.append(list(load_file(f"out/mnist/base-4/detections/{index}.txt")))

    point_color = colors.to_rgb("#34495e")
    for i, step in enumerate(progress):
        plt.subplot(
            len(args.n), len(progress), i + 1 + j * len(progress), aspect="equal"
        )
        score, x, y = zip(*step)
        x, y = y, x
        y = 1 - np.array(y)

        color = np.zeros((y.size, 4))
        color[:, :3] = point_color
        color[:, 3] = np.clip(score, 0, 1)

        # remove elements with too low score
        keep = color[:, 3] > 0.2
        x = np.array(x)[keep]
        y = np.array(y)[keep]
        color = color[keep]

        plt.scatter(x, y, marker=".", color=color, s=4, rasterized=True)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xticks([])
        plt.yticks([])
        if j == 0:
            if i == len(progress) - 2:
                plt.title(r"Target")
            elif i == len(progress) - 1:
                plt.title(r"Baseline")
            else:
                plt.title(f"Step {i}")
filename = "mnist.png" if len(args.n) < 4 else "mnist-full.pdf"
plt.savefig(filename, bbox_inches="tight", dpi=300)
