import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np
import os
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, nargs="*")
parser.add_argument("--keep", type=int, nargs="*")
args = parser.parse_args()

matplotlib.rc("text", usetex=True)
params = {"text.latex.preamble": [r"\usepackage{bm,amsmath,mathtools,amssymb}"]}
plt.rcParams.update(params)

base_path = "clevr/images/val"
val_images = sorted(os.listdir(base_path))


def load_file(path):
    with open(path) as fd:
        for line in fd:
            tokens = line.split(" ")
            if "detect" in path:
                _, score, x1, y1, x2, y2 = tokens
            else:
                _, x1, y1, x2, y2, score = tokens
            score = float(score)
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            yield score, x1, y1, x2, y2


indices_to_use = args.keep
indices_to_use.append(-2)
indices_to_use.append(-1)

for j, index in enumerate(args.n):
    progress = []
    log_dspn_path = "out/clevr-box/dspn-clevr-box-1-30"
    log_base_path = "out/clevr-box/base-clevr-box-1-10"
    for i in range(31):
        points = list(load_file(f"{log_dspn_path}/detections/{index}-step{i}.txt"))
        progress.append(points)
    progress.append(list(load_file(f"{log_base_path}/groundtruths/{index}.txt")))
    progress.append(list(load_file(f"{log_base_path}/detections/{index}.txt")))

    point_color = colors.to_rgb("#34495e")
    for i, progress_n in enumerate(indices_to_use):
        step = progress[progress_n]
        plt.figure(figsize=(2, 2))
        ax = plt.gca()
        score, x1, y1, x2, y2 = zip(*step)
        print(i, progress_n)

        img = Image.open(os.path.join(base_path, val_images[int(index)]))
        plt.imshow(img)
        for a, b, c, d, s in zip(x1, y1, x2, y2, np.clip(score, 0, 1)):
            a *= 480
            b *= 320
            c *= 480
            d *= 320
            rect = patches.Rectangle(
                (a, b),
                c - a,
                d - b,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
                alpha=s,
            )
            ax.add_patch(rect)
        plt.xticks([])
        plt.yticks([])
        filename = f"clevr-{index}-{progress_n}.pdf"
        plt.savefig(filename, bbox_inches="tight", dpi=600)
