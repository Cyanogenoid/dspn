import sys
import data
import numpy as np


def take(iterable, n):
    l = []
    for _ in range(n):
        l.append(next(iterable))
    return l


path = sys.argv[1]
with open(path) as fd:
    objects = []
    for f in fd:
        tokens = iter(f.strip().split(" "))
        take(tokens, 1)
        if "detect" in path:
            score = float(take(tokens, 1)[0])
            if score < 0.5:
                continue
        else:
            score = 1.0
        coord = take(tokens, 3)
        material = np.argmax(take(tokens, 2))
        color = np.argmax(take(tokens, 8))
        shape = np.argmax(take(tokens, 3))
        size = np.argmax(take(tokens, 2))
        access = lambda x, i: data.CLASSES[x][i]
        objects.append(
            (
                "{:.2f} {:.2f} {:.2f}\t".format(*map(float, coord)),
                access("size", size),
                access("color", color),
                access("material", material),
                access("shape", shape),
                "{:.2f}".format(score),
            )
        )
objects.sort()
for o in objects:
    print(" ".join(o))
