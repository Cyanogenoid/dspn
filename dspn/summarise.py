import glob
import pandas as pd


data = []
for filename in glob.glob("out/*/*/*.txt"):
    _, _, folder, ap_name = filename.split("/")
    tokens = folder.split("-")
    model = tokens[0]
    dataset = tokens[2]
    run = int(tokens[-2])
    iters = int(tokens[-1])
    precision = float(ap_name.split("-")[1][:-4])
    with open(filename, "r") as fd:
        result_line = fd.readlines()[1]
        result = float(result_line.split(" ")[1][:-1])
    datapoint = model, dataset, iters, precision, run, result
    data.append(datapoint)

data = pd.DataFrame(data)
data.columns = ["model", "dataset", "iters", "threshold", "run", "ap"]
mean = data.groupby(["dataset", "model", "iters", "threshold"]).mean()
std = data.groupby(["dataset", "model", "iters", "threshold"]).std()
print(std.round(1))
print(mean.round(1))
