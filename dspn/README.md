# Reproducing the experiments

This directory contains the main training code for running the experiments.
In particular, `train.py` is the main training loop, `model.py` contains the network, and `data.py` contains the datasets.

Make sure that you have the requirements listed below installed.

You can keep track of model training with Tensorboard.
Just point it at the `runs` directory that is created when training a model and you can view losses, how good the DSPN model thinks the initial and predicted sets are (`eval_first` and `eval_last`), and view intermediate predictions throughout training.

## MNIST

### Training

From this directory, run:
```
scripts/mnist.sh
```

This runs `train.py` with the right arguments for MNIST using both the DSPN and MLP set decoders.
The models are saved to `logs/dspn-mnist` and `logs/base-mnist` for the two versions respectively.
Note that while the models are trained for 100 epochs, their loss stops changing much earlier.

Predictions of the trained models are saved to `out/mnist/{dspn,base}/detections`.
Each text file in here contains the predicted set coordinates and the mask.
For example, `2.txt` corresponds to the same image as `../groundtruths/2.txt`.
For DSPN, there are additional `2-step{0..10}.txt` files that correspond to the intermediate sets obtained during the inner optimisation.

### Visualisation

The predictions of the trained models can be visualised with:

`python plot-mnist-progress.py [image_index_1, image_index_2, ...]`

where the arguments are a list of integer indices like `10 11 12`.
The file is saved as `mnist.pdf` or `mnist-full.pdf`, depending on how many arguments are used.


## CLEVR

### Set-up

First, download the [CLEVR v1.0][0] dataset.
Make sure that the dataset is available in this directory as `clevr` directory, either by renaming the CLEVR directory in this directory or by placing a `clevr` symlink to the main CLEVR directory.
To be clear, there should now be a `clevr` directory in this folder that contains a `image` and a `scenes` directory.
Now, pre-process the images by running:

```
python preprocess-images.py
```

### Training
Once you have setup the dataset, run:

```
scripts/clevr.sh clevr-box 1
scripts/clevr.sh clevr-box 1 baseline
scripts/clevr.sh clevr-state 1
scripts/clevr.sh clevr-state 1 baseline
```

or alternatively, `scripts/all-jobs.sh`.
The first argument specifies whether the bounding box version or the state prediction version of CLEVR is used.
The second argument specifies the run number so that results aren't being overwritten.
To do more runs, simply change the 1 in the arguments to 2, 3, etc.
The third argument specifies whether the DSPN model or the baseline model is used.

### Evaluation

Once the models are done training, it is time for evaluation.
To do this, run:

```
scripts/eval-clevr.sh clevr-box 1
scripts/eval-clevr.sh clevr-box 1 baseline
scripts/eval-clevr.sh clevr-state 1
scripts/eval-clevr.sh clevr-state 1 baseline
```

or alternatively, `scripts/all-eval-jobs.sh`.
This will load the trained models, export their predictions into `out/clevr-{box,state}`, and run the evaluation scripts for measuring average precision.
The DSPN models are evaluated at 10, 20, and 30 iterations (the model is trained with only 10).

The evaluation code in `Object-Detection-Metrics` for bounding boxes and `Object-Detection-Metrics-State` for state prediction is slightly modified from [rafaelpadilla/Object-Detection-Metrics][1] for bounding boxes.

### Visualisation

There are several ways of visualising prediction results.
To summarise the results of all runs on `clevr-box` and `clevr-state`, run:

```
python summarise.py
```

This prints two tables: the first table shows the standard deviations over runs, while the second table shows the average results over runs.
These tables contain all the different combinations of models, datasets, and evaluation thresholds used.

Next up, you can plot the bounding box predictions with:

```
python train.py --show --loss hungarian --encoder RNFSEncoder --dim 512 --dataset clevr-box --epochs 100 --latent 512 --supervised --inner-lr 800 --resume logs/dspn-clevr-box-1 --iters 30 --name test --eval-only --export-dir out/clevr-box/dspn-clevr-box-1-30  --decoder DSPN --mask-feature --export-progress --export-n 100
python plot-box-progress.py [image_index_1, image_index2, ...] --keep 0 5 10 20
```

The first command exports intermediate sets because they would take up too much space if they were done in the evaluation stage.
The image indices are a list of integers like `10 11 12`.
The numbers after `--keep` specify which steps of the inner optimisation to plot.
The file is saved as `clevr.pdf` or `clevr-full.pdf` depending on how many image indices are used.

State predictions can be viewed with:

```
python view-state.py {path to prediction}
```

For example, calling this with `out/clevr-state/dspn-clevr-state-1-30/detections/42.txt` as argument displays the predicted object attributes for the DSPN model from run 1 after 30 inner optimisation steps for image index 42.
You can compare this against the groundtruth by replacing `detections` in the path with `groundtruths`.
Here is an example output:

```
-0.62 -0.30 0.23	 large yellow rubber sphere 1.00
-0.70 0.30 0.12	 small yellow metal sphere 1.00
0.11 0.98 0.12	 small blue metal cylinder 1.00
0.23 -0.30 0.23	 large brown rubber cylinder 1.00
0.62 -0.06 0.12	 small cyan rubber cube 1.00
```

Lastly, you can run:

```
python train.py --show --loss hungarian --encoder RNFSEncoder --dim 512 --dataset clevr-state --epochs 100 --latent 512 --supervised --inner-lr 800 --resume logs/dspn-clevr-state-1 --iters 30 --name test --eval-only --export-dir out/clevr-state/dspn-clevr-state-1-30  --decoder DSPN --mask-feature --export-progress --export-n 100
python plot-state-progress.py [image_index_1, image_index2, ...] --keep 5 10 20
```

to format predictions into a LaTeX table.
The arguments mean the same thing as for `plot-box-progress.py`.
If you pipe the output of this into a `.tex` file, you can `\input` it and it gets rendered as a LaTeX table.


## Requirements
- Python 3.6+
- PyTorch 1.0+
- torchvision
- Pillow
- tensorboardX
- h5py
- matplotlib
- numpy
- pandas
- scipy


[0]: https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
[1]: https://github.com/rafaelpadilla/Object-Detection-Metrics
