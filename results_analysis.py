# TODO
# 1. Do a t-SNE visualization of the features that you learn from your contrastive learning method.
# 2. Compute the instance-level variability that is coming from the data augmentation in your learned representations.

# you can find some other nice docs in the plotter in tools/
import os
import shutil
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import product as cprod
from pprint import PrettyPrinter

pp = PrettyPrinter()


def plot_mat_points(mat, filename, idx2class=None):
    if instanceof(mat, torch.tensor):
        mat = mat.detach().cpu().clone().numpy()

    if len(mat.shape) != 2:
        raise ValueError(
            "Expected shape to have length 2. Instead have shape {}.".format(mat.shape)
        )
    num_embds, embd_dim = mat.shape
    if embd_dim != 2:
        raise ValueError(
            "Got {} embeddings with dimension {}. Cannot plot dimension {}.".format(
                embd_dim
            )
        )

    X = mat[:, 0]
    Y = mat[:, 1]
    if len(X) != len(Y):
        raise ValueError(
            "Somehow the first column and second column had unequal length: {}, {}".format(
                X, Y
            )
        )

    # TODO find a better coloring scheme if possible?
    # i.e. hardcode an MNIST coloring scheme and be able to store it somewhere
    if idx2class:
        # NOTE: this colors thing will only work if you relable the ids
        colors = np.array([idx2class[idx] for idx in range(mat.shape[0])])
        colors = colors / np.max(colors)  # this will not be helpful for representing...
        plt.scatter(X, Y, c=colors)
    else:
        plt.scatter(X, Y)
    plt.savefig(filename)
    plt.clf()  # clear the figure so we can plot additional plots


def tsne(mat, pca_dim=None, **kwargs):
    pca = PCA(n_components=pca_dim).fit_transform(mat) if pca_dim else mat
    tsne = TSNE(n_components=2, random_state=None, **kwargs).fit_transform(pca)
    return tsne


def kwargs2names(kwargs):
    # the sorting is more or less an arbitrary key but is consistent
    return "_".join(sorted(["{}_{}".format(k, str(v)) for k, v in kwargs.items()]))


def tsne_hyperparameter_search(
    mat,
    tsne_dir="../../outputs/tsne",
    plot_root_name="",
    perplexities=[10, 30, 90],
    num_iters=[100, 500, 100, 2000],
    learning_rates=[100.0, 200.0, 300.0],
    early_exaggerations=[6.0, 12.0, 18.0],
    pca_dims=[4, 8, 16, None],
):
    # create a 4-D grid using the cartesian product
    # you are encouraged to use a SMALL grid otherwise this might take a LONG time
    hyperparameters = [
        {
            "perplexity": perp,
            "n_iter": n_iters,
            "learning_rate": lr,
            "early_exaggeration": early_ex,
            "pca_dim": pca_dim,
        }
        for perp, n_iters, lr, early_ex, pca_dim in cprod(
            perplexities, num_iters, learning_rates, early_exaggerations
        )
    ]

    if os.isdir(tsne_dir):
        shutil.rmtree(tsne_dir)
    os.mkdir(tsne_dir)

    idx2class = None
    for tsne_mat, kwargs in map(
        lambda kwargs: tsne(mat, **kwargs), kwargs, hyperparameters
    ):
        filename = plot_root_name + "_tsne_" + kwargs2names(kwargs) + ".jpeg"
        plot_mat_points(
            tsne_mat, filenames, idx2class=idx2class
        )  # TODO add class coloring


# this is for fc2
def load_rep_mat():
    # TODO load this from a saved version of the model then test it
    # note that index idx corresponds to digit idx
    # NOTE: don't forget to make sure that the output dir exists (etc)
    raise NotImplementedError


# expect sample to be a dictionary of
# {label_idx : [a list of input images]}
def instance_level_variability(model, samples):
    # get the variability within single instances
    keys = list(samples.keys())

    # TODO has to yield the representation
    outputs = {idx: [model(sample).numpy() for sample in samples[idx]] for idx in keys}

    # in the future may want to do cross-instance variability too
    # avgs = {
    #     idx : np.average(np.array(outputs[idx]), axis=1) for idx in keys
    # }
    stdevs = {idx: np.std(np.array(outputs[idx]), axis=1) for idx in keys}

    print("Standard deviations within classes.")
    pp.pprint(stdevs)
    print("")


if __name__ == "__main__":
    # TODO goals for this file
    # 1. plot the weights with TSNE
    # 2. plot the outputs of fc2 using a subset of the dataset with TSNE
    # 3. calculate the instance level variability
    raise NotImplementedError
