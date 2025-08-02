import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def moons_plot(X_train, X_test, Y_train, Y_test, width=600):
    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        s=30,
        c=Y_train,
        cmap=plt.cm.Spectral,
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        s=20,
        color="black",
        cmap=plt.cm.Spectral,
    )
    ax.plot()
    fig.savefig("twomoonsdataset.pdf", format="pdf", bbox_inches="tight")


def moons_plotcontour(
    net, X_train, X_test, Y_train, Y_test, width=600, device="cpu", method=""
):
    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    min1 = -2
    max1 = 3
    min2 = -1.5
    max2 = 1.75

    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    outputs = net(torch.tensor(grid, device=device, dtype=torch.float))
    zz = F.softmax(outputs, 1)[:, 1]
    _, predicted = torch.max(
        F.softmax(net(torch.tensor(X_test, dtype=torch.float, device=device)), 1), 1
    )
    correct = predicted.cpu() == Y_test
    color = ["white" if correct[i] else "black" for i in range(Y_test.shape[0])]
    zz = zz.reshape(xx.shape).cpu().detach().numpy()
    c = ax.contourf(xx, yy, zz, 200, cmap="RdBu", vmin=0, vmax=1)
    ax.contour(xx, yy, zz, levels=[0.5], color="k", linewidths=2)
    fig.colorbar(c)
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        s=30,
        c=Y_train,
        cmap=plt.cm.Spectral,
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        s=20,
        c=color,
        cmap=plt.cm.Spectral,
    )
    fig.savefig(
        "twomoonsdb2" + "_" + method + ".pdf", format="pdf", bbox_inches="tight"
    )
