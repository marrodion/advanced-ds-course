import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import patches
import numpy as np


def draw_boxes(img, boxes, labels, ax=None):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    if ax is None:
        _, ax = plt.subplots(1, figsize=(12, 20))
    ax.imshow(img)
    for box, label in zip(boxes, labels):
        box = box.numpy()
        h = box[2] - box[0]
        w = box[3] - box[1]
        x, y = box[0], box[1]
        rec = patches.Rectangle((x, y), w, h, lw=2.0, color='r', fill=False)
        ax.add_patch(rec)
        ax.text(x+5, y+5, label.item(), fontsize=12, color='blue')
        plt.axis('off')
    return ax


def plot_grid(batch, rows=2, cols=2):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1)
    n = len(batch[0])
    
    for i, ax in zip(range(n), grid):
        img = batch[0][i].squeeze()
        boxes = batch[1][i]['boxes']
        labels = batch[1][i]['labels']
        draw_boxes(img, boxes, labels, ax=ax)
    plt.axis('off')
    plt.show()

