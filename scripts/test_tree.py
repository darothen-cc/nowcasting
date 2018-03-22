
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.colors import BoundaryNorm

import numpy as np
import tqdm
import xarray as xr

from ccpy.obs import n0q_to_refl
from ccpy.plot import make_precip_colormap, get_figsize
from nowcast.houwang2017 import (medfilt2d_dataarray, create_storm_tree,
                                 find_stratiform_regions, find_storm_cells,
                                 find_storm_regions)


def plot_original(refl, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    cmap_kws = make_precip_colormap()
    refl.plot.pcolormesh(ax=ax, infer_intervals=True, add_colorbar=True, **cmap_kws)
    ax.set_title("Original Image")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    return ax


def plot_regions(regions, ax=None, patches=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Set up colormap, if patches
    if patches:
        cmap = plt.get_cmap('tab20')
        levels = range(20)
        norm = BoundaryNorm(levels, ncolors=cmap.N)
        cmap_kws = dict(cmap=cmap, levels=levels, norm=norm)

    # Else we use the normal rainfall one
    else:
        cmap_kws = make_precip_colormap()

    for r in regions:
        # Plot labeled regions
        if patches:
            _da = r.region.intensity_image.copy()
            _id = np.random.randint(20)
            _da[:] = r.region.image * _id
            _da = _da.where(_da == _id)
            _ = _da.plot.pcolormesh(ax=ax, infer_intervals=True,
                                    add_colorbar=False, **cmap_kws)
            # Plot original intensity image
        else:
            _ = r.region.intensity_image.plot.pcolormesh(
                ax=ax, add_colorbar=False, **cmap_kws
            )
    plt.colorbar(_, ax=ax)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    return ax

def plot_storm_regions(tree, ax=None, patches=False):
    regions = find_storm_regions(tree)
    ax = plot_regions(regions, ax, patches)
    ax.set_title("Convetive Regions")
    return ax

def plot_storm_cells(tree, ax=None, patches=False):
    regions = find_storm_cells(tree)
    ax = plot_regions(regions, ax, patches)
    ax.set_title("Storm Cells")
    return ax

def plot_stratiform(tree, ax=None, patches=False):
    regions = find_stratiform_regions(tree)
    ax = plot_regions(regions, ax, patches)
    ax.set_title("Stratiform Regions")
    return ax


def plot_4panel(refl, tree, patches=False):
    figsize = get_figsize(2, 2, size=4, aspect=2.)
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axs = axs.ravel()
    plot_original(refl, axs[0])
    plot_stratiform(tree, axs[1], patches)
    plot_storm_regions(tree, axs[2], patches)
    plot_storm_cells(tree, axs[3], patches)


def plot_labeled(tree):
    """ Create a 2D plot with each segmented region labeled by its sequential
    ID number """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap_kws = make_precip_colormap()
    total_nodes = len(tree)
    for v in tqdm.tqdm(tree, total=total_nodes):
        if v.node_depth() > 100:
            continue
        _da = v.region.intensity_image.copy()
        _da[:] = v.region.image * v.nid
        _da = _da.where(_da == v.nid)
        # print(_da.shape, v.nid)
        _da.plot.pcolormesh(ax=ax, infer_intervals=True,
                            add_colorbar=False, vmin=1,
                            vmax=total_nodes, cmap='Spectral')
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    return ax


# Open up some data
# fn = "../data/boston_nov_3/n0q.subset.nc"
fn = "../data/eusa_feb_7/n0q.subset.nc"
print("Opening", fn)
ds = xr.open_dataset(fn, chunks={'time': 1})
ds['refl'] = n0q_to_refl(ds.data)  # May need to retrieve 'n0q' instead

# Smart subset
xlo, xhi = map(float, [ds.x.min(), ds.x.max()])
ylo, yhi = map(float, [ds.y.min(), ds.y.max()])
it = 12
print(ds.time.isel(time=it).values)
ds = (
    ds
    # Smaller lat/lon box
    .sel(x=slice(xlo, xhi), y=slice(yhi, ylo))
    # Single timestep
    .isel(time=slice(it, it+1))
)

# Pre-process
# 1) Median filter over time
# print("Applying median filter")
ds['refl_med'] = medfilt2d_dataarray(ds['refl'], 'time')
# 2) Re-shape
with ProgressBar():
    da = ds['refl_med'].squeeze().compute()
# 3) Threshold lowest level (superfluous?)
da = da.where(da > 0.)

print("Creating tree...")
tree = create_storm_tree(da)
# print([v.nid for v in tree])
print("""\nSUMMARY
----------------------------
Total Nodes: {}
 Top-level Regions: {}
Stratiform Regions: {}
Convective Regions: {}
       Storm Cells: {} 
""".format(
    len(tree), tree.degree,
    len(find_stratiform_regions(tree)),
    len(find_storm_regions(tree)),
    len(find_storm_cells(tree))
))

print("Plotting...")
ax = plot_4panel(da, tree, patches=True)
plt.tight_layout()





