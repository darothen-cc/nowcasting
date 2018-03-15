""" Utilities for implementing the tree-based storm tracking algorithm
from Hou and Wang, 2017.

"""

import numpy as np
from scipy.signal import medfilt2d
from skimage.measure import label, regionprops
from skimage.color import label2rgb

import xarray as xr

#: Algorithm reflectivity level-set thresholds, in dBz
_THRESHOLDS = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60]


class Node(object):
    """ Simple class for constructing and containing tree-like graphs.

    Attributes
    ----------
    parent : Node, or None
        The parent of this node on a tree; if 'None', then this is the root
        of a sub-tree.
    children : list of Nodes
        The children which claim this Node as a parent
    _data : dict
        A dictionary containing any data attached to this Node

    """

    def __init__(self, parent=None, data={}):
        self.parent = parent
        self.children = []

        self._data = data

    @property
    def data(self):
        return self._data

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return not self.children

    def is_ancestor(self, n):
        if self.is_root():
            return False
        elif self.parent.equals(n):
            return True
        else:
            return self.parent.is_ancestor(n)

    def is_descendant(self, n):
        if not self.children:
            return False

        for child in self.children:
            if child.equals(n):
                return True
        else:
            return np.any([child.is_descendant(n) for child in self.children])

    def degree(self):
        return len(self.children)

    def node_depth(self):
        if self.is_root():
            return 1
        else:
            return 1 + self.parent.node_depth()

    def tree_depth(self):
        if not self.children:
            return 1
        else:
            return 1 + np.max([child.depth() for child in self.children])


def medfilt2d_dataarray(da, dim='time', **kwargs):
    """ Apply scipy.signal.medfilt2d efficiently against a DataArray.

    Parameters
    ----------
    da : DataArray
        Data to be filtered
    dim : str
        Dimension along-which to apply the filter

    Returns
    -------
    Original datarray with the 2d median filter applied along the requested
    dimension.

    """

    def looped_medfilt2d(data, **kwargs):
        """ Apply `medfilt2d` to each slice along the last dimension of
        a given ndarray """
        n = data.shape[-1]
        return np.dstack([medfilt2d(data[...,i], **kwargs) for i in range(n)])

    return xr.apply_ufunc(looped_medfilt2d, da, kwargs=kwargs,
                          input_core_dims=[[dim, ]],
                          output_core_dims=[[dim, ]]),
                          dask='allowed')