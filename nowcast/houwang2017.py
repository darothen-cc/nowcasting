""" Utilities for implementing the tree-based storm tracking algorithm
from Hou and Wang, 2017.

"""

from itertools import chain

import numpy as np
from scipy.signal import medfilt2d
from skimage.measure import label, regionprops
from skimage.color import label2rgb

import xarray as xr

#: Algorithm reflectivity level-set thresholds, in dBz
DEFAULT_THRESHOLDS = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60]


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

    def __init__(self, parent=None, **kw_attrs):
        self.parent = parent
        self.children = []

        self._data = kw_attrs
        self.__dict__.update(**self._data)

    # @property
    # def data(self):
    #     return self._data

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
            return 1 + np.max([child.tree_depth() for child in self.children])

    def find_deepest_child(self):
        """ Bread-first search to find the node in a tree furthest from the root
        which still contains children. """

        md = 0
        child = None

        if not self.children:
            return child

        for c in self.children:
            depth = c.tree_depth()
            if depth > md:
                md = depth
                child = c

        return child

    def __iter__(self):
        """ Traverse the structure via the iterator protocol, for convenience """
        for v in chain(
            *map(iter, self.children)
        ):
            yield v
        yield self


def create_storm_tree(image, thresholds=DEFAULT_THRESHOLDS):
    """ Recursively bisect an image to construct the Hou and Wang (2017)
    tree based on a set of level-set thresholds

    Parameters
    ----------
    image : ndarray
        The image to use for calculating the level-set tree
    thresholds : list of floats
        Level-set boundary values

    Returns
    -------
    A Node object containing the level-set tree for a given image

    """

    # Identify regions where we're below the lowest threshold
    image_to_label = np.where(image >= thresholds[0], 1, 0)

    # Initialize the head root node
    root = Node(data='root')
    root.nid = 0

    def _fit_labels(image, i, parent, count=1):

        local_count = count+1

        # Don't further segment if the region is small or contains no data.
        if (np.product(image.shape) < 10) or (np.min(image.shape) == 1):
            return parent

        # Re-label this image
        try:
            _thresh_image = np.where(image >= thresholds[i], 1, 0)
        except:
            return parent

        labels, n = label(_thresh_image, return_num=True)
        image_label_overlay = label2rgb(labels, image=image)

        if n > 0:
            regions = regionprops(labels, intensity_image=image)
            V = []
            for region in regions:
                _node = Node(parent,
                             region=region,
                             image_label_overlay=image_label_overlay,
                             nid=local_count)
                local_count += 1
                V.append(_node)
            parent.children = V

            for Vi in V:
                # print(Vi.nid, Vi.region.intensity_image.shape)
                _fit_labels(Vi.region.intensity_image, i=i + 1, parent=Vi,
                            count=local_count)

        return parent

    return _fit_labels(image, i=1, parent=root)


def threshold_dataarray(da, coords=['y', 'x'], thresholds=DEFAULT_THRESHOLDS):
    """ For a given DataArray, compute level-set areas and save as a new
    dimension in the DataArray. """

    Pis = [np.where(da >= gi, i, 0) for i, gi in enumerate(thresholds)]
    Pis = np.stack(Pis)

    new_coords = {'threshold': thresholds}
    new_coords.update({k: da[k] for k in coords})

    Pis_da = xr.DataArray(Pis, name='Pi',
                          coords=new_coords,
                          dims=['threshold'] + coords)

    return Pis_da


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
                          output_core_dims=[[dim, ]],
                          dask='allowed')


