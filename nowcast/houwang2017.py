""" Utilities for implementing the tree-based storm tracking algorithm
from Hou and Wang, 2017.

"""

import numpy as np


class Node(object):

    def __init__(self, parent=None, data=None):
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
            return self .parent.is_ancestor(n)

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
