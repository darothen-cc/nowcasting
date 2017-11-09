import os
from textwrap import dedent

from setuptools import setup

MAJOR = 0
MINOR = 0
MICRO = 1
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MICRO)

setup(
    name = "nowcast",
    author = "Daniel Rothenberg",
    author_email = "daniel@climacell.co",
    maintainer = "Daniel Rothenberg",
    maintainer_email = "daniel@climacell.co",
    version = VERSION,
    packages = ["nowcast", ],
    package_data = {},
    entry_points = {},
    install_requires = []
)
