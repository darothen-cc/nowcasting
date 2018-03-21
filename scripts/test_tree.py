
from dask.diagnostics import ProgressBar
import xarray as xr

from ccpy.obs import n0q_to_refl
from nowcast.houwang2017 import medfilt2d_dataarray, create_storm_tree

# Open up some data
ds = xr.open_dataset("../data/boston_nov_3/n0q.subset.nc", chunks={'time': 1})
ds['refl'] = n0q_to_refl(ds.n0q)

# Smart subset
xlo, xhi = map(float, [ds.x.min(), ds.x.max()])
ylo, yhi = map(float, [ds.y.min(), ds.y.max()])
ds = (
    ds
    # Smaller lat/lon box
    .sel(x=slice(xhi-5, xhi), y=slice(yhi, yhi-3))
    # Single timestep
    .isel(time=slice(12, 13))
)

# Pre-process
# 1) Median filter over time
ds['refl_med'] = medfilt2d_dataarray(ds['refl'], 'time')

# Create a storm tree -> need to grab a 2D slice for things to be kosher
tree = create_storm_tree(ds['refl_med'].squeeze())
print([v.nid for v in tree])

