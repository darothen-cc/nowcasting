""" Useful utility functions """

import xarray as xr


def make_advection_dataset(q,  u, v, x, y, t):
    """ Package a set of results from an advection solver into a Dataset for
    convenient export and visualization

    Parameters
    ----------
    q : 3D array
        Time-varying tracer field defined on a regular, equally-spaced rectilinear
        grid, with dimensions [time, x, y]
    x, y : 2D array
        2D arrays containing the x- and y-coordinates of the data grid points,
        respectively (m)
    u, v : 2D array
        2D arrays containing the steady-state zonal (u) and meridional (v)
        velocity fields used in the advection calculation (m/s)
    t : 1D array
        Timesteps corresponding to the output slices in the tracer field

    """

    ds = xr.Dataset(
        {'q': (('time', 'x', 'y'), q),
         'u': (('x', 'y'), u),
         'v': (('x', 'y'), v)},
        {'time': t, 'x': x, 'y': y},
    )
    ds['time'].attrs.update(
        {'long_name': 'time', 'units': 'seconds since 2000-01-01 0:0:0'})
    ds['x'].attrs.update({'long_name': 'x-coordinate', 'units': 'm'})
    ds['y'].attrs.update({'long_name': 'y-coordinate', 'units': 'm'})
    ds['u'].attrs.update({'long_name': 'zonal wind', 'units': 'm/s'})
    ds['v'].attrs.update({'long_name': 'meridional wind', 'units': 'm/s'})
    ds.attrs.update({
        'Conventions': 'CF-1.7'
    })

    return ds