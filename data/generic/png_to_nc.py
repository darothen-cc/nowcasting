#!/usr/bin/env python
""" Convert a PNG with associated world file to NetCDF. """
import click
import os
import pandas as pd
import xarray as xr

@click.command()
@click.argument("filename")
@click.option("--name", default='subset',
              help="Name to use for indicating subset.")
@click.option("--field", default="data",
              help="Name of field to use in output file.")
@click.option("--geometry", type=float, nargs=4)
def png_to_nc(filename, name, field, geometry=[50, 50, 2.5, 2.5],
              help="[clat clon dlat dlon]"):

    print("Reading", filename)
    da = xr.open_rasterio(filename)
    da.name = field

    clat, clon, dlat, dlon = geometry

    # Subset the data
    da = da.sel(y=slice(clat+dlat, clat-dlat),
                x=slice(clon-dlon, clon+dlon))
    da = da.isel(band=0)
    da.squeeze()

    base, ext = os.path.splitext(filename)
    out_fn = base+".{}.nc".format(name)
    print("Writing to", out_fn)

    da.to_netcdf(out_fn)


if __name__ == "__main__":
    png_to_nc()
