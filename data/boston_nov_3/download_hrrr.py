""" Snakefile sentinel script for downloading a 'triangular' HRRR dataset.

"""
import os
import pandas as pd
from ccpy.util import wget

from collections import namedtuple

Forecast = namedtuple('forecast', ['year', 'month', 'day', 'hour', 'fcst_hour'])

#: Base pattern for accessing an archival HRRR output from Utah server, using
#: Forecast objects
UTAH_CHPC_PATTERN = (
    "https://pando-rgw01.chpc.utah.edu/HRRR/oper/sfc/"
    "2017{fcst.month:02d}{fcst.day:02d}/"
    "hrrr.t{fcst.hour:02d}z.wrfsfcf{fcst.fcst_hour:02d}.grib2"
)

#: Base pattern for output files, using Forecast objects
FN_PATTERN = (
    "hrrr.{fcst.year:4d}{fcst.month:02d}{fcst.day:02d}."
    "t{fcst.hour:02d}z.f{fcst.fcst_hour:02d}.grib2"
)

def download_hrrr(t_begin, t_end):

    for t in pd.date_range(t_begin - pd.Timedelta('1H'), t_end, freq='1H'):
        nhours = int((t_end - t).total_seconds() / 3600.)
        print(t)
        for i in range(nhours+1):
            print(i)
            fcst = Forecast(t.year, t.month, t.day, t.hour, i)

            url = UTAH_CHPC_PATTERN.format(fcst=fcst)
            filename = FN_PATTERN.format(fcst=fcst)
            full_filename = "hrrr/" + filename
            print("{} -> {}".format(url, full_filename))

            _ = wget(url, full_filename)


if __name__ == "__main__":
    print(snakemake)
    print(snakemake.params)

    if not os.path.exists("hrrr"):
        os.makedirs("hrrr")
    download_hrrr(snakemake.params['t_begin'], snakemake.params['t_end'])
