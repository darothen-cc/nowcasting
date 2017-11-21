""" Configuration setup for animation pipeline

"""

import pandas as pd

# LOCATION CONFIG
NAME = "chicago"
CLON, CLAT = -87.5, 41.75
DLON, DLAT = 2.5, 1.5

STATIONS = {
    'KORD': (),
    'KMDW': (),
}

# TIMERANGE CONFIG
TIME_BEGIN = pd.Timestamp(2017, 11, 15, 12)
TIME_END = pd.Timestamp(2017, 11, 15, 16)

MRMS_TIMES = pd.date_range(TIME_BEGIN, TIME_END, freq='120S')
N0Q_TIMES = pd.date_range(TIME_BEGIN, TIME_END, freq='300S')
INTEGRATED_TIMES = pd.date_range(TIME_BEGIN, TIME_END, freq='1H')
NOWCAST_TIMES = [
    pd.Timestamp(2017, 11, 15, 12, 1, 54),
    pd.Timestamp(2017, 11, 15, 13, 2, 40),
    pd.Timestamp(2017, 11, 15, 14, 0, 56),
    pd.Timestamp(2017, 11, 15, 15, 2, 40)
]
