from datetime import datetime, timedelta
import glob
import numpy as np
from sustain_drag_2020.wavewire import read_wavewire_from_toa5
import xarray as xr

DATAPATH = '/home/milan/Work/sustain/data/sustain-drag-2020/20191121'

RUN_SECONDS = 600

start_time = datetime(2019, 11, 21, 16, 10)
end_time = start_time + 13 * timedelta(seconds=RUN_SECONDS)

files = glob.glob(DATAPATH + '/TOA5_OSSwavex4.*.dat')
time, data = read_wavewire_from_toa5(files)

mask = (time >= start_time) & (time <= end_time)

ttime = time[mask]
seconds = np.array([(t - start_time).total_seconds() for t in ttime])

ds = xr.Dataset(
    {'w1': ('time', data['w1'][mask]), 
     'w2': ('time', data['w2'][mask]), 
     'w3': ('time', data['w3'][mask]), 
     'd1': ('time', data['d1'][mask]),
     'd2': ('time', data['d2'][mask]),
     'd3': ('time', data['d3'][mask])},
    coords = {'time': seconds})

ds.to_netcdf('wavewire.nc', 'w', 'NETCDF4')
