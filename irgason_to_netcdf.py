from datetime import datetime, timedelta
import glob
import numpy as np
from sustain_drag_2020.irgason import read_irgason_from_toa5
import xarray as xr

DATAPATH = '/home/milan/Work/sustain/data/sustain-drag-2020/20191121'

RUN_SECONDS = 600

start_time = datetime(2019, 11, 21, 16, 10)
end_time = start_time + 13 * timedelta(seconds=RUN_SECONDS)

files = glob.glob(DATAPATH + '/TOA5_SUSTAIN_Wind.FAST*.dat')
time, irg1, irg2 = read_irgason_from_toa5(files, valid_flag=11)

mask = (time >= start_time) & (time <= end_time)

ttime = time[mask]
seconds = np.array([(t - start_time).total_seconds() for t in ttime])

ds = xr.Dataset(
    {'u': ('time', irg1['u'][mask]), 
     'v': ('time', irg1['v'][mask]), 
     'w': ('time', irg1['w'][mask]), 
     'T': ('time', irg1['T'][mask])},
    coords = {'time': seconds})

ds.to_netcdf('irgason.nc', 'w', 'NETCDF4')
