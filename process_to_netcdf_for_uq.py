from datetime import datetime, timedelta
import glob
import numpy as np
from sustain_drag_2020.irgason import read_irgason_from_toa5
from sustain_drag_2020.wavewire import read_wavewire_from_toa5
import xarray as xr

DATAPATH = '/home/milan/Work/sustain/data/sustain-drag-2020/20191121'
RUN_SECONDS = 600
FREQUENCY = 20

start_time = datetime(2019, 11, 21, 16, 10)
end_time = start_time + 13 * timedelta(seconds=RUN_SECONDS)

files = glob.glob(DATAPATH + '/TOA5_SUSTAIN_Wind.FAST*.dat')
files.sort()
time, irg1, irg2 = read_irgason_from_toa5(files, valid_flag=11)

mask = (time >= start_time) & (time <= end_time)

ttime = time[mask]
irg_seconds = np.array([(t - start_time).total_seconds() for t in ttime])

seconds = np.linspace(0, irg_seconds.max(), int(irg_seconds.max() * FREQUENCY) + 1, endpoint=True)

u1 = np.interp(seconds, irg_seconds, irg1['u'][mask])
u2 = np.interp(seconds, irg_seconds, irg2['u'][mask])
v1 = np.interp(seconds, irg_seconds, irg1['v'][mask])
v2 = np.interp(seconds, irg_seconds, irg2['v'][mask])
w1 = np.interp(seconds, irg_seconds, irg1['w'][mask])
w2 = np.interp(seconds, irg_seconds, irg2['w'][mask])
T1 = np.interp(seconds, irg_seconds, irg1['T'][mask])
T2 = np.interp(seconds, irg_seconds, irg2['T'][mask])

files = glob.glob(DATAPATH + '/TOA5_OSSwavex4.*.dat')
files.sort()
time, data = read_wavewire_from_toa5(files)

mask = (time >= start_time) & (time <= end_time)

ttime = time[mask]
ww_seconds = np.array([(t - start_time).total_seconds() for t in ttime])

e1 = np.interp(seconds, ww_seconds, data['w1'][mask]) 
e2 = np.interp(seconds, ww_seconds, data['w2'][mask]) 
e3 = np.interp(seconds, ww_seconds, data['w3'][mask]) 

# make elevation start from zero
e1 -= np.mean(e1[seconds < RUN_SECONDS])
e2 -= np.mean(e2[seconds < RUN_SECONDS])
e3 -= np.mean(e3[seconds < RUN_SECONDS])

ds = xr.Dataset(
    {
        'u1': ('time', u1), 
        'u2': ('time', u2), 
        'v1': ('time', v1), 
        'v2': ('time', v2), 
        'w1': ('time', w1), 
        'w2': ('time', w2), 
        'T1': ('time', T1),
        'T2': ('time', T2),
        'e1': ('time', e1),
        'e2': ('time', e2),
        'e3': ('time', e3),
    },
    coords = {'time': seconds}
    )

ds['time'].attrs['name'] = 'Time since start of experiment'
ds['time'].attrs['units'] = 's'
ds['u1'].attrs['name'] = 'Along-tank velocity from IRGASON 1'
ds['u1'].attrs['units'] = 'm/s'
ds['u1'].attrs['fetch'] = 7.15
ds['u2'].attrs['name'] = 'Along-tank velocity from IRGASON 2'
ds['u2'].attrs['units'] = 'm/s'
ds['u2'].attrs['fetch'] = 11.98
ds['v1'].attrs['name'] = 'Cross-tank velocity from IRGASON 1'
ds['v1'].attrs['units'] = 'm/s'
ds['v1'].attrs['fetch'] = 7.15
ds['v2'].attrs['name'] = 'Cross-tank velocity from IRGASON 2'
ds['v2'].attrs['units'] = 'm/s'
ds['v2'].attrs['fetch'] = 11.98
ds['w1'].attrs['name'] = 'Vertical velocity from IRGASON 1'
ds['w1'].attrs['units'] = 'm/s'
ds['w1'].attrs['fetch'] = 7.15
ds['w2'].attrs['name'] = 'Vertical velocity from IRGASON 2'
ds['w2'].attrs['units'] = 'm/s'
ds['w2'].attrs['fetch'] = 11.98
ds['T1'].attrs['name'] = 'Air temperature from IRGASON 1'
ds['T1'].attrs['units'] = 'K'
ds['T1'].attrs['fetch'] = 7.15
ds['T2'].attrs['name'] = 'Air temperature from IRGASON 2'
ds['T2'].attrs['units'] = 'K'
ds['T2'].attrs['fetch'] = 11.98
ds['e1'].attrs['name'] = 'Water elevation from wave-wire 1'
ds['e1'].attrs['units'] = 'm'
ds['e1'].attrs['fetch'] = 9.8
ds['e2'].attrs['name'] = 'Water elevation from wave-wire 2'
ds['e2'].attrs['units'] = 'm'
ds['e2'].attrs['fetch'] = 10.1
ds['e3'].attrs['name'] = 'Water elevation from wave-wire 3'
ds['e3'].attrs['units'] = 'm'
ds['e3'].attrs['fetch'] = 10.1

ds.attrs['experiment_name'] = 'wind-only_fresh-water_20191121'
ds.attrs['experiment_time'] = start_time.strftime('%Y-%m-%d_%H:%M:%S')
ds.attrs['institution'] = 'University of Miami'
ds.attrs['facility'] = 'SUSTAIN Laboratory'
ds.attrs['tank'] = 'SUSTAIN'
ds.attrs['contact'] = 'Milan Curcic <mcurcic@miami.edu>'

ds.to_netcdf('sustain_drag_' + start_time.strftime('%Y%m%d')  + '.nc', 'w', 'NETCDF4')
