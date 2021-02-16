from datetime import datetime, timedelta
import glob
import numpy as np
import pandas as pd
from sustain_drag_2020.fetch import fetch_20201106
from sustain_drag_2020.irgason import read_irgason_from_toa5
from sustain_drag_2020.udm import clean_elevation_from_udm, read_udm_from_toa5
from sustain_drag_2020.wavewire import read_wavewire_from_toa5
import xarray as xr

DATAPATH = '/home/milan/Work/sustain/data/sustain-drag-2020/20201106'
RUN_SECONDS = 600
FREQUENCY = 20
FREQUENCY_PRESSURE = 10
NUM_RUNS = 11

start_time = datetime(2020, 11, 6, 17, 50)
end_time = start_time + NUM_RUNS * timedelta(seconds=RUN_SECONDS)

# IRGASON
files = glob.glob(DATAPATH + '/TOA5_SUSTAIN_Wind.FAST*.dat')
files.sort()
time, irg1, irg2 = read_irgason_from_toa5(files, valid_flag=11)

mask = (time >= start_time) & (time <= end_time)

ttime = time[mask]
irg_seconds = np.array([(t - start_time).total_seconds() for t in ttime])

seconds = np.linspace(0, irg_seconds.max(), int(irg_seconds.max() * FREQUENCY) + 1, endpoint=True)

fan = (seconds // RUN_SECONDS) * 5
fan[-1] = 50

u = np.interp(seconds, irg_seconds, irg2['u'][mask])
v = np.interp(seconds, irg_seconds, irg2['v'][mask])
w = np.interp(seconds, irg_seconds, irg2['w'][mask])
T = np.interp(seconds, irg_seconds, irg2['T'][mask])

# Wave wires
files = glob.glob(DATAPATH + '/TOA5_OSSwave*.dat')
files.sort()
time, data = read_wavewire_from_toa5(files)

mask = (time >= start_time) & (time <= end_time)

ttime = time[mask]
ww_seconds = np.array([(t - start_time).total_seconds() for t in ttime])

eta_w = np.zeros((4, seconds.size))
eta_w[0,:] = np.interp(seconds, ww_seconds, data['w2'][mask])
eta_w[1,:] = np.interp(seconds, ww_seconds, data['w4'][mask])
eta_w[2,:] = np.interp(seconds, ww_seconds, data['w1'][mask])
eta_w[3,:] = np.interp(seconds, ww_seconds, data['w3'][mask])

for n in range(4):
    eta_w[n,:] -= np.mean(eta_w[n, seconds < RUN_SECONDS])

fetch_w = fetch_20201106['wave_wire']

# UDM
files = glob.glob(DATAPATH + '/TOA5_SUSTAIN_ELEV*.dat')
files.sort()
time, u1, u2, u3, u4, u5, u6 = read_udm_from_toa5(files)

mask = (time >= start_time) & (time <= end_time)
ttime = time[mask]
udm_seconds = np.array([(t - start_time).total_seconds() for t in ttime])

eta_u = np.zeros((5, seconds.size))
eta_u[0,:] = np.interp(seconds, udm_seconds, u3[mask])
eta_u[1,:] = np.interp(seconds, udm_seconds, u4[mask])
eta_u[2,:] = np.interp(seconds, udm_seconds, u1[mask])
eta_u[3,:] = np.interp(seconds, udm_seconds, u6[mask])
eta_u[4,:] = np.interp(seconds, udm_seconds, u5[mask])

for n in range(5):
    eta_u[n,:] = clean_elevation_from_udm(eta_u[n,:])

fetch_u = fetch_20201106['udm']

# Static pressure
data = pd.read_csv(DATAPATH + '/scanivalve_mps_' + start_time.strftime('%Y%m%d') + '.csv')
time = data['FTime']

fan_p = (time // RUN_SECONDS) * 5
fan[-1] = 50

fetch_p = fetch_20201106['static_pressure']

p = np.zeros((9, time.size))

p[0,:] = data['33Press'][:]
p[1,:] = data['34Press'][:]
p[2,:] = data['35Press'][:]
p[3,:] = data['36Press'][:]
p[4,:] = data['37Press'][:]
p[5,:] = data['38Press'][:]
p[6,:] = data['39Press'][:]
p[7,:] = data['40Press'][:]
p[8,:] = data['41Press'][:]

# Create dataset
ds = xr.Dataset(
    {
        'fan': ('time', fan), 
        'fan_p': ('time_p', fan_p), 
        'u': ('time', u), 
        'v': ('time', v), 
        'w': ('time', w), 
        'T': ('time', T),
        'eta_w': (['fetch_wavewire', 'time'], eta_w),
        'eta_u': (['fetch_udm', 'time'], eta_u),
        'p': (['fetch_pressure', 'time_p'], p),
    },
    coords = {
        'time': seconds,
        'time_p': time.to_numpy(),
        'fetch_wavewire': fetch_w,
        'fetch_udm': fetch_u,
        'fetch_pressure': fetch_p
    }
)

# Add metadata
ds['time'].attrs['name'] = 'Time since start of experiment'
ds['time'].attrs['units'] = 's'
ds['time_p'].attrs['name'] = 'Time since start of experiment, static pressure only'
ds['time_p'].attrs['units'] = 's'
ds['fetch_wavewire'].attrs['name'] = 'Fetch of wave wires'
ds['fetch_wavewire'].attrs['units'] = 'm'
ds['fetch_udm'].attrs['name'] = 'Fetch of UDM'
ds['fetch_udm'].attrs['units'] = 'm'
ds['fan'].attrs['name'] = 'Fan speed'
ds['fan'].attrs['units'] = 'Hz'
ds['fan_p'].attrs['name'] = 'Fan speed in pressure time coordinate'
ds['fan_p'].attrs['units'] = 'Hz'
ds['u'].attrs['name'] = 'Along-tank velocity from IRGASON'
ds['u'].attrs['units'] = 'm/s'
ds['u'].attrs['fetch'] = 9.55
ds['v'].attrs['name'] = 'Cross-tank velocity from IRGASON'
ds['v'].attrs['units'] = 'm/s'
ds['v'].attrs['fetch'] = 9.55
ds['w'].attrs['name'] = 'Vertical velocity from IRGASON'
ds['w'].attrs['units'] = 'm/s'
ds['w'].attrs['fetch'] = 9.55
ds['T'].attrs['name'] = 'Air temperature from IRGASON'
ds['T'].attrs['units'] = 'K'
ds['T'].attrs['fetch'] = 9.55
ds['p'].attrs['name'] = 'Static pressure at the ceiling'
ds['p'].attrs['units'] = 'Pa'
ds['eta_w'].attrs['name'] = 'Water elevation from wave wires'
ds['eta_w'].attrs['units'] = 'm'
ds['eta_u'].attrs['name'] = 'Water elevation from UDM'
ds['eta_u'].attrs['units'] = 'm'

ds.attrs['experiment_name'] = 'wind-only_fresh-water_' + start_time.strftime('%Y%m%d')
ds.attrs['experiment_time'] = start_time.strftime('%Y-%m-%d_%H:%M:%S')
ds.attrs['water_type'] = 'fresh'
ds.attrs['initial_water_depth'] = 0.8
ds.attrs['institution'] = 'University of Miami'
ds.attrs['facility'] = 'SUSTAIN Laboratory'
ds.attrs['tank'] = 'SUSTAIN'
ds.attrs['contact'] = 'Milan Curcic <mcurcic@miami.edu>'

ds.to_netcdf('sustain_drag_' + start_time.strftime('%Y%m%d')  + '.nc', 'w', 'NETCDF4')
