from datetime import datetime, timedelta
import glob
import numpy as np
import os

def read_irgason_from_toa5(filenames, valid_flag=16):
    """Reads data from IRGASON output file(s) in TOA5 format.
    If filenames is a string, process a single file. If it is
    a list of strings, process files in order and concatenate
    valid_flag is the largest value of diagnostic flag to include.
    Default is 16, which means do not remove any data
    valid_flag=0 means include only data without any issue
    valid_flag=11 seems to product reasonable values."""
    if type(filenames) is str:
        print('Reading ', filenames)
        data = [line.rstrip() for line in open(filenames).readlines()[4:]]
    elif type(filenames) is list:
        data = []
        for filename in filenames:
            print('Reading ', os.path.basename(filename))
            data += [line.rstrip() for line in open(filename).readlines()[4:]]
    else:
        raise RuntimeError('filenames must be string or list')

    times = []
    irgason1 = {'u': [], 'v': [], 'w': [], 'T': [], 'flag': []}
    irgason2 = {'u': [], 'v': [], 'w': [], 'T': [], 'flag': []}

    print('Processing IRGASON time series..')

    for line in data:
        line = line.replace('"', '').split(',')
        timestr = line[0]
        if len(timestr) == 19:
            time = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
        elif len(timestr) == 21:
            time = datetime.strptime(timestr[:19], '%Y-%m-%d %H:%M:%S')
            time += timedelta(seconds=float(timestr[-2:]))
        else:
            time = datetime.strptime(timestr[:19], '%Y-%m-%d %H:%M:%S')
            time += timedelta(seconds=float(timestr[-3:]))
        times.append(time)
        irgason1['u'].append(float(line[14].strip('"')))
        irgason1['v'].append(float(line[15].strip('"')))
        irgason1['w'].append(float(line[16].strip('"')))
        irgason1['T'].append(float(line[17].strip('"')))
        irgason1['flag'].append(int(line[18].strip('"').replace('NAN', '16')))
        irgason2['u'].append(float(line[26].strip('"')))
        irgason2['v'].append(float(line[27].strip('"')))
        irgason2['w'].append(float(line[28].strip('"')))
        irgason2['T'].append(float(line[29].strip('"')))
        irgason2['flag'].append(int(line[30].strip('"').replace('NAN', '16')))

    times = np.array(times)
    for var in ['u', 'v', 'w', 'T', 'flag']:
        irgason1[var] = np.array(irgason1[var])
        irgason2[var] = np.array(irgason2[var])

    for var in ['u', 'v', 'w', 'T']:
        irgason1[var][irgason1['flag'] > valid_flag] = np.nan 
        irgason2[var][irgason2['flag'] > valid_flag] = np.nan 

    return times, irgason1, irgason2



def rotate(u, w, th):
    """Rotates the vector (u, w) by angle th."""
    ur =  np.cos(th) * u + np.sin(th) * w
    wr = -np.sin(th) * u + np.cos(th) * w
    return ur, wr


def eddy_covariance_flux(irg, time, t0, t1, fan):
    """Eddy covariance flux from IRGASON."""
    U, Ustd, Wstd, uw = [], [], [], []
    max_u_gust = 10
    max_w_gust = 5
    for n in range(len(fan)):
        mask = (time >= t0[n]) & (time <= t1[n])
        u, v, w = irg['u'][mask][:], irg['v'][mask][:], irg['w'][mask][:]

        # clean up
        um, vm, wm = np.nanmean(u), np.nanmean(v), np.nanmean(w)
        u[u > um + max_u_gust] = um + max_u_gust
        u[u < um - max_u_gust] = um - max_u_gust
        v[v > vm + max_u_gust] = vm + max_u_gust
        v[v < vm - max_u_gust] = vm - max_u_gust
        w[w > wm + max_w_gust] = wm + max_w_gust
        w[w < wm - max_w_gust] = wm - max_w_gust

        # horizontal velocity
        u = np.sqrt(u**2 + v**2)

        # rotate
        angle = np.arctan2(np.nanmean(w), np.nanmean(u))
        u, w = rotate(u, w, angle)

        # time average
        um, wm = np.nanmean(u), np.nanmean(w)

        up, wp = u - um, w - wm
        U.append(um)
        Ustd.append(np.nanstd(u))
        Wstd.append(np.nanstd(w))
        uw.append(np.nanmean(up * wp))

    return np.array(U), np.array(Ustd), np.array(Wstd), np.array(uw)
