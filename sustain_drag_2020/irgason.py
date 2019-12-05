from datetime import datetime, timedelta
import glob
import numpy as np
import os

def read_irgason_from_toa5(filenames):
    """Reads data from IRGASON output file(s) in TOA5 format.
    If filenames is a string, process a single file. If it is
    a list of strings, process files in order and concatenate."""
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
    irgason1 = {'u': [], 'v': [], 'w': [], 'T': []}
    irgason2 = {'u': [], 'v': [], 'w': [], 'T': []}

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
        irgason2['u'].append(float(line[26].strip('"')))
        irgason2['v'].append(float(line[27].strip('"')))
        irgason2['w'].append(float(line[28].strip('"')))
        irgason2['T'].append(float(line[29].strip('"')))

    times = np.array(times)
    for var in ['u', 'v', 'w', 'T']:
        irgason1[var] = np.array(irgason1[var])
        irgason2[var] = np.array(irgason2[var])

    return times, irgason1, irgason2



