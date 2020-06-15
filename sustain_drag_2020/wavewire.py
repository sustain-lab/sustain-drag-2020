from datetime import datetime, timedelta
import numpy as np
import os

def read_wavewire_from_toa5(filenames):
    """Reads data from wave wire output file(s) in TOA5 format.
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
    d = {'w1': [], 'w2': [], 'w3': [], 'd1': [], 'd2': [], 'd3': []}

    print('Processing wave wire time series..')

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
        d['w1'].append(float(line[2].strip('"')))
        d['w2'].append(float(line[3].strip('"')))
        d['w3'].append(float(line[4].strip('"')))
        d['d1'].append(float(line[6].strip('"')))
        d['d2'].append(float(line[7].strip('"')))
        d['d3'].append(float(line[8].strip('"')))

    for key in d.keys():
        d[key] = np.array(d[key])
        for i in range(1, d[key].size -1, 1):
            if d[key][i] < 0.2:
                d[key][i] = 0.5 * (d[key][i-1] + d[key][i+1])

    return np.array(times), d
