import numpy as np

# the following fetch values were valid for all experiments in 2019
fetch_2019 = {
    'irgason': np.array([7.15, 11.98]),
    'pitot': 9.85,
    'static_pressure': np.array([1.98, 3.81, 5.64, 7.47, 9.00, 11.13, 12.96, 14.79]),
    'udm': np.array([3.34, 7.08, 9.87, 11.94, 14.90]),
    'wave_wire': np.array([9.8, 10.1, 10.1])
}

fetch_20201106 = {
    'irgason': 9.55,
    'pitot': 9.85,
    'static_pressure': np.array([1.68, 3.51, 5.34, 7.17, 9.00, 10.83, 12.66, 14.49, 16.32]),
    'udm': np.array([3.40, 6.25, 8.85, 11.31, 13.96]),
    'wave_wire': np.array([1.53, 6.14, 9.30, 10.80])
}

# same as previous, except the wavewire from 6.14 m moved to 14 m.
fetch_20201118 = {
    'irgason': 9.55,
    'pitot': 9.85,
    'static_pressure': np.array([1.68, 3.51, 5.34, 7.17, 9.00, 10.83, 12.66, 14.49, 16.32]),
    'udm': np.array([3.40, 6.25, 8.85, 11.31, 13.96]),
    'wave_wire': np.array([1.53, 9.30, 10.80, 14.0])
}
