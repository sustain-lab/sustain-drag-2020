from asist.utility import power_spectrum
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import beta
from sustain_drag_2020.irgason import read_irgason_from_toa5, rotate
from sustain_drag_2020.udm import read_udm_from_toa5
from sustain_drag_2020.fetch import fetch_20201118
from sustain_drag_2020.dispersion import w2k
import xarray as xr

import warnings
warnings.filterwarnings('ignore')

import matplotlib
font = {'size': 16}
matplotlib.rc('font', **font)

GRAV = 9.8

def cp_cg(F, f, depth):
    """Return phase and group speeds for a wave spectrum."""
    w = 2 * np.pi * f
    k = w2k(w, depth)[0]
    cp = w[1:] / k[1:]
    cg = np.diff(w) / np.diff(k)
    return cp, cg


def radiation_stress(F, f, df, depth, rhow=1000):
    """Returns radiation stress."""
    cp, cg = cp_cg(F, f, depth)
    rad_stress_fac = 2 * cg / cp - 0.5
    return rhow * GRAV * np.sum(rad_stress_fac * F[1:] * df)


def stress_from_momentum_budget(H, dhdx, dSxxdx, dpdx, taub, rhow):
    """Returns the stress estimate from momentum budget components."""
    return rhow * GRAV * H * dhdx + H * dpdx + dSxxdx - taub


def eddy_covariance_flux(ds, t0, t1, irgnum=None):
    """Eddy covariance flux from IRGASON, for a time subset between t0 and t1."""
    time = np.array(ds.time)
    mask = (time >= t0) & (time <= t1)
    if irgnum:
        u, v, w = ds.u[irgnum-1,mask][:], ds.v[irgnum-1,mask][:], ds.w[irgnum-1,mask][:]
    else:
        u, v, w = ds.u[mask][:], ds.v[mask][:], ds.w[mask][:]
    u = np.sqrt(u**2 + v**2) # horizontal velocity
    angle = np.arctan2(np.nanmean(w), np.nanmean(u))
    u, w = rotate(u, w, angle)
    U, W = np.nanmean(u), np.nanmean(w) # time mean
    up, wp = u - U, w - W # deviations from the mean
    uw = np.nanmean(up * wp) # stress
    good = ~np.isnan(u) & ~np.isnan(w)
    perc_good = (u[good].size / u.size) * 100 # percentage of data that is not bad
    return U, uw, perc_good


def mean_pressure(p, time, fan, run_seconds, offset):
    pmean = np.zeros((len(fan), p.shape[0]))
    pstd = np.zeros((len(fan), p.shape[0]))
    for n in range(len(fan)):
        t0 = n * run_seconds + offset
        t1 = t0 + run_seconds - offset
        mask = (time >= t0) & (time < t1)
        for i in range(p.shape[0]):
            pp = p[i,mask]
            pmean[n,i] = np.mean(pp)
            pstd[n,i] = np.std(pp)
    return pmean, pstd


RUN_DURATION = 600
RUN_OFFSET = 60
NUM_RUNS = 11
FAN = range(0, 5 * NUM_RUNS, 5) 

ds = xr.open_dataset('data/sustain_drag_20201118.nc')

# wind speed
U = np.zeros(NUM_RUNS)
perc_good = np.zeros(NUM_RUNS)
for n in range(NUM_RUNS):
    t0 = n * RUN_DURATION + RUN_OFFSET
    t1 = t0 + RUN_DURATION - RUN_OFFSET
    U[n], _, perc_good[n] = eddy_covariance_flux(ds, t0, t1)

# static pressure
pmean, pstd = mean_pressure(np.array(ds.p), np.array(ds.time_p), FAN, RUN_DURATION, RUN_OFFSET)

good = [1, 2, 3, 4, 5, 6]

fig = plt.figure(figsize=(16, 2))
dpdx = np.zeros(NUM_RUNS)
dpdx1 = np.zeros(NUM_RUNS)
for n in range(NUM_RUNS):
    ax = plt.subplot2grid((1, NUM_RUNS), (0, n))
    plt.plot(ds.fetch_pressure[good], pmean[n,good], marker='o')
    p = np.polyfit(ds.fetch_pressure[good], pmean[n,good], 1)
    dpdx[n] = p[0]
    dpdx1[n] = (pmean[n,good][-1] - pmean[n,good][0]) \
            / (ds.fetch_pressure[good][-1] - ds.fetch_pressure[good][0])
    plt.plot(ds.fetch_pressure, np.polyval(p, ds.fetch_pressure), lw=1)
    plt.xlabel('Fetch [m]')
    if n == 0: plt.ylabel('Pressure [Pa]')
    plt.title(str(FAN[n]) + ' Hz')
    plt.grid()


# wavewire
h = np.zeros((NUM_RUNS, 4))
Sxx = np.zeros((NUM_RUNS, 4))
for n in range(NUM_RUNS):
    t0 = n * RUN_DURATION + RUN_OFFSET
    t1 = t0 + RUN_DURATION - RUN_OFFSET
    mask = (ds.time >= t0) & (ds.time < t1)
    for i in range(ds.fetch_wavewire.size):
        h[n,i] = np.mean(ds.eta_w[i,mask])
        eta = detrend(ds.eta_w[i,mask])
        F, f, df = power_spectrum(eta, 1 / 20, binsize=1)
        fmask = (f > 0.5) & (f < 10)
        Sxx[n,i] = radiation_stress(F[fmask], f[fmask], np.diff(f)[0], 0.8, rhow=1000)

# UDM
h_udm = np.zeros((NUM_RUNS, ds.fetch_udm.size))
for n in range(NUM_RUNS):
    t0 = n * RUN_DURATION + RUN_OFFSET
    t1 = t0 + RUN_DURATION - RUN_OFFSET
    mask = (ds.time >= t0) & (ds.time < t1)
    for i in range(ds.fetch_udm.size):
        h_udm[n,i] = np.mean(ds.eta_u[i,mask])


plt.figure(figsize=(16, 6))
for n, f in enumerate(FAN):
    plt.plot(ds.fetch_wavewire, h[n,:], marker='o', label='%2i Hz' % f)
plt.legend(ncol=3)
plt.grid()
plt.xlabel('Fetch [m]')
plt.ylabel('Mean elevation [m]')
plt.title('Mean elevation from wave wire')



good = [0, 1, 2, 3]
dhdx = np.zeros((NUM_RUNS))
dSxxdx = np.zeros((NUM_RUNS))
for n in range(NUM_RUNS):
    x = ds.fetch_wavewire[good]
    #dhdx[n] = np.polyfit(x, h[n,good], 1)[0]
    dhdx[n] = (h[n,-1] - h[n,0]) / (x[-1] - x[0])
    #dSxxdx[n] = np.polyfit(x, Sxx[n,good], 1)[0]
    dSxxdx[n] = (Sxx[n,-1] - Sxx[n,0]) / (x[-1] - x[0])


plt.figure(figsize=(16, 6))
for n, f in enumerate(FAN):
    plt.plot(ds.fetch_udm, h_udm[n,:], marker='o', label='%2i Hz' % f)
plt.legend(ncol=3)
plt.grid()
plt.xlabel('Fetch [m]')
plt.ylabel('Mean elevation [m]')
plt.title('Mean elevation from UDM')


dhdx_udm = np.zeros((NUM_RUNS))
for n in range(NUM_RUNS):
    x = ds.fetch_udm
    dhdx_udm[n] = (h_udm[n,-1] - h_udm[n,0]) / (x[-1] - x[0])

H = np.mean(h, axis=1) + 0.8
rhow = 1e3
rhoa = 1.2

# Load eddy covariance data
ec = xr.open_dataset('data/sustain_eddy_covariance_cd.nc')


fig = plt.figure(figsize=(8, 6))
plt.plot(U, rhow * GRAV * H * dhdx, marker='o', label=r'$\rho_w g H \dfrac{\partial h}{\partial x}$')
plt.plot(U, H * dpdx, marker='o', label='$H\dfrac{\partial p}{\partial x}$')
plt.plot(U, dSxxdx, marker='o', label='$\dfrac{\partial S_{xx}}{\partial x}$')
plt.plot(U, rhow * GRAV * H * dhdx + H * dpdx + dSxxdx, 'k-', marker='o', label='Total stress')
plt.plot(U, rhow * GRAV * H * dhdx + H * dpdx1 + dSxxdx, 'k--', marker='o', label='Total stress')
plt.legend(ncol=2, prop={'size': 14})
plt.xlabel('Fan speed [Hz]')
plt.ylabel('Stress component [$N/m^2$]')
plt.plot([0, 45], [0, 0], 'k--')
plt.ylim(-20, 40)
plt.grid()
