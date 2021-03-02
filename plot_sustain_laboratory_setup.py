#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Width and height of individual panels of the ASIST tank
PANEL_WIDTH = 23 / 11
PANEL_HEIGHT = 2

# We have a total of 19 panels
SUSTAIN_WIDTH = 23
SUSTAIN_HEIGHT = 2

# Resting water depth, need to fix KRD
H = 0.8

#irgason, combined positions from two runs
irgason1 = 7.15, SUSTAIN_HEIGHT - 0.6
irgason2 = 9.55, SUSTAIN_HEIGHT - 0.6
irgason3 = 11.98, SUSTAIN_HEIGHT - 0.6

# UDM positions
udm = [3.40, 6.25, 8.85, 11.31, 13.96]

# Wave wires (combined posistions from 2 runs)
wave_wire = [1.53, 6.14, 9.30, 10.80, 14.0]

# Pressure ports
static_pressure = [1.68, 3.51, 5.34, 7.17, 9.00, 10.83, 12.66, 14.49, 16.32]

# Pitot + hotfilm
pitot = 9.85

fig = plt.figure(figsize=(16, 5))
ax = fig.add_subplot(111, xlim=(-1, SUSTAIN_WIDTH + 1), ylim=(0, SUSTAIN_HEIGHT))

for u in udm:
    plt.plot(u, 1.94, 'ws', mec='k', ms=12, clip_on=False, zorder=5)
    plt.plot([u, u - 0.1], [2, H], 'k:', lw=1)
    plt.plot([u, u + 0.1], [2, H], 'k:', lw=1)

#irgason placement
plt.plot(irgason1[0], irgason1[1], 'k*', ms=12, clip_on=False, zorder=5)
plt.plot(irgason2[0], irgason2[1], 'k*', ms=12, clip_on=False, zorder=5)
plt.plot(irgason3[0], irgason3[1], 'k*', ms=12, clip_on=False, zorder=5)

#water
plt.plot([-0.5, SUSTAIN_WIDTH], [H, H], 'c-', lw=2)
plt.fill_between([-0.5, SUSTAIN_WIDTH], [H, H], color='c', alpha=0.5)

# beach
x = np.linspace(19,23, 30)
y = (-1/16)*(x-23)**2+1
plt.plot(x,y, 'k:', lw=3)
plt.text(18, -0.55, 'Porous beach', fontsize=16)

# inlet
plt.plot([-1, 0], [1, 1], 'k-', lw=3)
plt.plot([-0.5, -0.5], [0, 1], 'k-', lw=2)
plt.plot([-1, -0.5], [0.25, 0.25], 'k-', lw=2)
plt.arrow(-0.5, 1.5, 1, 0, width=0.2, head_width=0.5, clip_on=False, zorder=5)
plt.text(-1, 2.2, 'Wind inlet', zorder=10, fontsize=16, ha='center')
plt.text(-1, -0.55, 'Wavemaker', fontsize=16, ha='center')

# outlet
plt.plot([SUSTAIN_WIDTH, SUSTAIN_WIDTH], [0, 1], 'k-', lw=3)
plt.plot([SUSTAIN_WIDTH, SUSTAIN_WIDTH + 1], [1, 1], 'k-', lw=3)
plt.arrow(SUSTAIN_WIDTH - 1, 1.5, 1, 0, width=0.2, head_width=0.5, clip_on=False, zorder=5)
plt.text(SUSTAIN_WIDTH, 2.2, 'Outlet', fontsize=16)

# tank
plt.plot([-1, SUSTAIN_WIDTH + 1], [2, 2], 'k-', lw=3, clip_on=False)
plt.plot([-1, SUSTAIN_WIDTH], [0, 0], 'k-', lw=3, clip_on=False)

for w in wave_wire:
    plt.plot(w, 1.94, 'kv', ms=12, clip_on=False, zorder=5)
    plt.plot([w, w ], [2, 0], 'k', lw=1)

for p in static_pressure:
    plt.plot(p, 2, 'wo', mec='k', ms=12, clip_on=False, zorder=5)

plt.plot(pitot, 1.2, 'kx', ms=12, clip_on=False, zorder=5)
plt.plot([pitot,pitot], [2, 0], 'k', linewidth = 3)

plt.plot(np.nan, np.nan, 'k*', ms=16, label='Sonic anemometer')
plt.plot(np.nan, np.nan, 'ws', mec='k', ms=16, label='Ultrasonic Distance Meter')
plt.plot(np.nan, np.nan, 'kv', ms=16, label="Wave Wire")
plt.plot(np.nan, np.nan, 'wo', mec='k', ms=16, label="Static Pressure Ports")
plt.plot(np.nan, np.nan, 'kx', ms=16, label="Pitot + hot-film")

plt.legend(bbox_to_anchor=(0.2, 0.8), bbox_transform=plt.gcf().transFigure,
           prop={'size': 16}, ncol=3, fancybox=True, shadow=True)

plt.xlabel('Fetch [m]', fontsize=16)
plt.ylabel('Height [m]', fontsize=16)
plt.xlim(-1, 24)
plt.xticks(range(0, 24, 2))

ax.tick_params(axis='both', labelsize=16)

fig.subplots_adjust(left=0.1, bottom=0.2, top=0.7, right=0.95)
plt.savefig('laboratory_setup.png', dpi=300)
plt.close(fig)
