import sys
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from numpy import pi


with np.load(sys.argv[1]) as f:
    wind_speeds = f['wind_speeds']
    axial_induction = f['axial_induction']
    derivs = f['derivs']


fig, ax = plt.subplots(figsize=(4, 2.2))
fig.subplots_adjust(bottom=0.18, top=0.96)
ax.contourf(wind_speeds, axial_induction, derivs.T,
            np.arange(-1.4, 1.41, 0.2), cmap='coolwarm')
CS = ax.contour(wind_speeds, axial_induction, derivs.T,
                [-0.8, -0.4, 0, 0.4, 0.8], colors='k')
ax.clabel(CS, fmt=r'$%.1f$')
ax.set_xlabel(r'Wind speed $U_\infty$ [m/s]')
ax.set_ylabel(r'Axial induction $u$ [m/s]')
# ax.set_title(r'Contours of $\dot{u}$')

ax.add_artist(
    patches.Rectangle((10, 1), 15, 1.4, lw=0.5, ec='gray', fc='none'))
ax.annotate('Area shown\nin Figure 8', (10.5, 2.4), fontsize='small',
            va='bottom', ha='left', color='gray')

try:
    plt.savefig(sys.argv[2])
except Exception as e:
    print(e.args[0])
    raise e
