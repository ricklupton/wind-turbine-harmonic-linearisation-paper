import sys
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, newaxis
from matplotlib.patches import ConnectionPatch
import h5py
#matplotlib.rc('font', family='Bitstream Vera Sans')


def eval_harmonic(w, harmonic, t):
    mean, amp = harmonic
    return np.real(mean[newaxis, :]) + \
        np.real(amp[newaxis, :] * np.exp(1j * w * t)[:, newaxis])


def plot_results(ax, w, U0, Ub):
    g = f['fixedspeed_simpblade_frozen/w%.2f/mean%.1f/amp%.1f' % (w, U0, Ub)]
    t = np.arange(0, g['mbwind'].attrs['t1'], g['mbwind'].attrs['dt'])
    U = eval_harmonic(w, (np.array([U0]), np.array([Ub])), t)

    cmap = plt.cm.PiYG
    norm = matplotlib.colors.Normalize(vmin=-180, vmax=180)

    for ir in range(10, len(model.bem.radii), 2):
        x = model.bem.radii[ir] + (U - U0)/2

        # mbwind results have been rotated to give out-of-plane and
        # in-plane loading already
        ax.plot(x, g['mbwind/blade_loading'][:, ir, 0], 'k')

        # Harmonic results
        fh = eval_harmonic(w, (-g['mbwind_harmonic'].attrs['f0'][ir, 2:3],
                               -g['mbwind_harmonic'].attrs['fb'][ir, 2:3]), t)
        polygon = matplotlib.patches.Polygon(np.c_[x, fh], edgecolor='r',
                                             facecolor='r', alpha=0.4)
        polygon.set_clip_on(False)
        ax.add_artist(polygon)

    ax.locator_params(nbins=5, tight=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


freqs = np.power(10, [-1.5, -1.0, -0.5, 0.0, 0.5])[::2]
f = h5py.File(sys.argv[1], 'r')

from whales import aeroelastic
model_file = sys.argv[2]
# not using FE, centrifugal stiffening doesn't matter
model = aeroelastic.AeroelasticModel.from_yaml(model_file, 0)
r = np.array(model.bem.radii)

U0 = 8
Ub = 5
fig, ax = plt.subplots(len(freqs), sharex=True, sharey=True, figsize=(6, 4))
fig.subplots_adjust(left=0.12, top=0.95, right=0.85)

for i in range(len(freqs)):
    plot_results(ax[i], freqs[i], U0, Ub)
    ax[i].annotate('{:.2f} rad/s'.format(freqs[i]), (1.05, 0.5),
                   xycoords="axes fraction")


ax[0].annotate('', (15, 1), (15+Ub, 1),
               ("data", "axes fraction"), ("data", "axes fraction"),
               arrowprops=dict(arrowstyle='|-|,widthA=0.5,widthB=0.5'))
ax[0].annotate('Wind speed\nvariation', (15 + Ub/2, 0.93),
               xycoords=("data", "axes fraction"),
               ha='center', va='top')


ax[-1].set_xlabel('Blade radius [m]')
ax[1].set_ylabel('Out-of-plane force [N/m]')

# For a in ax.flat:
#     a.locator_params(nbins=6, axis='y', tight=False)
#     a.set_xticks([])
#     plt.setp(list(a.spines.values()), color='gray', lw=0.5)
#     a.tick_params(color='gray')
#     a.yaxis.set_ticks_position('left' if a.is_first_col() else 'none')
#     a.xaxis.set_ticks_position('none')


# ax[1, -1].annotate('Harmonic', (0.7, 0), (1.15, 0), color='red', va='center',
#                    arrowprops=dict(arrowstyle='-', color='r',
#                                    alpha=0.5, lw=0.5))
# ax[1, -1].annotate('Nonlinear', (0.9, 2000), (1.15, 2000),
#                    color='k', va='center',
#                    arrowprops=dict(arrowstyle='-', color='k',
#                                    alpha=0.5, lw=0.5))
# ax[1, -1].annotate('Tangent', (0.65, -1000), (1.15, -2000),
#                    color='blue', va='center',
#                    arrowprops=dict(arrowstyle='-', color='b',
#                                    alpha=0.5, lw=0.5))

try:
    plt.savefig(sys.argv[3])
except Exception as e:
    print(e.args[0])
    raise e
