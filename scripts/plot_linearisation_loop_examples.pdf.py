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


def plot_results_loop_ax(ax, w, U0, Ub, var, ir=0):
    g = f['fixedspeed_simpblade_frozen/w%.2f/mean%.1f/amp%.1f' % (w, U0, Ub)]
    rqs = f['fixedspeed_simpblade_frozen/mbwind_quasistatic/mean%.1f' % U0]

    t = np.arange(0, g['mbwind'].attrs['t1'], g['mbwind'].attrs['dt'])
    U = eval_harmonic(w, (np.array([U0]), np.array([Ub])), t)

    hattr = g['mbwind_harmonic'].attrs
    tattr = g['mbwind_tangent'].attrs
    modeh = eval_harmonic(w, (hattr['x0'][:], hattr['xb'][:]), t)
    modet = eval_harmonic(w, (tattr['x0'][:], tattr['xb'][:]), t)
    frh = eval_harmonic(w, (hattr['fr0'][ir, :], hattr['frb'][ir, :]), t)
    frt = eval_harmonic(w, (tattr['fr0'][ir, :], tattr['frb'][ir, :]), t)
    tiph = eval_harmonic(w, (g['mbwind_harmonic/tipdefl_x0'][:],
                             g['mbwind_harmonic/tipdefl_xb'][:]), t)
    tipt = eval_harmonic(w, (g['mbwind_tangent/tipdefl_x0'][:],
                             g['mbwind_tangent/tipdefl_xb'][:]), t)
    rotorh = eval_harmonic(w, (g['mbwind_harmonic/rotor0'][:],
                               g['mbwind_harmonic/rotorb'][:]), t)
    rotort = eval_harmonic(w, (g['mbwind_tangent/rotor0'][:],
                               g['mbwind_tangent/rotorb'][:]), t)
    harmonic, tangent, qs = {
        'rotor_thrust': (rotorh[:, 0], rotort[:, 0], rqs['rotor'][:, 0]),
        'rotor_torque': (rotorh[:, 1], rotort[:, 1], rqs['rotor'][:, 1]),
        'tipdefl_x': (tiph[:, 0], tipt[:, 0], None),
        'tipdefl_y': (tiph[:, 1], tipt[:, 1], None),
        'mode1': (modeh[:, 0], modet[:, 0], None),
        'mode2': (modeh[:, 1], modet[:, 1], None),
        'mode3': (modeh[:, 2], modet[:, 2], None),
        'mode4': (modeh[:, 3], modet[:, 3], None),
        'fx': (frh[:, 0], frt[:, 0], None),
        'fy': (frh[:, 1], frt[:, 1], None),
    }[var]
    scale = 1e3 if var in ('rotor_thrust', 'rotor_torque') else 1
    if var.startswith('mode'):
        nonlinear = g['mbwind/strains'][:, int(var[-1:]) - 1]
    elif var == 'fx':
        nonlinear = g['mbwind/blade_loading'][:, ir, 0]
    elif var == 'fy':
        nonlinear = g['mbwind/blade_loading'][:, ir, 1]
    else:
        nonlinear = g['mbwind'][var][...]
    nonlinear /= scale
    harmonic /= scale
    tangent /= scale

    #if qs is not None:
    #    ax.plot(rqs['wind_speeds'], qs / scale, 'k', lw=0.5, alpha=0.4)

    ax.plot(U, nonlinear, 'k', lw=0.8)

    polygon = matplotlib.patches.Polygon(
        np.c_[U, harmonic], edgecolor='r', facecolor='r', alpha=0.4)
    ax.add_artist(polygon)

    ax.plot(U, tangent, 'b', lw=1, alpha=0.4)

    ax.set_xlim(U0 - Ub*1.5, U0 + Ub*1.5)
    yrange = harmonic.max() - harmonic.min()
    # ax.set_ylim(harmonic.min() - 0.2*yrange,
    #             harmonic.max() + 0.2*yrange)


def plot_results(ax, names, U0=8, Ub=5):
    for iw in range(len(freqs)):
        ax[0, iw].set_title('{:.2f} rad/s'.format(freqs[iw]))
        for iv in range(len(names)):
            plot_results_loop_ax(ax[iv, iw], freqs[iw], U0, Ub, names[iv][0])
    for iv in range(len(names)):
        ax[iv, 0].set_ylabel(names[iv][1], rotation=0, ha='center', va='center')
        ax[iv, 0].yaxis.set_label_coords(-1.0, 0.5)
    ax[-1, 2].set_xlabel('Wind speed [m/s]')


freqs = np.power(10, [-1.5, -1.0, -0.5, 0.0, 0.5])
names = [
    ('rotor_thrust', 'Rotor thrust\n[kN]'),
    ('rotor_torque', 'Rotor torque\n[kNm]'),
    ('tipdefl_x', 'OOP tip deflection\n[m]'),
    ('tipdefl_y', 'IP tip deflection\n[m]'),
]
f = h5py.File(sys.argv[1], 'r')

fig, ax = plt.subplots(len(names), len(freqs), figsize=(6.7, 5),
                       sharex=True, sharey='row')
fig.subplots_adjust(left=0.21, top=0.95, right=0.89)
plot_results(ax, names, 8, 5)


for a in ax.flat:
    a.locator_params(nbins=6, axis='y', tight=False)
    a.set_xticks([8-5, 8, 8+5])
    plt.setp(list(a.spines.values()), color='gray', lw=0.5)
    a.tick_params(color='gray')
    a.yaxis.set_ticks_position('left' if a.is_first_col() else 'none')
    a.xaxis.set_ticks_position('bottom' if a.is_last_row() else 'none')


ax[1, -1].annotate('Harmonic', (11, 4000), (18, 4000), color='red', va='center',
                   arrowprops=dict(arrowstyle='-', color='r',
                                   alpha=0.5, lw=0.5))
ax[1, -1].annotate('Nonlinear', (12, 2000), (18, 2000), color='k', va='center',
                   arrowprops=dict(arrowstyle='-', color='k',
                                   alpha=0.5, lw=0.5))
ax[1, -1].annotate('Tangent', (8, 0000), (18, 0000), color='blue', va='center',
                   arrowprops=dict(arrowstyle='-', color='b',
                                   alpha=0.5, lw=0.5))

try:
    plt.savefig(sys.argv[2])
except Exception as e:
    print(e.args[0])
    raise e
