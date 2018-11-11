import sys
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LinearLocator
import numpy as np
from numpy import pi
import h5py
#matplotlib.rc('font', family='Bitstream Vera Sans')


def eval_harmonic(w, harmonic, t):
    mean, amp = harmonic[:]
    return np.real(mean) + np.real(amp * np.exp(1j * w * t))


def plot_example(fi, fh, axes_row, im, iw, ia, converged=True):
    var = ['generator_torque', 'aero_thrust', 'rotor_speed', 'pitch_angle']
    scales = [1e-6, 1e-3, 30/pi, 180/pi]
    titles = ['Rotor torque\n[MNm]', 'Rotor thrust\n[kN]',
              'Rotor speed\n[rpm]', 'Pitch angle\n[deg]']
    U0 = MEAN_WIND_SPEEDS[im]
    Ub = AMPLITUDES[ia]
    w = FREQUENCIES[iw]
    path = 'NREL5MW_simpblade_model_coarse/w%.2f/mean%.1f/amp%.1f' % \
        (w, U0, Ub)
    gi = fi[path]
    gh = fh[path]

    for ivar in range(len(titles)):
        dt = np.diff(gi['t'][:2])
        n = len(gi['t']) - int(2 * (2*pi/w) / dt)
        t = gi['t'][n:]

        ax = axes_row[ivar]
        if ax.is_first_col():
            ax.set_ylabel(titles[ivar])
            ax.yaxis.set_label_coords(-0.3, 0.5)

        scale = scales[ivar]
        yh = eval_harmonic(w, gh[var[ivar]], t) * scale
        if var[ivar] == 'generator_torque':
            yi = gi['shub'][n:, 3] * scale
            yh *= 97  # referred to hub
        elif var[ivar] == 'aero_thrust':
            yi = gi['shub'][n:, 0] * scale
        else:
            yi = gi[var[ivar]][n:] * scale

        ax.plot(t, yi, 'k')
        ax.plot(t, yh, 'r', alpha=(1 if converged else 0.5))
        ax.axhline(np.mean(yi), c='k', ls=':')
        ax.axhline(np.mean(yh), c='r', ls=':', alpha=(1 if converged else 0.5))

        ax.set_xlim(t[0], t[-1])
        ax.set_xticks([])

    label = '{:.0f} ± {:.0f} m/s\n{:.2f} rad/s'.format(U0, Ub, w)
    if not converged:
        label = '(not converged)\n' + label
    axes_row[0].set_title(label, color=('k' if converged else 'gray'))

    # axes_row[-1].annotate(label, (1.1, 0.5), xycoords="axes fraction",
    #                       ha="left", va="center")

FREQUENCIES = [10**(-1.5), 10**(-1.0), 10**(-0.5), 10**(0.0), 10**(0.5)]
MEAN_WIND_SPEEDS = np.arange(6, 15.1, 1)  # [5., 7., 9., 13., 15.]
AMPLITUDES = [1, 2, 3]  # , 2.0, 3.0, 4.0, 5.0]

fi = h5py.File(sys.argv[1], 'r')
fh = h5py.File(sys.argv[2], 'r')


def plot(tag, examples):
    fig, axes = plt.subplots(4, 4, sharex=False, sharey='row', figsize=(6, 6))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.05,
                        wspace=0.2)

    for i in range(4):
        if examples[i] == (4, 1, 2):
            converged = False
        else:
            converged = True
        plot_example(fi, fh, axes[:, i], *examples[i], converged=converged)

    for a in axes.flat:
        a.locator_params(nbins=6, axis='y', tight=False)
        a.set_xticks([])
        plt.setp(list(a.spines.values()), color='gray', lw=0.5)
        a.tick_params(color='gray')
        a.yaxis.set_ticks_position('left' if a.is_first_col() else 'none')
        a.xaxis.set_ticks_position('none')

    axes[0, 0].set_ylim((0, 5))    # torque
    axes[1, 0].set_ylim((0, 800))  # thrust
    axes[2, 0].set_ylim((7, 14))   # rotor speed
    axes[3, 0].set_ylim((-5, 20))  # pitch
    for a in axes[2, :]:
        a.axhline(12.11, alpha=0.3, lw=0.5)  # rated
    axes[2, 0].annotate('Rated speed', (0.1, 12.2),
                        xycoords=('axes fraction', 'data'),
                        fontsize='x-small', alpha=0.5, color='b')

    try:
        plt.savefig(sys.argv[3].replace('.pdf', '_{}.pdf'.format(tag)))
    except Exception as e:
        print(e.args[0])
        raise e


examples1 = [
    (2, 2, 2),
    (4, 2, 2),
    (6, 2, 2),
    (9, 2, 2),
]
examples2 = [
    (2, 3, 2),
    (4, 3, 2),
    (6, 3, 2),
    (9, 3, 2),
]
examples3 = [
    (2, 1, 2),
    (4, 1, 2),  # None,
    (6, 1, 2),
    (9, 1, 2),
]


plot('midfreq', examples1)
plot('highfreq', examples2)
plot('lowfreq', examples3)


with open(sys.argv[3], 'wt') as f:
    f.write('Keeping SCons dependencies happy')
