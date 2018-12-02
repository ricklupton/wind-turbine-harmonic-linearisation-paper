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
    return (np.real(mean[newaxis, :]) +
            np.real(amp[newaxis, :] * np.exp(1j * w * t)[:, newaxis]))


def calc_overall_errors(error_func):
    error = np.nan * np.zeros((2, 4, len(freqs), len(means), len(amps)))
    for iw in range(error.shape[2]):
        w = freqs[iw]
        for im in range(error.shape[3]):
            for ia in range(error.shape[4]):
                g = f['fixedspeed_simpblade_frozen/w%.2f/mean%.1f/amp%.1f' %
                      (w, means[im], amps[ia])]
                t = np.arange(0, g['mbwind'].attrs['t1'],
                              g['mbwind'].attrs['dt'])
                tiph = eval_harmonic(w, (g['mbwind_harmonic/tipdefl_x0'][:],
                                         g['mbwind_harmonic/tipdefl_xb'][:]),
                                     t)
                rotorh = eval_harmonic(w, (g['mbwind_harmonic/rotor0'][:],
                                           g['mbwind_harmonic/rotorb'][:]), t)
                tipt = eval_harmonic(w, (g['mbwind_tangent/tipdefl_x0'][:],
                                         g['mbwind_tangent/tipdefl_xb'][:]), t)
                rotort = eval_harmonic(w, (g['mbwind_tangent/rotor0'][:],
                                           g['mbwind_tangent/rotorb'][:]), t)

                error[0, 0, iw, im, ia] = error_func(
                    g['mbwind/tipdefl_x'][:], tiph[:, 0])
                error[1, 0, iw, im, ia] = error_func(
                    g['mbwind/tipdefl_x'][:], tipt[:, 0])
                error[0, 1, iw, im, ia] = error_func(
                    g['mbwind/tipdefl_y'][:], tiph[:, 1])
                error[1, 1, iw, im, ia] = error_func(
                    g['mbwind/tipdefl_y'][:], tipt[:, 1])
                error[0, 2, iw, im, ia] = error_func(
                    g['mbwind/rotor_thrust'][:], rotorh[:, 0])
                error[1, 2, iw, im, ia] = error_func(
                    g['mbwind/rotor_thrust'][:], rotort[:, 0])
                error[0, 3, iw, im, ia] = error_func(
                    g['mbwind/rotor_torque'][:], rotorh[:, 1])
                error[1, 3, iw, im, ia] = error_func(
                    g['mbwind/rotor_torque'][:], rotort[:, 1])
    return error


def err_norm_rms(y, z):
    return np.sqrt(np.mean((y-z)**2)) / np.std(y)
err_norm_rms.label = 'RMS'


def err_norm_peak(y, z):
    ypp = y.max() - y.min()
    zpp = z.max() - z.min()
    return abs(ypp - zpp) / ypp  # np.std(y)
err_norm_peak.label = 'peak-peak'


def err_norm_mean(y, z):
    return abs(y.mean() - z.mean()) / abs(np.mean(y))
err_norm_mean.label = 'mean value'


CMAP = 'viridis'

def plot_overall_errors_contour(axes, Umean, error_func, levels=None):
    var = ['tipdefl_x', 'tipdefl_y', 'rotor_thrust', 'rotor_torque']
    titles = ['Out-of-plane\ntip deflection', 'In-plane\ntip deflection',
              'Rotor\nthrust', 'Rotor\ntorque']
    error = calc_overall_errors(error_func) * 100

    print("\nAt ", Umean)
    print("Harmonic max error: ")
    for ia, amp in enumerate(amps):
        print(r"$A<$ \SI{{ {} }}{{\metre\per\second}}".format(amp), end='')
        for iv, v in enumerate(var):
            print(r"& \SI{{ {:.1f} }}{{\%}}"
                  .format(error[0, iv, :, Umean, :ia+1].max()), end='')
        print(r"\\")
    print("Tangent max error: ")
    for ia, amp in enumerate(amps):
        print(r"$A<$ \SI{{ {} }}{{\metre\per\second}}".format(amp), end='')
        for iv, v in enumerate(var):
            print(r"& \SI{{ {:.1f} }}{{\%}}"
                  .format(error[1, iv, :, Umean, :ia+1].max()), end='')
        print(r"\\")

    # norm = matplotlib.colors.Normalize(vmin=error[0].min(),
    #                                    vmax=error[0].max()*1.1)
    norm = matplotlib.colors.Normalize(vmin=levels[0], vmax=levels[-1])
    for ivar in range(len(var)):
        ax = axes[:, ivar]
        ax[0].contour(amps, freqs, error[0, ivar, :, Umean, :],
                      levels, norm=norm, cmap=CMAP,
                      extend='max', zorder=-10)
        CS = ax[0].contourf(amps, freqs, error[0, ivar, :, Umean, :],
                            levels, norm=norm, cmap=CMAP, extend='max')
        # CS.cmap.set_over('#333333')
        ax[1].contour(amps, freqs, error[1, ivar, :, Umean, :],
                      levels, norm=norm, cmap=CMAP, extend='max',
                      zorder=-10)
        ax[1].contourf(amps, freqs, error[1, ivar, :, Umean, :],
                       levels, norm=norm, cmap=CMAP, extend='max')
        ax[0].set_title(titles[ivar])
        for a in ax:
            a.set_yscale('log')
            a.set_yticks(freqs[1:])
            a.set_yticklabels(['%.1f' % w for w in freqs[1:]])

    axes[0, 0].set_xticks([1, 3, 5])
    axes[0, 0].annotate('Harmonic', (-0.7, 0.5), (-0.85, 0.5), "axes fraction",
                        ha="right", va="center", fontsize='small',
                        arrowprops=dict(arrowstyle='-[,widthB=4'))
    axes[1, 0].annotate('Tangent', (-0.7, 0.5), (-0.85, 0.5), "axes fraction",
                        ha="right", va="center", fontsize='small',
                        arrowprops=dict(arrowstyle='-[,widthB=4'))
    return CS


means = [8.0, 16.0]
amps = [1., 2., 3., 4., 5.]
freqs = np.power(10, [-1.5, -1.0, -0.5, 0.0, 0.5])[::2]

f = h5py.File(sys.argv[1], 'r')


def plot(Umean, levels):
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(6, 2.5))
    fig.subplots_adjust(left=0.22, right=0.85, bottom=0.12, top=0.86)

    CS = plot_overall_errors_contour(axes, Umean, err_norm_peak, levels)

    plt.colorbar(CS, cax=fig.add_axes([0.9, 0.12, 0.03, 0.75])) \
       .set_label('Normalised peak-peak error [%]')
    fig.text((0.2 + 0.9) / 2, 0.0,
             'Wind speed variation [m/s]', ha='center', va='bottom')
    fig.text(0.17, 0.5, 'Frequency [rad/s]',
             rotation=90, ha='right', va='center')

    try:
        plt.savefig(sys.argv[2].replace('.pdf', '_{}.pdf'.format(Umean)))
    except Exception as e:
        print(e.args[0])
        raise e


plot(0, np.arange(0, 25.1, 1))
plot(1, np.arange(0, 7.1, 0.2))
with open(sys.argv[2], 'wt') as f:
    f.write('Keeping SCons dependencies happy')
