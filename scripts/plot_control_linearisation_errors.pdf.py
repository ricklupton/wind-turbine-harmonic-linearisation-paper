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


def calc_overall_errors(error_func):
    error = np.nan * np.zeros((1, 4, len(FREQUENCIES),
                               len(MEAN_WIND_SPEEDS), len(AMPLITUDES)))
    success = np.zeros((len(FREQUENCIES), len(MEAN_WIND_SPEEDS),
                        len(AMPLITUDES)), dtype=bool)
    for iw in range(error.shape[2]):
        w = FREQUENCIES[iw]
        for im in range(error.shape[3]):
            U0 = MEAN_WIND_SPEEDS[im]
            for ia in range(error.shape[4]):
                Ub = AMPLITUDES[ia]
                path = 'NREL5MW_simpblade_model_coarse/w%.2f/mean%.1f/amp%.1f'\
                       % (w, U0, Ub)
                gi = fi[path]
                gh = fh[path]
                dt = np.diff(gi['t'][:2])
                n = len(gi['t']) - int(2 * (2*pi/w) / dt)
                t = gi['t'][n:]

                # Refer generator torque to low-speed side to get rotor torque
                rotor_thrust_h = eval_harmonic(w, gh['aero_thrust'], t)
                rotor_torque_h = eval_harmonic(w, gh['generator_torque'], t)*97
                pitch_angle_h = eval_harmonic(w, gh['pitch_angle'], t)
                rotor_speed_h = eval_harmonic(w, gh['rotor_speed'], t)

                suc = gh.attrs['success']
                # this doesn't look right even if it says it worked
                if U0 == 11 and iw <= 1:
                    suc = False
                success[iw, im, ia] = suc
                # if not suc:
                #     continue

                error[0, 0, iw, im, ia] = error_func(gi['shub'][n:, 3],
                                                     rotor_torque_h)
                error[0, 1, iw, im, ia] = error_func(gi['shub'][n:, 0],
                                                     rotor_thrust_h)
                error[0, 2, iw, im, ia] = error_func(gi['rotor_speed'][n:],
                                                     rotor_speed_h)
                error[0, 3, iw, im, ia] = error_func(gi['pitch_angle'][n:],
                                                     pitch_angle_h)
    return error, success


def err_norm_rms(y, z):
    err = np.sqrt(np.mean((y-z)**2))
    if np.std(y) != 0:
        err /= np.std(y)
    return err
err_norm_rms.label = 'RMS'


def err_normmean_rms(y, z):
    err = np.sqrt(np.mean((y-z)**2))
    if np.mean(y) != 0:
        err /= np.mean(y)
    return err
err_normmean_rms.label = 'meanRMS'


def err_norm_peak(y, z):
    ypp = y.max() - y.min()
    zpp = z.max() - z.min()
    err = abs(ypp - zpp)
    if np.std(y) != 0:
        err /= np.std(y)
    return err
err_norm_peak.label = 'peak-peak'


def err_peak(y, z):
    ypp = y.max() - y.min()
    zpp = z.max() - z.min()
    err = abs(ypp - zpp)
    return err
err_peak.label = 'peak-peak'


def err_norm_mean(y, z):
    err = abs(y.mean() - z.mean())
    ymean = abs(np.mean(y))
    if ymean != 0:
        err /= ymean
    return err
err_norm_mean.label = 'mean value'


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
#    None,
    (6, 1, 2),
    (9, 1, 2),
]


def plot_overall_errors_contour_notnorm(error_func, scales, labels, ranges):
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(6, 5))
    fig.subplots_adjust(left=0.1, right=0.85, top=0.75, bottom=0.1)
    #, hspace=0.05, wspace=0.05)

    titles = ['Rotor torque', 'Rotor thrust', 'Rotor speed', 'Pitch angle']
    error, success = calc_overall_errors(error_func)

    for ivar in range(len(titles)):
        err = error[0, ivar] * scales[ivar] * 100
        # norm = matplotlib.colors.Normalize(vmin=np.nanmin(err),
        #                                    vmax=np.nanmax(err))
        # levels = np.linspace(ranges[ivar][0] * scales[ivar],
        #                      ranges[ivar][1] * scales[ivar], 20)
        levels = np.linspace(0, ranges[ivar], 10)
        norm = matplotlib.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        for iamp in range(len(AMPLITUDES)):
            ax = axes[iamp, ivar]
            ax.contour(MEAN_WIND_SPEEDS, FREQUENCIES, err[:, :, iamp],
                       levels, norm=norm, cmap='coolwarm', zorder=-10)
            CS = ax.contourf(MEAN_WIND_SPEEDS, FREQUENCIES, err[:, :, iamp],
                             levels, norm=norm, cmap='coolwarm', extend='max')
            # for c in CS.collections:
            #     c.set_rasterized(True)
            if ax.is_first_row():
                ax.set_title(titles[ivar])
            ax.set_yscale('log')
            ax.set_yticks(FREQUENCIES[1:])
            ax.set_yticklabels(['%.1f' % w for w in FREQUENCIES[1:]])
            ax.axvline(11.4, c='w', alpha=0.7)
            ax.locator_params(axis='x', nbins=6)
            for iw in range(len(FREQUENCIES)):
                for im in range(len(MEAN_WIND_SPEEDS)):
                    if not success[iw, im, iamp]:
                        ax.plot([MEAN_WIND_SPEEDS[im]], [FREQUENCIES[iw]],
                                'ro', ms=5, mec='w', zorder=10, clip_on=False)

        CS.cmap.set_over('#333333')
        #CS.set_clim(0, ranges[ivar][1])
        pos = axes[0, ivar].get_position()
        cax = fig.add_axes([pos.x0, 0.90, pos.width, 0.03])
        ticks = [0, ranges[ivar]/2, ranges[ivar]]
        cbar = plt.colorbar(CS, cax=cax, orientation='horizontal', ticks=ticks)
        cbar.set_ticklabels(ticklabels=['%d%%' % x for x in ticks])
        #cbar.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%g%%'))
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(labels[ivar], fontsize=8)
        cax.xaxis.set_ticks_position('top')
        # if ivar % 2 == 0:
        #     cax.xaxis.set_ticks_position('top')
        # else:
        #     cax.xaxis.set_label_position('top')

    for iamp in range(len(AMPLITUDES)):
        axes[iamp, -1].annotate('Amplitude\n{} m/s'.format(AMPLITUDES[iamp]),
                                (1.1, 0.5), xycoords="axes fraction",
                                ha="left", va="center")

    # for im, iw, ia in (examples1 + examples2 + examples3):
    #     for ivar in range(len(titles)):
    #         axes[ia, ivar].plot([MEAN_WIND_SPEEDS[im]], [FREQUENCIES[iw]],
    #                             'ks', ms=5, mec='w', zorder=10, clip_on=False)

    fig.text((0.1 + 0.85) / 2, 0.02, 'Mean wind speed [m/s]',
             ha='center', va='bottom')
    fig.text(0.05, (0.1 + 0.75) / 2, 'Wind variation frequency [rad/s]',
             rotation=90, ha='right', va='center')

FREQUENCIES = [10**(-1.5), 10**(-1.0), 10**(-0.5), 10**(0.0), 10**(0.5)]
MEAN_WIND_SPEEDS = np.arange(6, 15.1, 1)  # [5., 7., 9., 13., 15.]
AMPLITUDES = [1, 2, 3]  # , 2.0, 3.0, 4.0, 5.0]

RATED_TORQUE = 4.18e6      # Nm (rotor)
RATED_THRUST = 721e3       # N
RATED_ROTOR_SPEED = 1.267  # rad/s
MAX_PITCH_ANGLE = 23.2 * pi/180  # rad (at cutout)

fi = h5py.File(sys.argv[1], 'r')
fh = h5py.File(sys.argv[2], 'r')


plot_overall_errors_contour_notnorm(
    err_peak,
    scales=[1/RATED_TORQUE, 1/RATED_THRUST,
            1/RATED_ROTOR_SPEED, 1/MAX_PITCH_ANGLE],
    labels=['of rated torque', 'of rated thrust',
            'of rated speed', 'of cut-out pitch'],
    ranges=[30, 40, 40, 30])


try:
    plt.savefig(sys.argv[3])
except Exception as e:
    print(e.args[0])
    raise e
