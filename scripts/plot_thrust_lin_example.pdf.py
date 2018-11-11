import sys
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin
from matplotlib.patches import ConnectionPatch


exp = np.exp(1j * np.linspace(0, 2*pi))
E = lambda harmonic: np.real(harmonic[0]) + np.real(harmonic[1] * exp)


def remove_top_right_spines(ax):
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def plot_linearisation(mU, A, w, show_tangent=True, show_harmonic=True):
    mU = min(wind_speeds[-1], max(wind_speeds[0], mU))
    if mU - A <= wind_speeds[0]:
        A = mU - wind_speeds[0]
    if mU + A > wind_speeds[-1]:
        A = wind_speeds[-1] - mU

    t = np.linspace(0, 2*pi/w, 50)
    U = mU + A*sin(w*t)

    f0 = (w/2/pi) * np.trapz(F(U), t)
    fb = (w/pi) * np.trapz(F(U) * np.exp(-1j*w*t), t)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5.5, 2.5))
    fig.subplots_adjust(bottom=0.17, left=0.1, right=0.85, top=0.95)
    ax[0].plot(wind_speeds, F(wind_speeds), 'k')
    #ax[0].plot(U, F(U), 'g.', alpha=0.4)
    ax[0].axvspan(U.min(), U.max(), fc='k', alpha=0.05)
    #ax[0].plot(mU, F(mU), 'bo')
    ax[0].axvline(mU, c='k', lw=0.3, alpha=0.3)
    ax[0].set_xlabel('Wind speed [m/s]')
    ax[0].set_ylabel('Thrust [kN]')
    ax[0].set_ylim(0, 7)

    ax[0].plot(U, 0.5 + 1.5*t*w/2/pi, 'g', alpha=0.8)
    ax[0].text(17.5, 1.25, 'Sinusoidal\nwind speed\ninput',
               color='g', va='center', fontsize='small')

    ax[1].plot(t, F(U), 'k', label='Non-linear function')
    #ax[1].plot(t, F(U), 'g.', alpha=0.4)
    ax[1].set_xlabel('Time [s]')
    ax[1].set_xlim(0, t[-1])

    if show_tangent:
        ax[0].plot(wind_speeds, F(mU) + dFdU(mU) * (wind_speeds - mU), 'b--')
        ax[1].plot(t, F(mU) + dFdU(mU) * (U-mU), 'b--',
                   label='Tangent linearisation')
        con = ConnectionPatch(xyB=(0, F(mU)), xyA=(t[-1], F(mU)),
                              coordsB="data", coordsA="data",
                              axesB=ax[0], axesA=ax[1],
                              color='b', lw=0.3, alpha=0.2)
        ax[1].add_artist(con)

    if show_harmonic:
        # ax[0].axhline(f0, c='r', ls=':')
        # ax[1].axhline(f0, c='r', ls=':')
        #ax[0].plot(mU, f0, 'ro')
        ax[0].plot(U, np.real(fb * np.exp(1j*w*t)) + f0, 'r')
        ax[1].plot(t, np.real(fb * np.exp(1j*w*t)) + f0, 'r',
                   label='Harmonic linearisation')
        con = ConnectionPatch(xyB=(0, f0), xyA=(t[-1], f0),
                              coordsB="data", coordsA="data",
                              axesB=ax[0], axesA=ax[1],
                              color='r', lw=0.3, alpha=0.2)
        ax[1].add_artist(con)

    ax[1].legend(loc='upper right', fontsize='small', frameon=False,
                 bbox_to_anchor=(1.3, 1.1), handlelength=3, labelspacing=0.2)
    #fig.suptitle((r'\SI[separate-uncertainty]{%g+-%g}{\metre\per\second}'
    #              r', at \SI{%g}{\radian\per\second}')
    #             % (mU, A, w))
    return fig, ax


from bem import bem

# Load blade & aerofoil definitions
blade_filename = sys.argv[1]
aerofoil_database_filename = sys.argv[2]
root_length = 1.5

blade_definition = bem.Blade.from_yaml(blade_filename)
aerofoil_database = bem.AerofoilDatabase(aerofoil_database_filename)

model = bem.BEMModel(blade_definition, root_length, 3, aerofoil_database)

rotor_speed = 9.45 * (np.pi/30)  # rad/s
pitch_angle = 0
wind_speeds = np.arange(1, 25, 1)
ia = 30

forces = np.zeros((len(wind_speeds), 2))
for i in range(len(wind_speeds)):
    fac = model.solve(wind_speeds[i], rotor_speed, pitch_angle, annuli=[ia])[0]
    forces[i] = model.forces(wind_speeds[i], rotor_speed, pitch_angle, 1.225,
                             [fac], annuli=[ia])[0]
forces /= 1e3

from scipy.interpolate import interp1d
dFdU = interp1d(wind_speeds,
                np.gradient(forces[:, 0], wind_speeds[1] - wind_speeds[0]))
F = interp1d(wind_speeds, forces[:, 0])

fig, ax = plot_linearisation(12, 5, 1)
for a in ax:
    remove_top_right_spines(a)

try:
    plt.savefig(sys.argv[3])
except Exception as e:
    print(e.args[0])
    raise e
