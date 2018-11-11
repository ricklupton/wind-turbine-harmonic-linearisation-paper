import sys
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi


exp = np.exp(1j * np.linspace(0, 2*pi))
E = lambda harmonic: np.real(harmonic[0]) + np.real(harmonic[1] * exp)


def plot_harmonic_and_integrated_solutions(ax, w):
    with np.load('data/wake/nonlinear_induction_solution_16pm4_%.2f.npz' % w) as f:
        t, U, x = f['t'], f['U'], f['x']
    with np.load('data/wake/harmonic_induction_solution_16pm4_%.2f.npz' % w) as f:
        harmonic_wind_speed = f['harmonic_wind_speed']
        harmonic_induction = f['harmonic_induction']
    title = r'\SI{%g}{\radian\per\second}' % w
    # r'$16 \pm 4$ m/s at %g rad/s' % w
    levels = np.r_[np.arange(-0.8, 0, 0.2), np.arange(0, 0.9, 0.2)]
    CS = ax.contour(wind_speeds, axial_induction, derivs.T, levels,
                    cmap='coolwarm')
    fmt = lambda x: (r'$\dot{u} = 0$' if x == 0 else r'$%g' % x)
    ax.clabel(CS, levels[[3, 4, 5]], colors='k', fmt=fmt,
              manual=[(23, 1.4), (22, 1.2), (19, 1.1)], inline_spacing=0)
    ax.set_title(title)
    ax.locator_params(nbins=4)
    ax.plot(E(harmonic_wind_speed), E(harmonic_induction), '.-g',
            ms=3, lw=0.5)
    ax.plot(U[-51:], x[-51:], 'k')


with np.load(sys.argv[1]) as f:
    wind_speeds = f['wind_speeds']
    axial_induction = f['axial_induction']
    derivs = f['derivs']

fig, ax = plt.subplots(1, 3, figsize=(6, 3), sharex=True, sharey=True)
fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95, wspace=0.1)
for a, w in zip(ax, [0.01, 1, 4]):
    plot_harmonic_and_integrated_solutions(a, w)

ax[1].set_xlabel('Wind speed $U_\infty$ [m/s]')
ax[0].set_ylabel('Axial induction $u$ [m/s]')

ax[0].annotate("Nonlinear", (12.2, 2.2), (15, 2.2), va='center',
               bbox=dict(fc='white', ec='none'),
               arrowprops=dict(arrowstyle='->'))
ax[0].annotate("Harmonic", (16, 1.8), (18.5, 1.8), color='g', va='center',
               bbox=dict(fc='white', ec='none'),
               arrowprops=dict(arrowstyle='->', color='g'))

ax[0].set_xlim(10, 25)
ax[0].set_ylim(1.0, 2.4)

try:
    plt.savefig(sys.argv[2])
except Exception as e:
    print(e.args[0])
    raise e
