import sys
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from numpy import pi


from harmonic_controller import HarmonicTorqueController
from controller import TorqueController, CombinedController


# Inputs
freqs = [0.32, 1.0, 3.16]
speeds = [
    (80, 15),
    (105, 6),
    (125.5, 10),
]
model_file = sys.argv[1]

controller = TorqueController.from_yaml(model_file)
hcont = HarmonicTorqueController(controller.timestep, controller.params)


def controller_characteristic():
    p = controller.params
    kq = hcont.kq

    spds = np.arange(40, p['rated speed'] * 1.2, 0.2)
    Q = kq * spds**2

    # Zero below minimum
    Q[spds < p['cut in speed']] = 0

    # Linear ramp from minimum
    slin1 = np.logical_and(p['cut in speed'] <= spds,
                           spds < p['opt min speed'])
    lin1 = (kq*p['opt min speed']**2 *
            (spds - p['cut in speed']) /
            (p['opt min speed'] - p['cut in speed']))
    Q[slin1] = lin1[slin1]

    # Linear ramp to rated
    slin2 = p['opt max speed'] <= spds
    lin2 = kq*p['opt max speed']**2 + (
        (p['rated power']/p['rated speed'] - kq*p['opt max speed']**2) *
        (spds - p['opt max speed']) /
        (p['rated speed'] - p['opt max speed']))
    Q[slin2] = lin2[slin2]

    # Constant power above rated
    above_rated = spds > p['rated speed']
    Q[above_rated] = p['rated power'] / spds[above_rated]

    return spds, Q


def plot_torque_background(ax):
    spds, Q = controller_characteristic()
    ax.plot(spds, hcont.kq*spds**2 / 1e3, 'b', alpha=0.1, lw=0.5)
    ax.plot(spds, controller.params['rated power']/spds / 1e3,
            'b', alpha=0.1, lw=0.5)
    ax.plot(spds, Q / 1e3, 'b', alpha=0.4)
    ax.set_xlim(50, 150)
    ax.set_ylim(-5, controller.params['torque max'] * 1.1 / 1e3)


def plot_torque(ax, w, Og0, Ogb, force_const_power=False, show_inset=True):
    controller.reset()
    t = np.arange(0, 2*pi/w, controller.timestep * 10)
    gen_speed = Og0 + np.real(Ogb * np.exp(1j * w * t))

    # Harmonic result
    Q0, Qb = hcont(w, (Og0, Ogb), force_const_power)
    harmonic = (Q0 + np.real(Qb * np.exp(1j * w * t)))
    polygon = Polygon(np.c_[gen_speed, harmonic/1e3],
                      facecolor='red', edgecolor='red', alpha=0.4)
    ax.add_artist(polygon)

    # Stepped result
    result = np.zeros_like(t)
    for n in range(5):
        for i in range(len(t)):
            controller.step((2*pi/w * n) + t[i],
                            gen_speed[i], force_const_power)
            result[i] = controller.torque_demand
    ax.plot(gen_speed, result / 1e3, 'k', lw=1.2)


# Plot
fig, ax = plt.subplots(len(freqs), figsize=(5, 7), sharex=True, sharey=True)
fig.subplots_adjust(right=0.95, top=0.98, bottom=0.06, left=0.1)

for i in range(len(freqs)):
    plot_torque_background(ax[i])
    for Og0, Ogb in speeds:
        plot_torque(ax[i], freqs[i], Og0, Ogb)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].xaxis.set_ticks_position('bottom')
    ax[i].yaxis.set_ticks_position('left')
    ax[i].annotate(r'$\omega = {:.2f}$ rad/s'.format(freqs[i]),
                   (0.95, 0.1), xycoords='axes fraction', ha='right',
                   bbox=dict(boxstyle="round", fc="w"))
ax[0].annotate(r'$80\pm 15$ rpm', (85, 6))
ax[0].annotate(r'$105\pm 6$ rpm', (105, 20))
ax[0].annotate(r'$125\pm 10$ rpm', (125, 30))
ax[0].annotate('Nonlinear', (119, 41), (100, 41), ha='right', va='center',
               arrowprops=dict(arrowstyle='->'))
ax[0].annotate('Harmonic', (115.5, 35), (100, 35), ha='right',
               va='center', color='red',
               arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
ax[1].set_ylabel('Generator torque [kNm]')
ax[2].set_xlabel('Generator speed [rpm]')

try:
    plt.savefig(sys.argv[2])
except Exception as e:
    print(e.args[0])
    raise e
