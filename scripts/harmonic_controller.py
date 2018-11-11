"""Find the harmonic linearisation of the non-linear controller (NREL 5MW).

Copyright (C) Richard Lupton 2018.

"""

import numpy as np
from scipy.optimize import brentq, root
from scipy.integrate import ode
from scipy import interpolate
from controller import TorqueController
from harmonic_linearisation import calc_first_harmonic


class HarmonicTorqueController:
    def __init__(self, timestep, tc_params):
        self.params = tc_params
        self.kq = tc_params['opt constant']
        self.wc = tc_params['speed filter corner freq']
        self.nonlinear_controller = TorqueController(timestep, tc_params)

    def __call__(self, w, harmonic_generator_speed, force_const_power=False):
        """Calculate the harmonic generator torque corresponding to the
        harmonic generator speed"""
        Og0, Ogb = harmonic_generator_speed

        # Filtered harmonic generator speed
        Of0 = Og0
        Ofb = Ogb / (1 + 1j*w/self.wc)

        # Do the harmonic linearisation of the non-linear torque characteristic
        torque = np.vectorize(lambda x: \
            self.nonlinear_controller.get_torque(x, force_const_power))
        Q0, Qb = calc_first_harmonic(torque, Of0, Ofb)

        return (Q0[0], Qb[0])


def calc_harmonic_aero_rotor_loads(aero, rotor, w, harmonic_wind_speed,
                                   harmonic_rotor_speed, harmonic_pitch_angle):
    def torque(x):
        y = np.zeros((6, x.shape[1]))
        for i in range(x.shape[1]):
            aero_forces = aero.forces(x[0, i], x[1, i], x[2, i], rho=1.225)
            # interp = interpolate.interp1d(
            #     aero.bem_model.radii, aero_forces, axis=0,
            #     copy=False, assume_sorted=True)
            # forces_at_FE = interp(rotor.blade_fe.q0[::6])
            y[:, i] = rotor.hub_forces_from_aero_loading(aero_forces)
        return y
    means, amps = zip(*[harmonic_wind_speed, harmonic_rotor_speed,
                        harmonic_pitch_angle])
    f0, fb = calc_first_harmonic(torque, means, amps)
    return f0, fb


class HarmonicSolver:
    def __init__(self, model, aero, gear_ratio, w,
                 harmonic_wind_speed, harmonic_pitch_angle,
                 harmonic_torque_controller):
        self.model = model
        self.aero = aero
        self.gear_ratio = gear_ratio
        self.w = w
        self.harmonic_wind_speed = harmonic_wind_speed
        self.harmonic_pitch_angle = harmonic_pitch_angle
        self.harmonic_controller = harmonic_torque_controller
        self.speed_range = (
            self.harmonic_controller.params['cut in speed']/97,
            self.harmonic_controller.params['rated speed']*1.1/97
        )
        self.harmonic_rotor_speed = (
            (self.speed_range[0] + self.speed_range[1]) / 2, 0)

    def calc_torques(self):
        self.Fr0, self.Frb = calc_harmonic_aero_rotor_loads(
            self.aero, self.model.rotor, self.w,
            self.harmonic_wind_speed, self.harmonic_rotor_speed,
            self.harmonic_pitch_angle)
        self.Qr0, self.Qrb = self.Fr0[3], self.Frb[3]  # hub Mx
        harmonic_generator_speed = tuple([x * self.gear_ratio
                                          for x in self.harmonic_rotor_speed])
        self.Qg0, self.Qgb = self.harmonic_controller(self.w,
                                                      harmonic_generator_speed)

    def get_resultant_torque(self):
        return (self.Qr0 - self.gear_ratio * self.Qg0,
                self.Qrb - self.gear_ratio * self.Qgb)

    def solve(self, ftol=1e-5, disp=False):
        """Step using ND root finder"""

        # Rescaling
        self.calc_torques()
        typical_rotor_speed = np.mean(self.speed_range)
        typical_torque = self.Qr0

        # Error in mean and harmonic part of resultant torque
        def func(x):
            # Rescale rotor speed
            Omega0 = typical_rotor_speed * np.real(x[0])
            Omegab = typical_rotor_speed * complex(x[1], x[2])
            self.harmonic_rotor_speed = (Omega0, Omegab)

            # Calculate resultant torque
            self.calc_torques()
            Q0, Qb = self.get_resultant_torque()
            J = self.model.rotor_inertia
            dQb = Qb - 1j * self.w * J * self.harmonic_rotor_speed[1]
            fval = np.array([Q0, dQb.real, dQb.imag]) / typical_torque

            self._log.append([Omega0, Omegab,
                              self.Qr0, self.Qrb,
                              self.Qg0, self.Qgb])
            return fval

        self._log = []
        Omega0, Omegab = self.harmonic_rotor_speed
        x0 = np.array([Omega0, Omegab.real, Omegab.imag]) / typical_rotor_speed
        sol = root(func, x0, method='broyden1',
                   jac=False, options=dict(disp=disp, fatol=ftol, maxiter=50))
        self._log = np.array(self._log)

        # Undo rescalingn
        sol.x *= typical_rotor_speed
        self.harmonic_rotor_speed = (sol.x[0], complex(sol.x[1], sol.x[2]))
        return sol.success, sol.nit
