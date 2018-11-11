"""
Implementation of the NREL 5MW wind turbine controller.

Copyright (C) Richard Lupton 2018.

"""

import numpy as np
import yaml

def saturate(x, a, b):
    return min(max(x, a), b)


class PitchController:
    def __init__(self, timestep, params):
        self.params = params
        self.timestep = timestep
        self.reset()

    def reset(self):
        # Values from the previous timestep
        self.last_time = None
        self.speed_error_int = None
        self.pitch_demand = None
        self.speed_filtered = None

    def get_scheduled_gain(self, pitch):
        GK = 1.0 / (1.0 + pitch / self.params['pitch schedule doubled angle'])
        return GK

    def initialise(self, time, measured_speed, measured_pitch):
        self.last_time = time - self.timestep
        self.speed_filtered = measured_speed
        self.pitch_demand = measured_pitch

        # Initialise integral speed error. This will ensure that the
        # pitch angle is unchanged if the initial speed_error is zero
        GK = self.get_scheduled_gain(measured_pitch)
        self.speed_error_int = (measured_pitch /
                                (GK * self.params['integral gain']))

    def get_pitch_demand(self, speed_error, speed_error_int, GK):
        # Compute the pitch commands associated with the proportional
        # and integral gains:
        demand_p = GK * self.params['proportional gain'] * speed_error
        demand_i = GK * self.params['integral gain'] * speed_error_int

        # Superimpose the individual commands to get the total pitch command;
        # saturate the overall command using the pitch angle limits:
        demand = saturate(demand_p + demand_i,
                          self.params['pitch angle min'],
                          self.params['pitch angle max'])

        return demand

    def step(self, time, measured_speed, measured_pitch):
        # First run?
        if self.last_time is None:
            self.initialise(time, measured_speed, measured_pitch)

        # Check if enough time has elapsed
        elapsed_time = time - self.last_time
        if elapsed_time < self.timestep:
            return

        # Update filtered speed
        alpha = np.exp(-elapsed_time * self.params['speed filter corner freq'])
        self.speed_filtered = ((1 - alpha) * measured_speed +
                               alpha * self.speed_filtered)

        # Compute the current speed error and its integral
        # w.r.t. time; saturate the integral term using the pitch
        # angle limits:
        GK = self.get_scheduled_gain(self.pitch_demand)
        speed_error = self.speed_filtered - self.params['rated speed']
        self.speed_error_int += (speed_error * elapsed_time)
        self.speed_error_int = saturate(
            self.speed_error_int,
            self.params['pitch angle min'] / (GK*self.params['integral gain']),
            self.params['pitch angle max'] / (GK*self.params['integral gain']))

        # Saturate the overall commanded pitch using the pitch rate limit:
        demand = self.get_pitch_demand(speed_error, self.speed_error_int, GK)
        pitch_rate = saturate((demand - measured_pitch) / elapsed_time,
                              -self.params['pitch rate limit'],
                              +self.params['pitch rate limit'])
        self.pitch_demand = measured_pitch + pitch_rate * elapsed_time
        self.last_time = time

    @classmethod
    def from_yaml(cls, filename):
        """Read controller params from 'controller' section of YAML file"""
        with open(filename) as f:
            config = yaml.safe_load(f)
        c = config['controller']
        return cls(c['timestep'], c['pitch controller'])


class TorqueController:
    def __init__(self, timestep, params):
        self.params = params
        self.timestep = timestep

        # Calculate maximum optimum-torque speed to achieve slope
        Qrated = self.params['rated power'] / params['rated speed']
        sync_speed = params['rated speed'] / (1 + params['slip percent']/100)
        slope25 = Qrated / (self.params['rated speed'] - sync_speed)
        kopt = params['opt constant']
        params['opt max speed'] = (
            (slope25 - np.sqrt(slope25*(slope25 - 4*kopt*sync_speed))) /
            (2 * kopt))

        # For Hywind: optionally use constant torque instead of constant power
        self.constant_torque = params.get('constant torque', None)
        assert self.constant_torque is None or self.constant_torque > 0

        # Check values
        assert params['speed filter corner freq'] > 0
        assert timestep > 0
        assert params['slip percent'] > 0
        assert params['opt constant'] > 0
        assert params['torque rate limit'] > 0
        assert (0 < params['cut in speed']
                  < params['opt min speed']
                  < params['opt max speed']
                  < params['rated speed'])
        assert (0 < (kopt * params['rated speed']**2)
                  < Qrated
                  < params['torque max'])

        self.reset()

    def reset(self):
        # Values from the previous timestep
        self.last_time = None
        self.torque_demand = None
        self.speed_filtered = None

    def _optQ(self, speed):
        return self.params['opt constant'] * speed**2

    def get_torque(self, spd, const_power):
        Vin = self.params['cut in speed']
        Vo1 = self.params['opt min speed']
        Vo2 = self.params['opt max speed']
        Vrated = self.params['rated speed']
        Qrated = self.params['rated power'] / Vrated

        if spd >= Vrated or const_power:
            # Region 3 - constant power
            if spd <= 0:
                # Needed for harmonic linearisation
                torque = self.params['torque max']
            elif self.constant_torque is not None:
                torque = self.constant_torque
            else:
                torque = self.params['rated power'] / spd
        elif spd < Vo1:
            # Region 1 to 1.5 - linear ramp from cut-in to optimal region
            torque = np.interp(spd, [Vin, Vo1], [0, self._optQ(Vo1)])
        elif spd < Vo2:
            # Region 2 - optimal control
            torque = self._optQ(spd)
        else:
            # Region 2.5 - linear ramp
            torque = np.interp(spd, [Vo2, Vrated], [self._optQ(Vo2), Qrated])

        # Limit to maximum torque
        torque = saturate(torque, 0, self.params['torque max'])

        return torque

    def initialise(self, time, measured_speed):
        self.last_time = time - self.timestep
        self.speed_filtered = measured_speed

    def step(self, time, measured_speed, force_constant_power):
        # First run?
        if self.last_time is None:
            self.initialise(time, measured_speed)

        # Check if enough time has elapsed
        elapsed_time = time - self.last_time
        if elapsed_time < self.timestep:
            return

        # Update filtered speed
        alpha = np.exp(-elapsed_time * self.params['speed filter corner freq'])
        self.speed_filtered = ((1 - alpha) * measured_speed +
                               alpha * self.speed_filtered)

        # Choose the desired torque & limit
        torque = self.get_torque(self.speed_filtered, force_constant_power)

        # Saturate the commanded torque using the rate limit
        if self.torque_demand is not None:
            rate = saturate((torque - self.torque_demand) / elapsed_time,
                            -self.params['torque rate limit'],
                            +self.params['torque rate limit'])
            torque = self.torque_demand + rate * elapsed_time

        self.torque_demand = torque
        self.last_time = time

    @classmethod
    def from_yaml(cls, filename):
        """Read controller params from 'controller' section of YAML file"""
        with open(filename) as f:
            config = yaml.safe_load(f)
        c = config['controller']
        return cls(c['timestep'], c['torque controller'])

class CombinedController:
    def __init__(self, torque_params, pitch_params, torque_timestep,
                 pitch_timestep=None, const_power_min_pitch=0):
        if pitch_timestep is None:
            pitch_timestep = torque_timestep
        self.const_power_min_pitch = const_power_min_pitch
        self.c_torque = TorqueController(torque_timestep, torque_params)
        self.c_pitch = PitchController(pitch_timestep, pitch_params)

    def step(self, time, measured_speed, measured_pitch):
        self.c_pitch.step(time, measured_speed, measured_pitch)
        force_constant_power = (self.c_pitch.pitch_demand >=
                                self.const_power_min_pitch)
        self.c_torque.step(time, measured_speed, force_constant_power)

    @property
    def torque_demand(self):
        return self.c_torque.torque_demand

    @property
    def pitch_demand(self):
        return self.c_pitch.pitch_demand

    @classmethod
    def from_yaml(cls, filename):
        """Read controller params from 'controller' section of YAML file"""
        with open(filename) as f:
            config = yaml.safe_load(f)
        c = config['controller']
        torque_params = c['torque controller']
        pitch_params = c['pitch controller']
        timestep = c['timestep']
        const_power_min_pitch = c['force const power above pitch']
        return cls(torque_params, pitch_params, timestep, timestep,
                   const_power_min_pitch)
