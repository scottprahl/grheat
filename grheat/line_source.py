# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for z-line source in infinite media.

More documentation at <https://grheat.readthedocs.io>

Typical usage::

    import grheat
    import numpy as np
    import matplotlib.pyplot as plt

    r = np.array([0, 0, 0])
    rp = np.array([0, 0, 1])
    t = np.linspace(0, 2, 100)
    tp = 0

    T =
    plt.plot(t, T, color='blue')
    plt.show()

"""

import scipy.special
import numpy as np

__all__ = ('instantaneous',
           'continuous',
           'pulsed',
           )


water_heat_capacity = 4.184 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


def _instantaneous(x, y, t, xp, yp, tp):
    """
    Calculate temperature rise due to a 1 J/m instantaneous z-line source at time t.

    The line parallel to the z-axis and passes through xp, yp.

    Carslaw and Jaeger page 258, 10.3(1)

    Parameters:
        x, y: location for desired temperature [meters]
        t: time of desired temperature [seconds]
        xp, yp: location of z-line source [meters]
        tp: time of source impulse [seconds]

    Returns:
        normalized temperature
    """
    if t <= tp:
        return 0

    kappa = water_thermal_diffusivity  # m**2/s
    rho_cee = water_heat_capacity      # J/degree/m**3
    r = np.sqrt((x - xp)**2 + (y - yp)**2)
    factor = rho_cee * 4 * np.pi * kappa * (t - tp)
    return 1 / factor * np.exp(-r**2 / (4 * kappa * (t - tp)))


def instantaneous(x, y, t, xp, yp, tp):
    """
    Calculate temperature rise due to a 1 J/m instant z-line source

    Parameters:
        x, y: location for desired temperature [meters]
        t: time of desired temperature [seconds]
        xp, yp: location of z-line source [meters]
        tp: time of source impulse [seconds]

    Returns:
        Temperature Increase [°C]
    """
    if np.isscalar(t):
        T = _instantaneous(x, y, t, xp, yp, tp)

    else:
        T = np.empty_like(t)
        for i, tt in enumerate(t):
            T[i] = _instantaneous(x, y, tt, xp, yp, tp)
    return T


def _continuous(x, y, t, xp, yp):
    """
    Calculate temperature rise due to a 1W/m z-line source at single time point.

    Carslaw and Jaeger page 261, 10.4(5)

    Parameters:
        x, y: location for desired temperature [meters]
        t: time of desired temperature [seconds]
        xp, yp: location of z-line source [meters]

    Returns:
        Temperature Increase [°C]
    """
    if t <= 0:
        return 0

    kappa = water_thermal_diffusivity  # m**2/s
    rho_cee = water_heat_capacity      # J/degree/m**3
    r = np.sqrt((x - xp)**2 + (y - yp)**2)
    factor = -rho_cee * 4 * np.pi * kappa
    return 1 / factor * scipy.special.expi(-r**2 / (4 * kappa * t))


def continuous(x, y, t, xp, yp):
    """
    Calculate temperature rise due to a 1W/m continuous z-line source.

    The z-line source turns on at t=0 and passes through (xp, yp).

    Parameters:
        x, y: location for desired temperature [meters]
        t: time(s) of desired temperature [seconds]
        xp, yp: location of z-line source [meters]

    Returns:
        Temperature Increase [°C]
    """
    if np.isscalar(t):
        T = _continuous(x, y, t, xp, yp)

    else:
        T = np.empty_like(t)
        for i, tt in enumerate(t):
            T[i] = _continuous(x, y, tt, xp, yp)
    return T


def _pulsed(x, y, t, xp, yp, t_pulse):
    """
    Calculate temperature rise due to a 1 J/m pulsed z-line source at time(s) t.

    1 J/m of heat deposited along z-line passing through (xp, yp) from t=0 to t=t_pulse.

    Parameters:
        x, y: location for desired temperature [meters]
        t: time of desired temperature [seconds]
        xp, yp: location of z-line source [meters]
        t_pulse: duration of pulse [seconds]

    Returns:
        Temperature Increase [°C]
    """
    if t <= 0:
        T = 0
    else:
        T = _continuous(x, y, t, xp, yp)
        if t > t_pulse:
            T -= _continuous(x, y, t - t_pulse, xp, yp)
    return T / t_pulse


def pulsed(x, y, t, xp, yp, t_pulse):
    """
    Calculate temperature rise due to a 1 J/m pulsed z-line source.

    Parameters:
        x, y: location for desired temperature [meters]
        t: time(s) of desired temperature [seconds]
        xp, yp: location of z-line source [meters]
        t_pulse: duration of pulse [seconds]

    Returns
        Temperature Increase [°C]
    """
    if np.isscalar(t):
        T = _pulsed(x, y, t, xp, yp, t_pulse)
    else:
        T = np.empty_like(t)
        for i, tt in enumerate(t):
            T[i] = _pulsed(x, y, tt, xp, yp, t_pulse)
    return T
