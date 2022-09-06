# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for point source in infinite media.

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


def _instantaneous(x, y, z, t, xp, yp, zp, tp):
    """
    Calculate temperature for point source for scalar t and tp.

    1J of heat deposited at (xp, yp, zp) at time t=tp.

    See Carslaw and Jaeger page 256 - 259

    Get real temperature by multiplying by

    Parameters:
        x, y, z: location for desired temperature [meters]
        t: time of desired temperature [seconds]
        xp, yp, zp: location of point source [meters]
        tp: time of source impulse time [seconds]

    Returns:
        normalized temperature

    """
    if t <= tp:
        return 0

    kappa = water_thermal_diffusivity  # m**2/s
    rho_cee = water_heat_capacity      # J/degree/m**3
    rr = np.sqrt((x - xp)**2 + (y - yp)**2 + (z - zp)**2)
    tt = kappa * (t - tp)
    factor = 1 / rho_cee / (8 * np.pi * tt)**1.5
    return factor * np.exp(-rr**2 / (4 * tt))


def instantaneous(x, y, z, t, xp, yp, zp, tp):
    """
    Return normalized temperature for point source.

    1J of heat deposited at (xp, yp, zp) at time t=tp.

    See Carslaw and Jaeger page 256 - 259

    Get real temperature by multiplying by

    Parameters:
        x, y, z: location for desired temperature
        t: time(s) of desired temperature
        xp, yp, zp: location of point source
        tp: time of source impulse time

    Returns:
        normalized temperature
    """
    if np.isscalar(t):
        return _instantaneous(x, y, z, t, xp, yp, zp, tp)

    T = np.empty_like(t)
    for i, tt in enumerate(t):
        T[i] = _instantaneous(x, y, z, tt, xp, yp, zp, tp)

    return T


def _continuous(x, y, z, t, xp, yp, zp):
    """
    Return normalized temperature for continuous point source.

    1W of heat deposited at (xp, yp, zp) continuously starting at t=0.

    See Carslaw and Jaeger page 261.

    Point source starts at t=0.

    Parameters:
        x, y, z: location for desired temperature
        t: time(s) of desired temperature
        xp, yp, zp: location of point source

    Returns:
        normalized temperature
    """
    if t <= 0:
        return 0

    kappa = water_thermal_diffusivity  # J/degree / gm
    rho_cee = water_heat_capacity      # J/degree/m**3
    rr = np.sqrt((x - xp)**2 + (y - yp)**2 + (z - zp)**2)
    tt = kappa * t
    factor = 1 / rho_cee / (4 * np.pi * kappa * rr)

    return factor * scipy.special.erfc(rr / np.sqrt(4 * kappa * tt))


def continuous(x, y, z, t, xp, yp, zp):
    """
    Return normalized temperature for continuous point source.

    See Carslaw and Jaeger page 256 - 259

    1W of heat deposited at (xp, yp, zp) continuously starting at t=0.

    Get real temperature by multiplying by

    Parameters:
        x, y, z: location for desired temperature
        t: time(s) of desired temperature
        xp, yp, zp: location of point source

    Returns
        Normalized Temperature
    """
    if np.isscalar(t):
        return _continuous(x, y, z, t, xp, yp, zp)

    T = np.empty_like(t)
    for i, tt in enumerate(t):
        T[i] = _continuous(x, y, z, tt, xp, yp, zp)

    return T


def _pulsed(x, y, z, t, xp, yp, zp, t_pulse):
    """
    Calc temperature for pulsed point source.

    1J of heat deposited at (xp, yp, zp) from t=0 to t=t_pulse.

    Parameters:
        x, y, z: location for desired temperature
        t: time(s) of desired temperature
        xp, yp, zp: location of point source
        t_pulse: duration of pulse

    Returns:
        normalized temperature
    """
    if t <= 0:
        return 0

    T = _continuous(x, y, z, t, xp, yp, zp)

    if t < t_pulse:
        return T

    return T - _continuous(x, y, z, t - t_pulse, xp, yp, zp)


def pulsed(x, y, z, t, xp, yp, zp, t_pulse):
    """
    Calc temperature for 1J pulsed point source.

    1J of heat is deposited at (xp, yp, zp) from t=0 to t=t_pulse.

    See Carslaw and Jaeger page 256 - 259

    Get real temperature by multiplying by

    Parameters:
        x, y, z: location for desired temperature
        t: time(s) of desired temperature
        xp, yp, zp: location of point source
        t_pulse: duration of pulse

    Returns
        Normalized Temperature
    """
    if np.isscalar(t):
        return _pulsed(x, y, z, t, xp, yp, zp, t_pulse)

    T = np.empty_like(t)
    for i, tt in enumerate(t):
        T[i] = _pulsed(x, y, z, tt, xp, yp, zp, t_pulse)

    return T
