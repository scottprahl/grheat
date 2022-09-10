# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for xy-planar source in infinite media.

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


def _instantaneous(z, t, zp, tp):
    """
    Calculate temperature rise due to a 1 J/m² instantaneous xy-planar source at time t.

    The plane is parallel to the surface and passes through zp.

    Carslaw and Jaeger page 259, 10.3(4)

    Parameters:
        z: depth of desired temperature [meters]
        t: time of desired temperature [seconds]
        zp: depth of xy-planar source [meters]
        tp: time of source impulse [seconds]

    Returns:
        Temperature increase [°C]
    """
    if t <= tp:
        return 0

    kappa = water_thermal_diffusivity  # m**2/s
    rho_cee = water_heat_capacity      # J/degree/m**3
    r = np.abs(z - zp)
    factor = rho_cee * 2 * np.sqrt(np.pi * kappa * (t - tp))
    return 1 / factor * np.exp(-r**2 / (4 * kappa * (t - tp)))


def instantaneous(z, t, zp, tp):
    """
    Calculate temperature rise due to a 1 J/m² instant xy-planar source

    The plane is parallel to the surface and passes through zp.

    Parameters:
        z: depth for desired temperature [meters]
        t: time of desired temperature [seconds]
        zp: depth of xy-planar source [meters]
        tp: time of source impulse [seconds]

    Returns:
        Temperature increase [°C]
    """
    if np.isscalar(t):
        T = _instantaneous(z, t, zp, tp)

    else:
        T = np.empty_like(t)
        for i, tt in enumerate(t):
            T[i] = _instantaneous(z, tt, zp, tp)
    return T


def _continuous(z, t, zp):
    """
    Calculate temperature rise due to a 1 W/m² xy-planar source.

    The xy-planar source is located at a depth zp and is turned on at t=0.
    It remains on until t=t.
    
    Equation obtained by integrating planar Green's function from 0 to t in
    Mathematica.

    Parameters:
        z: depth for desired temperature [meters]
        t: time of desired temperature [seconds]
        zp: depth of xy-planar source [meters]

    Returns:
        Temperature increase [°C]
    """
    if t <= 0:
        return 0

    kappa = water_thermal_diffusivity  # m**2/s
    rho_cee = water_heat_capacity      # J/degree/m**3
    alpha = np.sqrt((z - zp)**2 / (4 * kappa * t))
    T = np.exp(-alpha**2) / np.sqrt(np.pi) - alpha * scipy.special.erfc(alpha)
    return np.sqrt(t / kappa) / rho_cee * T


def continuous(z, t, zp):
    """
    Calculate temperature rise due to a 1W/m² continuous xy-planar source.

    The xy-planar source turns on at t=0 and passes through zp.

    Parameters:
        z: depth for desired temperature [meters]
        t: time(s) of desired temperature [seconds]
        zp: depth of xy-planar source [meters]

    Returns:
        Temperature Increase [°C]
    """
    if np.isscalar(t):
        T = _continuous(z, t, zp)

    else:
        T = np.empty_like(t)
        for i, tt in enumerate(t):
            T[i] = _continuous(z, tt, zp)
    return T


def _pulsed(z, t, zp, t_pulse):
    """
    Calculate temperature rise due to a 1 J/m² pulsed xy-planar source at time t.

    1 J/m² of heat deposited in an xy-plane passing through (zp) from t=0 to t=t_pulse.

    Parameters:
        z: depth for desired temperature [meters]
        t: time of desired temperature [seconds]
        zp: depth of xy-planar source [meters]
        t_pulse: duration of pulse [seconds]

    Returns:
        Temperature Increase [°C]
    """
    if t <= 0:
        T = 0
    else:
        T = _continuous(z, t, zp)
        if t > t_pulse:
            T -= _continuous(z, t - t_pulse, zp)
    return T / t_pulse


def pulsed(z, t, zp, t_pulse):
    """
    Calculate temperature rise due to a 1 J/m² pulsed xy-planar source.

    Parameters:
        z: depth for desired temperature [meters]
        t: time(s) of desired temperature [seconds]
        zp: depth of xy-planar source [meters]
        t_pulse: duration of pulse [seconds]

    Returns
        Temperature Increase [°C]
    """
    if np.isscalar(t):
        T = _pulsed(z, t, zp, t_pulse)
    else:
        T = np.empty_like(t)
        for i, tt in enumerate(t):
            T[i] = _pulsed(z, tt, zp, t_pulse)
    return T
