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

__all__ = ('theta_exp',
           'theta_exp_pulsed',
           )


water_heat_capacity = 4.184 * 1000          # J/degree / kg
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s

def _instantaneous(z, t, tp, mua):
    """
    Calculate temperature rise due to a 1 J/m² instantaneous irradiance.

    Prahl, "Charts to rapidly estimate temperature following laser irradiation" 1995.

    \frac{1}{2} e^{\text{mua} (\kappa  \text{mua} t-z)} \text{erfc}\left(\frac{2 \kappa  \text{mua} t-z}{2 \sqrt{\kappa  t}}\right)

    Parameters:
        z: depth for desired temperature [meters]
        t: time of desired temperature [seconds]
        tp: time of source impulse [seconds]
        mua: absorption coefficient [1/m]

    Returns:
        Temperature Increase [°C]
    """
    if t <= tp:
        return 0

    kappa = water_thermal_diffusivity  # m**2/s
    rho_cee = water_heat_capacity      # J/degree/m**3
    tau = mua**2 * kappa * (t-tp)
    zeta = mua * z
    factor = 1 / 2 / rho_cee * np.exp(-zeta + tau)
    T = factor * scipy.special.erfc((-zeta + 2 * tau)/(2 * np.sqrt(tau)))
    return T


def instantaneous(z, t, tp, mua):
    """
    Calculate temperature rise due to a 1 J/m² radiant exposure on absorber.
    
    Parameters:
        z: depth for desired temperature [meters]
        t: time of desired temperature [seconds]
        tp: time of source impulse [seconds]

    Returns:
        Temperature Increase [°C]
    """
    if np.isscalar(t):
        T = _instantaneous(z, t, tp, mua)

    else:
        T = np.empty_like(t)
        for i, tt in enumerate(t):
            T[i] = _instantaneous(z, tt, tp, mua)
    return T


def _theta_exp_dimensionless(xi, tau):
    """
    Handy dimensionless temperature*tau for illumination of absorber.
    
    Parameters:
        xi: dimensionless depth (mu_a*x)
        tau: dimensionless time (mu_a**2 * kappa * t)
    
    Returns: 
        dimensionless temperature*tau
    """
    if tau <= 0:
        return 0

    xx = xi/2/np.sqrt(tau)
    if abs(xx) > 20:
        expxx = 0
    else:
        expxx = np.exp(-xx**2)

    print(np.sqrt(tau)-xx, np.sqrt(tau)+xx)
    T = 2*np.sqrt(tau/np.pi)*np.exp(-xx**2) 
    T -= xi * scipy.special.erfc(xx)
    T -= np.exp(-xi) 
    T += 1/2*expxx*scipy.special.erfcx(np.sqrt(tau)-xx)
    T += 1/2*expxx*scipy.special.erfcx(np.sqrt(tau)+xx)

#    T += 1/2*np.exp(tau-xi)*scipy.special.erfc(np.sqrt(tau)-xx)
#    T += 1/2*np.exp(tau+xi)*scipy.special.erfc(np.sqrt(tau)+xx)
    return T


def _theta_exp_pulsed_dimensionless(xi, tau, tau_pulse):
    """
    Find Temperature of pulsed illumination of absorber.
    
    Parameters:
        xi: dimensionless depth (mu_a*x)
        tau: dimensionless time (mu_a**2 * kappa * t)
    
    Returns: 
        dimensionless temperature
    """
    if tau <= 0:
        return 0
    
    T = theta_exp_dimensionless(xi, tau)
    
    if tau <= tau_pulse:
        return T/tau
    
    T = T - theta_exp_dimensionless(xi, tau-tau_pulse)

    return T/tau_pulse


def theta_exp(x, t, mu_a):
    """
    Find dimensionless temperature of illuminated absorber.
    
    Parameters:
        x:  depth [m]
        t:  time of illumination [s]
        mu_a: absorption coefficient [1/m]
    
    Returns: temperature
    """
    kappa = water_thermal_diffusivity
    xi = mu_a * x
    tau = t * kappa * mu_a**2
    
    if np.isscalar(tau):
        return _theta_exp_dimensionless(xi, tau)/tau
    
    T = np.empty_like(tau)
    for i, tt in enumerate(tau):
        T[i] = _theta_exp_dimensionless(xi, tt)/tau
    return T

    
def _exp_pulsed(x, t, t_pulse, mu_a):
    """
    Calculate temperature increase due to 1 J/m**2 pulsed irradiance on absorber.
    
    Parameters:
        x:  depth [m]
        t:  time of illumination [s]
        t_pulse: duration of pulse [s]
        mu_a: absorption coefficient [1/m]
        kappa: thermal diffusivity [m**2/s]
    
    Returns: dimensionless temperature
    """
    kappa = water_thermal_diffusivity
    xi = mu_a * x
    tau = t * kappa * mu_a**2
    tau_pulse = t_pulse * kappa * mu_a**2

    if np.scalar(tau):
        return theta_exp_pulsed_dimensionless(xi, tau, tau_pulse)

    T = np.empty_like(tau)
    for i, tt in enumerate(tau):
        T[i] = _theta_exp_dimensionless(xi, tt)/tau_pulse
    return T


def exp_pulsed(x, t, t_pulse, mu_a):
    """
    Calculate temperature increase due to 1 J/m**2 pulsed irradiance on absorber.
    
    Parameters:
        x: depth of temperature [m]
        t: time(s) of temperature [s]
        t_pulse: duration of pulse [s]
        mu_a: absorption coefficient [1/m]
    
    Returns: 
        Temperature increase [°C]
    """
    kappa = water_thermal_diffusivity
    rho_c = water_thermal_capacity
    return 1/(kappa*rho_c*mu_a)*theta_exp_pulsed(x, t, t_pulse, mu_a)


