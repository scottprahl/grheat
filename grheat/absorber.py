# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Heat Transfer Solutions for Exponential Heating of Semi-Infinite Media
======================================================================

This module provides heat transfer solutions for the problem of uniform
illumination of an absorbing semi-infinite medium. The solutions are
based on the mathematical formulations provided in the 1995 SPIE paper by Prahl.

Three types of illumination are supported:

- **Instantaneous**:
  Represents a single, instantaneous pulse of light on the surface at time ``tp``.

- **Continuous**:
  Represents continuous illumination of the surface starting at ``t=0``.

- **Pulsed**:
  Represents a pulse of light on the surface from ``t=0`` to ``t=t_pulse``.

Each of these illumination types can be analyzed under different boundary conditions at ``z=0``:

- `'infinite'`: No boundary (infinite medium).
- `'adiabatic'`: No heat flow across the boundary.
- `'zero'`: Boundary is fixed at ``T=0``.

Any other boundary condition will trigger a ValueError.

Reference:

Scott A. Prahl "Charts to rapidly estimate temperature following laser irradiation",
`Proc. SPIE 2391, Laser-Tissue Interaction VI, (22 May 1995) <https://doi.org/10.1117/12.209919>`_.

More documentation can be found at `grheat Documentation <https://grheat.readthedocs.io>`_.

"""
import scipy.special
import numpy as np

water_heat_capacity = 4.184 * 1e6          # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


class Absorber:
    """
    Heat transfer solutions for exponential heating of semi-infinite absorbing-only media.

    This class provides solutions for the temperature rise in a semi-infinite medium due to
    uniform illumination over its surface. The illumination is exponentially absorbed within
    the medium, with the volumetric heating being proportional to mu_a * exp(-mu_a * z).

    This is a one-dimensional solution in the depth z.

    The solutions are applicable under three different boundary conditions at z=0:
        - 'infinite': No boundary (infinite medium).
        - 'adiabatic': No heat flow across the boundary.
        - 'zero': Boundary is fixed at T=0.

    Attributes
    ----------
        mu_a (float): Exponential attenuation coefficient [1/meters].
        tp (float): Time of source impulse [seconds].
        diffusivity (float): Thermal diffusivity [m**2/s].
        capacity (float): Volumetric heat capacity [J/degree/m**3].
        boundary (str): Boundary condition at z=0 ('infinite', 'adiabatic', or 'zero').
    """

    def __init__(self,
                 mu_a,
                 tp=0,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initialize an exponential heating object.

        Args:
            mu_a (float): Exponential absorption coefficient, which dictates how the
                illumination attenuates with depth in the medium [1/meters].
            tp (float, optional): Time of source impulse, which is assumed to be
                instantaneous at tp [seconds]. Defaults to 0.
            diffusivity (float, optional): Thermal diffusivity of the medium [m**2/s].
                Defaults to water_thermal_diffusivity.
            capacity (float, optional): Volumetric heat capacity of the medium
                [J/degree/m**3]. Defaults to water_heat_capacity.
            boundary (str, optional): Boundary condition at z=0.
                Can be 'infinite', 'adiabatic', or 'zero'. Defaults to 'infinite'.

        Raises:
            ValueError: If the specified boundary condition is not one of the
                recognized types ('infinite', 'adiabatic', 'zero').
        """
        self.mu_a = mu_a                   # 1/meter
        self.tp = tp                       # seconds
        self.diffusivity = diffusivity     # m**2/s
        self.capacity = capacity           # J/degree/m**3
        self.boundary = boundary.lower()   # infinite, adiabatic, zero
        if self.boundary not in ['infinite', 'adiabatic', 'zero']:
            raise ValueError("boundary must be 'infinite', 'adiabatic', or 'zero'")

    def _instantaneous(self, z, t, tp):
        """
        Calculate temperature rise due to instant surface exposure (scalar t and tp).

        The radiant exposure occurs instantaneously at time tp.  The resulting
        instant volumetric heating is proportional to mu_a * exp(-mu_a * z) [J/m³].

        Args:
            z (float or array): Depth at which the temperature is desired [meters].
            t (float): Time at which the temperature is desired [seconds].
            tp (float): Time at which the source impulse occurs [seconds].

        Returns:
            Temperature increase at depth z and time t due to the radiant exposure [°C].
        """
        zeta = self.mu_a * z

        if t <= tp:
            if np.isscalar(zeta):
                return 0
            else:
                return np.zeros_like(zeta)

        tau = self.mu_a**2 * self.diffusivity * (t - tp)
        factor = (self.mu_a / 2 / self.capacity) * np.exp(tau - zeta)
        T = factor * scipy.special.erfc((2 * tau - zeta) / (2 * np.sqrt(tau)))

        if self.boundary != 'infinite':
            factor = (self.mu_a / 2 / self.capacity) * np.exp(tau + zeta)
            T1 = factor * scipy.special.erfc((2 * tau + zeta) / (2 * np.sqrt(tau)))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def instantaneous(self, z, t):
        """
        Calculate temperature rise due to an instant surface exposure.

        This method computes the temperature increase at a specified depth `z` and time `t`
        following an instantaneous radiant exposure of 1 J/m² on the medium's surface. The
        computation is based on the formulations provided in Prahl's 1995 SPIE paper,
        "Charts to rapidly estimate temperature following laser irradiation".

        The method handles scalar or array-like inputs for `z` and `t`, allowing for
        the calculation of temperature increases at multiple depths and/or times.

        The `self.tp` attribute of the Absorber object specifies the time of the source
        impulse, which is used in the underlying `_instantaneous` method call.

        Args:
            z (float or array-like): Depth(s) at which the temperature is desired [meters].
            t (float or array-like): Time(s) at which the temperature is desired [seconds].

        Returns:
            float or array-like: Temperature increase at the specified depth(s) and time(s) [°C].

        Example:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                z = 0                                 # meters
                mu_a = 0.25 * 1000                    # 1/m
                tp = 0.1                              # seconds
                t = np.linspace(0, 500, 100) / 1000   # seconds

                medium = grheat.Absorber(mua, tp)
                T = medium.continuous(z, t)

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("1 J/m² Instant Radiant Exposure at %.1f seconds" % tp)
                plt.show()
        """
        if np.isscalar(t):
            if np.isscalar(self.tp):
                T = self._instantaneous(z, t, self.tp)
            else:
                T = np.empty_like(self.tp)
                for i, tt in enumerate(self.tp):
                    T[i] = self._instantaneous(z, t, tt)
        else:
            if np.isscalar(self.tp):
                T = np.empty_like(t)
                for i, tt in enumerate(t):
                    T[i] = self._instantaneous(z, tt, self.tp)
            else:
                raise ValueError('One of t or self.tp must be a scalar.')
        return T

    def _continuous(self, z, t):
        """
        Calculate temperature rise due to a continuous 1 W/m² surface exposure.

        The method computes the temperature increase at a specified depth `z` and time `t`
        following a continuous radiant exposure of 1 W/m² on the medium's surface. The
        volumetric heating within the medium decreases as self.mu_a * exp(-self.mu * z) [W/m³],
        The temperatures are calculated as in Prahl's 1995 SPIE paper, "Charts to rapidly
        estimate temperature following laser irradiation".

        The method handles scalar or array-like inputs for `z`, allowing for
        the calculation of temperature increases at multiple depths and/or times.

        Args:
            z (float or array-like): Depth(s) at which the temperature is desired [meters].
            t (float): Time(s) at which the temperature is desired [seconds].

        Returns:
            float or array-like: Temperature increase at the specified depth(s) and time(s) [°C].
        """
        zeta = self.mu_a * z

        if t <= 0:
            if np.isscalar(zeta):
                return 0
            else:
                return np.zeros_like(zeta)

        tau = self.mu_a**2 * self.diffusivity * t
        zz = zeta / np.sqrt(4 * tau)

        T = 2 * np.sqrt(tau / np.pi) * np.exp(-zz**2)
        T += (-1 + zeta) * scipy.special.erfc(-zz)
        T += np.exp(-zz**2) * scipy.special.erfcx(np.sqrt(tau) - zz)

        if self.boundary != 'infinite':
            zeta = -zeta
            zz = zeta / np.sqrt(4 * tau)
            T1 = 2 * np.sqrt(tau / np.pi) * np.exp(-zz**2)
            T1 += (-1 + zeta) * scipy.special.erfc(-zz)
            T1 += np.exp(-zz**2) * scipy.special.erfcx(np.sqrt(tau) - zz)

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        T /= 2 * self.diffusivity * self.capacity * self.mu_a
        return T

    def continuous(self, z, t):
        """
        Calculate temperature rise due to continuous surface irradiance on an absorber.

        The method computes the temperature increase at a specified depth `z` and time `t`
        due to a continuous irradiance of 1 W/m² on the absorber's surface. The
        volumetric heating within the medium decreases exponentially with depth, following
        the expression: mu_a * exp(-mu * z) [W/m³]. The heating starts at t=0 and continues
        up to the specified time `t`.

        The temperatures are calculated as in Prahl's 1995 SPIE paper, "Charts to rapidly
        estimate temperature following laser irradiation".

        The method handles scalar or array-like inputs for `z` and `t`, allowing for
        the calculation of temperature increases at multiple depths and/or times.

        Args:
            z (float or array-like): Depth(s) at which the temperature is desired [meters].
            t (float or array-like): Time(s) at which the temperature is desired [seconds].

        Returns:
            float or array-like: Temperature increase at the specified depth(s) and time(s) [°C].

        Example:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                z = 0                                 # meters
                mu_a = 0.25 * 1000                    # 1/m
                t = np.linspace(0, 500, 100) / 1000   # seconds

                medium = grheat.Absorber(mua)
                T = medium.continuous(z, t)

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("1 W/m² Surface Irradiance")
                plt.show()
        """
        if np.isscalar(t):
            T = self._continuous(z, t)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(z, tt)
        return T

    def _pulsed(self, z, t, t_pulse):
        """
        Calculate temperature rise due to pulsed radiant exposure on absorber surface.

        The method computes the temperature increase at a specified depth `z` and time `t`
        due to a pulsed irradiance of 1 J/m² on the absorber's surface. The irradiance
        starts at t=0 and continues up to time `t_pulse`. The volumetric heating within
        the medium decreases exponentially with depth, following the expression:
        mu_a * exp(-mu * z) [W/m³].

        The temperatures are calculated as in Prahl's 1995 SPIE paper, "Charts to rapidly
        estimate temperature following laser irradiation".  The calculations of temperature
        should work for times before, during, and after the pulse.

        Parameters:
            z (float or array-like): Depth(s) at which the temperature is desired [meters].
            t (float or array-like): Time(s) at which the temperature is desired [seconds].
            t_pulse (float): Duration of the irradiance pulse [seconds].

        Returns:
            float or array-like: Temperature increase at the specified depth(s) and time(s) [°C].

        Notes:
            - If the specified time `t` exceeds the pulse duration `t_pulse`, the method
              subtracts the temperature increase due to continuous irradiance after `t_pulse`
              from the total temperature increase.

        """
        T = self._continuous(z, t)
        if t > t_pulse:
            T -= self._continuous(z, t - t_pulse)
        return T / t_pulse

    def pulsed(self, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m² pulsed radiant exposure on the absorber surface.

        This method computes the temperature increase at specified depth(s) `z` and time(s) `t`
        due to a pulsed irradiance of 1 J/m² on the absorber's surface. The irradiance starts at t=0
        and continues up to time `t_pulse`. The volumetric heating within the medium decreases
        exponentially with depth, following the expression: mu_a * exp(-mu * z) [W/m³].

        The temperatures are calculated based on the formulations in Prahl's 1995 SPIE paper,
        "Charts to rapidly estimate temperature following laser irradiation". The calculations 
        of temperature should work for times before, during, and after the pulse.

        Parameters:
            z (float or array-like): Depth(s) at which the temperature is desired [meters].
            t (float or array-like): Time(s) at which the temperature is desired [seconds].
            t_pulse (float): Duration of the irradiance pulse [seconds].

        Returns:
            float or array-like: Temperature increase at the specified depth(s) and time(s) [°C].

        Example:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                z = 0                                 # meters
                mu_a = 0.25 * 1000                    # 1/m
                t = np.linspace(0, 500, 100) / 1000   # seconds
                t_pulse = 0.100                       # seconds

                medium = grheat.Absorber(mua)
                T = medium.pulsed(z, t, t_pulse)

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("1 J/m² pulse lasting %.0f ms" % t_pulse)
                plt.show()
        """
        if np.isscalar(t):
            T = self._pulsed(z, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(z, tt, t_pulse)
        return T
