# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Heat transfer solutions for uniform illumination of an absorbing semi-infinite absorber.

Three types of illumination are supported:

- **Instantaneous**:
  Represents an instantaneous pulse of light on the surface at time(s) ``tp``.

- **Continuous**:
  Represents continuous illumination of the surface starting at time(s) ``tp``.

- **Pulsed**:
  Represents a pulses of light at times ``t=tp`` to ``t=tp+t_pulse``.

Each of these illumination types can be analyzed under different boundary conditions at ``z=0``:

- `'infinite'`: No boundary (infinite medium).
- `'adiabatic'`: No heat flow across the boundary.
- `'zero'`: Boundary is fixed at ``T=0``.

Reference:

Scott A. Prahl "Charts to rapidly estimate temperature following laser irradiation",
`Proc. SPIE 2391, Laser-Tissue Interaction VI, (22 May 1995) <https://doi.org/10.1117/12.209919>`_.

More documentation can be found at <https://grheat.readthedocs.io>
"""

import scipy.special
import numpy as np

water_heat_capacity = 4.186 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.145 * 1e-6    # m**2/s


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

    Attributes:
    ----------
        mu_a (scalar): Exponential attenuation coefficient [1/meters].
        tp (scalar): Time of source impulse [seconds].
        diffusivity (scalar): Thermal diffusivity [m**2/s].
        capacity (scalar): Volumetric heat capacity [J/degree/m**3].
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
            mu_a (scalar): Exponential absorption coefficient, which dictates how the
                illumination attenuates with depth in the medium [1/meters].
            tp (scalar, optional): Time of source impulse, which is assumed to be
                instantaneous at tp [seconds]. Defaults to 0.
            diffusivity (scalar, optional): Thermal diffusivity of the medium [m**2/s].
                Defaults to water_thermal_diffusivity.
            capacity (scalar, optional): Volumetric heat capacity of the medium (𝜌c)
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

    def __str__(self):
        """Create string for object."""
        return (f"Absorber Properties:\n"
                f"mu_a: {self.mu_a} 1/meters\n"
                f"tp: {self.tp} seconds\n"
                f"diffusivity: {self.diffusivity} m^2/s\n"
                f"capacity: {self.capacity} J/degree/m^3\n"
                f"boundary: {self.boundary}\n")

    def _instantaneous_scalar_no_bndry(self, z, t, tp):
        """
        Calculate temperature rise due to instant surface exposure (scalar z, t, and tp).

        The radiant exposure occurs instantaneously at time tp.  The resulting
        instant volumetric heating is proportional to mu_a * exp(-mu_a * z) [J/m³].

        Args:
            z (scalar): Depth at which the temperature is desired [meters].
            t (scalar): Time at which the temperature is desired [seconds].
            tp (scalar): Time at which the source impulse occurs [seconds].

        Returns:
            Temperature increase at depth z and time t due to the radiant exposure [°C].
        """
        zeta = self.mu_a * z
        tau = self.mu_a**2 * self.diffusivity * (t - tp)
        scale = self.mu_a / self.capacity

        # before pulse, no temperature rise
        if tau < 0:
            return 0

        # at pulse, exponential heating for positive z
        if tau == 0:
            if z < 0:
                return 0
            return scale * np.exp(-zeta)

        sqrt_tau = np.sqrt(tau)
        zz = zeta / 2 / sqrt_tau
        arg = sqrt_tau - zz

        # stable calculations require erfcx() for positive arg
        if arg >= 0:
            T = 0.5 * scale * np.exp(-zz**2) * scipy.special.erfcx(arg)
        else:
            T = 0.5 * scale * np.exp(tau - zeta) * scipy.special.erfc(arg)
        return T

    def _instantaneous_scalar(self, z, t, tp):
        """
        Calculate temperature rise due to instant surface exposure (scalar z, t, and tp).

        The radiant exposure occurs instantaneously at time tp.  The resulting
        instant volumetric heating is proportional to mu_a * exp(-mu_a * z) [J/m³].

        Args:
            z (scalar): Depth at which the temperature is desired [meters].
            t (scalar): Time at which the temperature is desired [seconds].
            tp (scalar): Time at which the source impulse occurs [seconds].

        Returns:
            Temperature increase at depth z and time t due to the radiant exposure [°C].
        """
        T = self._instantaneous_scalar_no_bndry(z, t, tp)

        if self.boundary != 'infinite':
            T1 = self._instantaneous_scalar_no_bndry(-z, t, tp)

            if t > 0:   # only use method of images after pulse
                if self.boundary == 'adiabatic':
                    T += T1

                if self.boundary == 'zero':
                    T -= T1

        return T

    def _instantaneous(self, z, t, tp):
        """
        Calculate temperature rise due to instant surface exposure (scalar t and tp).

        The radiant exposure occurs instantaneously at time tp.  The resulting
        instant volumetric heating is proportional to mu_a * exp(-mu_a * z) [J/m³].

        Args:
            z (scalar or array): Depth at which the temperature is desired [meters].
            t (scalar): Time at which the temperature is desired [seconds].
            tp (scalar): Time at which the source impulse occurs [seconds].

        Returns:
            Temperature increase at depth z and time t due to the radiant exposure [°C].
        """
        if np.isscalar(z):
            T = self._instantaneous_scalar(z, t, tp)
        else:
            T = np.zeros_like(z)
            for i, zz in enumerate(z):
                T[i] = self._instantaneous_scalar(zz, t, tp)
        return T

    def instantaneous(self, z, t):
        """
        Calculate temperature rise due to instantaneous pulse on semi-infinite absorber.

        This method computes the temperature increase at a specified depth `z` and time `t`
        following an instantaneous radiant exposure of 1 J/m² on the medium's surface.
        Multiply the result of this function by the radiant exposure in J/m² to get the
        temperature increase.

        The method handles scalar or array-like inputs for `z` and `t`, allowing for
        the calculation of temperature increases at multiple depths and/or times.

        The `self.tp` attribute of the Absorber object specifies the time of the source
        impulse, which is used in the underlying `_instantaneous` method call.

        Args:
            z (scalar or array): Depth(s) at which the temperature is desired [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].

        Returns:
            scalar or array: Temperature increase at the specified depth(s) and time(s) [°C].

        Example:
            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                z = 0                                 # meters
                mu_a = 0.25 * 1000                    # 1/m
                tp = 0.1                              # seconds
                radiant_exposure = 1 / (0.01 * 0.01)  # 1 J/cm² = 10⁴ J/m²
                t = np.linspace(0, 500, 100) / 1000   # seconds

                medium = grheat.Absorber(mua, tp)
                T = medium.instantaneous(z, t) * radiant_exposure

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("1 J/cm² Instant Radiant Exposure at %.1f seconds" % tp)
                plt.show()
        """
        if np.isscalar(t):
            T = 0                # return a scalar
            if np.isscalar(self.tp):
                T += self._instantaneous(z, t, self.tp)
            else:
                for i, tp in enumerate(self.tp):
                    T += self._instantaneous(z, t, tp)

        else:
            T = np.zeros_like(t)  # return an array
            for i, tt in enumerate(t):
                if np.isscalar(self.tp):
                    T[i] += self._instantaneous(z, tt, self.tp)
                else:
                    for tp in self.tp:
                        T[i] += self._instantaneous(z, tt, tp)
        return T

    def _continuous_scalar_zero(self, z, t):
        """Calculate the temperature assuming z=0 has T=0."""
        tau = self.mu_a**2 * self.diffusivity * t
        zeta = self.mu_a * z
        scale = 1 / (2 * self.diffusivity * self.capacity * self.mu_a)

        if t <= 0:
            return zeta * 0

        sqrt_tau = np.sqrt(tau)
        zz = zeta / 2 / sqrt_tau

        T = 2 * scipy.special.erfc(zz)
        T -= 2 * np.exp(-zeta)

        # stable calculations require erfcx() for positive arg
        arg = sqrt_tau - zz
        if arg >= 0:
            T += np.exp(-zz**2) * scipy.special.erfcx(arg)
        else:
            T += np.exp(tau - zeta) * scipy.special.erfc(arg)

        arg = sqrt_tau + zz
        if arg >= 0:
            T -= np.exp(-zz**2) * scipy.special.erfcx(arg)
        else:
            T -= np.exp(tau + zeta) * scipy.special.erfc(arg)

        T *= scale
        return T

    def _continuous_scalar_adiabatic(self, z, t):
        """Calculate the temperature assuming z=0 has dT/dz=0."""
        tau = self.mu_a**2 * self.diffusivity * t
        zeta = self.mu_a * z
        scale = 1 / (2 * self.diffusivity * self.capacity * self.mu_a)

        if t <= 0:
            return zeta * 0

        sqrt_tau = np.sqrt(tau)
        zz = zeta / 2 / sqrt_tau

        T = 4 * np.sqrt(tau / np.pi) * np.exp(-zz**2)
        T -= 2 * zeta * scipy.special.erfc(zz)
        T -= 2 * np.exp(-zeta)

        # stable calculations require erfcx() for positive arg
        arg = sqrt_tau - zz
        if arg >= 0:
            T += np.exp(-zz**2) * scipy.special.erfcx(arg)
        else:
            T += np.exp(tau - zeta) * scipy.special.erfc(arg)

        arg = sqrt_tau + zz
        if arg >= 0:
            T += np.exp(-zz**2) * scipy.special.erfcx(arg)
        else:
            T += np.exp(tau + zeta) * scipy.special.erfc(arg)

        T *= scale
        return T

    def _continuous_scalar_infinite(self, z, t):
        """Calculate the temperature assuming no boundary at z=0."""
        tau = self.mu_a**2 * self.diffusivity * t
        zeta = self.mu_a * z
        scale = 1 / (2 * self.diffusivity * self.capacity * self.mu_a)

        if t <= 0:
            return zeta * 0

        sqrt_tau = np.sqrt(tau)
        zz = zeta / 2 / sqrt_tau

        T = 2 * np.sqrt(tau / np.pi) * np.exp(-zz**2)

        arg = sqrt_tau - zz
        if arg >= 0:
            T += np.exp(-zz**2) * scipy.special.erfcx(arg)
        else:
            T += np.exp(tau - zeta) * scipy.special.erfc(arg)

        if z < 0:
            T += (zeta - 1) * scipy.special.erfc(-zz)
        else:
            T -= 2 * np.exp(-zeta)
            T -= (zeta - 1) * scipy.special.erfc(zz)

        T *= scale
        return T

    def _continuous_scalar(self, z, t):
        """Calculate temperature rise due to a continuous 1 W/m² surface exposure."""
        if self.boundary == 'adiabatic':
            return self._continuous_scalar_adiabatic(z, t)

        if self.boundary == 'zero':
            return self._continuous_scalar_zero(z, t)

        return self._continuous_scalar_infinite(z, t)

    def _continuous(self, z, t):
        """
        Calculate temperature rise due to a continuous 1 W/m² surface exposure.

        The method computes the temperature increase at a specified depth `z` and time `t`
        following a continuous radiant exposure of 1 W/m² on the medium's surface. The
        volumetric heating within the medium decreases as mu_a * exp(-mu_a * z) [W/m³],

        The method handles scalar or array-like inputs for `z`, allowing for
        the calculation of temperature increases at multiple depths and/or times.

        Args:
            z (scalar or array): Depth(s) at which the temperature is desired [meters].
            t (scalar): Time(s) at which the temperature is desired [seconds].

        Returns:
            scalar or array: Temperature increase at the specified depth(s) and time(s) [°C].
        """
        if np.isscalar(z):
            T = self._continuous_scalar(z, t)
        else:
            T = np.zeros_like(z)
            for i, zz in enumerate(z):
                T[i] = self._continuous_scalar(zz, t)
        return T

    def continuous(self, z, t):
        """
        Calculate temperature rise due to steady irradiance of semi-infinite absorber.

        The method computes the temperature increase at a specified depth `z` and time `t`
        due to a continuous irradiance of 1 W/m² on the absorber's surface. The heating
        starts at t=self.tp and continues to the specified time `t`.

        The volumetric heating within the medium decreases exponentially with depth,
        following the expression: mu_a * exp(-mu_a * z) [W/m³]. Multiply the result
        of this function by the irradiance in W/m² to get the temperature increase.

        The method handles scalar or array-like inputs for `z` and `t`, allowing for
        the calculation of temperature increases at multiple depths and/or times.

        Args:
            z (scalar or array): Depth(s) at which the temperature is desired [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].

        Returns:
            scalar or array: Temperature increase at the specified depth(s) and time(s) [°C].

        Example:
            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                z = 0                                 # meters
                mu_a = 0.25 * 1000                    # 1/m
                irradiance = 1 / (0.01 * 0.01)        # 1 W/cm² = 10⁴ W/m²
                t = np.linspace(0, 500, 100) / 1000   # seconds

                medium = grheat.Absorber(mua)
                T = medium.continuous(z, t) * irradiance

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("1 W/cm² Surface Irradiance")
                plt.show()
        """
        if np.isscalar(t):
            T = 0                # return a scalar
            if np.isscalar(self.tp):
                T += self._continuous(z, t - self.tp)
            else:
                for i, tp in enumerate(self.tp):
                    T += self._continuous(z, t - tp)

        else:
            T = np.zeros_like(t)  # return an array
            for i, tt in enumerate(t):
                if np.isscalar(self.tp):
                    T[i] += self._continuous(z, tt - self.tp)
                else:
                    for tp in self.tp:
                        T[i] += self._continuous(z, tt - tp)
        return T

    def pulsed(self, z, t, t_pulse):
        """
        Calculate temperature rise due to pulsed irradiance of semi-infinite absorber.

        The method computes the temperature increase at a specified depth `z` and time `t`
        due to a pulsed irradiance of 1 J/m² on the absorber's surface. The irradiance
        starts at t=self.tp and continues up to time `t_pulse`. The volumetric heating within
        the medium decreases exponentially with depth, following the expression:
        mu_a * exp(-mu * z) [J/m³].

        Multiply the result of this function by the radiant exposure of the pulse
        in J/m² to get the temperature increase.

        The temperatures are calculated as in Prahl's 1995 SPIE paper, "Charts to rapidly
        estimate temperature following laser irradiation".  The calculations of temperature
        should work for times before, during, and after the pulse.

        Args:
            z (scalar or array): Depth(s) at which the temperature is desired [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].
            t_pulse (scalar): Duration of the irradiance pulse [seconds].

        Returns:
            scalar or array: Temperature increase at the specified depth(s) and time(s) [°C].

        Example:
            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                z = 0                                 # meters
                mu_a = 0.25 * 1000                    # 1/m
                t_pulse = 0.100                       # seconds
                radiant_exposure = 1 / (0.01 * 0.01)  # 1 J/cm²
                t = np.linspace(0, 500, 100) / 1000   # seconds

                medium = grheat.Absorber(mua)
                T = medium.pulsed(z, t, t_pulse) * radiant_exposure

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("1 J/cm² pulse lasting %.0f ms" % t_pulse)
                plt.show()
        """
        T = self.continuous(z, t)
        T -= self.continuous(z, t - t_pulse)
        return T / t_pulse
