"""Shared defaults and validation helpers for grheat models."""

from __future__ import annotations

import operator

water_heat_capacity = 4.184 * 1e6  # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


def validate_positive(name, value):
    """Return a validated positive scalar value.

    Args:
        name (str): Parameter name used in error messages.
        value (scalar): Value to validate.

    Returns:
        scalar: The original value when it is strictly positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_positive_int(name, value):
    """Return a validated positive integer value.

    Args:
        name (str): Parameter name used in error messages.
        value (object): Value to validate.

    Returns:
        int: The validated integer value.
    """
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc

    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def validate_pulse_duration(t_pulse):
    """Validate a pulse duration before dividing by it.

    Args:
        t_pulse (scalar): Pulse duration in seconds.

    Returns:
        scalar: The original pulse duration when it is strictly positive.
    """
    if t_pulse <= 0:
        raise ValueError("Pulse duration (%f) must be positive" % t_pulse)
    return t_pulse
