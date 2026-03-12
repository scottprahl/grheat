"""Regression tests for numeric edge cases and constructor validation."""

import numpy as np
import pytest

import grheat


@pytest.mark.parametrize(
    ("call_with_int", "call_with_float"),
    [
        (
            lambda: grheat.Line(0, 1e-3).instantaneous(0, 0, np.array([1, 2, 3], dtype=int)),
            lambda: grheat.Line(0, 1e-3).instantaneous(0, 0, np.array([1.0, 2.0, 3.0])),
        ),
        (
            lambda: grheat.Line(0, 1e-3).continuous(0, 0, np.array([1, 2, 3], dtype=int)),
            lambda: grheat.Line(0, 1e-3).continuous(0, 0, np.array([1.0, 2.0, 3.0])),
        ),
        (
            lambda: grheat.Line(0, 1e-3).pulsed(0, 0, np.array([1, 2, 3], dtype=int), 0.5),
            lambda: grheat.Line(0, 1e-3).pulsed(0, 0, np.array([1.0, 2.0, 3.0]), 0.5),
        ),
        (
            lambda: grheat.Plane(1e-3).instantaneous(0, np.array([1, 2, 3], dtype=int)),
            lambda: grheat.Plane(1e-3).instantaneous(0, np.array([1.0, 2.0, 3.0])),
        ),
        (
            lambda: grheat.Plane(1e-3).continuous(0, np.array([1, 2, 3], dtype=int)),
            lambda: grheat.Plane(1e-3).continuous(0, np.array([1.0, 2.0, 3.0])),
        ),
        (
            lambda: grheat.Plane(1e-3).pulsed(0, np.array([1, 2, 3], dtype=int), 0.5),
            lambda: grheat.Plane(1e-3).pulsed(0, np.array([1.0, 2.0, 3.0]), 0.5),
        ),
        (
            lambda: grheat.ExponentialVolumeSource(1000).instantaneous(0, np.array([1, 2, 3], dtype=int)),
            lambda: grheat.ExponentialVolumeSource(1000).instantaneous(
                0, np.array([1.0, 2.0, 3.0])
            ),
        ),
        (
            lambda: grheat.ExponentialVolumeSource(1000).continuous(0, np.array([1, 2, 3], dtype=int)),
            lambda: grheat.ExponentialVolumeSource(1000).continuous(0, np.array([1.0, 2.0, 3.0])),
        ),
        (
            lambda: grheat.ExponentialVolumeSource(1000).pulsed(0, np.array([1, 2, 3], dtype=int), 0.5),
            lambda: grheat.ExponentialVolumeSource(1000).pulsed(0, np.array([1.0, 2.0, 3.0]), 0.5),
        ),
        (
            lambda: grheat.ExponentialVolumeSource(1000).instantaneous(np.array([0, 1, 2], dtype=int), 1.0),
            lambda: grheat.ExponentialVolumeSource(1000).instantaneous(np.array([0.0, 1.0, 2.0]), 1.0),
        ),
        (
            lambda: grheat.ExponentialVolumeSource(1000).continuous(np.array([0, 1, 2], dtype=int), 1.0),
            lambda: grheat.ExponentialVolumeSource(1000).continuous(np.array([0.0, 1.0, 2.0]), 1.0),
        ),
    ],
    ids=[
        "line_instantaneous",
        "line_continuous",
        "line_pulsed",
        "plane_instantaneous",
        "plane_continuous",
        "plane_pulsed",
        "volume_instantaneous_time",
        "volume_continuous_time",
        "volume_pulsed_time",
        "volume_instantaneous_depth",
        "volume_continuous_depth",
    ],
)
def test_integer_arrays_match_float_results(call_with_int, call_with_float):
    """Integer-valued arrays should produce the same float results as float arrays."""
    actual = call_with_int()
    expected = call_with_float()

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda: grheat.Point(0, 0, 1e-3).pulsed(0, 0, 0, 1.0, 0.0),
        lambda: grheat.Line(0, 1e-3).pulsed(0, 0, 1.0, 0.0),
        lambda: grheat.Plane(1e-3).pulsed(0, 1.0, 0.0),
        lambda: grheat.ExponentialVolumeSource(1000).pulsed(0, 1.0, 0.0),
        lambda: grheat.ExponentialColumnSource(1000, 0, 0).pulsed(0, 0, 0, 1.0, 0.0),
    ],
    ids=["point", "line", "plane", "volume", "column"],
)
def test_zero_duration_pulses_raise_value_error(call):
    """Zero-duration pulses should fail fast with a clear error."""
    with pytest.raises(ValueError, match="Pulse duration"):
        call()


@pytest.mark.parametrize("mu_a", [0, -1])
def test_exponential_volume_source_requires_positive_absorption(mu_a):
    """The exponential volume source should reject non-positive absorption values."""
    with pytest.raises(ValueError, match="mu_a must be positive"):
        grheat.ExponentialVolumeSource(mu_a)


@pytest.mark.parametrize("mu_a", [0, -1])
def test_exponential_column_source_requires_positive_absorption(mu_a):
    """The exponential column source should reject non-positive absorption values."""
    with pytest.raises(ValueError, match="mu_a must be positive"):
        grheat.ExponentialColumnSource(mu_a, 0, 0)


@pytest.mark.parametrize("n_quad", [0, -1])
def test_exponential_column_source_requires_positive_quadrature_count(n_quad):
    """The exponential column source should require at least one quadrature node."""
    with pytest.raises(ValueError, match="n_quad must be a positive integer"):
        grheat.ExponentialColumnSource(1000, 0, 0, n_quad=n_quad)


def test_all_models_share_default_water_properties():
    """Default water properties should remain consistent across all source models."""
    models = [
        grheat.Point(0, 0, 1e-3),
        grheat.Line(0, 1e-3),
        grheat.Plane(1e-3),
        grheat.ExponentialVolumeSource(1000),
        grheat.ExponentialColumnSource(1000, 0, 0),
    ]

    capacities = {model.capacity for model in models}
    diffusivities = {model.diffusivity for model in models}

    assert len(capacities) == 1
    assert len(diffusivities) == 1
