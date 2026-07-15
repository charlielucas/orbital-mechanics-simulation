"""Acceleration models and conserved quantities for orbital propagation."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
AccelerationModel = Callable[[FloatArray], FloatArray]


def _position_vector(position_km: ArrayLike) -> FloatArray:
    position = np.asarray(position_km, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("position must contain exactly three components")
    if not np.all(np.isfinite(position)):
        raise ValueError("position must contain only finite values")
    if np.linalg.norm(position) == 0.0:
        raise ValueError("position magnitude must be nonzero")
    return position


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def two_body_acceleration(
    position_km: ArrayLike,
    gravitational_parameter_km3_s2: float,
) -> FloatArray:
    """Return point-mass acceleration in km/s^2."""

    position = _position_vector(position_km)
    mu = _positive_finite(gravitational_parameter_km3_s2, "gravitational parameter")
    radius = float(np.linalg.norm(position))
    return -mu * position / radius**3


def j2_perturbation(
    position_km: ArrayLike,
    gravitational_parameter_km3_s2: float,
    equatorial_radius_km: float,
    j2: float,
) -> FloatArray:
    """Return the first-order J2 perturbing acceleration in km/s^2."""

    position = _position_vector(position_km)
    mu = _positive_finite(gravitational_parameter_km3_s2, "gravitational parameter")
    equatorial_radius = _positive_finite(equatorial_radius_km, "equatorial radius")
    coefficient = float(j2)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("J2 must be finite and nonnegative")

    radius = float(np.linalg.norm(position))
    x, y, z = position
    z_ratio_squared = (z / radius) ** 2
    scale = 1.5 * coefficient * mu * equatorial_radius**2 / radius**5
    return scale * np.array(
        [
            x * (5.0 * z_ratio_squared - 1.0),
            y * (5.0 * z_ratio_squared - 1.0),
            z * (5.0 * z_ratio_squared - 3.0),
        ],
        dtype=np.float64,
    )


def two_body_j2_acceleration(
    position_km: ArrayLike,
    gravitational_parameter_km3_s2: float,
    equatorial_radius_km: float,
    j2: float,
) -> FloatArray:
    """Return point-mass plus first-order J2 acceleration in km/s^2."""

    return two_body_acceleration(position_km, gravitational_parameter_km3_s2) + j2_perturbation(
        position_km,
        gravitational_parameter_km3_s2,
        equatorial_radius_km,
        j2,
    )


def j2_secular_raan_rate(
    semi_major_axis_km: float,
    eccentricity: float,
    inclination_rad: float,
    gravitational_parameter_km3_s2: float,
    equatorial_radius_km: float,
    j2: float,
) -> float:
    """Return the first-order secular J2 RAAN rate in radians per second."""

    semi_major_axis = _positive_finite(semi_major_axis_km, "semi-major axis")
    mu = _positive_finite(gravitational_parameter_km3_s2, "gravitational parameter")
    equatorial_radius = _positive_finite(equatorial_radius_km, "equatorial radius")
    coefficient = float(j2)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("J2 must be finite and nonnegative")
    if not np.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= eccentricity < 1")
    if not np.isfinite(inclination_rad) or not 0.0 <= inclination_rad <= np.pi:
        raise ValueError("inclination must be in the closed interval [0, pi]")

    mean_motion = np.sqrt(mu / semi_major_axis**3)
    semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)
    return float(
        -1.5
        * coefficient
        * mean_motion
        * (equatorial_radius / semi_latus_rectum) ** 2
        * np.cos(inclination_rad)
    )


def specific_orbital_energy(
    position_km: ArrayLike,
    velocity_km_s: ArrayLike,
    gravitational_parameter_km3_s2: float,
) -> float:
    """Return specific mechanical energy for the two-body problem in km^2/s^2."""

    position = _position_vector(position_km)
    velocity = np.asarray(velocity_km_s, dtype=np.float64)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must contain exactly three finite components")
    mu = _positive_finite(gravitational_parameter_km3_s2, "gravitational parameter")
    return 0.5 * float(np.dot(velocity, velocity)) - mu / float(np.linalg.norm(position))


def specific_angular_momentum(position_km: ArrayLike, velocity_km_s: ArrayLike) -> FloatArray:
    """Return the specific angular-momentum vector in km^2/s."""

    position = _position_vector(position_km)
    velocity = np.asarray(velocity_km_s, dtype=np.float64)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must contain exactly three finite components")
    return np.cross(position, velocity)
