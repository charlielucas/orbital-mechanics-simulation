"""Conversions between classical orbital elements and Cartesian state vectors."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
TWO_PI = 2.0 * np.pi
SINGULARITY_TOLERANCE = 1e-10


def _wrap_angle(angle_rad: float) -> float:
    return float(angle_rad % TWO_PI)


def _vector3(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three components")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True, slots=True)
class ClassicalOrbitalElements:
    """Elliptic classical orbital elements in kilometers and radians.

    Singular circular/equatorial cases use explicit conventions in
    :func:`cartesian_to_elements`: undefined angles are set to zero and the
    remaining defined longitude carries the orientation.
    """

    semi_major_axis_km: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    argument_of_periapsis_rad: float
    true_anomaly_rad: float

    def __post_init__(self) -> None:
        values = (
            self.semi_major_axis_km,
            self.eccentricity,
            self.inclination_rad,
            self.raan_rad,
            self.argument_of_periapsis_rad,
            self.true_anomaly_rad,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("orbital elements must be finite")
        if self.semi_major_axis_km <= 0.0:
            raise ValueError("semi-major axis must be positive")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("only elliptic orbits with 0 <= eccentricity < 1 are supported")
        if not 0.0 <= self.inclination_rad <= np.pi:
            raise ValueError("inclination must be in the closed interval [0, pi]")


def elements_to_cartesian(
    elements: ClassicalOrbitalElements,
    gravitational_parameter_km3_s2: float,
) -> tuple[FloatArray, FloatArray]:
    """Convert elliptic classical elements to inertial position and velocity."""

    mu = float(gravitational_parameter_km3_s2)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("gravitational parameter must be finite and positive")

    semi_major_axis = elements.semi_major_axis_km
    eccentricity = elements.eccentricity
    true_anomaly = elements.true_anomaly_rad
    parameter = semi_major_axis * (1.0 - eccentricity**2)
    denominator = 1.0 + eccentricity * np.cos(true_anomaly)

    position_perifocal = np.array(
        [
            parameter * np.cos(true_anomaly) / denominator,
            parameter * np.sin(true_anomaly) / denominator,
            0.0,
        ],
        dtype=np.float64,
    )
    velocity_scale = np.sqrt(mu / parameter)
    velocity_perifocal = velocity_scale * np.array(
        [-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0],
        dtype=np.float64,
    )

    raan = elements.raan_rad
    inclination = elements.inclination_rad
    argument_of_periapsis = elements.argument_of_periapsis_rad
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)
    cos_argp, sin_argp = np.cos(argument_of_periapsis), np.sin(argument_of_periapsis)

    perifocal_to_inertial = np.array(
        [
            [
                cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
                -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
                sin_raan * sin_i,
            ],
            [
                sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
                -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
                -cos_raan * sin_i,
            ],
            [sin_argp * sin_i, cos_argp * sin_i, cos_i],
        ],
        dtype=np.float64,
    )
    return perifocal_to_inertial @ position_perifocal, perifocal_to_inertial @ velocity_perifocal


def cartesian_to_elements(
    position_km: ArrayLike,
    velocity_km_s: ArrayLike,
    gravitational_parameter_km3_s2: float,
    *,
    singularity_tolerance: float = SINGULARITY_TOLERANCE,
) -> ClassicalOrbitalElements:
    """Convert an inertial state to elliptic classical orbital elements.

    For equatorial orbits, right ascension of the ascending node is set to
    zero. For circular orbits, argument of periapsis is set to zero and true
    anomaly stores argument of latitude (inclined) or true longitude
    (equatorial). For eccentric equatorial orbits, argument of periapsis stores
    longitude of periapsis. Equatorial longitudes preserve the handedness of
    prograde (positive h-z) and retrograde (negative h-z) states.
    """

    position = _vector3(position_km, "position")
    velocity = _vector3(velocity_km_s, "velocity")
    mu = float(gravitational_parameter_km3_s2)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("gravitational parameter must be finite and positive")
    if singularity_tolerance <= 0.0:
        raise ValueError("singularity tolerance must be positive")

    radius = float(np.linalg.norm(position))
    if radius <= singularity_tolerance:
        raise ValueError("position magnitude must be nonzero")

    angular_momentum = np.cross(position, velocity)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if angular_momentum_norm <= singularity_tolerance:
        raise ValueError("state must have nonzero angular momentum")

    node = np.cross(np.array([0.0, 0.0, 1.0]), angular_momentum)
    node_norm = float(np.linalg.norm(node))
    node_is_defined = node_norm > singularity_tolerance * angular_momentum_norm
    eccentricity_vector = np.cross(velocity, angular_momentum) / mu - position / radius
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    if eccentricity >= 1.0:
        raise ValueError("only bound elliptic states are supported")

    specific_energy = 0.5 * float(np.dot(velocity, velocity)) - mu / radius
    if specific_energy >= 0.0:
        raise ValueError("only bound elliptic states are supported")
    semi_major_axis = -mu / (2.0 * specific_energy)
    inclination = float(np.arccos(np.clip(angular_momentum[2] / angular_momentum_norm, -1.0, 1.0)))
    equatorial_orientation = 1.0 if angular_momentum[2] >= 0.0 else -1.0

    raan = _wrap_angle(float(np.arctan2(node[1], node[0]))) if node_is_defined else 0.0

    if eccentricity > singularity_tolerance and node_is_defined:
        argument_of_periapsis = _wrap_angle(
            float(
                np.arctan2(
                    np.dot(np.cross(node, eccentricity_vector), angular_momentum)
                    / (node_norm * eccentricity * angular_momentum_norm),
                    np.dot(node, eccentricity_vector) / (node_norm * eccentricity),
                )
            )
        )
    elif eccentricity > singularity_tolerance:
        argument_of_periapsis = _wrap_angle(
            equatorial_orientation
            * float(np.arctan2(eccentricity_vector[1], eccentricity_vector[0]))
        )
    else:
        argument_of_periapsis = 0.0

    if eccentricity > singularity_tolerance:
        true_anomaly = _wrap_angle(
            float(
                np.arctan2(
                    np.dot(np.cross(eccentricity_vector, position), angular_momentum)
                    / (eccentricity * radius * angular_momentum_norm),
                    np.dot(eccentricity_vector, position) / (eccentricity * radius),
                )
            )
        )
    elif node_is_defined:
        true_anomaly = _wrap_angle(
            float(
                np.arctan2(
                    np.dot(np.cross(node, position), angular_momentum)
                    / (node_norm * radius * angular_momentum_norm),
                    np.dot(node, position) / (node_norm * radius),
                )
            )
        )
    else:
        true_anomaly = _wrap_angle(
            equatorial_orientation * float(np.arctan2(position[1], position[0]))
        )

    return ClassicalOrbitalElements(
        semi_major_axis_km=semi_major_axis,
        eccentricity=eccentricity,
        inclination_rad=inclination,
        raan_rad=raan,
        argument_of_periapsis_rad=argument_of_periapsis,
        true_anomaly_rad=true_anomaly,
    )
