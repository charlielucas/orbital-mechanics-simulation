"""Interactive explorer scenarios and diagnostics built on the validated propagator."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orbital_mechanics.constants import EARTH
from orbital_mechanics.dynamics import (
    j2_secular_raan_rate,
    two_body_acceleration,
    two_body_j2_acceleration,
)
from orbital_mechanics.elements import ClassicalOrbitalElements, elements_to_cartesian
from orbital_mechanics.propagation import Trajectory, propagate_rk4

FloatArray = NDArray[np.float64]
ForceModel = Literal["two_body", "j2"]
SECONDS_PER_DAY = 86_400.0
MAX_PROPAGATION_STEPS = 20_160
MINIMUM_PERIGEE_ALTITUDE_KM = 120.0
TWO_BODY_DRIFT_LIMIT = 1e-7
J2_RATE_ERROR_LIMIT = 0.02


def keplerian_period_s(semi_major_axis_km: float) -> float:
    """Return the Earth-centered Keplerian period for a positive semi-major axis."""

    semi_major_axis = float(semi_major_axis_km)
    if not np.isfinite(semi_major_axis) or semi_major_axis <= 0.0:
        raise ValueError("semi-major axis must be finite and positive")
    return float(2.0 * np.pi * np.sqrt(semi_major_axis**3 / EARTH.gravitational_parameter_km3_s2))


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    """Inputs for one bounded explorer propagation."""

    elements: ClassicalOrbitalElements
    force_model: ForceModel
    duration_s: float
    step_count: int

    def __post_init__(self) -> None:
        if self.force_model not in ("two_body", "j2"):
            raise ValueError("force model must be 'two_body' or 'j2'")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration must be finite and positive")
        if type(self.step_count) is not int or not 1 <= self.step_count <= MAX_PROPAGATION_STEPS:
            raise ValueError(f"step count must be an integer between 1 and {MAX_PROPAGATION_STEPS}")

        perigee_radius = self.elements.semi_major_axis_km * (1.0 - self.elements.eccentricity)
        perigee_altitude = perigee_radius - EARTH.equatorial_radius_km
        if perigee_altitude < MINIMUM_PERIGEE_ALTITUDE_KM:
            raise ValueError(
                f"perigee altitude must be at least {MINIMUM_PERIGEE_ALTITUDE_KM:.0f} km"
            )

    @property
    def step_s(self) -> float:
        """Return the exact fixed step implied by the duration and step count."""

        return self.duration_s / self.step_count

    @property
    def orbital_period_s(self) -> float:
        """Return the Keplerian period associated with the semi-major axis."""

        return keplerian_period_s(self.elements.semi_major_axis_km)


@dataclass(frozen=True, slots=True)
class GuidedScenario:
    """A fixed scenario with a clear numerical question."""

    key: str
    name: str
    summary: str
    question: str
    request: SimulationRequest


@dataclass(frozen=True, slots=True)
class ExplorerResult:
    """Trajectory plus diagnostics used by the app and its downloads."""

    request: SimulationRequest
    trajectory: Trajectory
    radius_km: FloatArray
    altitude_km: FloatArray
    relative_energy_drift: FloatArray | None
    relative_angular_momentum_vector_drift: FloatArray | None
    raan_change_deg: FloatArray | None
    theoretical_raan_change_deg: FloatArray | None
    fitted_raan_rate_deg_day: float | None
    theoretical_raan_rate_deg_day: float | None
    relative_raan_rate_error: float | None

    @property
    def max_relative_energy_drift(self) -> float | None:
        if self.relative_energy_drift is None:
            return None
        return float(np.max(np.abs(self.relative_energy_drift)))

    @property
    def max_relative_angular_momentum_vector_drift(self) -> float | None:
        if self.relative_angular_momentum_vector_drift is None:
            return None
        return float(np.max(self.relative_angular_momentum_vector_drift))


def _two_body_model(position_km: FloatArray) -> FloatArray:
    return two_body_acceleration(position_km, EARTH.gravitational_parameter_km3_s2)


def _j2_model(position_km: FloatArray) -> FloatArray:
    return two_body_j2_acceleration(
        position_km,
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )


def _fit_slope(times_s: FloatArray, values: FloatArray) -> float:
    centered_times = times_s - float(np.mean(times_s))
    slope = np.linalg.lstsq(centered_times[:, np.newaxis], values - np.mean(values), rcond=None)[0][
        0
    ]
    return float(slope)


def _raan_series(trajectory: Trajectory) -> FloatArray:
    angular_momentum = np.cross(trajectory.positions_km, trajectory.velocities_km_s)
    return np.unwrap(np.arctan2(angular_momentum[:, 0], -angular_momentum[:, 1]))


def run_explorer(request: SimulationRequest) -> ExplorerResult:
    """Run one explorer propagation and calculate model-appropriate diagnostics."""

    position, velocity = elements_to_cartesian(
        request.elements,
        EARTH.gravitational_parameter_km3_s2,
    )
    acceleration = _two_body_model if request.force_model == "two_body" else _j2_model
    trajectory = propagate_rk4(
        position,
        velocity,
        request.duration_s,
        request.step_s,
        acceleration,
    )

    radius = np.linalg.norm(trajectory.positions_km, axis=1)
    altitude = radius - EARTH.equatorial_radius_km

    if request.force_model == "two_body":
        speed_squared = np.einsum(
            "ij,ij->i",
            trajectory.velocities_km_s,
            trajectory.velocities_km_s,
        )
        energy = 0.5 * speed_squared - EARTH.gravitational_parameter_km3_s2 / radius
        energy_drift = (energy - energy[0]) / abs(energy[0])
        angular_momentum = np.cross(trajectory.positions_km, trajectory.velocities_km_s)
        initial_momentum_norm = float(np.linalg.norm(angular_momentum[0]))
        momentum_drift = (
            np.linalg.norm(angular_momentum - angular_momentum[0], axis=1) / initial_momentum_norm
        )
        return ExplorerResult(
            request=request,
            trajectory=trajectory,
            radius_km=radius,
            altitude_km=altitude,
            relative_energy_drift=energy_drift,
            relative_angular_momentum_vector_drift=momentum_drift,
            raan_change_deg=None,
            theoretical_raan_change_deg=None,
            fitted_raan_rate_deg_day=None,
            theoretical_raan_rate_deg_day=None,
            relative_raan_rate_error=None,
        )

    if abs(np.sin(request.elements.inclination_rad)) <= 1e-8:
        return ExplorerResult(
            request=request,
            trajectory=trajectory,
            radius_km=radius,
            altitude_km=altitude,
            relative_energy_drift=None,
            relative_angular_momentum_vector_drift=None,
            raan_change_deg=None,
            theoretical_raan_change_deg=None,
            fitted_raan_rate_deg_day=None,
            theoretical_raan_rate_deg_day=None,
            relative_raan_rate_error=None,
        )

    raan = _raan_series(trajectory)
    fitted_rate_rad_s = _fit_slope(trajectory.times_s, raan)
    theoretical_rate_rad_s = j2_secular_raan_rate(
        request.elements.semi_major_axis_km,
        request.elements.eccentricity,
        request.elements.inclination_rad,
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )
    fitted_rate_deg_day = float(np.rad2deg(fitted_rate_rad_s) * SECONDS_PER_DAY)
    theoretical_rate_deg_day = float(np.rad2deg(theoretical_rate_rad_s) * SECONDS_PER_DAY)
    relative_rate_error = (
        abs((fitted_rate_rad_s - theoretical_rate_rad_s) / theoretical_rate_rad_s)
        if abs(theoretical_rate_rad_s) > 1e-14
        else None
    )

    return ExplorerResult(
        request=request,
        trajectory=trajectory,
        radius_km=radius,
        altitude_km=altitude,
        relative_energy_drift=None,
        relative_angular_momentum_vector_drift=None,
        raan_change_deg=np.rad2deg(raan - raan[0]),
        theoretical_raan_change_deg=np.rad2deg(theoretical_rate_rad_s * trajectory.times_s),
        fitted_raan_rate_deg_day=fitted_rate_deg_day,
        theoretical_raan_rate_deg_day=theoretical_rate_deg_day,
        relative_raan_rate_error=relative_rate_error,
    )


def chart_indices(length: int, max_points: int = 1_200) -> NDArray[np.int64]:
    """Return deterministic indices that retain both endpoints for plotting."""

    if length < 1:
        raise ValueError("length must be positive")
    if max_points < 2:
        raise ValueError("max points must be at least two")
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=np.int64))


def trajectory_csv(result: ExplorerResult) -> str:
    """Serialize the complete trajectory and diagnostics for download."""

    output = io.StringIO()
    fieldnames = [
        "time_s",
        "elapsed_orbits",
        "x_km",
        "y_km",
        "z_km",
        "vx_km_s",
        "vy_km_s",
        "vz_km_s",
        "radius_km",
        "altitude_km",
    ]
    if result.relative_energy_drift is not None:
        fieldnames.extend(
            [
                "relative_energy_drift",
                "relative_angular_momentum_vector_drift",
            ]
        )
    if result.raan_change_deg is not None:
        fieldnames.extend(["raan_change_deg", "theoretical_raan_change_deg"])

    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    period = result.request.orbital_period_s
    for index, time_s in enumerate(result.trajectory.times_s):
        position = result.trajectory.positions_km[index]
        velocity = result.trajectory.velocities_km_s[index]
        row = {
            "time_s": float(time_s),
            "elapsed_orbits": float(time_s / period),
            "x_km": float(position[0]),
            "y_km": float(position[1]),
            "z_km": float(position[2]),
            "vx_km_s": float(velocity[0]),
            "vy_km_s": float(velocity[1]),
            "vz_km_s": float(velocity[2]),
            "radius_km": float(result.radius_km[index]),
            "altitude_km": float(result.altitude_km[index]),
        }
        if result.relative_energy_drift is not None:
            row["relative_energy_drift"] = float(result.relative_energy_drift[index])
            row["relative_angular_momentum_vector_drift"] = float(
                result.relative_angular_momentum_vector_drift[index]
            )
        if result.raan_change_deg is not None:
            row["raan_change_deg"] = float(result.raan_change_deg[index])
            row["theoretical_raan_change_deg"] = float(result.theoretical_raan_change_deg[index])
        writer.writerow(row)
    return output.getvalue()


def _request_for_orbits(
    elements: ClassicalOrbitalElements,
    force_model: ForceModel,
    orbit_count: int,
    steps_per_orbit: int,
) -> SimulationRequest:
    period = keplerian_period_s(elements.semi_major_axis_km)
    return SimulationRequest(
        elements=elements,
        force_model=force_model,
        duration_s=orbit_count * period,
        step_count=orbit_count * steps_per_orbit,
    )


GUIDED_SCENARIOS = (
    GuidedScenario(
        key="circular_reference",
        name="Circular reference",
        summary="Inspect fixed-step RK4 behavior in a circular orbit.",
        question="Does the numerical trajectory preserve the two-body invariants over two orbits?",
        request=_request_for_orbits(
            ClassicalOrbitalElements(
                semi_major_axis_km=7_000.0,
                eccentricity=0.0,
                inclination_rad=np.deg2rad(28.5),
                raan_rad=np.deg2rad(20.0),
                argument_of_periapsis_rad=0.0,
                true_anomaly_rad=0.0,
            ),
            "two_body",
            orbit_count=2,
            steps_per_orbit=720,
        ),
    ),
    GuidedScenario(
        key="eccentric_conservation",
        name="Eccentric conservation",
        summary="Propagate an eccentric, inclined orbit while measuring numerical drift.",
        question="Do energy and angular momentum remain stable across three complete orbits?",
        request=_request_for_orbits(
            ClassicalOrbitalElements(
                semi_major_axis_km=10_000.0,
                eccentricity=0.2,
                inclination_rad=np.deg2rad(50.0),
                raan_rad=np.deg2rad(40.0),
                argument_of_periapsis_rad=np.deg2rad(30.0),
                true_anomaly_rad=np.deg2rad(15.0),
            ),
            "two_body",
            orbit_count=3,
            steps_per_orbit=600,
        ),
    ),
    GuidedScenario(
        key="j2_precession",
        name="J2 nodal precession",
        summary="Measure how Earth's oblateness changes the orbital plane over seven days.",
        question="Does the fitted RAAN drift agree with first-order J2 theory?",
        request=SimulationRequest(
            elements=ClassicalOrbitalElements(
                semi_major_axis_km=7_000.0,
                eccentricity=0.001,
                inclination_rad=np.deg2rad(98.0),
                raan_rad=np.deg2rad(20.0),
                argument_of_periapsis_rad=np.deg2rad(30.0),
                true_anomaly_rad=0.0,
            ),
            force_model="j2",
            duration_s=7.0 * SECONDS_PER_DAY,
            step_count=5_040,
        ),
    ),
)

GUIDED_SCENARIOS_BY_KEY = {scenario.key: scenario for scenario in GUIDED_SCENARIOS}
