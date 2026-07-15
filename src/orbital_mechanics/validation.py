"""Reproducible numerical validation suite and evidence artifact generation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from orbital_mechanics.constants import EARTH
from orbital_mechanics.dynamics import (
    j2_secular_raan_rate,
    two_body_acceleration,
    two_body_j2_acceleration,
)
from orbital_mechanics.elements import ClassicalOrbitalElements, elements_to_cartesian
from orbital_mechanics.propagation import Trajectory, propagate_rk4

FloatArray = NDArray[np.float64]
SECONDS_PER_DAY = 86_400.0
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRID = "#D9DEE7"
GENERATOR_SOURCE_FILES = (
    "constants.py",
    "dynamics.py",
    "elements.py",
    "propagation.py",
    "validation.py",
)


def generator_source_sha256() -> str:
    """Hash the source modules that define validation data and figures."""

    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in GENERATOR_SOURCE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((package_dir / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def orbital_period_s(semi_major_axis_km: float, gravitational_parameter_km3_s2: float) -> float:
    """Return the Keplerian period of a bound orbit in seconds."""

    if semi_major_axis_km <= 0.0 or gravitational_parameter_km3_s2 <= 0.0:
        raise ValueError("semi-major axis and gravitational parameter must be positive")
    return float(2.0 * np.pi * np.sqrt(semi_major_axis_km**3 / gravitational_parameter_km3_s2))


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": "#4C566A",
            "axes.grid": True,
            "axes.labelcolor": "#202634",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "text.color": "#202634",
        }
    )


def _two_body_model(position_km: FloatArray) -> FloatArray:
    return two_body_acceleration(position_km, EARTH.gravitational_parameter_km3_s2)


def _j2_model(position_km: FloatArray) -> FloatArray:
    return two_body_j2_acceleration(
        position_km,
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )


def _propagate_elements(
    elements: ClassicalOrbitalElements,
    orbit_count: int,
    steps_per_orbit: int,
    acceleration: Callable[[FloatArray], FloatArray],
) -> tuple[Trajectory, float]:
    position, velocity = elements_to_cartesian(elements, EARTH.gravitational_parameter_km3_s2)
    period = orbital_period_s(elements.semi_major_axis_km, EARTH.gravitational_parameter_km3_s2)
    duration = orbit_count * period
    step = period / steps_per_orbit
    return propagate_rk4(position, velocity, duration, step, acceleration), period


def _fit_slope(times_s: FloatArray, values: FloatArray) -> tuple[float, float]:
    centered_times = times_s - float(np.mean(times_s))
    design = np.column_stack((centered_times, np.ones_like(centered_times)))
    slope, centered_intercept = np.linalg.lstsq(design, values, rcond=None)[0]
    intercept = centered_intercept - slope * float(np.mean(times_s))
    return float(slope), float(intercept)


def _raan_series(trajectory: Trajectory) -> FloatArray:
    angular_momentum = np.cross(trajectory.positions_km, trajectory.velocities_km_s)
    raan = np.arctan2(angular_momentum[:, 0], -angular_momentum[:, 1])
    return np.unwrap(raan)


def _metric(
    scenario: str,
    name: str,
    value: float,
    unit: str,
    threshold: float,
    comparison: str,
) -> dict[str, Any]:
    if comparison == "<=":
        passed = value <= threshold
    elif comparison == ">=":
        passed = value >= threshold
    elif comparison == "sign":
        passed = int(np.sign(value)) == int(np.sign(threshold))
    else:
        raise ValueError(f"unsupported comparison: {comparison}")
    return {
        "scenario": scenario,
        "name": name,
        "value": float(value),
        "unit": unit,
        "threshold": float(threshold),
        "comparison": comparison,
        "passed": bool(passed),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_circular(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    radius_km = 7_000.0
    orbit_count = 5
    steps_per_orbit = 720
    speed_km_s = np.sqrt(EARTH.gravitational_parameter_km3_s2 / radius_km)
    initial_position = np.array([radius_km, 0.0, 0.0])
    initial_velocity = np.array([0.0, speed_km_s, 0.0])
    period = orbital_period_s(radius_km, EARTH.gravitational_parameter_km3_s2)
    step = period / steps_per_orbit
    trajectory = propagate_rk4(
        initial_position,
        initial_velocity,
        orbit_count * period,
        step,
        _two_body_model,
    )

    mean_motion = np.sqrt(EARTH.gravitational_parameter_km3_s2 / radius_km**3)
    phase = mean_motion * trajectory.times_s
    analytic_positions = np.column_stack(
        (radius_km * np.cos(phase), radius_km * np.sin(phase), np.zeros_like(phase))
    )
    analytic_velocities = np.column_stack(
        (-speed_km_s * np.sin(phase), speed_km_s * np.cos(phase), np.zeros_like(phase))
    )
    position_errors = np.linalg.norm(trajectory.positions_km - analytic_positions, axis=1)
    velocity_errors = np.linalg.norm(trajectory.velocities_km_s - analytic_velocities, axis=1)

    metrics = [
        _metric(
            "circular_two_body",
            "maximum_position_error",
            float(np.max(position_errors)),
            "km",
            0.02,
            "<=",
        ),
        _metric(
            "circular_two_body",
            "maximum_velocity_error",
            float(np.max(velocity_errors)),
            "km/s",
            2e-5,
            "<=",
        ),
    ]

    rows = [
        {
            "time_s": float(time),
            "orbit_fraction": float(time / period),
            "x_numeric_km": float(numeric[0]),
            "y_numeric_km": float(numeric[1]),
            "x_analytic_km": float(analytic[0]),
            "y_analytic_km": float(analytic[1]),
            "position_error_km": float(position_error),
            "velocity_error_km_s": float(velocity_error),
        }
        for time, numeric, analytic, position_error, velocity_error in zip(
            trajectory.times_s,
            trajectory.positions_km,
            analytic_positions,
            position_errors,
            velocity_errors,
            strict=True,
        )
    ]
    _write_csv(
        output_dir / "circular_two_body.csv",
        list(rows[0]),
        rows,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(
        analytic_positions[:, 0],
        analytic_positions[:, 1],
        color=ORANGE,
        linewidth=2.8,
        label="Analytic",
    )
    axes[0].plot(
        trajectory.positions_km[:, 0],
        trajectory.positions_km[:, 1],
        color=BLUE,
        linewidth=1.2,
        linestyle="--",
        label="RK4",
    )
    axes[0].scatter([0.0], [0.0], color=GREEN, marker="o", s=65, label="Earth center", zorder=3)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("ECI x (km)")
    axes[0].set_ylabel("ECI y (km)")
    axes[0].set_title("Five-orbit trajectory overlay")
    axes[0].legend()

    axes[1].semilogy(
        trajectory.times_s / period,
        np.maximum(position_errors / 0.02, 1e-14),
        color=BLUE,
        label="Position / 0.02 km limit",
    )
    axes[1].semilogy(
        trajectory.times_s / period,
        np.maximum(velocity_errors / 2e-5, 1e-14),
        color=VERMILLION,
        label="Velocity / 2e-5 km/s limit",
    )
    axes[1].axhline(
        1.0,
        color="#4C566A",
        linewidth=1.0,
        linestyle=":",
        label="Acceptance limit",
    )
    axes[1].set_xlabel("Elapsed orbits")
    axes[1].set_ylabel("Error / acceptance limit")
    axes[1].set_title("Whole-arc error margin")
    axes[1].legend(fontsize=8)
    figure.suptitle("Circular two-body validation", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(
        output_dir / "circular_two_body.png",
        dpi=200,
        metadata={"Creator": "orbital-mechanics-simulation"},
    )
    plt.close(figure)

    return (
        {
            "parameters": {
                "radius_km": radius_km,
                "orbit_count": orbit_count,
                "steps_per_orbit": steps_per_orbit,
                "analytic_period_s": period,
                "step_s": step,
            },
            "metrics": metrics,
        },
        metrics,
    )


def _validate_conservation(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    elements = ClassicalOrbitalElements(
        semi_major_axis_km=10_000.0,
        eccentricity=0.2,
        inclination_rad=np.deg2rad(50.0),
        raan_rad=np.deg2rad(40.0),
        argument_of_periapsis_rad=np.deg2rad(30.0),
        true_anomaly_rad=np.deg2rad(15.0),
    )
    orbit_count = 10
    steps_per_orbit = 600
    trajectory, period = _propagate_elements(
        elements,
        orbit_count,
        steps_per_orbit,
        _two_body_model,
    )
    radius = np.linalg.norm(trajectory.positions_km, axis=1)
    speed_squared = np.einsum("ij,ij->i", trajectory.velocities_km_s, trajectory.velocities_km_s)
    energy = 0.5 * speed_squared - EARTH.gravitational_parameter_km3_s2 / radius
    angular_momentum_vectors = np.cross(trajectory.positions_km, trajectory.velocities_km_s)
    angular_momentum_magnitudes = np.linalg.norm(angular_momentum_vectors, axis=1)
    energy_drift = (energy - energy[0]) / abs(energy[0])
    angular_momentum_magnitude_drift = (
        angular_momentum_magnitudes - angular_momentum_magnitudes[0]
    ) / angular_momentum_magnitudes[0]
    angular_momentum_vector_drift = (
        np.linalg.norm(angular_momentum_vectors - angular_momentum_vectors[0], axis=1)
        / angular_momentum_magnitudes[0]
    )

    metrics = [
        _metric(
            "two_body_conservation",
            "maximum_relative_specific_energy_drift",
            float(np.max(np.abs(energy_drift))),
            "dimensionless",
            1e-7,
            "<=",
        ),
        _metric(
            "two_body_conservation",
            "maximum_relative_specific_angular_momentum_vector_drift",
            float(np.max(angular_momentum_vector_drift)),
            "dimensionless",
            1e-7,
            "<=",
        ),
    ]

    rows = [
        {
            "time_s": float(time),
            "orbit_fraction": float(time / period),
            "specific_energy_km2_s2": float(energy_value),
            "relative_energy_drift": float(energy_delta),
            "specific_angular_momentum_magnitude_km2_s": float(momentum_magnitude),
            "relative_angular_momentum_magnitude_drift": float(momentum_magnitude_delta),
            "relative_angular_momentum_vector_drift": float(momentum_vector_delta),
        }
        for (
            time,
            energy_value,
            energy_delta,
            momentum_magnitude,
            momentum_magnitude_delta,
            momentum_vector_delta,
        ) in zip(
            trajectory.times_s,
            energy,
            energy_drift,
            angular_momentum_magnitudes,
            angular_momentum_magnitude_drift,
            angular_momentum_vector_drift,
            strict=True,
        )
    ]
    _write_csv(output_dir / "two_body_conservation.csv", list(rows[0]), rows)

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    axes[0].plot(trajectory.times_s / period, energy_drift, color=BLUE, linewidth=1.3)
    axes[0].axhline(0.0, color="#4C566A", linewidth=0.8)
    axes[0].set_ylabel("Relative energy drift")
    axes[0].set_title("Specific mechanical energy")
    axes[1].plot(
        trajectory.times_s / period,
        angular_momentum_vector_drift,
        color=GREEN,
        linewidth=1.3,
    )
    axes[1].axhline(0.0, color="#4C566A", linewidth=0.8)
    axes[1].set_xlabel("Elapsed orbits")
    axes[1].set_ylabel("Relative vector drift")
    axes[1].set_title("Specific angular momentum vector")
    figure.suptitle("Ten-orbit conservation validation", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(
        output_dir / "two_body_conservation.png",
        dpi=200,
        metadata={"Creator": "orbital-mechanics-simulation"},
    )
    plt.close(figure)

    return (
        {
            "parameters": {
                "elements": {
                    "semi_major_axis_km": elements.semi_major_axis_km,
                    "eccentricity": elements.eccentricity,
                    "inclination_deg": float(np.rad2deg(elements.inclination_rad)),
                    "raan_deg": float(np.rad2deg(elements.raan_rad)),
                    "argument_of_periapsis_deg": float(
                        np.rad2deg(elements.argument_of_periapsis_rad)
                    ),
                    "true_anomaly_deg": float(np.rad2deg(elements.true_anomaly_rad)),
                },
                "orbit_count": orbit_count,
                "steps_per_orbit": steps_per_orbit,
                "period_s": period,
                "step_s": period / steps_per_orbit,
            },
            "metrics": metrics,
        },
        metrics,
    )


def _propagate_j2_case(inclination_deg: float) -> dict[str, Any]:
    elements = ClassicalOrbitalElements(
        semi_major_axis_km=7_000.0,
        eccentricity=0.001,
        inclination_rad=np.deg2rad(inclination_deg),
        raan_rad=np.deg2rad(20.0),
        argument_of_periapsis_rad=np.deg2rad(30.0),
        true_anomaly_rad=0.0,
    )
    duration_s = 14.0 * SECONDS_PER_DAY
    step_s = 60.0
    position, velocity = elements_to_cartesian(elements, EARTH.gravitational_parameter_km3_s2)
    trajectory = propagate_rk4(position, velocity, duration_s, step_s, _j2_model)
    raan = _raan_series(trajectory)
    fitted_rate, fitted_intercept = _fit_slope(trajectory.times_s, raan)
    analytic_rate = j2_secular_raan_rate(
        elements.semi_major_axis_km,
        elements.eccentricity,
        elements.inclination_rad,
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )
    analytic_raan = fitted_intercept + analytic_rate * trajectory.times_s
    fitted_raan = fitted_intercept + fitted_rate * trajectory.times_s
    relative_error = abs((fitted_rate - analytic_rate) / analytic_rate)
    return {
        "inclination_deg": inclination_deg,
        "trajectory": trajectory,
        "raan_rad": raan,
        "analytic_raan_rad": analytic_raan,
        "fitted_raan_rad": fitted_raan,
        "fitted_rate_rad_s": fitted_rate,
        "analytic_rate_rad_s": analytic_rate,
        "relative_rate_error": relative_error,
        "fitted_intercept_rad": fitted_intercept,
    }


def _validate_j2(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prograde = _propagate_j2_case(60.0)
    retrograde = _propagate_j2_case(120.0)
    cases = {"prograde": prograde, "retrograde": retrograde}
    metrics: list[dict[str, Any]] = []
    expected_signs = {"prograde": -1.0, "retrograde": 1.0}

    for name, case in cases.items():
        metrics.extend(
            [
                _metric(
                    "j2_raan_drift",
                    f"{name}_relative_rate_error",
                    case["relative_rate_error"],
                    "dimensionless",
                    0.02,
                    "<=",
                ),
                _metric(
                    "j2_raan_drift",
                    f"{name}_fitted_rate_sign",
                    case["fitted_rate_rad_s"],
                    "rad/s",
                    expected_signs[name],
                    "sign",
                ),
            ]
        )

    rows: list[dict[str, float | str]] = []
    prograde_trajectory = prograde["trajectory"]
    for index, time in enumerate(prograde_trajectory.times_s):
        row: dict[str, float | str] = {
            "time_s": float(time),
            "time_days": float(time / SECONDS_PER_DAY),
        }
        for name, case in cases.items():
            numeric_deg = float(np.rad2deg(case["raan_rad"][index]))
            theory_deg = float(np.rad2deg(case["analytic_raan_rad"][index]))
            fitted_deg = float(np.rad2deg(case["fitted_raan_rad"][index]))
            row[f"{name}_numeric_raan_deg"] = numeric_deg
            row[f"{name}_theory_raan_deg"] = theory_deg
            row[f"{name}_theory_residual_deg"] = numeric_deg - theory_deg
            row[f"{name}_fitted_raan_deg"] = fitted_deg
            row[f"{name}_fitted_residual_deg"] = numeric_deg - fitted_deg
        rows.append(row)
    _write_csv(output_dir / "j2_raan_drift.csv", list(rows[0]), rows)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    for name, color in (("prograde", BLUE), ("retrograde", ORANGE)):
        case = cases[name]
        times_days = case["trajectory"].times_s / SECONDS_PER_DAY
        numeric_degrees = np.rad2deg(case["raan_rad"] - case["raan_rad"][0])
        analytic_degrees = np.rad2deg(case["analytic_rate_rad_s"] * case["trajectory"].times_s)
        label = f"{name.capitalize()} ({case['inclination_deg']:.0f}°)"
        axes[0].plot(times_days, numeric_degrees, color=color, linewidth=1.5, label=f"{label} RK4")
        axes[0].plot(
            times_days,
            analytic_degrees,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{label} theory",
        )
        residual_degrees = np.rad2deg(case["raan_rad"] - case["fitted_raan_rad"])
        short_period_mask = case["trajectory"].times_s <= 12.0 * 3_600.0
        axes[1].plot(
            case["trajectory"].times_s[short_period_mask] / 3_600.0,
            residual_degrees[short_period_mask],
            color=color,
            linewidth=1.0,
            label=label,
        )

    axes[0].set_xlabel("Elapsed time (days)")
    axes[0].set_ylabel("RAAN change (deg)")
    axes[0].set_title("Numerical drift vs first-order theory")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Elapsed time (hours)")
    axes[1].set_ylabel("RAAN minus fitted line (deg)")
    axes[1].set_title("First 12 hours: residual about OLS fit")
    axes[1].legend()
    figure.suptitle("Fourteen-day J2 nodal precession", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(
        output_dir / "j2_raan_drift.png",
        dpi=200,
        metadata={"Creator": "orbital-mechanics-simulation"},
    )
    plt.close(figure)

    parameters: dict[str, Any] = {
        "semi_major_axis_km": 7_000.0,
        "eccentricity": 0.001,
        "duration_days": 14.0,
        "step_s": 60.0,
        "rate_formula": "-1.5 * J2 * n * (Re / p)^2 * cos(i)",
        "fit_method": "ordinary least squares over all unwrapped osculating RAAN samples",
    }
    case_evidence = {
        name: {
            "inclination_deg": case["inclination_deg"],
            "analytic_rate_deg_day": float(
                np.rad2deg(case["analytic_rate_rad_s"]) * SECONDS_PER_DAY
            ),
            "fitted_rate_deg_day": float(np.rad2deg(case["fitted_rate_rad_s"]) * SECONDS_PER_DAY),
            "relative_rate_error": case["relative_rate_error"],
        }
        for name, case in cases.items()
    }
    return {"parameters": parameters, "cases": case_evidence, "metrics": metrics}, metrics


def run_validation(output_dir: Path) -> dict[str, Any]:
    """Run all validation cases and write JSON, CSV, and PNG evidence."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _configure_plots()

    circular, circular_metrics = _validate_circular(destination)
    conservation, conservation_metrics = _validate_conservation(destination)
    j2, j2_metrics = _validate_j2(destination)
    metrics = circular_metrics + conservation_metrics + j2_metrics
    overall_pass = all(metric["passed"] for metric in metrics)

    summary: dict[str, Any] = {
        "schema_version": 2,
        "generator": {
            "source_files": list(GENERATOR_SOURCE_FILES),
            "source_sha256": generator_source_sha256(),
        },
        "units": {"distance": "km", "time": "s", "angle": "rad unless labeled otherwise"},
        "constants": {
            "earth_gravitational_parameter_km3_s2": EARTH.gravitational_parameter_km3_s2,
            "earth_equatorial_radius_km": EARTH.equatorial_radius_km,
            "earth_j2": EARTH.j2,
        },
        "integrator": "classical fixed-step fourth-order Runge-Kutta",
        "overall_pass": overall_pass,
        "scenarios": {
            "circular_two_body": circular,
            "two_body_conservation": conservation,
            "j2_raan_drift": j2,
        },
        "artifacts": [
            "validation_summary.json",
            "validation_metrics.csv",
            "circular_two_body.csv",
            "circular_two_body.png",
            "two_body_conservation.csv",
            "two_body_conservation.png",
            "j2_raan_drift.csv",
            "j2_raan_drift.png",
        ],
    }
    (destination / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        destination / "validation_metrics.csv",
        ["scenario", "name", "value", "unit", "threshold", "comparison", "passed"],
        metrics,
    )
    return summary
