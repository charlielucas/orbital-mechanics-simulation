"""Streamlit interface for the orbital mechanics explorer."""

from __future__ import annotations

import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from orbital_mechanics.constants import EARTH
from orbital_mechanics.elements import ClassicalOrbitalElements
from orbital_mechanics.explorer import (
    GUIDED_SCENARIOS,
    GUIDED_SCENARIOS_BY_KEY,
    J2_RATE_ERROR_LIMIT,
    SECONDS_PER_DAY,
    TWO_BODY_DRIFT_LIMIT,
    ExplorerResult,
    SimulationRequest,
    chart_indices,
    keplerian_period_s,
    run_explorer,
    trajectory_csv,
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#4C566A"
SOURCE_URL = "https://github.com/charlielucas/orbital-mechanics-simulation"
CUSTOM_SCENARIO_KEY = "custom"


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_run(request: SimulationRequest) -> ExplorerResult:
    return run_explorer(request)


def _force_model_name(force_model: str) -> str:
    return "Two-body" if force_model == "two_body" else "Two-body plus J2"


def _duration_label(duration_s: float) -> str:
    if duration_s >= SECONDS_PER_DAY:
        return f"{duration_s / SECONDS_PER_DAY:.1f} days"
    return f"{duration_s / 3_600.0:.1f} hours"


def _custom_request() -> SimulationRequest:
    force_label = st.sidebar.radio(
        "Force model",
        ["Two-body", "Two-body plus J2"],
        help="J2 adds the first-order effect of Earth's oblateness.",
    )
    force_model = "two_body" if force_label == "Two-body" else "j2"
    semi_major_axis_km = st.sidebar.slider(
        "Semi-major axis (km)",
        min_value=6_600,
        max_value=30_000,
        value=7_000,
        step=100,
    )
    eccentricity = st.sidebar.slider(
        "Eccentricity",
        min_value=0.0,
        max_value=0.6,
        value=0.01,
        step=0.01,
        format="%.2f",
    )
    inclination_deg = st.sidebar.slider(
        "Inclination (degrees)",
        min_value=0.0,
        max_value=180.0,
        value=50.0,
        step=0.5,
    )

    with st.sidebar.expander("Orientation"):
        raan_deg = st.slider(
            "RAAN (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=20.0,
            step=1.0,
        )
        argument_of_periapsis_deg = st.slider(
            "Argument of periapsis (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=30.0,
            step=1.0,
        )
        true_anomaly_deg = st.slider(
            "True anomaly (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=0.0,
            step=1.0,
        )

    elements = ClassicalOrbitalElements(
        semi_major_axis_km=float(semi_major_axis_km),
        eccentricity=float(eccentricity),
        inclination_rad=np.deg2rad(inclination_deg),
        raan_rad=np.deg2rad(raan_deg),
        argument_of_periapsis_rad=np.deg2rad(argument_of_periapsis_deg),
        true_anomaly_rad=np.deg2rad(true_anomaly_deg),
    )
    period_s = keplerian_period_s(semi_major_axis_km)

    if force_model == "two_body":
        orbit_count = st.sidebar.slider(
            "Propagation length (orbits)",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
        )
        duration_s = orbit_count * period_s
        step_count = orbit_count * 600
    else:
        duration_days = st.sidebar.slider(
            "Propagation length (days)",
            min_value=1,
            max_value=14,
            value=7,
            step=1,
        )
        duration_s = duration_days * SECONDS_PER_DAY
        step_count = duration_days * 720

    return SimulationRequest(
        elements=elements,
        force_model=force_model,
        duration_s=duration_s,
        step_count=step_count,
    )


def _orbit_figure(result: ExplorerResult) -> plt.Figure:
    positions = result.trajectory.positions_km
    indices = chart_indices(len(positions))
    sampled = positions[indices]

    figure = plt.figure(figsize=(8.0, 7.0))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(
        sampled[:, 0],
        sampled[:, 1],
        sampled[:, 2],
        color=BLUE,
        linewidth=1.6,
        label="Propagated state",
    )
    axis.scatter(
        [sampled[0, 0]],
        [sampled[0, 1]],
        [sampled[0, 2]],
        color=GREEN,
        s=42,
        label="Start",
    )
    axis.scatter(
        [sampled[-1, 0]],
        [sampled[-1, 1]],
        [sampled[-1, 2]],
        color=ORANGE,
        s=42,
        label="End",
    )

    longitude = np.linspace(0.0, 2.0 * np.pi, 48)
    latitude = np.linspace(0.0, np.pi, 24)
    earth_x = EARTH.equatorial_radius_km * np.outer(np.cos(longitude), np.sin(latitude))
    earth_y = EARTH.equatorial_radius_km * np.outer(np.sin(longitude), np.sin(latitude))
    earth_z = EARTH.equatorial_radius_km * np.outer(
        np.ones_like(longitude),
        np.cos(latitude),
    )
    axis.plot_surface(
        earth_x,
        earth_y,
        earth_z,
        color="#B8D8E8",
        alpha=0.28,
        linewidth=0.0,
    )

    extent = max(
        EARTH.equatorial_radius_km,
        float(np.max(np.abs(sampled))),
    )
    limit = 1.08 * extent
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xlabel("ECI x (km)")
    axis.set_ylabel("ECI y (km)")
    axis.set_zlabel("ECI z (km)")
    axis.set_title("Earth-centered inertial trajectory")
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


def _diagnostics_figure(result: ExplorerResult) -> plt.Figure:
    indices = chart_indices(len(result.trajectory.times_s))
    times = result.trajectory.times_s[indices]
    elapsed_orbits = times / result.request.orbital_period_s
    if result.request.force_model == "j2":
        altitude_time = times / SECONDS_PER_DAY
        altitude_time_label = "Elapsed days"
    else:
        altitude_time = elapsed_orbits
        altitude_time_label = "Elapsed orbits"

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0))
    sampled_altitude = result.altitude_km[indices]
    altitude_span_km = float(np.ptp(sampled_altitude))
    if altitude_span_km < 1.0:
        altitude_values = 1_000.0 * (sampled_altitude - sampled_altitude[0])
        altitude_label = "Change from start (m)"
        altitude_title = "Numerical altitude change from start"
    else:
        altitude_values = sampled_altitude
        altitude_label = "Altitude (km)"
        altitude_title = "Altitude above the equatorial radius"
    axes[0].plot(
        altitude_time,
        altitude_values,
        color=BLUE,
        linewidth=1.4,
    )
    axes[0].set_xlabel(altitude_time_label)
    axes[0].set_ylabel(altitude_label)
    axes[0].set_title(altitude_title)
    axes[0].grid(True, color="#D9DEE7", linewidth=0.7)

    if result.relative_energy_drift is not None:
        axes[1].semilogy(
            elapsed_orbits,
            np.maximum(np.abs(result.relative_energy_drift[indices]), 1e-16),
            color=BLUE,
            linewidth=1.4,
            label="Specific energy drift",
        )
        axes[1].semilogy(
            elapsed_orbits,
            np.maximum(
                result.relative_angular_momentum_vector_drift[indices],
                1e-16,
            ),
            color=GREEN,
            linewidth=1.4,
            label="Angular momentum vector drift",
        )
        axes[1].axhline(
            TWO_BODY_DRIFT_LIMIT,
            color=GRAY,
            linestyle=":",
            linewidth=1.2,
            label="Reference limit",
        )
        axes[1].set_xlabel("Elapsed orbits")
        axes[1].set_ylabel("Relative drift")
        axes[1].set_title("Two-body numerical conservation")
        axes[1].legend()
        axes[1].grid(True, color="#D9DEE7", linewidth=0.7)
    elif result.raan_change_deg is not None:
        times_days = times / SECONDS_PER_DAY
        axes[1].plot(
            times_days,
            result.raan_change_deg[indices],
            color=BLUE,
            linewidth=1.5,
            label="RK4 plus J2",
        )
        axes[1].plot(
            times_days,
            result.theoretical_raan_change_deg[indices],
            color=ORANGE,
            linewidth=1.8,
            linestyle="--",
            label="First-order theory",
        )
        axes[1].set_xlabel("Elapsed days")
        axes[1].set_ylabel("RAAN change (degrees)")
        axes[1].set_title("Nodal precession")
        axes[1].legend()
        axes[1].grid(True, color="#D9DEE7", linewidth=0.7)
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "RAAN is undefined for an equatorial orbit.",
            ha="center",
            va="center",
        )

    figure.tight_layout()
    return figure


def _preview_rows(result: ExplorerResult) -> list[dict[str, float]]:
    indices = chart_indices(len(result.trajectory.times_s), max_points=12)
    rows: list[dict[str, float]] = []
    for index in indices:
        position = result.trajectory.positions_km[index]
        rows.append(
            {
                "time (hours)": round(float(result.trajectory.times_s[index] / 3_600.0), 3),
                "x (km)": round(float(position[0]), 3),
                "y (km)": round(float(position[1]), 3),
                "z (km)": round(float(position[2]), 3),
                "altitude (km)": round(float(result.altitude_km[index]), 3),
            }
        )
    return rows


def _render_diagnostic_summary(result: ExplorerResult, *, is_guided: bool) -> None:
    if result.relative_energy_drift is not None:
        energy_drift = result.max_relative_energy_drift
        momentum_drift = result.max_relative_angular_momentum_vector_drift
        columns = st.columns(2)
        columns[0].metric("Maximum relative energy drift", f"{energy_drift:.2e}")
        columns[1].metric("Maximum angular momentum drift", f"{momentum_drift:.2e}")
        passed = energy_drift <= TWO_BODY_DRIFT_LIMIT and momentum_drift <= TWO_BODY_DRIFT_LIMIT
        if is_guided and passed:
            st.success("Both measured drifts stay below the 1e-7 reference limit.")
        elif is_guided:
            st.warning("At least one measured drift exceeds the 1e-7 reference limit.")
        else:
            st.info(
                "These measurements describe the custom run. They are not a checked-in "
                "acceptance case."
            )
        return

    if result.raan_change_deg is None:
        st.info("RAAN is undefined for this equatorial orbit, so no nodal rate is reported.")
        return

    fitted_rate = result.fitted_raan_rate_deg_day
    theoretical_rate = result.theoretical_raan_rate_deg_day
    if fitted_rate is None or theoretical_rate is None:
        st.warning("Nodal-rate diagnostics are unavailable for this run.")
        return

    columns = st.columns(2)
    columns[0].metric("Fitted RAAN rate", f"{fitted_rate:.4f}°/day")
    columns[1].metric(
        "First-order theory",
        f"{theoretical_rate:.4f}°/day",
    )
    if result.relative_raan_rate_error is None:
        st.info(
            "The first-order nodal rate is effectively zero for this near-polar orbit, "
            "so relative rate error is not reported."
        )
        return

    st.metric("Relative rate error", f"{100.0 * result.relative_raan_rate_error:.2f}%")
    if is_guided and result.relative_raan_rate_error <= J2_RATE_ERROR_LIMIT:
        st.success("The fitted nodal rate is within 2% of first-order J2 theory.")
    elif is_guided:
        st.warning("The fitted nodal rate is more than 2% from first-order J2 theory.")
    else:
        st.info(
            "The rate comparison describes the custom run. Short runs and near-polar "
            "configurations can make relative error less informative."
        )


def main() -> None:
    st.set_page_config(
        page_title="Orbital Mechanics Explorer",
        layout="wide",
    )
    st.title("Orbital Mechanics Explorer")
    st.write(
        "Change an orbit, propagate it with fixed-step RK4, and inspect the numerical "
        "checks behind the result."
    )
    st.caption(
        "The models are implemented directly in NumPy. This is an educational numerical "
        "tool, not mission design software."
    )

    scenario_labels = {scenario.key: scenario.name for scenario in GUIDED_SCENARIOS} | {
        CUSTOM_SCENARIO_KEY: "Custom orbit"
    }
    scenario_key = st.sidebar.selectbox(
        "Scenario",
        list(scenario_labels),
        format_func=scenario_labels.get,
    )

    scenario = GUIDED_SCENARIOS_BY_KEY.get(scenario_key)
    if scenario is None:
        st.sidebar.caption(
            "Custom runs use 600 steps per orbit for two-body propagation and a "
            "120-second step for J2 propagation."
        )
        try:
            request = _custom_request()
        except ValueError as error:
            st.error(str(error))
            st.stop()
        is_guided = False
        scenario_name = "Custom orbit"
        summary = "Explore a bounded configuration with the same deterministic propagator."
        question = "What changes when you adjust the orbit or force model?"
    else:
        request = scenario.request
        is_guided = True
        scenario_name = scenario.name
        summary = scenario.summary
        question = scenario.question

    st.sidebar.markdown(f"[View the source on GitHub]({SOURCE_URL})")

    st.subheader(scenario_name)
    st.write(summary)
    st.markdown(f"**Question:** {question}")

    with st.spinner("Propagating the orbit and calculating diagnostics..."):
        result = _cached_run(request)

    metric_columns = st.columns(2)
    metric_columns[0].metric("Force model", _force_model_name(request.force_model))
    metric_columns[1].metric("Orbital period", f"{request.orbital_period_s / 60.0:.1f} min")
    metric_columns = st.columns(2)
    metric_columns[0].metric("Duration", _duration_label(request.duration_s))
    metric_columns[1].metric("Fixed step", f"{request.step_s:.1f} s")

    detail_columns = st.columns(2)
    detail_columns[0].metric("Minimum altitude", f"{np.min(result.altitude_km):,.1f} km")
    detail_columns[1].metric("Maximum altitude", f"{np.max(result.altitude_km):,.1f} km")
    detail_columns = st.columns(2)
    detail_columns[0].metric("Propagation steps", f"{request.step_count:,}")
    detail_columns[1].metric(
        "Inclination",
        f"{np.rad2deg(request.elements.inclination_rad):.1f}°",
    )

    _render_diagnostic_summary(result, is_guided=is_guided)

    orbit_tab, diagnostics_tab, data_tab = st.tabs(["Orbit", "Diagnostics", "Trajectory data"])
    with orbit_tab:
        orbit_figure = _orbit_figure(result)
        st.pyplot(orbit_figure, width="stretch")
        plt.close(orbit_figure)
    with diagnostics_tab:
        diagnostics_figure = _diagnostics_figure(result)
        st.pyplot(diagnostics_figure, width="stretch")
        plt.close(diagnostics_figure)
        if result.relative_energy_drift is not None:
            st.write(
                "For the two-body model, specific mechanical energy and the angular "
                "momentum vector should remain constant. The plotted drift measures "
                "numerical integration error."
            )
        elif result.raan_change_deg is not None:
            st.write(
                "For the J2 model, the orbital plane precesses. The comparison fits the "
                "unwrapped numerical RAAN series and checks it against the first-order "
                "secular rate."
            )
        else:
            st.write(
                "RAAN is undefined for an equatorial orbit. The trajectory still includes "
                "the J2 acceleration, but no nodal-rate comparison is shown."
            )
    with data_tab:
        st.dataframe(_preview_rows(result), width="stretch", hide_index=True)
        st.download_button(
            "Download complete trajectory",
            data=trajectory_csv(result),
            file_name=f"orbital_explorer_{scenario_key}.csv",
            mime="text/csv",
        )

    with st.expander("What this app does and does not show"):
        st.markdown(
            """
- The state conversion, force models, and fixed-step RK4 propagator come from this repo.
- Guided cases use explicit numerical checks instead of judging a plot by eye.
- The Earth model includes point-mass gravity and an optional first-order J2 term.
- The model omits drag, higher-order gravity, third bodies, radiation pressure, and maneuvers.
- Every trajectory is generated from the controls. No mission or employer data is used.
"""
        )


if __name__ == "__main__":
    main()
