import csv
import io

import numpy as np
import pytest

from orbital_mechanics.constants import EARTH
from orbital_mechanics.elements import ClassicalOrbitalElements
from orbital_mechanics.explorer import (
    GUIDED_SCENARIOS,
    J2_RATE_ERROR_LIMIT,
    MAX_PROPAGATION_STEPS,
    MINIMUM_PERIGEE_ALTITUDE_KM,
    TWO_BODY_DRIFT_LIMIT,
    SimulationRequest,
    chart_indices,
    keplerian_period_s,
    run_explorer,
    trajectory_csv,
)


def test_guided_scenarios_are_unique_physical_and_bounded() -> None:
    assert len({scenario.key for scenario in GUIDED_SCENARIOS}) == len(GUIDED_SCENARIOS)

    for scenario in GUIDED_SCENARIOS:
        request = scenario.request
        perigee_radius = request.elements.semi_major_axis_km * (1.0 - request.elements.eccentricity)

        assert perigee_radius - EARTH.equatorial_radius_km >= MINIMUM_PERIGEE_ALTITUDE_KM
        assert 1 <= request.step_count <= MAX_PROPAGATION_STEPS
        assert request.step_s > 0.0


@pytest.mark.parametrize(
    "scenario",
    [scenario for scenario in GUIDED_SCENARIOS if scenario.request.force_model == "two_body"],
    ids=lambda scenario: scenario.key,
)
def test_guided_two_body_scenarios_meet_conservation_limits(scenario) -> None:
    result = run_explorer(scenario.request)

    assert result.max_relative_energy_drift is not None
    assert result.max_relative_energy_drift <= TWO_BODY_DRIFT_LIMIT
    assert result.max_relative_angular_momentum_vector_drift is not None
    assert result.max_relative_angular_momentum_vector_drift <= TWO_BODY_DRIFT_LIMIT
    assert result.raan_change_deg is None


def test_guided_j2_scenario_matches_first_order_rate() -> None:
    scenario = next(
        scenario for scenario in GUIDED_SCENARIOS if scenario.request.force_model == "j2"
    )
    result = run_explorer(scenario.request)

    assert result.fitted_raan_rate_deg_day is not None
    assert result.fitted_raan_rate_deg_day > 0.0
    assert result.theoretical_raan_rate_deg_day is not None
    assert result.relative_raan_rate_error is not None
    assert result.relative_raan_rate_error <= J2_RATE_ERROR_LIMIT
    assert result.raan_change_deg is not None
    assert result.theoretical_raan_change_deg is not None


def test_equatorial_j2_case_does_not_report_undefined_raan() -> None:
    elements = ClassicalOrbitalElements(
        semi_major_axis_km=7_000.0,
        eccentricity=0.001,
        inclination_rad=0.0,
        raan_rad=0.0,
        argument_of_periapsis_rad=0.0,
        true_anomaly_rad=0.0,
    )
    request = SimulationRequest(
        elements=elements,
        force_model="j2",
        duration_s=5_800.0,
        step_count=290,
    )

    result = run_explorer(request)

    assert result.raan_change_deg is None
    assert result.fitted_raan_rate_deg_day is None
    assert result.relative_raan_rate_error is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"force_model": "unknown"}, "force model"),
        ({"duration_s": 0.0}, "duration"),
        ({"step_count": 0}, "step count"),
        ({"step_count": MAX_PROPAGATION_STEPS + 1}, "step count"),
    ],
)
def test_invalid_simulation_requests_are_rejected(changes: dict, message: str) -> None:
    values = {
        "elements": ClassicalOrbitalElements(
            semi_major_axis_km=7_000.0,
            eccentricity=0.001,
            inclination_rad=np.deg2rad(45.0),
            raan_rad=0.0,
            argument_of_periapsis_rad=0.0,
            true_anomaly_rad=0.0,
        ),
        "force_model": "two_body",
        "duration_s": 5_800.0,
        "step_count": 290,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        SimulationRequest(**values)


def test_request_rejects_orbit_that_intersects_earth() -> None:
    elements = ClassicalOrbitalElements(
        semi_major_axis_km=7_000.0,
        eccentricity=0.2,
        inclination_rad=np.deg2rad(45.0),
        raan_rad=0.0,
        argument_of_periapsis_rad=0.0,
        true_anomaly_rad=0.0,
    )

    with pytest.raises(ValueError, match="perigee altitude"):
        SimulationRequest(
            elements=elements,
            force_model="two_body",
            duration_s=5_800.0,
            step_count=290,
        )


def test_keplerian_period_matches_circular_reference() -> None:
    assert keplerian_period_s(7_000.0) == pytest.approx(5_828.516637686015)

    with pytest.raises(ValueError, match="semi-major axis"):
        keplerian_period_s(0.0)


def test_chart_indices_retain_endpoints_and_respect_limit() -> None:
    indices = chart_indices(10_001, max_points=250)

    assert indices[0] == 0
    assert indices[-1] == 10_000
    assert len(indices) <= 250
    assert np.all(np.diff(indices) > 0)
    np.testing.assert_array_equal(chart_indices(3, max_points=10), [0, 1, 2])


def test_trajectory_csv_contains_every_state_and_model_diagnostics() -> None:
    scenario = GUIDED_SCENARIOS[0]
    result = run_explorer(scenario.request)

    rows = list(csv.DictReader(io.StringIO(trajectory_csv(result))))

    assert len(rows) == scenario.request.step_count + 1
    assert rows[0]["time_s"] == "0.0"
    assert float(rows[-1]["time_s"]) == pytest.approx(scenario.request.duration_s)
    assert "relative_energy_drift" in rows[0]
    assert "relative_angular_momentum_vector_drift" in rows[0]
    assert "raan_change_deg" not in rows[0]


def test_j2_trajectory_csv_contains_rate_series() -> None:
    scenario = next(
        scenario for scenario in GUIDED_SCENARIOS if scenario.request.force_model == "j2"
    )
    rows = list(csv.DictReader(io.StringIO(trajectory_csv(run_explorer(scenario.request)))))

    assert len(rows) == scenario.request.step_count + 1
    assert "raan_change_deg" in rows[0]
    assert "theoretical_raan_change_deg" in rows[0]
    assert "relative_energy_drift" not in rows[0]
