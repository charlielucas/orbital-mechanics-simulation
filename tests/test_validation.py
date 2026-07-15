import csv
import json
from pathlib import Path

import numpy as np
import pytest

from orbital_mechanics.validation import run_validation


@pytest.fixture(scope="module")
def evidence_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("validation")
    summary = run_validation(output_dir)
    assert summary["overall_pass"] is True
    return output_dir


def test_all_scientific_acceptance_metrics_pass(evidence_dir: Path) -> None:
    summary = json.loads((evidence_dir / "validation_summary.json").read_text(encoding="utf-8"))
    metrics = [
        metric for scenario in summary["scenarios"].values() for metric in scenario["metrics"]
    ]
    by_name = {metric["name"]: metric for metric in metrics}

    assert all(metric["passed"] for metric in metrics)
    assert by_name["maximum_position_error"]["value"] <= 0.02
    assert by_name["maximum_velocity_error"]["value"] <= 2e-5
    assert by_name["maximum_relative_specific_energy_drift"]["value"] <= 1e-7
    assert by_name["maximum_relative_specific_angular_momentum_vector_drift"]["value"] <= 1e-7
    assert by_name["prograde_relative_rate_error"]["value"] <= 0.02
    assert by_name["retrograde_relative_rate_error"]["value"] <= 0.02
    assert by_name["prograde_fitted_rate_sign"]["value"] < 0.0
    assert by_name["retrograde_fitted_rate_sign"]["value"] > 0.0


def test_plot_evidence_is_backed_by_matching_csv_extrema(evidence_dir: Path) -> None:
    summary = json.loads((evidence_dir / "validation_summary.json").read_text(encoding="utf-8"))
    with (evidence_dir / "circular_two_body.csv").open(encoding="utf-8", newline="") as handle:
        circular_rows = list(csv.DictReader(handle))
    position_max = max(float(row["position_error_km"]) for row in circular_rows)
    velocity_max = max(float(row["velocity_error_km_s"]) for row in circular_rows)
    metrics = {
        metric["name"]: metric["value"]
        for metric in summary["scenarios"]["circular_two_body"]["metrics"]
    }

    assert position_max == pytest.approx(metrics["maximum_position_error"], rel=1e-15)
    assert velocity_max == pytest.approx(metrics["maximum_velocity_error"], rel=1e-15)


def test_j2_fitted_residual_contains_only_short_period_signal(evidence_dir: Path) -> None:
    with (evidence_dir / "j2_raan_drift.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([float(row["time_s"]) for row in rows])
    centered_times = times - float(np.mean(times))

    for direction in ("prograde", "retrograde"):
        residual = np.array([float(row[f"{direction}_fitted_residual_deg"]) for row in rows])
        residual_slope = float(
            np.dot(centered_times, residual) / np.dot(centered_times, centered_times)
        )
        assert abs(float(np.mean(residual))) < 1e-10
        assert abs(residual_slope) < 1e-16
        assert float(np.ptp(residual)) > 0.03


def test_expected_machine_readable_and_plot_artifacts_exist(evidence_dir: Path) -> None:
    expected = {
        "validation_summary.json",
        "validation_metrics.csv",
        "circular_two_body.csv",
        "circular_two_body.png",
        "two_body_conservation.csv",
        "two_body_conservation.png",
        "j2_raan_drift.csv",
        "j2_raan_drift.png",
    }
    assert expected == {path.name for path in evidence_dir.iterdir()}
    for png_name in ("circular_two_body.png", "two_body_conservation.png", "j2_raan_drift.png"):
        assert (evidence_dir / png_name).stat().st_size > 20_000
