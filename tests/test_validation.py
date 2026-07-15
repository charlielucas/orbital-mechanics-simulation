import csv
import json
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from orbital_mechanics.evidence import compare_validation_directories
from orbital_mechanics.validation import generator_source_sha256, run_validation


@pytest.fixture(scope="module")
def evidence_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("validation")
    summary = run_validation(output_dir)
    assert summary["overall_pass"] is True
    return output_dir


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metric_by_identity(
    metrics: list[dict[str, object]],
    scenario: str,
    name: str,
) -> dict[str, object]:
    return next(
        metric for metric in metrics if metric["scenario"] == scenario and metric["name"] == name
    )


def _offset_circular_metric(directory: Path, name: str, delta: float) -> None:
    metrics_path = directory / "validation_metrics.csv"
    fieldnames, metrics = _read_csv(metrics_path)
    csv_metric = _metric_by_identity(metrics, "circular_two_body", name)
    csv_metric["value"] = str(float(csv_metric["value"]) + delta)
    _write_csv(metrics_path, fieldnames, metrics)

    summary_path = directory / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_metric = _metric_by_identity(
        summary["scenarios"]["circular_two_body"]["metrics"],
        "circular_two_body",
        name,
    )
    summary_metric["value"] = float(summary_metric["value"]) + delta
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def test_checked_in_evidence_matches_regeneration(evidence_dir: Path) -> None:
    reference_dir = Path("artifacts/validation")
    assert compare_validation_directories(reference_dir, evidence_dir) == []


def test_summary_binds_evidence_to_generator_source(evidence_dir: Path) -> None:
    summary = json.loads((evidence_dir / "validation_summary.json").read_text(encoding="utf-8"))

    assert summary["generator"]["source_sha256"] == generator_source_sha256()


def test_evidence_comparison_rejects_material_numeric_change(
    evidence_dir: Path,
    tmp_path: Path,
) -> None:
    changed_dir = tmp_path / "changed"
    shutil.copytree(evidence_dir, changed_dir)
    summary_path = changed_dir / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["scenarios"]["circular_two_body"]["metrics"][0]["value"] = 1.0
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    errors = compare_validation_directories(Path("artifacts/validation"), changed_dir)

    assert any("metrics[0].value" in error for error in errors)


def test_evidence_comparison_allows_micrometer_position_deltas(tmp_path: Path) -> None:
    reference_dir = Path("artifacts/validation")
    changed_dir = tmp_path / "position-error-roundoff"
    shutil.copytree(reference_dir, changed_dir)
    csv_path = changed_dir / "circular_two_body.csv"
    fieldnames, rows = _read_csv(csv_path)
    position_fields = (
        "x_numeric_km",
        "y_numeric_km",
        "x_analytic_km",
        "y_analytic_km",
        "position_error_km",
    )
    for field in position_fields:
        near_zero_row = min(rows, key=lambda row: abs(float(row[field])))
        near_zero_row[field] = str(float(near_zero_row[field]) + 5e-10)
    _write_csv(csv_path, fieldnames, rows)
    _offset_circular_metric(changed_dir, "maximum_position_error", 5e-10)

    assert compare_validation_directories(reference_dir, changed_dir) == []


def test_evidence_comparison_allows_sub_microarcsecond_j2_residual_deltas(
    tmp_path: Path,
) -> None:
    reference_dir = Path("artifacts/validation")
    changed_dir = tmp_path / "j2-residual-roundoff"
    shutil.copytree(reference_dir, changed_dir)
    csv_path = changed_dir / "j2_raan_drift.csv"
    fieldnames, rows = _read_csv(csv_path)
    residual_fields = (
        "prograde_theory_residual_deg",
        "prograde_fitted_residual_deg",
        "retrograde_theory_residual_deg",
        "retrograde_fitted_residual_deg",
    )
    for field in residual_fields:
        near_zero_row = min(rows, key=lambda row: abs(float(row[field])))
        near_zero_row[field] = str(float(near_zero_row[field]) + 5e-11)
    _write_csv(csv_path, fieldnames, rows)

    assert compare_validation_directories(reference_dir, changed_dir) == []


@pytest.mark.parametrize(
    ("csv_name", "field", "delta"),
    [
        ("circular_two_body.csv", "x_numeric_km", 1.1e-9),
        ("j2_raan_drift.csv", "prograde_fitted_residual_deg", 1.1e-10),
    ],
)
def test_evidence_comparison_rejects_roundoff_above_scoped_tolerance(
    csv_name: str,
    field: str,
    delta: float,
    tmp_path: Path,
) -> None:
    reference_dir = Path("artifacts/validation")
    changed_dir = tmp_path / f"above-tolerance-{field}"
    shutil.copytree(reference_dir, changed_dir)
    csv_path = changed_dir / csv_name
    fieldnames, rows = _read_csv(csv_path)
    near_zero_row = min(rows, key=lambda row: abs(float(row[field])))
    near_zero_row[field] = str(float(near_zero_row[field]) + delta)
    _write_csv(csv_path, fieldnames, rows)

    errors = compare_validation_directories(reference_dir, changed_dir)

    assert any(csv_name in error and f":{field}:" in error for error in errors)


def test_evidence_comparison_rejects_position_metric_delta_above_tolerance(
    tmp_path: Path,
) -> None:
    reference_dir = Path("artifacts/validation")
    changed_dir = tmp_path / "position-metric-above-tolerance"
    shutil.copytree(reference_dir, changed_dir)
    _offset_circular_metric(changed_dir, "maximum_position_error", 1.1e-9)

    errors = compare_validation_directories(reference_dir, changed_dir)

    assert any("validation_metrics.csv" in error and ":value:" in error for error in errors)
    assert any("validation_summary.json" in error and ".value:" in error for error in errors)


def test_position_metric_tolerance_does_not_apply_to_velocity_metric(tmp_path: Path) -> None:
    reference_dir = Path("artifacts/validation")
    changed_dir = tmp_path / "velocity-metric-roundoff"
    shutil.copytree(reference_dir, changed_dir)
    _offset_circular_metric(changed_dir, "maximum_velocity_error", 5e-10)

    errors = compare_validation_directories(reference_dir, changed_dir)

    assert any("validation_metrics.csv" in error and ":value:" in error for error in errors)
    assert any("validation_summary.json" in error and ".value:" in error for error in errors)


def test_evidence_comparison_rejects_removed_conservation_series(
    evidence_dir: Path,
    tmp_path: Path,
) -> None:
    changed_dir = tmp_path / "removed-conservation-series"
    shutil.copytree(evidence_dir, changed_dir)
    csv_path = changed_dir / "two_body_conservation.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    drift_fields = (
        "relative_energy_drift",
        "relative_angular_momentum_magnitude_drift",
        "relative_angular_momentum_vector_drift",
    )
    for row in rows:
        for field in drift_fields:
            row[field] = "0.0"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    errors = compare_validation_directories(Path("artifacts/validation"), changed_dir)

    assert any("relative_energy_drift" in error for error in errors)


def test_evidence_comparison_rejects_blank_plot(evidence_dir: Path, tmp_path: Path) -> None:
    changed_dir = tmp_path / "blank-plot"
    shutil.copytree(evidence_dir, changed_dir)
    plot_path = changed_dir / "circular_two_body.png"
    image = mpimg.imread(plot_path)
    mpimg.imsave(plot_path, np.ones_like(image))

    errors = compare_validation_directories(Path("artifacts/validation"), changed_dir)

    assert any("circular_two_body.png" in error for error in errors)


def test_evidence_comparison_rejects_low_resolution_plot(
    evidence_dir: Path,
    tmp_path: Path,
) -> None:
    changed_dir = tmp_path / "low-resolution-plot"
    shutil.copytree(evidence_dir, changed_dir)
    plot_path = changed_dir / "circular_two_body.png"
    image = mpimg.imread(plot_path)
    mpimg.imsave(plot_path, image[::2, ::2])

    errors = compare_validation_directories(Path("artifacts/validation"), changed_dir)

    assert any("plot dimensions differ" in error for error in errors)


@pytest.mark.parametrize(
    ("plot_name", "series_color", "region"),
    [
        ("circular_two_body.png", "#0072B2", (0.0, 1.0, 0.0, 0.5)),
        ("circular_two_body.png", "#E69F00", (0.0, 1.0, 0.0, 0.5)),
        ("circular_two_body.png", "#009E73", (0.0, 1.0, 0.0, 0.5)),
        ("circular_two_body.png", "#0072B2", (0.0, 1.0, 0.5, 1.0)),
        ("circular_two_body.png", "#D55E00", (0.0, 1.0, 0.5, 1.0)),
        ("two_body_conservation.png", "#0072B2", (0.0, 0.5, 0.0, 1.0)),
        ("two_body_conservation.png", "#009E73", (0.5, 1.0, 0.0, 1.0)),
        ("j2_raan_drift.png", "#0072B2", (0.0, 1.0, 0.0, 0.5)),
        ("j2_raan_drift.png", "#E69F00", (0.0, 1.0, 0.0, 0.5)),
        ("j2_raan_drift.png", "#0072B2", (0.0, 1.0, 0.5, 1.0)),
        ("j2_raan_drift.png", "#E69F00", (0.0, 1.0, 0.5, 1.0)),
    ],
)
def test_evidence_comparison_rejects_missing_plot_series(
    evidence_dir: Path,
    tmp_path: Path,
    plot_name: str,
    series_color: str,
    region: tuple[float, float, float, float],
) -> None:
    changed_dir = tmp_path / "missing-series"
    shutil.copytree(evidence_dir, changed_dir)
    plot_path = changed_dir / plot_name
    image = mpimg.imread(plot_path)
    target = (
        np.array(
            [int(series_color[index : index + 2], 16) for index in (1, 3, 5)],
            dtype=np.float64,
        )
        / 255.0
    )
    y0, y1, x0, x1 = region
    image_region = image[
        round(y0 * image.shape[0]) : round(y1 * image.shape[0]),
        round(x0 * image.shape[1]) : round(x1 * image.shape[1]),
        :3,
    ]
    matching_color = np.max(np.abs(image_region - target), axis=2) <= 0.05
    image_region[matching_color] = 1.0
    mpimg.imsave(plot_path, image)

    errors = compare_validation_directories(Path("artifacts/validation"), changed_dir)

    assert any("series coverage differs" in error for error in errors)
