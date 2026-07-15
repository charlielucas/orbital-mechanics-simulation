"""Cross-platform validation-evidence comparison helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
from numpy.typing import NDArray

RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 1e-12
POSITION_ABSOLUTE_TOLERANCE_KM = 1e-9
RAAN_RESIDUAL_ABSOLUTE_TOLERANCE_DEG = 1e-10
CSV_ABSOLUTE_TOLERANCES = {
    ("circular_two_body.csv", "x_numeric_km"): POSITION_ABSOLUTE_TOLERANCE_KM,
    ("circular_two_body.csv", "y_numeric_km"): POSITION_ABSOLUTE_TOLERANCE_KM,
    ("circular_two_body.csv", "x_analytic_km"): POSITION_ABSOLUTE_TOLERANCE_KM,
    ("circular_two_body.csv", "y_analytic_km"): POSITION_ABSOLUTE_TOLERANCE_KM,
    ("circular_two_body.csv", "position_error_km"): POSITION_ABSOLUTE_TOLERANCE_KM,
    ("j2_raan_drift.csv", "prograde_theory_residual_deg"): (RAAN_RESIDUAL_ABSOLUTE_TOLERANCE_DEG),
    ("j2_raan_drift.csv", "prograde_fitted_residual_deg"): (RAAN_RESIDUAL_ABSOLUTE_TOLERANCE_DEG),
    ("j2_raan_drift.csv", "retrograde_theory_residual_deg"): (RAAN_RESIDUAL_ABSOLUTE_TOLERANCE_DEG),
    ("j2_raan_drift.csv", "retrograde_fitted_residual_deg"): (RAAN_RESIDUAL_ABSOLUTE_TOLERANCE_DEG),
}
METRIC_ABSOLUTE_TOLERANCES = {
    ("circular_two_body", "maximum_position_error"): POSITION_ABSOLUTE_TOLERANCE_KM,
}
MAX_ASPECT_RATIO_DELTA = 0.01
MAX_DIMENSION_DELTA = 0.02
MAX_PNG_MEAN_ABSOLUTE_ERROR = 0.03
MAX_PNG_CHANGED_FRACTION = 0.15
MAX_PNG_INK_RELATIVE_DELTA = 0.20
MAX_SERIES_COLOR_RELATIVE_DELTA = 0.35
PNG_CHANGE_THRESHOLD = 0.10
PNG_INK_THRESHOLD = 0.05
PNG_SAMPLE_SIZE = 256
SERIES_COLOR_TOLERANCE = 0.05
MIN_SERIES_COLOR_COVERAGE = 1e-4
SERIES_COLORS = {
    "blue": np.array([0x00, 0x72, 0xB2], dtype=np.float64) / 255.0,
    "orange": np.array([0xE6, 0x9F, 0x00], dtype=np.float64) / 255.0,
    "green": np.array([0x00, 0x9E, 0x73], dtype=np.float64) / 255.0,
    "vermillion": np.array([0xD5, 0x5E, 0x00], dtype=np.float64) / 255.0,
}
SERIES_REGIONS = {
    "circular_two_body.png": (
        ("trajectory panel", 0.0, 1.0, 0.0, 0.5, ("blue", "orange", "green")),
        ("error panel", 0.0, 1.0, 0.5, 1.0, ("blue", "vermillion")),
    ),
    "two_body_conservation.png": (
        ("energy panel", 0.0, 0.5, 0.0, 1.0, ("blue",)),
        ("momentum panel", 0.5, 1.0, 0.0, 1.0, ("green",)),
    ),
    "j2_raan_drift.png": (
        ("drift panel", 0.0, 1.0, 0.0, 0.5, ("blue", "orange")),
        ("residual panel", 0.0, 1.0, 0.5, 1.0, ("blue", "orange")),
    ),
}


def _compare_json(
    reference: Any,
    candidate: Any,
    location: str,
    errors: list[str],
    *,
    absolute_tolerance: float = ABSOLUTE_TOLERANCE,
) -> None:
    if isinstance(reference, bool) or isinstance(candidate, bool):
        if type(reference) is not type(candidate) or reference != candidate:
            errors.append(f"{location}: {reference!r} != {candidate!r}")
        return
    if isinstance(reference, int) or isinstance(candidate, int):
        if type(reference) is not type(candidate) or reference != candidate:
            errors.append(f"{location}: {reference!r} != {candidate!r}")
        return
    if isinstance(reference, float) or isinstance(candidate, float):
        if not isinstance(reference, (int, float)) or not isinstance(candidate, (int, float)):
            errors.append(f"{location}: numeric type mismatch")
            return
        if not math.isfinite(float(reference)) or not math.isfinite(float(candidate)):
            errors.append(f"{location}: evidence must be finite")
            return
        if not math.isclose(
            float(reference),
            float(candidate),
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=absolute_tolerance,
        ):
            errors.append(f"{location}: {reference!r} != {candidate!r} within tolerance")
        return
    if isinstance(reference, dict) or isinstance(candidate, dict):
        if not isinstance(reference, dict) or not isinstance(candidate, dict):
            errors.append(f"{location}: object type mismatch")
            return
        if reference.keys() != candidate.keys():
            errors.append(f"{location}: object keys differ")
            return
        for key in reference:
            child_tolerance = (
                _metric_absolute_tolerance(reference, candidate)
                if key == "value"
                else ABSOLUTE_TOLERANCE
            )
            _compare_json(
                reference[key],
                candidate[key],
                f"{location}.{key}",
                errors,
                absolute_tolerance=child_tolerance,
            )
        return
    if isinstance(reference, list) or isinstance(candidate, list):
        if not isinstance(reference, list) or not isinstance(candidate, list):
            errors.append(f"{location}: array type mismatch")
            return
        if len(reference) != len(candidate):
            errors.append(f"{location}: array lengths differ")
            return
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            _compare_json(left, right, f"{location}[{index}]", errors)
        return
    if type(reference) is not type(candidate) or reference != candidate:
        errors.append(f"{location}: {reference!r} != {candidate!r}")


def _numeric_cell(value: str) -> tuple[bool, float]:
    try:
        return True, float(value)
    except ValueError:
        return False, 0.0


def _metric_absolute_tolerance(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    reference_scenario = reference.get("scenario")
    reference_name = reference.get("name")
    candidate_scenario = candidate.get("scenario")
    candidate_name = candidate.get("name")
    if not all(
        isinstance(value, str)
        for value in (reference_scenario, reference_name, candidate_scenario, candidate_name)
    ):
        return ABSOLUTE_TOLERANCE
    reference_identity = (reference_scenario, reference_name)
    candidate_identity = (candidate_scenario, candidate_name)
    if reference_identity != candidate_identity:
        return ABSOLUTE_TOLERANCE
    return METRIC_ABSOLUTE_TOLERANCES.get(reference_identity, ABSOLUTE_TOLERANCE)


def _csv_absolute_tolerance(
    reference_path: Path,
    reference_row: dict[str, str],
    candidate_row: dict[str, str],
    field: str,
) -> float:
    artifact_tolerance = CSV_ABSOLUTE_TOLERANCES.get((reference_path.name, field))
    if artifact_tolerance is not None:
        return artifact_tolerance
    if reference_path.name == "validation_metrics.csv" and field == "value":
        return _metric_absolute_tolerance(reference_row, candidate_row)
    return ABSOLUTE_TOLERANCE


def _compare_csv(reference_path: Path, candidate_path: Path, errors: list[str]) -> None:
    with reference_path.open(encoding="utf-8", newline="") as handle:
        reference_reader = csv.DictReader(handle)
        reference_fields = reference_reader.fieldnames
        reference_rows = list(reference_reader)
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        candidate_reader = csv.DictReader(handle)
        candidate_fields = candidate_reader.fieldnames
        candidate_rows = list(candidate_reader)

    if reference_fields != candidate_fields:
        errors.append(f"{reference_path.name}: CSV columns differ")
        return
    if len(reference_rows) != len(candidate_rows):
        errors.append(f"{reference_path.name}: CSV row counts differ")
        return
    assert reference_fields is not None
    for row_number, (reference_row, candidate_row) in enumerate(
        zip(reference_rows, candidate_rows, strict=True),
        start=2,
    ):
        for field in reference_fields:
            reference_value = reference_row[field]
            candidate_value = candidate_row[field]
            reference_is_numeric, reference_numeric = _numeric_cell(reference_value)
            candidate_is_numeric, candidate_numeric = _numeric_cell(candidate_value)
            if reference_is_numeric or candidate_is_numeric:
                matches = (
                    reference_is_numeric
                    and candidate_is_numeric
                    and math.isfinite(reference_numeric)
                    and math.isfinite(candidate_numeric)
                    and math.isclose(
                        reference_numeric,
                        candidate_numeric,
                        rel_tol=RELATIVE_TOLERANCE,
                        abs_tol=_csv_absolute_tolerance(
                            reference_path,
                            reference_row,
                            candidate_row,
                            field,
                        ),
                    )
                )
            else:
                matches = reference_value == candidate_value
            if not matches:
                errors.append(
                    f"{reference_path.name}:{row_number}:{field}: "
                    f"{reference_value} != {candidate_value} within CSV tolerance"
                )
                return


def _rgb_image(path: Path) -> NDArray[np.float64]:
    image = np.asarray(mpimg.imread(path), dtype=np.float64)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"{path.name}: expected an RGB or RGBA image")
    rgb = image[:, :, :3]
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4]
        rgb = rgb * alpha + (1.0 - alpha)
    return np.clip(rgb, 0.0, 1.0)


def _sample_image(image: NDArray[np.float64]) -> NDArray[np.float64]:
    y_indices = np.linspace(0, image.shape[0] - 1, PNG_SAMPLE_SIZE).round().astype(int)
    x_indices = np.linspace(0, image.shape[1] - 1, PNG_SAMPLE_SIZE).round().astype(int)
    return image[y_indices][:, x_indices]


def _ink_metrics(image: NDArray[np.float64]) -> tuple[float, float]:
    ink = np.max(1.0 - image, axis=2)
    return float(np.mean(ink)), float(np.mean(ink > PNG_INK_THRESHOLD))


def _series_color_coverage(image: NDArray[np.float64], color: NDArray[np.float64]) -> float:
    distance = np.max(np.abs(image - color), axis=2)
    return float(np.mean(distance <= SERIES_COLOR_TOLERANCE))


def _region(
    image: NDArray[np.float64],
    y_start: float,
    y_end: float,
    x_start: float,
    x_end: float,
) -> NDArray[np.float64]:
    y0 = round(y_start * image.shape[0])
    y1 = round(y_end * image.shape[0])
    x0 = round(x_start * image.shape[1])
    x1 = round(x_end * image.shape[1])
    return image[y0:y1, x0:x1]


def _compare_series_regions(
    reference_path: Path,
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    errors: list[str],
) -> bool:
    contracts = SERIES_REGIONS.get(reference_path.name)
    if contracts is None:
        errors.append(f"{reference_path.name}: no plot-series contract")
        return False
    for region_name, y0, y1, x0, x1, color_names in contracts:
        reference_region = _region(reference, y0, y1, x0, x1)
        candidate_region = _region(candidate, y0, y1, x0, x1)
        for color_name in color_names:
            color = SERIES_COLORS[color_name]
            reference_coverage = _series_color_coverage(reference_region, color)
            candidate_coverage = _series_color_coverage(candidate_region, color)
            if reference_coverage < MIN_SERIES_COLOR_COVERAGE:
                errors.append(
                    f"{reference_path.name}: {region_name} has no reference {color_name} series"
                )
                return False
            color_delta = abs(reference_coverage - candidate_coverage) / reference_coverage
            if color_delta > MAX_SERIES_COLOR_RELATIVE_DELTA:
                errors.append(
                    f"{reference_path.name}: {region_name} {color_name} series coverage "
                    f"differs ({color_delta:.2%})"
                )
                return False
    return True


def _compare_png(reference_path: Path, candidate_path: Path, errors: list[str]) -> None:
    try:
        reference = _rgb_image(reference_path)
        candidate = _rgb_image(candidate_path)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return

    reference_ratio = reference.shape[1] / reference.shape[0]
    candidate_ratio = candidate.shape[1] / candidate.shape[0]
    ratio_delta = abs(reference_ratio - candidate_ratio) / reference_ratio
    if ratio_delta > MAX_ASPECT_RATIO_DELTA:
        errors.append(f"{reference_path.name}: plot aspect ratio differs")
        return

    height_delta = abs(reference.shape[0] - candidate.shape[0]) / reference.shape[0]
    width_delta = abs(reference.shape[1] - candidate.shape[1]) / reference.shape[1]
    if max(height_delta, width_delta) > MAX_DIMENSION_DELTA:
        errors.append(f"{reference_path.name}: plot dimensions differ")
        return

    if not _compare_series_regions(reference_path, reference, candidate, errors):
        return

    reference_ink, reference_coverage = _ink_metrics(reference)
    candidate_ink, candidate_coverage = _ink_metrics(candidate)
    if reference_ink == 0.0 or reference_coverage == 0.0:
        errors.append(f"{reference_path.name}: reference plot contains no visible content")
        return
    ink_delta = abs(reference_ink - candidate_ink) / reference_ink
    coverage_delta = abs(reference_coverage - candidate_coverage) / reference_coverage
    if max(ink_delta, coverage_delta) > MAX_PNG_INK_RELATIVE_DELTA:
        errors.append(
            f"{reference_path.name}: visible plot content differs "
            f"(ink={ink_delta:.2%}, coverage={coverage_delta:.2%})"
        )
        return

    difference = np.abs(_sample_image(reference) - _sample_image(candidate))
    mean_absolute_error = float(np.mean(difference))
    changed_fraction = float(np.mean(np.max(difference, axis=2) > PNG_CHANGE_THRESHOLD))
    if (
        mean_absolute_error > MAX_PNG_MEAN_ABSOLUTE_ERROR
        or changed_fraction > MAX_PNG_CHANGED_FRACTION
    ):
        errors.append(
            f"{reference_path.name}: perceptual difference too large "
            f"(mean={mean_absolute_error:.4f}, changed={changed_fraction:.2%})"
        )


def compare_validation_directories(reference_dir: Path, candidate_dir: Path) -> list[str]:
    """Return material differences between checked-in and regenerated evidence."""

    reference = Path(reference_dir)
    candidate = Path(candidate_dir)
    errors: list[str] = []
    reference_files = {path.name for path in reference.iterdir() if path.is_file()}
    candidate_files = {path.name for path in candidate.iterdir() if path.is_file()}
    if reference_files != candidate_files:
        missing = sorted(reference_files - candidate_files)
        extra = sorted(candidate_files - reference_files)
        return [f"artifact set differs: missing={missing}, extra={extra}"]

    for name in sorted(reference_files):
        reference_path = reference / name
        candidate_path = candidate / name
        if name.endswith(".json"):
            reference_json = json.loads(reference_path.read_text(encoding="utf-8"))
            candidate_json = json.loads(candidate_path.read_text(encoding="utf-8"))
            _compare_json(reference_json, candidate_json, name, errors)
        elif name.endswith(".csv"):
            _compare_csv(reference_path, candidate_path, errors)
        elif name.endswith(".png"):
            _compare_png(reference_path, candidate_path, errors)
        else:
            errors.append(f"{name}: unsupported evidence format")
    return errors
