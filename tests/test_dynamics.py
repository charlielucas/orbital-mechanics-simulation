import numpy as np
import pytest

from orbital_mechanics.constants import EARTH
from orbital_mechanics.dynamics import (
    j2_perturbation,
    j2_secular_raan_rate,
    specific_angular_momentum,
    specific_orbital_energy,
    two_body_acceleration,
    two_body_j2_acceleration,
)


def test_two_body_acceleration_points_toward_origin() -> None:
    radius = 7_000.0
    acceleration = two_body_acceleration([radius, 0.0, 0.0], EARTH.gravitational_parameter_km3_s2)
    expected = -EARTH.gravitational_parameter_km3_s2 / radius**2
    np.testing.assert_allclose(acceleration, [expected, 0.0, 0.0], rtol=1e-15, atol=0.0)


def test_j2_equatorial_and_polar_signs_match_closed_form() -> None:
    radius = 7_000.0
    equatorial = j2_perturbation(
        [radius, 0.0, 0.0],
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )
    polar = j2_perturbation(
        [0.0, 0.0, radius],
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )

    assert equatorial[0] < 0.0
    assert equatorial[1] == 0.0
    assert equatorial[2] == 0.0
    assert polar[0] == 0.0
    assert polar[1] == 0.0
    assert polar[2] > 0.0


def test_combined_acceleration_is_sum_of_components() -> None:
    position = np.array([6_900.0, -800.0, 1_200.0])
    expected = two_body_acceleration(
        position, EARTH.gravitational_parameter_km3_s2
    ) + j2_perturbation(
        position,
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )
    actual = two_body_j2_acceleration(
        position,
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )
    np.testing.assert_array_equal(actual, expected)


def test_raan_rate_has_expected_prograde_and_retrograde_signs() -> None:
    common = (
        7_000.0,
        0.001,
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )
    prograde = j2_secular_raan_rate(common[0], common[1], np.deg2rad(60.0), *common[2:])
    retrograde = j2_secular_raan_rate(common[0], common[1], np.deg2rad(120.0), *common[2:])

    assert prograde < 0.0
    assert retrograde > 0.0
    assert abs(prograde) == pytest.approx(abs(retrograde), rel=1e-14)


def test_j2_raan_rate_matches_98_degree_hard_number_benchmark() -> None:
    rate_rad_s = j2_secular_raan_rate(
        7_000.0,
        0.001,
        np.deg2rad(98.0),
        EARTH.gravitational_parameter_km3_s2,
        EARTH.equatorial_radius_km,
        EARTH.j2,
    )
    rate_deg_day = float(np.rad2deg(rate_rad_s) * 86_400.0)

    assert rate_deg_day == pytest.approx(1.00132687, abs=3e-7)


def test_specific_energy_and_momentum_for_circular_orbit() -> None:
    radius = 7_000.0
    speed = np.sqrt(EARTH.gravitational_parameter_km3_s2 / radius)
    energy = specific_orbital_energy(
        [radius, 0.0, 0.0], [0.0, speed, 0.0], EARTH.gravitational_parameter_km3_s2
    )
    momentum = specific_angular_momentum([radius, 0.0, 0.0], [0.0, speed, 0.0])

    assert energy == pytest.approx(-EARTH.gravitational_parameter_km3_s2 / (2.0 * radius))
    np.testing.assert_allclose(momentum, [0.0, 0.0, radius * speed])


@pytest.mark.parametrize("position", ([0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 2.0]))
def test_invalid_positions_are_rejected(position: list[float]) -> None:
    with pytest.raises(ValueError):
        two_body_acceleration(position, EARTH.gravitational_parameter_km3_s2)
