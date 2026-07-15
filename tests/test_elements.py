import numpy as np
import pytest

from orbital_mechanics.constants import EARTH
from orbital_mechanics.elements import (
    ClassicalOrbitalElements,
    cartesian_to_elements,
    elements_to_cartesian,
)

MU = EARTH.gravitational_parameter_km3_s2


@pytest.mark.parametrize(
    "elements",
    [
        ClassicalOrbitalElements(
            semi_major_axis_km=7_200.0,
            eccentricity=0.01,
            inclination_rad=np.deg2rad(28.5),
            raan_rad=np.deg2rad(42.0),
            argument_of_periapsis_rad=np.deg2rad(87.0),
            true_anomaly_rad=np.deg2rad(133.0),
        ),
        ClassicalOrbitalElements(
            semi_major_axis_km=26_560.0,
            eccentricity=0.74,
            inclination_rad=np.deg2rad(63.4),
            raan_rad=np.deg2rad(310.0),
            argument_of_periapsis_rad=np.deg2rad(270.0),
            true_anomaly_rad=np.deg2rad(5.0),
        ),
        ClassicalOrbitalElements(
            semi_major_axis_km=8_000.0,
            eccentricity=1e-8,
            inclination_rad=1e-7,
            raan_rad=np.deg2rad(15.0),
            argument_of_periapsis_rad=np.deg2rad(35.0),
            true_anomaly_rad=np.deg2rad(80.0),
        ),
    ],
)
def test_element_round_trip_reconstructs_state(elements: ClassicalOrbitalElements) -> None:
    position, velocity = elements_to_cartesian(elements, MU)
    recovered = cartesian_to_elements(position, velocity, MU)
    reconstructed_position, reconstructed_velocity = elements_to_cartesian(recovered, MU)

    position_relative_error = np.linalg.norm(reconstructed_position - position) / np.linalg.norm(
        position
    )
    velocity_relative_error = np.linalg.norm(reconstructed_velocity - velocity) / np.linalg.norm(
        velocity
    )
    assert position_relative_error <= 1e-9
    assert velocity_relative_error <= 1e-9


def test_circular_equatorial_state_uses_true_longitude_convention() -> None:
    radius = 7_000.0
    longitude = np.deg2rad(123.0)
    speed = np.sqrt(MU / radius)
    position = radius * np.array([np.cos(longitude), np.sin(longitude), 0.0])
    velocity = speed * np.array([-np.sin(longitude), np.cos(longitude), 0.0])

    elements = cartesian_to_elements(position, velocity, MU)
    reconstructed_position, reconstructed_velocity = elements_to_cartesian(elements, MU)

    assert elements.eccentricity == pytest.approx(0.0, abs=1e-12)
    assert elements.inclination_rad == pytest.approx(0.0, abs=1e-12)
    assert elements.raan_rad == 0.0
    assert elements.argument_of_periapsis_rad == 0.0
    assert elements.true_anomaly_rad == pytest.approx(longitude)
    np.testing.assert_allclose(reconstructed_position, position, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(reconstructed_velocity, velocity, rtol=1e-12, atol=1e-12)


def test_eccentric_equatorial_state_uses_longitude_of_periapsis() -> None:
    elements = ClassicalOrbitalElements(
        semi_major_axis_km=8_000.0,
        eccentricity=0.1,
        inclination_rad=0.0,
        raan_rad=0.0,
        argument_of_periapsis_rad=np.deg2rad(75.0),
        true_anomaly_rad=np.deg2rad(20.0),
    )
    position, velocity = elements_to_cartesian(elements, MU)
    recovered = cartesian_to_elements(position, velocity, MU)

    assert recovered.raan_rad == 0.0
    assert recovered.argument_of_periapsis_rad == pytest.approx(elements.argument_of_periapsis_rad)


def test_eccentric_retrograde_equatorial_state_reconstructs_with_canonical_angles() -> None:
    elements = ClassicalOrbitalElements(
        semi_major_axis_km=8_000.0,
        eccentricity=0.1,
        inclination_rad=np.pi,
        raan_rad=np.deg2rad(40.0),
        argument_of_periapsis_rad=np.deg2rad(75.0),
        true_anomaly_rad=np.deg2rad(20.0),
    )
    position, velocity = elements_to_cartesian(elements, MU)
    recovered = cartesian_to_elements(position, velocity, MU)
    reconstructed_position, reconstructed_velocity = elements_to_cartesian(recovered, MU)

    assert recovered.inclination_rad == pytest.approx(np.pi)
    assert recovered.raan_rad == 0.0
    np.testing.assert_allclose(reconstructed_position, position, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(reconstructed_velocity, velocity, rtol=1e-12, atol=1e-12)


def test_circular_retrograde_equatorial_state_reconstructs_true_longitude() -> None:
    elements = ClassicalOrbitalElements(
        semi_major_axis_km=7_000.0,
        eccentricity=0.0,
        inclination_rad=np.pi,
        raan_rad=np.deg2rad(40.0),
        argument_of_periapsis_rad=np.deg2rad(75.0),
        true_anomaly_rad=np.deg2rad(20.0),
    )
    position, velocity = elements_to_cartesian(elements, MU)
    recovered = cartesian_to_elements(position, velocity, MU)
    reconstructed_position, reconstructed_velocity = elements_to_cartesian(recovered, MU)

    assert recovered.inclination_rad == pytest.approx(np.pi)
    assert recovered.raan_rad == 0.0
    assert recovered.argument_of_periapsis_rad == 0.0
    np.testing.assert_allclose(reconstructed_position, position, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(reconstructed_velocity, velocity, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semi_major_axis_km", 0.0),
        ("eccentricity", -0.01),
        ("eccentricity", 1.0),
        ("inclination_rad", -0.1),
        ("inclination_rad", np.pi + 0.1),
    ],
)
def test_invalid_elements_are_rejected(field: str, value: float) -> None:
    values = {
        "semi_major_axis_km": 7_000.0,
        "eccentricity": 0.01,
        "inclination_rad": 0.5,
        "raan_rad": 0.2,
        "argument_of_periapsis_rad": 0.3,
        "true_anomaly_rad": 0.4,
    }
    values[field] = value
    with pytest.raises(ValueError):
        ClassicalOrbitalElements(**values)


def test_radial_or_unbound_states_are_rejected() -> None:
    with pytest.raises(ValueError, match="angular momentum"):
        cartesian_to_elements([7_000.0, 0.0, 0.0], [1.0, 0.0, 0.0], MU)
    with pytest.raises(ValueError, match="elliptic"):
        cartesian_to_elements([7_000.0, 0.0, 0.0], [0.0, 12.0, 0.0], MU)
