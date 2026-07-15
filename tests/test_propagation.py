import numpy as np
import pytest

from orbital_mechanics.propagation import propagate_rk4


def test_rk4_integrates_constant_acceleration_exactly() -> None:
    acceleration = np.array([0.01, -0.02, 0.005])
    initial_position = np.array([1.0, 2.0, 3.0])
    initial_velocity = np.array([0.1, -0.2, 0.3])
    duration = 10.0

    trajectory = propagate_rk4(
        initial_position,
        initial_velocity,
        duration,
        0.25,
        lambda _position: acceleration,
    )

    expected_position = (
        initial_position + initial_velocity * duration + 0.5 * acceleration * duration**2
    )
    expected_velocity = initial_velocity + acceleration * duration
    np.testing.assert_allclose(trajectory.positions_km[-1], expected_position, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        trajectory.velocities_km_s[-1], expected_velocity, rtol=0.0, atol=1e-14
    )
    assert trajectory.times_s[-1] == duration


def test_propagation_is_deterministic_and_does_not_mutate_inputs() -> None:
    position = np.array([7_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7.5, 0.0])
    original_position = position.copy()
    original_velocity = velocity.copy()

    def acceleration(point: np.ndarray) -> np.ndarray:
        return -point / np.linalg.norm(point) ** 3

    first = propagate_rk4(position, velocity, 10.0, 1.0, acceleration)
    second = propagate_rk4(position, velocity, 10.0, 1.0, acceleration)

    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(position, original_position)
    np.testing.assert_array_equal(velocity, original_velocity)


@pytest.mark.parametrize(
    ("duration", "step"),
    [(0.0, 1.0), (10.0, 0.0), (10.0, -1.0), (10.0, 3.0), (np.inf, 1.0)],
)
def test_invalid_time_configuration_is_rejected(duration: float, step: float) -> None:
    with pytest.raises(ValueError):
        propagate_rk4([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], duration, step, lambda x: x)


def test_invalid_acceleration_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="acceleration model"):
        propagate_rk4(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            lambda _position: np.array([1.0, 2.0]),
        )
