"""Deterministic fixed-step Runge-Kutta orbital propagation."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
AccelerationModel = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Propagation output with seconds, kilometers, and kilometers per second."""

    times_s: FloatArray
    positions_km: FloatArray
    velocities_km_s: FloatArray

    @property
    def states(self) -> FloatArray:
        return np.column_stack((self.positions_km, self.velocities_km_s))


def _state_derivative(state: FloatArray, acceleration: AccelerationModel) -> FloatArray:
    result = np.empty(6, dtype=np.float64)
    result[:3] = state[3:]
    acceleration_value = np.asarray(acceleration(state[:3]), dtype=np.float64)
    if acceleration_value.shape != (3,) or not np.all(np.isfinite(acceleration_value)):
        raise ValueError("acceleration model must return exactly three finite components")
    result[3:] = acceleration_value
    return result


def propagate_rk4(
    initial_position_km: ArrayLike,
    initial_velocity_km_s: ArrayLike,
    duration_s: float,
    step_s: float,
    acceleration: AccelerationModel,
) -> Trajectory:
    """Propagate a Cartesian state using deterministic, fixed-step RK4.

    The duration must be an integer multiple of the time step. Requiring this
    explicitly avoids a hidden variable-size final step and keeps runs exactly
    reproducible.
    """

    position = np.asarray(initial_position_km, dtype=np.float64)
    velocity = np.asarray(initial_velocity_km_s, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("position and velocity must each contain exactly three components")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("initial state must contain only finite values")

    duration = float(duration_s)
    step = float(step_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")

    step_count_float = duration / step
    step_count = int(round(step_count_float))
    if step_count < 1 or not np.isclose(step_count_float, step_count, rtol=1e-12, atol=1e-12):
        raise ValueError("duration must be an integer multiple of step")

    times = np.arange(step_count + 1, dtype=np.float64) * step
    states = np.empty((step_count + 1, 6), dtype=np.float64)
    states[0, :3] = position
    states[0, 3:] = velocity

    for index in range(step_count):
        current = states[index]
        k1 = _state_derivative(current, acceleration)
        k2 = _state_derivative(current + 0.5 * step * k1, acceleration)
        k3 = _state_derivative(current + 0.5 * step * k2, acceleration)
        k4 = _state_derivative(current + step * k3, acceleration)
        states[index + 1] = current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return Trajectory(
        times_s=times,
        positions_km=states[:, :3],
        velocities_km_s=states[:, 3:],
    )
