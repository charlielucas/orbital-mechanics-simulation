"""Physical constants used by the simulation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CentralBody:
    """Constants for an oblate central body in kilometer-second units."""

    gravitational_parameter_km3_s2: float
    equatorial_radius_km: float
    j2: float


# WGS 84 geometry and GM with a rounded EGM96 degree-two gravity coefficient.
# See the source and precision notes in the project README.
EARTH = CentralBody(
    gravitational_parameter_km3_s2=398_600.4418,
    equatorial_radius_km=6_378.137,
    j2=1.082_626_68e-3,
)
