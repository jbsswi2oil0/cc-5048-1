"""Coordinate conventions and head-pose geometry for the binaural renderer.

Coordinate system (right-handed, meters):
    +X = in front of the listener
    +Y = toward the listener's left ear
    +Z = upward

Right-handedness check: cross(+X front, +Y left) = +Z up.

Direction angles are always the *relative* direction from the listener's head
to the source, expressed in the head-local frame:
    azimuth   az, degrees, positive = leftward  (front=0, left=+90,
               back=+/-180, right=-90)
    elevation el, degrees, positive = upward    (horizontal plane=0,
               zenith=+90, nadir=-90)

Head pose is given as intrinsic yaw/pitch/roll in degrees (applied in that
order). Yaw rotates the head to its left when positive (a source straight
ahead then appears on the listener's right, i.e. negative relative azimuth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EAR_DISTANCE = 0.0875  # ear radius (m), half the inter-ear distance (~17.5 cm)
SPEED_OF_SOUND = 343.0  # m/s


def normalize_angle(angle_deg: float) -> float:
    """Wrap an angle to the half-open interval (-180, 180] degrees."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    if a == -180.0:
        return 180.0
    return a


def head_rotation_matrix(yaw_deg: float, pitch_deg: float = 0.0,
                          roll_deg: float = 0.0) -> np.ndarray:
    """Rotation from world coordinates into head-local coordinates.

    Intrinsic yaw (around Z), pitch (around Y), roll (around X).
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    rz = np.array([
        [np.cos(yaw), np.sin(yaw), 0.0],
        [-np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    ry = np.array([
        [np.cos(pitch), 0.0, -np.sin(pitch)],
        [0.0, 1.0, 0.0],
        [np.sin(pitch), 0.0, np.cos(pitch)],
    ])
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(roll), np.sin(roll)],
        [0.0, -np.sin(roll), np.cos(roll)],
    ])
    return rx @ ry @ rz


@dataclass
class RelativeDirection:
    """Source direction relative to the head, plus range data."""

    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    unit_vector: np.ndarray  # head-local unit vector toward the source

    def as_tuple(self) -> tuple[float, float, float]:
        return self.azimuth_deg, self.elevation_deg, self.distance_m


def relative_direction(source_position: np.ndarray,
                       listener_position: np.ndarray,
                       yaw_deg: float = 0.0,
                       pitch_deg: float = 0.0,
                       roll_deg: float = 0.0) -> RelativeDirection:
    """Compute the head-relative direction of a source.

    Positions are world-frame XYZ vectors (see module docstring).
    """
    src = np.asarray(source_position, dtype=float)
    lis = np.asarray(listener_position, dtype=float)
    delta = src - lis
    distance = float(np.linalg.norm(delta))
    if distance < 1e-12:
        raise ValueError("source coincides with listener; direction undefined")
    world_unit = delta / distance
    rot = head_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    head_unit = rot @ world_unit

    x, y, z = head_unit
    elevation = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    azimuth = normalize_angle(np.degrees(np.arctan2(y, x)))
    return RelativeDirection(float(azimuth), float(elevation), distance,
                             head_unit)


def direction_to_unit_vector(azimuth_deg: float,
                             elevation_deg: float) -> np.ndarray:
    """Inverse of the angular part of :func:`relative_direction`."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    ce = np.cos(el)
    return np.array([
        ce * np.cos(az),
        ce * np.sin(az),
        np.sin(el),
    ])
