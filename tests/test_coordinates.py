import numpy as np
import pytest

from binaural.coordinates import (
    normalize_angle,
    head_rotation_matrix,
    relative_direction,
    direction_to_unit_vector,
)


def test_normalize_angle_wraps_to_open_interval():
    assert normalize_angle(0) == 0
    assert normalize_angle(180) == 180
    assert normalize_angle(-180) == 180
    assert normalize_angle(270) == -90
    assert normalize_angle(-450) == -90


def test_right_handed_axes():
    # cross(front, left) must point up
    front = np.array([1.0, 0, 0])
    left = np.array([0.0, 1, 0])
    up = np.cross(front, left)
    assert np.allclose(up, [0, 0, 1])


def test_cardinal_relative_directions():
    origin = np.zeros(3)
    assert relative_direction([1, 0, 0], origin).azimuth_deg == pytest.approx(0)
    assert relative_direction([0, 1, 0], origin).azimuth_deg == pytest.approx(90)
    assert relative_direction([0, -1, 0], origin).azimuth_deg == pytest.approx(-90)
    behind = relative_direction([-1, 0, 0], origin)
    assert abs(behind.azimuth_deg) == pytest.approx(180)
    up = relative_direction([0, 0, 1], origin)
    assert up.elevation_deg == pytest.approx(90)


def test_yaw_left_brings_front_source_to_right_azimuth():
    # Head turned 90 deg to its left: a world-front source sits at right ear.
    rel = relative_direction([1, 0, 0], np.zeros(3), yaw_deg=90)
    assert rel.azimuth_deg == pytest.approx(-90)


def test_head_turn_equivalent_to_source_motion():
    moved = relative_direction([0, -1, 0], np.zeros(3))
    turned = relative_direction([1, 0, 0], np.zeros(3), yaw_deg=90)
    assert moved.azimuth_deg == pytest.approx(turned.azimuth_deg)
    assert np.allclose(moved.unit_vector, turned.unit_vector, atol=1e-12)


def test_unit_vector_round_trip():
    for az, el in [(0, 0), (45, -30), (-120, 60), (180, 0)]:
        v = direction_to_unit_vector(az, el)
        rel = relative_direction(v, np.zeros(3))
        assert rel.azimuth_deg == pytest.approx(az, abs=1e-9)
        assert rel.elevation_deg == pytest.approx(el, abs=1e-9)


def test_rotation_matrix_orthonormal():
    r = head_rotation_matrix(37, -12, 5)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_coincident_source_rejected():
    with pytest.raises(ValueError):
        relative_direction(np.zeros(3), np.zeros(3))
