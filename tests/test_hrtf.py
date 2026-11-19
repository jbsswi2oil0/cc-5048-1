import numpy as np
import pytest

from binaural.hrtf import ProceduralHRTF


@pytest.fixture(scope="module")
def hrtf():
    return ProceduralHRTF()


def test_database_grid_and_shapes(hrtf):
    assert hrtf.hrir_length == 128
    assert hrtf.hrirs.shape[0] == len(hrtf.azimuths)
    assert hrtf.hrirs.shape[1] == len(hrtf.elevations)
    assert hrtf.hrirs.shape[2:] == (2, 128)


def test_left_right_mirror_symmetry(hrtf):
    left = hrtf.query(90, 0).hrir
    right = hrtf.query(-90, 0).hrir
    assert np.allclose(left[0], right[1], atol=1e-12)
    assert np.allclose(left[1], right[0], atol=1e-12)


def test_front_and_back_are_distinct(hrtf):
    front = hrtf.query(0, 0).hrir
    back = hrtf.query(180, 0).hrir
    assert not np.allclose(front, back, atol=1e-6)
    # behind stays perceptually centered: equal energy in both ears
    ratio = np.sum(back[0] ** 2) / np.sum(back[1] ** 2)
    assert 0.9 < ratio < 1.1


def test_interpolation_is_continuous_and_bounded(hrtf):
    angles = np.linspace(-180, 180, 361)
    prev = hrtf.query(angles[0], 0).hrir
    rms_jumps = []
    for az in angles[1:]:
        cur = hrtf.query(az, 0).hrir
        # RMS of the HRIR difference tolerates the genuine sub-sample
        # migration of the ITD onset while a true grid discontinuity (for
        # example a broken +/-180 seam) raises it several-fold.
        rms_jumps.append(np.sqrt(np.mean((cur - prev) ** 2)))
        prev = cur
    assert max(rms_jumps) < 0.04
    # the +/-180 seam is the same physical direction and must match exactly
    assert np.allclose(hrtf.query(-180, 0).hrir, hrtf.query(180, 0).hrir,
                       atol=1e-12)
    # global normalization keeps every stored/interpolated filter bounded
    assert np.max(np.abs(hrtf.hrirs)) <= 1.0
    assert np.max(np.abs(prev)) <= 1.0


def test_interpolation_matches_grid_points(hrtf):
    q = hrtf.query(hrtf.azimuths[7], hrtf.elevations[3])
    ia, ie = 7, 3
    assert np.allclose(q.hrir, hrtf.hrirs[ia, ie], atol=1e-12)


def test_itd_sign_and_magnitude(hrtf):
    # Left source: left ear HRIR peaks earlier than right ear HRIR.
    hrir = hrtf.query(90, 0).hrir
    left_onset = int(np.argmax(np.abs(hrir[0])))
    right_onset = int(np.argmax(np.abs(hrir[1])))
    assert left_onset < right_onset
    itd_seconds = (right_onset - left_onset) / hrtf.sample_rate
    assert 0.0 < itd_seconds < 0.001


def test_elevation_changes_spectrum(hrtf):
    low = np.abs(np.fft.rfft(hrtf.query(0, -40).hrir, n=512))
    high = np.abs(np.fft.rfft(hrtf.query(0, 40).hrir, n=512))
    assert not np.allclose(low, high, atol=1e-6)


def test_elevation_clamped_outside_range(hrtf):
    q = hrtf.query(0, 150)
    assert q.elevation_deg == hrtf.elevations[-1]
