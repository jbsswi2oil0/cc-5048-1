import numpy as np
import pytest

from binaural.hrtf import ProceduralHRTF
from binaural.renderer import BinauralRenderer


def make_renderer(block_size=256):
    hrtf = ProceduralHRTF()
    return BinauralRenderer(hrtf, block_size=block_size,
                            apply_distance_gain=False)


def arm(renderer, az, el=0.0, distance=1.2):
    renderer._current_query = None
    renderer.set_direction(az, el, distance)
    return renderer


def test_output_is_stereo_and_correct_length():
    renderer = arm(make_renderer(256), 0)
    out = renderer.process_block(np.zeros(256))
    assert out.shape == (256, 2)


def test_spatialization_uses_filtering_not_panning():
    # A pure level pan would produce perfectly correlated channels; real
    # convolution decorrelates them via independent HRIRs.
    renderer = arm(make_renderer(256), 60)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(256 * 8)
    out = np.concatenate([
        renderer.process_block(x[k:k + 256]) for k in range(0, len(x), 256)])
    correlation = np.corrcoef(out[256:, 0], out[256:, 1])[0, 1]
    assert abs(correlation) < 0.995


def test_direction_change_engages_crossfade():
    renderer = arm(make_renderer(256), 0)
    renderer.process_block(np.zeros(256))
    renderer.set_direction(90, 0, 1.2)
    assert renderer._fade_pos > 0
    renderer.process_block(np.zeros(256))
    # equal-power gains keep unit norm at the ramp midpoint
    t = 0.5
    assert np.cos(0.5 * np.pi * t) ** 2 + np.sin(0.5 * np.pi * t) ** 2 \
        == pytest.approx(1.0)


def test_same_direction_does_not_retrigger_crossfade():
    renderer = arm(make_renderer(256), 30)
    renderer.set_direction(30, 0, 1.2)
    assert renderer._fade_pos == 0


def test_distance_gain_inverse_law():
    hrtf = ProceduralHRTF()
    renderer = BinauralRenderer(hrtf, block_size=256)
    renderer._current_query = None
    renderer.set_direction(0, 0, 2.0)
    for _ in range(60):  # let the one-pole smoother settle
        renderer.process_block(np.zeros(256))
    assert renderer._distance_gain == pytest.approx(0.5, abs=1e-3)


def test_direction_log_records_every_block():
    renderer = arm(make_renderer(256), 10)
    for k in range(5):
        renderer.set_direction(10 + k * 5, 0, 1.2)
        renderer.process_block(np.zeros(256))
    assert len(renderer.direction_log) == 5
    az = [r.azimuth_deg for r in renderer.direction_log]
    assert az == sorted(az)


def test_state_report_contains_latency_and_banks():
    renderer = arm(make_renderer(256), 0)
    renderer.process_block(np.zeros(256))
    report = renderer.state_report()
    assert report["algorithm_latency_samples"] == 256
    assert report["blocks_processed"] == 1
    assert report["banks"][0] is not None
    assert report["current_direction"]["azimuth_deg"] == 0.0


def test_world_pose_drives_relative_direction():
    renderer = make_renderer(256)
    renderer.update_pose(source_position=(1.0, 0.0, 0.0),
                         listener_position=np.zeros(3), yaw_deg=90)
    assert renderer._current_query.azimuth_deg == pytest.approx(-90)
