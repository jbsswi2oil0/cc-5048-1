import inspect

import numpy as np

from binaural.hrtf import ProceduralHRTF
from binaural.realtime import RealtimeBinauralPlayer, audio_thread_io_audit
from binaural.renderer import BinauralRenderer
from binaural.signals import broadband_signal


def build_player(block_size=256):
    hrtf = ProceduralHRTF()
    renderer = BinauralRenderer(hrtf, block_size=block_size)
    renderer._current_query = None
    renderer.set_direction(0, 0, 1.2)
    signal = broadband_signal(duration_s=1.0)
    return RealtimeBinauralPlayer(renderer, signal, loop=True)


def test_callback_contains_no_file_io():
    audit = audio_thread_io_audit()
    assert audit["passed"]
    source = inspect.getsource(RealtimeBinauralPlayer._callback)
    for forbidden in ("open(", "soundfile", "subprocess", "urlopen",
                      "requests", "read_wav", "Path("):
        assert forbidden not in source


def test_simulated_playback_produces_stereo_blocks():
    player = build_player(256)
    out = player.simulate_blocks(6)
    assert out.shape == (6 * 256, 2)
    assert player.status.callback_blocks == 6


def test_looping_never_runs_out_of_signal():
    player = build_player(128)
    out = player.simulate_blocks(400)  # far more blocks than the signal
    assert out.shape[0] == 400 * 128
    assert np.isfinite(out).all()


def test_seek_stays_in_bounds():
    player = build_player(256)
    player.seek(player.signal.size + 999)
    block = player._next_block()
    assert block is not None and block.shape == (256,)


def test_latency_report_separates_device_and_algorithm():
    player = build_player(256)
    report = player.reported_latency()
    assert report["algorithm_latency_seconds"] == 256 / 48000
    assert report["total_latency_seconds"] >= report[
        "algorithm_latency_seconds"]
    assert report["backend"] in ("sounddevice", "simulated")


def test_empty_signal_rejected():
    hrtf = ProceduralHRTF()
    renderer = BinauralRenderer(hrtf, block_size=256)
    renderer._current_query = None
    renderer.set_direction(0, 0, 1.2)
    try:
        RealtimeBinauralPlayer(renderer, np.zeros(0))
    except ValueError:
        return
    raise AssertionError("empty signal should be rejected")
