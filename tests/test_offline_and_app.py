import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from binaural.gui import BinauralGUI, HeadlessGUI
from binaural.hrtf import ProceduralHRTF
from binaural.offline import (
    export_session,
    render_offline,
    static_trajectory,
    sweep_trajectory,
)
from binaural.realtime import RealtimeBinauralPlayer
from binaural.renderer import BinauralRenderer
from binaural.signals import (
    broadband_signal,
    build_default_library,
    impulse_signal,
    speech_signal,
)


def fresh_renderer(block_size=256):
    hrtf = ProceduralHRTF()
    return BinauralRenderer(hrtf, block_size=block_size)


def test_default_library_has_three_mono_signals():
    lib = build_default_library()
    assert set(lib) == {"impulse", "broadband", "speech"}
    for name, x in lib.items():
        assert x.ndim == 1
        assert np.max(np.abs(x)) > 0
        assert np.isfinite(x).all()


def test_impulse_has_single_unit_sample():
    x = impulse_signal()
    assert x[0] == 1.0
    assert np.count_nonzero(x) == 1


def test_broadband_is_loop_continuous():
    x = broadband_signal(duration_s=2.0)
    # constructed to match at the wrap boundary
    assert abs(x[0] - x[-1]) < 1e-6


def test_speech_signal_nonempty():
    x = speech_signal()
    assert x.size > 4800  # at least ~100 ms
    assert np.max(np.abs(x)) > 0.1


def test_offline_render_exports_manifest_and_hrirs(tmp_path):
    renderer = fresh_renderer(256)
    signal = broadband_signal(duration_s=1.0)
    traj = sweep_trajectory(-90, 90, 0, 20)
    renderer.reset()
    renderer.set_direction(traj[0]["azimuth_deg"], 0, 1.2)
    stereo, hrirs = render_offline(renderer, signal, traj)
    assert stereo.shape[1] == 2
    assert len(hrirs) > 1  # genuinely varying directions used

    manifest_path = export_session(
        tmp_path, "sweep", renderer, stereo, hrirs, "broadband")
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["block_size_samples"] == 256
    assert manifest["algorithm_latency_seconds"] == 256 / 48000
    assert Path(tmp_path, manifest["output_file"]).exists()
    for entry in manifest["hrirs_used"]:
        assert Path(tmp_path, entry["file"]).exists()
    log = Path(tmp_path, "direction_log_sweep.csv")
    assert log.exists() and len(log.read_text().splitlines()) > 20


def test_exported_wav_is_stereo_at_sample_rate(tmp_path):
    renderer = fresh_renderer(256)
    signal = impulse_signal(duration_s=0.5)
    traj = static_trajectory(0, 0, 4)
    renderer.reset()
    renderer.set_direction(0, 0, 1.2)
    stereo, hrirs = render_offline(renderer, signal, traj)
    export_session(tmp_path, "imp", renderer, stereo, hrirs, "impulse")
    data, fs = sf.read(str(tmp_path / "binaural_imp.wav"))
    assert fs == 48000
    assert data.ndim == 2 and data.shape[1] == 2


def test_headless_gui_presets_and_yaw():
    hrtf = ProceduralHRTF()
    renderer = BinauralRenderer(hrtf, block_size=256)
    renderer._current_query = None
    renderer.set_direction(0, 0, 1.2)
    player = RealtimeBinauralPlayer(
        renderer, broadband_signal(duration_s=0.5), loop=True)
    gui = BinauralGUI(player)

    gui.jump_preset("left")
    assert gui.state.azimuth_deg == pytest.approx(90)
    gui.jump_preset("right")
    assert gui.state.azimuth_deg == pytest.approx(-90)
    gui.jump_preset("back")
    assert abs(gui.state.azimuth_deg) == pytest.approx(180)

    headless = HeadlessGUI(player)
    audio = headless.scenario(
        [("preset", "front"), ("yaw", 90), ("source", (0.5, -0.5, 0))],
        blocks_per_move=3)
    assert audio.shape[1] == 2
    assert audio.shape[0] == 9 * 256


def test_unknown_preset_rejected():
    hrtf = ProceduralHRTF()
    renderer = BinauralRenderer(hrtf, block_size=256)
    renderer._current_query = None
    renderer.set_direction(0, 0, 1.2)
    player = RealtimeBinauralPlayer(
        renderer, broadband_signal(duration_s=0.5), loop=True)
    with pytest.raises(KeyError):
        BinauralGUI(player).jump_preset("above")
