"""Numerical self-checks backing the exhibit acceptance criteria."""

from __future__ import annotations

import numpy as np

from .convolution import measure_pipeline_latency
from .coordinates import relative_direction
from .hrtf import ProceduralHRTF
from .realtime import RealtimeBinauralPlayer, audio_thread_io_audit
from .renderer import BinauralRenderer
from .signals import broadband_signal


def _fresh(block_size: int) -> BinauralRenderer:
    hrtf = ProceduralHRTF()
    renderer = BinauralRenderer(hrtf, block_size=block_size,
                                crossfade_ms=5.0,
                                apply_distance_gain=False)
    renderer.set_direction(0.0, 0.0, 1.2)
    return renderer


def check_left_right_symmetry(block_size: int) -> dict:
    """L/R mirrored positions must produce channel-exchanged responses."""
    hrtf = ProceduralHRTF()
    q_l = hrtf.query(+90, 0)
    q_r = hrtf.query(-90, 0)
    hrir_swap = np.allclose(q_l.hrir[0], q_r.hrir[1], atol=1e-10) and \
        np.allclose(q_l.hrir[1], q_r.hrir[0], atol=1e-10)

    renderer = _fresh(block_size)
    n = block_size * 8
    x = broadband_signal(duration_s=n / renderer.sample_rate)[:n]
    outs = {}
    for label, az in (("L", +90), ("R", -90)):
        renderer.reset()
        # Force a clean initial arm on this direction (no crossfade), so the
        # comparison reflects the steady-state HRIR convolution only.
        renderer._current_query = None
        renderer.set_direction(az, 0, 1.2)
        chunks = [renderer.process_block(x[k:k + block_size])
                  for k in range(0, n, block_size)]
        outs[label] = np.concatenate(chunks)
    swapped = np.allclose(outs["L"][:, 0], outs["R"][:, 1], atol=1e-10) and \
        np.allclose(outs["L"][:, 1], outs["R"][:, 0], atol=1e-10)
    # ipsilateral channel must be louder than contralateral one
    ild = float(np.mean(outs["L"][:, 0] ** 2)
                - np.mean(outs["L"][:, 1] ** 2))
    return {"passed": bool(hrir_swap and swapped and ild > 0),
            "hrir_channel_swap": bool(hrir_swap),
            "render_channel_swap": bool(swapped),
            "left_ipsilateral_ild_db": float(10 * np.log10(
                np.mean(outs["L"][:, 0] ** 2)
                / max(np.mean(outs["L"][:, 1] ** 2), 1e-20)))}


def check_behind_difference(block_size: int) -> dict:
    """Front and behind at equal distance must be spectrally distinct, not a
    mere level change, and behind must keep centered (balanced) loudness."""
    hrtf = ProceduralHRTF()
    h_front = hrtf.query(0, 0).hrir
    h_back = hrtf.query(180, 0).hrir
    hf = np.fft.rfft(h_front, n=1024, axis=-1)
    hb = np.fft.rfft(h_back, n=1024, axis=-1)
    mag_f = np.abs(hf).mean(axis=0)
    mag_b = np.abs(hb).mean(axis=0)
    spectral_diff = float(np.sqrt(np.mean(
        (20 * np.log10(mag_b + 1e-12) - 20 * np.log10(mag_f + 1e-12)) ** 2)))
    balance_back = float(
        np.sum(h_back[0] ** 2) / max(np.sum(h_back[1] ** 2), 1e-20))
    return {"passed": bool(spectral_diff > 0.5
                           and 0.9 < balance_back < 1.1),
            "front_vs_back_spectral_rms_db": spectral_diff,
            "behind_l_over_r_energy_ratio": balance_back}


def check_real_convolution(block_size: int) -> dict:
    """Renderer output must match explicit FFT convolution of input x HRIR.

    The overlap-save renderer has one block of algorithmic latency, so its
    first output sample aligns with index ``block_size`` of the direct
    convolution.
    """
    renderer = _fresh(block_size)
    hrtf = renderer.hrtf
    q = hrtf.query(35, -10)
    # Build directly on the test direction so no crossfade contaminates the
    # pure convolution comparison.
    renderer._current_query = None
    renderer.set_direction(35, -10, 1.2)
    n = block_size * 6
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n)
    got = np.concatenate([
        renderer.process_block(x[k:k + block_size])
        for k in range(0, n, block_size)])

    nfft = 1 << int(np.ceil(np.log2(n + q.hrir.shape[-1])))
    ref = np.column_stack([
        np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(q.hrir[ch], nfft),
                     nfft)
        for ch in range(2)])
    # The renderer's explicit one-block output delay shifts every sample by
    # B: output block 1 (got[B:]) carries the direct convolution starting at
    # ref[0]. Compare equal-length slices (drop the first delayed block and
    # the final flush block beyond the reference span).
    valid = got.shape[0] - block_size
    got_tail = got[block_size: block_size + valid]
    ref_head = ref[:valid]
    err = float(np.max(np.abs(got_tail - ref_head)))
    return {"passed": bool(err < 1e-8),
            "max_abs_error_vs_fft_convolution": err}


def check_latency(block_size: int) -> dict:
    renderer = _fresh(block_size)
    report = measure_pipeline_latency(renderer, renderer.sample_rate)
    return {"passed": bool(report["matches_documentation"]), **report}

def check_head_turn_equivalence(block_size: int) -> dict:
    """Turning the head must give the same relative direction as moving the
    source: yaw +90 with source in front == source on the right (az -90)."""
    moved = relative_direction((0.0, -1.2, 0.0), (0, 0, 0))
    turned = relative_direction((1.2, 0.0, 0.0), (0, 0, 0), yaw_deg=90)
    az_match = abs(moved.azimuth_deg - turned.azimuth_deg) < 1e-9

    n = block_size * 6
    x = broadband_signal(duration_s=n / 48000)[:n]
    outs = []
    for az in (moved.azimuth_deg, turned.azimuth_deg):
        renderer = _fresh(block_size)
        renderer.set_direction(az, 0, 1.2)
        outs.append(np.concatenate([
            renderer.process_block(x[k:k + block_size])
            for k in range(0, n, block_size)]))
    audio_match = np.allclose(outs[0], outs[1], atol=1e-10)
    return {"passed": bool(az_match and audio_match),
            "moved_azimuth": moved.azimuth_deg,
            "turned_azimuth": turned.azimuth_deg,
            "audio_identical": bool(audio_match)}


def check_crossfade_no_click(block_size: int) -> dict:
    """A mid-stream direction switch must remain continuous.

    Two renders of the same smooth tone are compared: one holding a fixed
    direction, one switching direction at block 6. The switch is click-free
    when (a) the switched render's sample-to-sample jumps stay comparable to
    the fixed render's (no impulsive delta), and (b) the difference between
    the two renders itself ramps smoothly rather than stepping.
    """
    n_blocks = 12
    x = np.zeros(n_blocks * block_size)
    t = np.arange(x.size) / 48000
    x[:] = 0.5 * np.sin(2 * np.pi * 440 * t)

    def render(switch: bool) -> np.ndarray:
        renderer = _fresh(block_size)
        chunks = []
        for k in range(n_blocks):
            if switch and k == 6:
                renderer.set_direction(-120, 25, 1.4)
            chunks.append(renderer.process_block(
                x[k * block_size:(k + 1) * block_size]))
        return np.concatenate(chunks)

    fixed = render(False)
    switched = render(True)
    fixed_jump = np.max(np.abs(np.diff(fixed, axis=0)))
    switched_jump = np.max(np.abs(np.diff(switched, axis=0)))
    delta = switched - fixed
    delta_jump = np.max(np.abs(np.diff(delta, axis=0)))
    # A click would appear as a near full-scale discontinuity. The switched
    # render must stay well below the 0.5 signal peak (i.e. no impulse), and
    # its per-sample growth must remain bounded relative to a smooth render.
    signal_peak = float(np.max(np.abs(fixed)))
    smooth = (switched_jump < 0.8 * signal_peak
              and switched_jump <= fixed_jump * 5.0 + 1e-9
              and delta_jump <= signal_peak)
    return {"passed": bool(smooth),
            "signal_peak": signal_peak,
            "fixed_render_max_jump": float(fixed_jump),
            "switched_render_max_jump": float(switched_jump),
            "render_difference_max_jump": float(delta_jump)}


def check_callback_bounded_io(block_size: int) -> dict:
    audit = audio_thread_io_audit()
    renderer = _fresh(block_size)
    signal = broadband_signal(duration_s=1.0)
    player = RealtimeBinauralPlayer(renderer, signal, loop=True)
    out = player.simulate_blocks(6)
    return {"passed": bool(audit["passed"] and out.shape == (6 * block_size, 2)),
            **audit,
            "simulated_shape": list(out.shape)}


def run_all_checks(block_size: int = 512) -> dict:
    checks = {
        "left_right_symmetry": check_left_right_symmetry(block_size),
        "behind_difference": check_behind_difference(block_size),
        "real_convolution": check_real_convolution(block_size),
        "documented_latency": check_latency(block_size),
        "head_turn_equals_source_motion": check_head_turn_equivalence(
            block_size),
        "crossfade_no_click": check_crossfade_no_click(block_size),
        "callback_no_file_io": check_callback_bounded_io(block_size),
    }
    return {"block_size": block_size,
            "all_passed": all(v["passed"] for v in checks.values()),
            "checks": checks}
