"""Measurement helpers for the two handover implementations.

Everything here drives the real :class:`~binaural.renderer.BinauralRenderer`
-- no offline mock -- and reads back the quantities the exhibit promises:

* :func:`impulse_train` -- isolated pulses spaced wider than one HRIR plus one
  handover ramp, so every pulse's rendered response can be windowed on its
  own and its interaural delay read exactly.  A filter handover that keeps
  the old tail blends two HRIRs with different ITDs, which shows up as a
  measurable **ITD drift**; a state rebuild switches it cleanly.
* :func:`bass_tone` -- a sustained low tone.  Any cut, dropped tail or
  filter swap leaves a sample-to-sample **step**; the peak first difference
  bounds it.
* :func:`render_turn_scenario` -- renders two consecutive head turns and
  reports ITD drift, transient peak, and how many filters were simultaneously
  alive (read from the renderer's own bookkeeping).

Timings are absolute output-sample positions, so a scenario can be shifted by
an arbitrary offset -- including half a block -- and compared after
re-alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .handover import AUTO, estimate_itd
from .hrtf import ProceduralHRTF
from .renderer import BinauralRenderer

ALGORITHM_LATENCY_BLOCKS = 1


# ----------------------------------------------------------------------
# Test signals
# ----------------------------------------------------------------------
def impulse_train(sample_rate: int = 48000, duration_s: float = 0.7,
                  spacing_s: float = 0.08,
                  amplitude: float = 0.8) -> np.ndarray:
    """Isolated pulses: each response is one HRIR, windowable on its own.

    80 ms spacing is far longer than the 128-sample HRIR plus a 10 ms
    handover ramp, so neighbouring pulses never overlap.
    """
    n = int(round(duration_s * sample_rate))
    x = np.zeros(n)
    step = max(1, int(round(spacing_s * sample_rate)))
    x[::step] = amplitude
    return x


def bass_tone(sample_rate: int = 48000, duration_s: float = 0.7,
              frequency_hz: float = 80.0,
              amplitude: float = 0.6) -> np.ndarray:
    """Sustained low tone: smooth enough that a cut is an obvious step."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency_hz * t)


# ----------------------------------------------------------------------
# Per-pulse ITD readout
# ----------------------------------------------------------------------
def pulse_locations(signal: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Input-sample indices where ``signal`` contains a pulse."""
    idx = np.flatnonzero(np.abs(signal) > threshold * np.max(np.abs(signal)))
    if idx.size == 0:
        return idx
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return np.array([int(g[0]) for g in groups], dtype=int)


def per_pulse_itds(stereo: np.ndarray, pulse_input_indices: np.ndarray,
                   sample_rate: int, block_size: int,
                   hrir_length: int) -> list[tuple[int, float]]:
    """ITD of each pulse response, paired with its output sample position.

    The renderer has one block of algorithmic latency, so the input pulse at
    ``n`` first reaches the output at ``n + block_size``.  Each pulse's HRIR
    is short, so a window of one block either side isolates it.
    """
    n = stereo.shape[0]
    half = block_size
    out = []
    for p in pulse_input_indices:
        start = int(p) + block_size - half // 2
        stop = start + hrir_length + half
        if start < 0 or stop > n:
            continue
        seg = stereo[start:stop]
        if np.max(np.abs(seg)) < 1e-9:
            continue
        itd = estimate_itd(seg, sample_rate, max_lag=hrir_length // 2)
        out.append((int(p) + block_size, float(itd)))  # output sample
    return out


def measure_transient(stereo: np.ndarray) -> float:
    """Largest first difference across both channels (switch click bound)."""
    if stereo.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(stereo, axis=0))))


# ----------------------------------------------------------------------
# Scenario rendering
# ----------------------------------------------------------------------
@dataclass
class TurnResult:
    stereo: np.ndarray
    sample_rate: int
    block_size: int
    first_turn_sample: int
    second_turn_sample: int
    applied_second_turn_sample: int
    quantization_samples: int
    mode: str
    itd_drift_samples: float
    overlap_itd_drift_samples: float
    transient_peak: float
    max_alive_filters: int
    alive_at_second_turn: int
    alive_over_cap_samples: int
    alive_history: np.ndarray
    handover_log: list
    final_pose_itd_samples: float
    first_pose_itd_samples: float
    itd_track: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "block_size": self.block_size,
            "sample_rate": self.sample_rate,
            "first_turn_sample": self.first_turn_sample,
            "second_turn_sample": self.second_turn_sample,
            "applied_second_turn_sample": self.applied_second_turn_sample,
            "quantization_samples": self.quantization_samples,
            "second_turn_delay_samples": (
                self.second_turn_sample - self.first_turn_sample),
            "mode": self.mode,
            "itd_drift_samples": self.itd_drift_samples,
            "overlap_itd_drift_samples": self.overlap_itd_drift_samples,
            "transient_peak": self.transient_peak,
            "max_alive_filters": self.max_alive_filters,
            "alive_at_second_turn": self.alive_at_second_turn,
            "alive_over_cap_samples": self.alive_over_cap_samples,
            "final_pose_itd_samples": self.final_pose_itd_samples,
            "first_pose_itd_samples": self.first_pose_itd_samples,
            "itd_track": [{"sample": s, "itd_samples": v}
                          for s, v in self.itd_track],
            "handover_log": self.handover_log,
        }


def render_turn_scenario(block_size: int = 512, sample_rate: int = 48000,
                         signal: np.ndarray | None = None,
                         first_pose: tuple[float, float] = (0.0, 0.0),
                         second_pose: tuple[float, float] = (-90.0, 0.0),
                         final_pose: tuple[float, float] = (90.0, 0.0),
                         first_turn_sample: int = 4 * 512,
                         second_turn_delay_samples: int = 512,
                         mode: str = AUTO,
                         fade_ms: float = 5.0,
                         shift_samples: int = 0,
                         measure: str = "pulse") -> TurnResult:
    """Render ``first_pose -> second_pose -> final_pose`` and measure it.

    ``mode`` may be ``auto``, ``parallel_tail`` or ``state_rebuild``.  Calling
    again with ``shift_samples`` set renders the identical motion record
    displaced by that many samples, which is how the half-block alignment
    check is performed.
    """
    if signal is None:
        signal = (impulse_train(sample_rate) if measure == "pulse"
                  else bass_tone(sample_rate))

    hrtf = ProceduralHRTF(sample_rate=sample_rate)
    b = block_size

    # Shift the whole content timeline -- audio samples *and* the motion
    # record -- against the fixed block grid.  That is the invariant under
    # test: the head's motion relative to the sound is unchanged, and the
    # renderer should reproduce the same output slid by ``shift_samples``.
    at1 = int(first_turn_sample) + int(shift_samples)
    at2 = at1 + int(second_turn_delay_samples)
    shifted_signal = np.concatenate([
        np.zeros(int(shift_samples)), signal,
    ])

    renderer = BinauralRenderer(
        hrtf, block_size=b, crossfade_ms=fade_ms,
        apply_distance_gain=False, handover_mode=mode)
    renderer.set_direction(first_pose[0], first_pose[1], 1.2)
    renderer.schedule_direction(second_pose[0], second_pose[1], 1.2,
                                at_sample=at1, mode=mode)
    renderer.schedule_direction(final_pose[0], final_pose[1], 1.2,
                                at_sample=at2, mode=mode)

    n_blocks = int(np.ceil((at2 + 14 * b) / b))
    padded = np.zeros(n_blocks * b)
    take = min(shifted_signal.size, padded.size)
    padded[:take] = shifted_signal[:take]
    out = np.zeros((n_blocks * b, 2))
    for k in range(n_blocks):
        out[k * b:(k + 1) * b] = renderer.process_block(
            padded[k * b:(k + 1) * b])

    q_final = hrtf.query(final_pose[0], final_pose[1])
    q_first = hrtf.query(first_pose[0], first_pose[1])
    final_ref = estimate_itd(q_final.hrir.T, sample_rate)
    first_ref = estimate_itd(q_first.hrir.T, sample_rate)

    fade = renderer.parallel_fade_samples
    log = renderer.handover_log
    applied = int(log[-1]["sample"]) if log else at2
    quantum = 0  # sample-accurate engine: applied sample equals requested

    track: list[tuple[int, float]] = []
    overlap_track: list[tuple[int, float]] = []
    drift = 0.0
    overlap_drift = 0.0
    if measure == "pulse":
        locs = pulse_locations(padded)
        all_itds = per_pulse_itds(out, locs, sample_rate, b,
                                  hrtf.hrir_length)
        guard = renderer.rebuild_ramp_samples

        def fully_after_cut(sample: int) -> bool:
            # The per-pulse measurement window starts half a block before the
            # pulse; require the whole window to begin after the short
            # de-click ramp so the pulse's response is not itself split
            # across the cut (a split pulse legitimately reads a blended ITD
            # and would be mis-attributed to filter drift).
            return sample - b // 2 >= applied + guard

        lo = applied + guard
        hi = applied + fade
        track = [(s, v) for s, v in all_itds if fully_after_cut(s) and s <= hi]
        if track:
            drift = max(abs(v - final_ref) for _, v in track)
        # Drift attributable to *overlap* (three or more voices alive) rather
        # than to a single legitimate crossfade.
        alive_by_block = list(renderer.alive_history)
        for s, v in track:
            blk = s // b
            if 0 <= blk < len(alive_by_block) and alive_by_block[blk] > 2:
                overlap_track.append((s, v))
        if overlap_track:
            overlap_drift = max(abs(v - final_ref) for _, v in overlap_track)

    transient = (measure_transient(out[max(0, applied - b):
                                      applied + fade + b])
                 if measure == "bass" else 0.0)

    alive_at_turn = [
        int(a) for k, a in enumerate(renderer.alive_history)
        if k * b <= applied + fade and (k + 1) * b >= applied
    ]

    return TurnResult(
        stereo=out,
        sample_rate=sample_rate,
        block_size=b,
        first_turn_sample=at1,
        second_turn_sample=at2,
        applied_second_turn_sample=applied,
        quantization_samples=quantum,
        mode=mode,
        itd_drift_samples=float(drift),
        overlap_itd_drift_samples=float(overlap_drift),
        transient_peak=float(transient),
        max_alive_filters=int(renderer.max_alive_filters),
        alive_at_second_turn=max(alive_at_turn) if alive_at_turn else 0,
        alive_over_cap_samples=int(renderer.alive_over_cap_samples),
        alive_history=np.asarray(renderer.alive_history, dtype=int),
        handover_log=log,
        final_pose_itd_samples=float(final_ref),
        first_pose_itd_samples=float(first_ref),
        itd_track=track,
    )


def aligned_compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two renders that should differ only by a known sample shift."""
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return {"samples": 0, "max_abs_diff": 0.0, "rms_diff": 0.0}
    d = a[:n] - b[:n]
    return {
        "samples": int(n),
        "max_abs_diff": float(np.max(np.abs(d))),
        "rms_diff": float(np.sqrt(np.mean(d ** 2))),
        "max_abs_a": float(np.max(np.abs(a[:n]))),
        "max_abs_b": float(np.max(np.abs(b[:n]))),
    }
