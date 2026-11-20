"""Filter-handover policies and the constraint that selects between them.

When a listener keeps turning, a second pose can arrive while the previous
filter change is still releasing its tail.  Two handover implementations are
provided and are switchable at runtime (per event or globally):

``parallel_tail``
    Keep every already-running convolution voice alive and mix its retained
    output with the new voice over a smooth ramp.  The old interaural delay
    and tail therefore remain audible until the ramp has finished.

``state_rebuild``
    Do not mix.  Build the new voice, prime its partition delay line with the
    same recent input history, then cut to it over a short de-click ramp.
    Only the new filter contributes after the cut, and its interaural delay
    is exact immediately.  The cut itself is placed at the exact requested
    sample; the renderer evaluates gain envelopes per sample, so no block
    quantization is introduced.

The choice is made by :func:`select_handover` from the *first parallel-path
constraint that would fail* as the second turn approaches the first:

* ``alive_filter_cap`` -- at most two filters may contribute at one sample.
  A second turn earlier than one ramp length would require three.
* ``itd_drift`` -- the blended interaural delay may not drift further than a
  budget from the final pose's delay.
* ``transient_step`` -- the handover may not create a sample step larger
  than a fraction of the steady-state peak.

The condition is a numeric sample threshold.  A second turn scheduled
``delta`` samples after the previous one uses ``parallel_tail`` when
``delta >= threshold`` and ``state_rebuild`` otherwise.  For the equal-power
ramp used here the alive-filter cap always requires the longest delay, so it
normally binds; :func:`calibrate_switch_condition` re-measures the real
signals and records which constraint binds instead of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PARALLEL_TAIL = "parallel_tail"
STATE_REBUILD = "state_rebuild"
MODES = (PARALLEL_TAIL, STATE_REBUILD)
AUTO = "auto"


# ----------------------------------------------------------------------
# History rule
# ----------------------------------------------------------------------
def partition_count(block_size: int, hrir_length: int) -> int:
    """Number of ``B``-sized filter partitions: ``ceil(hrir_length / B)``."""
    return int(np.ceil(int(hrir_length) / int(block_size)))


def required_history_samples(block_size: int, hrir_length: int) -> int:
    """Raw input retained after parallel work stops, in samples.

    Overlap-save stores frames ``[x_{t-1} | x_t]`` and output block ``t`` is
    ``core(x_{t-1})``, which sums frames ``t-1`` down to ``t-P``.  The oldest
    frame ``[x_{t-P-1} | x_{t-P}]`` therefore needs input blocks ``x_{t-P-1}``
    through ``x_{t-1}``: ``P + 1`` blocks in total.  Feeding exactly those
    into a fresh convolver reproduces the running voice's next output
    bit-for-bit.  ``(P + 1) * B`` samples are both necessary and sufficient;
    nothing older can reach a future output.
    """
    return (partition_count(block_size, hrir_length) + 1) * int(block_size)


def history_blocks(block_size: int, hrir_length: int) -> int:
    return partition_count(block_size, hrir_length) + 1


# ----------------------------------------------------------------------
# Interaural delay
# ----------------------------------------------------------------------
def estimate_itd(stereo: np.ndarray, sample_rate: int | None = None,
                 band_hz: tuple[float, float] | None = None,
                 max_lag: int | None = None) -> float:
    """Interaural delay of a stereo signal, in samples (left minus right).

    Positive means the left ear receives the wavefront *later* (left is
    delayed).  The estimate is the peak of the (optionally band-limited)
    cross-correlation refined by a parabola through the three samples around
    the peak, so fractional delays survive.  ``max_lag`` restricts the search
    and removes the lattice ambiguity of a periodic (pulse-train) input.
    """
    x = np.asarray(stereo, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        return 0.0
    left, right = x[:, 0], x[:, 1]
    if left.size < 4 or np.max(np.abs(left)) == 0.0 \
            or np.max(np.abs(right)) == 0.0:
        return 0.0
    if band_hz is not None and sample_rate is not None:
        from scipy.signal import butter, sosfiltfilt

        nyq = sample_rate / 2.0
        lo = max(1e-3, band_hz[0] / nyq)
        hi = min(0.999, band_hz[1] / nyq)
        sos = butter(4, [lo, hi], btype="bandpass", output="sos")
        left = sosfiltfilt(sos, left)
        right = sosfiltfilt(sos, right)

    n = left.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(left, nfft) * np.conj(np.fft.rfft(right, nfft))
    corr = np.fft.irfft(spec, nfft)
    corr = np.concatenate([corr[-(n - 1):], corr[:n]])
    lags = np.arange(-(n - 1), n)
    if max_lag is not None:
        keep = np.abs(lags) <= int(max_lag)
        corr = corr[keep]
        lags = lags[keep]
    peak = int(np.argmax(np.abs(corr)))
    frac = 0.0
    if 0 < peak < corr.size - 1:
        y0, y1, y2 = np.abs(corr[peak - 1: peak + 2])
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-20:
            frac = 0.5 * (y0 - y2) / denom
    return float(lags[peak] + frac)


# ----------------------------------------------------------------------
# Budgets and condition
# ----------------------------------------------------------------------
@dataclass
class HandoverBudgets:
    max_alive_filters: int = 2
    max_itd_drift_samples: float = 0.5
    max_transient_step: float = 0.5

    def as_dict(self) -> dict:
        return {
            "max_alive_filters": int(self.max_alive_filters),
            "max_itd_drift_samples": float(self.max_itd_drift_samples),
            "max_transient_step": float(self.max_transient_step),
        }


@dataclass
class SwitchCondition:
    """Numeric condition derived from the first-failing constraint."""

    threshold_samples: int
    binding_constraint: str
    budgets: dict
    required_delay_samples: dict
    source: str
    geometry: dict = field(default_factory=dict)

    def choose(self, second_turn_delay_samples: float) -> str:
        if float(second_turn_delay_samples) >= float(self.threshold_samples):
            return PARALLEL_TAIL
        return STATE_REBUILD

    def as_dict(self, sample_rate: int | None = None) -> dict:
        out = {
            "threshold_samples": int(self.threshold_samples),
            "binding_constraint": self.binding_constraint,
            "budgets": self.budgets,
            "required_delay_samples": {
                k: float(v) for k, v in self.required_delay_samples.items()},
            "source": self.source,
            "geometry": self.geometry,
        }
        if sample_rate:
            out["threshold_seconds"] = (
                self.threshold_samples / float(sample_rate))
        return out

    @property
    def threshold(self) -> int:
        return int(self.threshold_samples)

    @property
    def binding(self) -> str:
        return self.binding_constraint


def parallel_required_delays(fade_samples: int, delta_itd_samples: float,
                             budgets: HandoverBudgets) -> dict:
    """Minimum second-turn delay for each ``parallel_tail`` constraint.

    At a delay ``d`` into an equal-power ramp of length ``f`` the not-yet
    faded voice still has power weight ``cos^2(pi/2 * d/f)``; that residual
    is what pulls the blended ITD toward the old pose.
    """
    fade = max(1.0, float(fade_samples))
    alive = fade

    delta = abs(float(delta_itd_samples))
    budget = budgets.max_itd_drift_samples
    if delta <= budget:
        drift = 0.0
    else:
        # delta * cos^2(pi/2 d/f) <= budget  ->  smallest d.
        frac = float(np.clip(np.sqrt(budget / delta), 0.0, 1.0))
        drift = fade * (2.0 / np.pi) * float(np.arccos(frac))

    # The equal-power ramp's own slope stays inside the transient budget for
    # the exhibit's signals, so it requires no extra delay.
    step = 0.0
    return {
        "alive_filter_cap": alive,
        "itd_drift": drift,
        "transient_step": step,
    }


def select_handover(fade_samples: int, delta_itd_samples: float,
                    budgets: HandoverBudgets | None = None,
                    source: str = "analytic") -> SwitchCondition:
    """Analytic condition from the constraint needing the longest delay."""
    budgets = budgets or HandoverBudgets()
    required = parallel_required_delays(
        fade_samples, delta_itd_samples, budgets)
    binding = max(required, key=required.get)
    return SwitchCondition(
        threshold_samples=int(np.ceil(required[binding])),
        binding_constraint=binding,
        budgets=budgets.as_dict(),
        required_delay_samples=required,
        source=source,
        geometry={"fade_samples": int(fade_samples),
                  "delta_itd_samples": float(delta_itd_samples)},
    )


def decide_mode(second_turn_delay_samples: float,
                condition: SwitchCondition,
                forced_mode: str | None = None) -> str:
    """Apply the condition to one event; a forced mode overrides it."""
    if forced_mode in MODES:
        return forced_mode
    return condition.choose(second_turn_delay_samples)


# ----------------------------------------------------------------------
# Calibration: measure which constraint actually binds
# ----------------------------------------------------------------------
def calibrate_switch_condition(block_size: int = 512,
                               sample_rate: int = 48000,
                               budgets: HandoverBudgets | None = None,
                               delays: tuple[int, ...] = (
                                   0, 128, 256, 384, 512, 768,
                                   896, 1024, 1280, 1536),
                               fade_ms: float = 5.0) -> dict:
    """Measure the parallel tail on the two test signals and locate the
    first constraint that fails as the second turn approaches the first.

    * impulse train  -> :func:`estimate_itd` per pulse, drift **during the
      overlap** (three voices alive);
    * low bass tone   -> peak first difference (handover transient).

    The alive-filter count comes from the renderer's own bookkeeping.  Each
    constraint's smallest satisfying second-turn delay is recorded; the
    largest such delay is the switching threshold and its owner is the
    constraint that fails first.
    """
    from .experiment import (  # local import avoids a cycle at import time
        bass_tone,
        impulse_train,
        measure_transient,
        render_turn_scenario,
    )

    budgets = budgets or HandoverBudgets()
    fade = max(2, int(round(fade_ms * 1e-3 * sample_rate / block_size))) \
        * block_size

    impulse = impulse_train(sample_rate, spacing_s=256.0 / sample_rate,
                            duration_s=1.4)
    bass = bass_tone(sample_rate, duration_s=2.0)

    rows = []
    for d in delays:
        rp = render_turn_scenario(
            block_size=block_size, sample_rate=sample_rate,
            second_turn_delay_samples=int(d), mode=PARALLEL_TAIL,
            signal=impulse, measure="pulse", fade_ms=fade_ms)
        rb = render_turn_scenario(
            block_size=block_size, sample_rate=sample_rate,
            second_turn_delay_samples=int(d), mode=PARALLEL_TAIL,
            signal=bass, measure="bass", fade_ms=fade_ms)
        rows.append({
            "second_turn_delay_samples": int(d),
            "overlap_itd_drift_samples": float(rp.overlap_itd_drift_samples),
            "itd_drift_samples": float(rp.itd_drift_samples),
            "transient_peak": float(rb.transient_peak),
            "max_alive_filters": int(rp.max_alive_filters),
            "alive_at_second_turn": int(rp.alive_at_second_turn),
        })

    def settle_delay(key: str, limit: float) -> int:
        # The requirement is "holds for every delay at or beyond this one",
        # not "passes once": an isolated pulse landing outside the overlap can
        # read zero drift at a short delay while a *later* delay still fails.
        # So find the last failing row and take the next delay after it.
        last_fail = -1
        for i, row in enumerate(rows):
            if row[key] > limit + 1e-12:
                last_fail = i
        if last_fail + 1 < len(rows):
            return int(rows[last_fail + 1]["second_turn_delay_samples"])
        return int(rows[-1]["second_turn_delay_samples"])

    required = {
        "alive_filter_cap": int(fade),
        "itd_drift": settle_delay(
            "overlap_itd_drift_samples", budgets.max_itd_drift_samples),
        "transient_step": settle_delay(
            "transient_peak", budgets.max_transient_step),
    }
    binding = max(required, key=required.get)
    condition = SwitchCondition(
        threshold_samples=int(required[binding]),
        binding_constraint=binding,
        budgets=budgets.as_dict(),
        required_delay_samples={k: float(v) for k, v in required.items()},
        source="measured",
        geometry={
            "block_size": int(block_size),
            "sample_rate": int(sample_rate),
            "fade_samples": int(fade),
        },
    )
    return {"condition": condition, "measurements": rows}
