"""Binaural renderer: live HRTF selection, interpolation and filter handover.

Spatialization always comes from two independently convolved channels -- no
stereo-balance shortcut anywhere in this module.  When the relative direction
changes, the renderer hands the stream from the running convolution voice(s)
to a new one using one of two implementations, selectable per event or
globally (see :mod:`binaural.handover`):

``parallel_tail``
    The new voice is primed from the retained input history, so it is in
    sync and exact, but the old voice is *kept and mixed* for the whole
    equal-power ramp.  Its tail and its interaural delay therefore remain
    audible across the whole overlap; a second turn inside that ramp leaves
    three voices contributing at once.

``state_rebuild``
    The new voice is rebuilt from exactly the same retained input history,
    then cut to over a very short de-click ramp.  No old tail is retained:
    after at most ``rebuild_ramp_samples`` only the new filter contributes and
    its interaural delay is exact.

``auto`` (default) applies the numeric :class:`SwitchCondition` derived from
the first parallel-path constraint that would fail; that condition is what
actually routes playback.  A second turn earlier than the condition's
threshold selects ``state_rebuild``, otherwise ``parallel_tail``.

Sample accuracy
---------------
Every live voice processes the same input block, so all voices are frame
aligned.  Handover events carry an absolute output-sample position; the gain
envelopes change on exactly that sample, inside whichever block contains it.
Shifting an entire content timeline (audio *and* motion) by any number of
samples therefore reproduces the same output shifted by that many samples --
including a half-block shift -- leaving only the filter cut itself as a
possible difference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .convolution import PartitionedConvolver
from .coordinates import relative_direction
from .handover import (
    AUTO,
    MODES,
    PARALLEL_TAIL,
    STATE_REBUILD,
    HandoverBudgets,
    SwitchCondition,
    decide_mode,
    estimate_itd,
    history_blocks as history_blocks_rule,
    required_history_samples,
    select_handover,
)
from .hrtf import HRTFBase, HRTFQuery


@dataclass
class DirectionRecord:
    block_index: int
    time_seconds: float
    world_source: tuple
    listener_position: tuple
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    distance_gain: float
    hrtf_source: str
    crossfading: bool
    handover_mode: str = ""
    alive_filters: int = 0


@dataclass
class _Voice:
    """One running binaural convolution voice and its gain envelope."""

    convs: list
    record: HRTFQuery
    born_at: int
    fade_in_len: int
    fade_out_start: Optional[int] = None
    fade_out_len: int = 0
    fade_out_g0: float = 1.0
    kill_at: Optional[int] = None

    def gain_at(self, t: np.ndarray) -> np.ndarray:
        """Raw (un-normalized) equal-power envelope for this voice.

        Callers normalize across all live voices per sample, so adding or
        dropping a voice at zero gain never leaves a level hole.
        """
        t = np.asarray(t, dtype=float)
        if self.fade_in_len > 0:
            u = np.clip((t - self.born_at) / float(self.fade_in_len), 0.0, 1.0)
            g = np.sin(0.5 * np.pi * u)
        else:
            g = (t >= self.born_at).astype(float)
        if self.fade_out_start is not None:
            u = np.clip((t - self.fade_out_start)
                        / float(max(1, self.fade_out_len)), 0.0, 1.0)
            g_out = self.fade_out_g0 * np.cos(0.5 * np.pi * u)
            g = np.where(t >= self.fade_out_start, g_out, g)
        return g

    def gain_scalar(self, t: int) -> float:
        return float(self.gain_at(np.array([float(t)]))[0])

    def remaining_fade(self, t: int) -> int:
        ends = [self.born_at + self.fade_in_len]
        if self.fade_out_start is not None:
            ends.append(self.fade_out_start + self.fade_out_len)
        return max(0, max(ends) - int(t))


@dataclass
class _Event:
    at: int
    query: HRTFQuery
    mode: str
    ramp: int
    distance_m: Optional[float]
    tag: str = "handover"


class BinauralRenderer:
    def __init__(self, hrtf: HRTFBase, block_size: int = 512,
                 crossfade_ms: float = 5.0,
                 reference_distance_m: float = 1.0,
                 apply_distance_gain: bool = True,
                 handover_mode: str = AUTO,
                 handover_budgets: Optional[HandoverBudgets] = None,
                 switch_condition: Optional[SwitchCondition] = None,
                 rebuild_ramp_samples: int = 32):
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError("block_size must be a positive power of two")
        if handover_mode not in (AUTO,) + MODES:
            raise ValueError(f"handover_mode must be one of {(AUTO,) + MODES}")
        self.hrtf = hrtf
        self.sample_rate = hrtf.sample_rate
        self.block_size = int(block_size)
        requested = max(
            1, int(round(crossfade_ms * 1e-3 * self.sample_rate / block_size)))
        # At least two blocks so the equal-power ramp spans a real interval.
        self.crossfade_blocks = max(2, requested)
        self.parallel_fade_samples = self.crossfade_blocks * self.block_size
        self.fade_samples = self.parallel_fade_samples
        self.rebuild_ramp_samples = max(1, int(rebuild_ramp_samples))
        self.reference_distance_m = float(reference_distance_m)
        self.apply_distance_gain = bool(apply_distance_gain)

        self.handover_mode = handover_mode
        self.handover_budgets = handover_budgets or HandoverBudgets()
        self.history_blocks = history_blocks_rule(
            self.block_size, hrtf.hrir_length)
        self.required_history_samples = required_history_samples(
            self.block_size, hrtf.hrir_length)

        if switch_condition is None:
            switch_condition = select_handover(
                self.parallel_fade_samples, self._representative_itd_spread(),
                self.handover_budgets)
        self.switch_condition = switch_condition

        # Bank view kept for audit/back-compat; mirrors the newest voice.
        self._banks = [[], []]
        self._active = 0
        self._fade_pos = 0
        self._current_query: Optional[HRTFQuery] = None
        self._voices: list[_Voice] = []
        self._pending: list[_Event] = []
        self._history: deque = deque(maxlen=self.history_blocks)
        self._last_handover_at: Optional[int] = None
        self._prev_event_sample: Optional[int] = None
        self._last_mode = ""
        self._distance_gain = 1.0
        self._target_gain = 1.0
        self._last_distance = None
        self._block_count = 0
        self.max_alive_filters = 0
        self.alive_over_cap_samples = 0
        self.alive_history: list[int] = []
        self.handover_log: list[dict] = []
        self.direction_log: list[DirectionRecord] = []

    # ------------------------------------------------------------------
    @property
    def algorithm_latency_samples(self) -> int:
        return self.block_size

    @property
    def algorithm_latency_seconds(self) -> float:
        return self.block_size / self.sample_rate

    @property
    def current_sample(self) -> int:
        """Absolute output-sample position of the next block to process."""
        return self._block_count * self.block_size

    def clear_pending(self) -> int:
        """Drop handovers that have not fired yet; returns the count removed.

        The listener UI uses this to re-arm a two-turn gesture when the user
        drags the second turn's moment: nothing that already happened is
        touched, only future events are withdrawn.
        """
        removed = len(self._pending)
        self._pending.clear()
        self._last_handover_at = self._prev_event_sample
        return removed

    def cancel_pending(self, tags=None) -> int:
        """Withdraw only the pending events whose ``tag`` is in ``tags``."""
        if tags is None:
            return self.clear_pending()
        keep = [e for e in self._pending if e.tag not in tags]
        removed = len(self._pending) - len(keep)
        self._pending = keep
        if removed:
            self._last_handover_at = max(
                [e.at for e in self._pending], default=self._prev_event_sample)
        return removed

    def pending_events(self) -> list[dict]:
        return [{"at": e.at, "mode": e.mode, "tag": e.tag,
                 "azimuth_deg": e.query.azimuth_deg,
                 "elevation_deg": e.query.elevation_deg}
                for e in self._pending]

    def active_hrir(self) -> np.ndarray:
        if self._current_query is None:
            raise RuntimeError("set_direction must be called before processing")
        return self._current_query.hrir

    # ------------------------------------------------------------------
    def _representative_itd_spread(self) -> float:
        try:
            a = estimate_itd(self.hrtf.query(90, 0).hrir.T, self.sample_rate)
            b = estimate_itd(self.hrtf.query(-90, 0).hrir.T, self.sample_rate)
            return abs(a - b)
        except Exception:  # pragma: no cover - exotic HRTF geometry
            return 1.0

    def _build_bank(self, hrir: np.ndarray):
        return [
            PartitionedConvolver(hrir[0], self.block_size),
            PartitionedConvolver(hrir[1], self.block_size),
        ]

    def _next_block_start(self) -> int:
        return self._block_count * self.block_size

    # ------------------------------------------------------------------
    # Direction scheduling
    # ------------------------------------------------------------------
    def set_direction(self, azimuth_deg: float, elevation_deg: float,
                      distance_m: Optional[float] = None,
                      at_sample: Optional[int] = None,
                      mode: Optional[str] = None) -> HRTFQuery:
        """Select a new direction and schedule its handover.

        ``at_sample`` is an absolute output-sample position (defaults to the
        start of the next block).  ``mode`` overrides the automatic condition
        for this event.  Re-selecting the identical direction is a no-op.
        """
        query = self.hrtf.query(azimuth_deg, elevation_deg)
        if self._current_query is None:
            self._current_query = query
            self._create_initial_voice(query)
            self._apply_distance(distance_m)
            return query

        if self._same_direction(query):
            self._apply_distance(distance_m)
            return query

        at = self._next_block_start() if at_sample is None else int(at_sample)
        self._schedule_event(query, at, mode, distance_m)
        # Legacy contract: the latest request is reported at once.
        self._current_query = query
        return query

    def schedule_direction(self, azimuth_deg: float, elevation_deg: float,
                           distance_m: Optional[float] = None,
                           at_sample: Optional[int] = None,
                           mode: Optional[str] = None,
                           tag: str = "handover") -> HRTFQuery:
        """Queue a handover without overwriting the reported current pose.

        The listening UI uses this for the *second* turn, which may be placed
        in the future while the first turn is still releasing.
        """
        if self._current_query is None:
            return self.set_direction(
                azimuth_deg, elevation_deg, distance_m, at_sample, mode)
        query = self.hrtf.query(azimuth_deg, elevation_deg)
        if self._same_direction(query):
            return query
        at = self._next_block_start() if at_sample is None else int(at_sample)
        self._schedule_event(query, at, mode, distance_m, tag)
        return query

    def _same_direction(self, query: HRTFQuery) -> bool:
        return (abs(query.azimuth_deg - self._current_query.azimuth_deg) <= 1e-12
                and abs(query.elevation_deg
                        - self._current_query.elevation_deg) <= 1e-12)

    def _schedule_event(self, query: HRTFQuery, at: int,
                        mode: Optional[str],
                        distance_m: Optional[float],
                        tag: str = "handover") -> None:
        if self._last_handover_at is None:
            delta: float = np.inf
        else:
            delta = float(at - self._last_handover_at)
        forced = mode if mode in MODES else (
            None if self.handover_mode == AUTO else self.handover_mode)
        resolved = decide_mode(delta, self.switch_condition, forced_mode=forced)
        ramp = (self.parallel_fade_samples if resolved == PARALLEL_TAIL
                else self.rebuild_ramp_samples)
        self._pending.append(
            _Event(at, query, resolved, ramp, distance_m, tag))
        self._pending.sort(key=lambda e: e.at)
        self._last_handover_at = at
        self._last_mode = resolved
        self._fade_pos = ramp

    def _create_initial_voice(self, query: HRTFQuery, at: int = 0) -> None:
        voice = _Voice(convs=self._build_bank(query.hrir), record=query,
                       born_at=at, fade_in_len=0)
        self._voices = [voice]
        self._history.clear()
        self._banks = [voice.convs, []]
        self._active = 0

    def _set_bank_view(self, voice: _Voice) -> None:
        self._active = 1 - self._active
        self._banks[self._active] = voice.convs

    def _apply_distance(self, distance_m: Optional[float]) -> None:
        if distance_m is not None:
            self._target_gain = self.reference_distance_m / max(
                distance_m, 1e-6)
            self._last_distance = float(distance_m)

    def update_pose(self, source_position, listener_position,
                    yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                    roll_deg: float = 0.0) -> HRTFQuery:
        """World-space convenience: geometry, then :meth:`set_direction`."""
        rel = relative_direction(
            np.asarray(source_position, dtype=float),
            np.asarray(listener_position, dtype=float),
            yaw_deg, pitch_deg, roll_deg)
        query = self.set_direction(rel.azimuth_deg, rel.elevation_deg,
                                   rel.distance_m)
        self._last_geometry = (
            tuple(np.asarray(source_position, dtype=float)),
            tuple(np.asarray(listener_position, dtype=float)),
            yaw_deg, pitch_deg, roll_deg, rel)
        return query

    # ------------------------------------------------------------------
    # Voice firing
    # ------------------------------------------------------------------
    def _prime(self, convs) -> None:
        """Replay retained input blocks so a new voice is frame-exact."""
        for block in self._history:
            convs[0].process_block(block)
            convs[1].process_block(block)

    def _fire(self, event: _Event, abs0: int) -> None:
        """Create the new voice for an event at sample ``event.at``.

        Called at the start of the block that contains ``event.at``, before
        any voice has consumed this block's input, so ``_history`` still ends
        with the previous block and priming is exact.  Every existing voice
        begins fading out on exactly the event sample; an already-running
        fade-out is never restarted.
        """
        s = int(event.at)
        for voice in self._voices:
            # A state rebuild cuts cleanly: retire *every* older voice over
            # the short de-click ramp, including one whose long ramp is still
            # running from an earlier turn.  Restarting a fade at its present
            # gain is continuous, so this cannot resurrect a decayed sample.
            # The parallel path instead leaves an in-progress fade alone, so
            # its old tail runs to its natural end and stays audible.
            if (event.mode == STATE_REBUILD
                    or voice.fade_out_start is None
                    or voice.fade_out_start > s):
                g0 = voice.gain_scalar(s)
                voice.fade_out_start = s
                voice.fade_out_len = event.ramp
                voice.fade_out_g0 = g0
                voice.kill_at = s + event.ramp

        convs = self._build_bank(event.query.hrir)
        # Both modes rebuild the new filter from the same retained history, so
        # it is frame-exact from its first sample; they differ in what happens
        # to the old voice, not in how the new one is started.
        self._prime(convs)
        voice = _Voice(convs=convs, record=event.query, born_at=s,
                       fade_in_len=event.ramp)
        self._voices.append(voice)
        self._set_bank_view(voice)
        if event.distance_m is not None:
            self._apply_distance(event.distance_m)

        self.handover_log.append({
            "tag": event.tag,
            "requested_sample": s,
            "sample": int(s),
            "mode": event.mode,
            "ramp_samples": int(event.ramp),
            "threshold_samples": int(self.switch_condition.threshold_samples),
            "binding_constraint": self.switch_condition.binding_constraint,
            "delta_samples": (None if self._prev_event_sample is None
                              else int(s - self._prev_event_sample)),
            "voices_before": int(len(self._voices) - 1),
            "history_blocks_used": int(len(self._history)),
        })
        self._prev_event_sample = int(s)
        self._last_mode = event.mode
        self._current_query = event.query

    # ------------------------------------------------------------------
    def process_block(self, mono_block: np.ndarray) -> np.ndarray:
        """Render one mono block to a (block_size, 2) stereo float64 block."""
        x = np.asarray(mono_block, dtype=np.float64).reshape(-1)
        n = self.block_size
        if x.size != n:
            raise ValueError(f"expected {n} samples, got {x.size}")
        if self._current_query is None or not self._voices:
            raise RuntimeError("set_direction must be called before processing")

        abs0 = self._block_count * n
        block_end = abs0 + n

        # Retire voices whose ramp finished before this block.
        self._voices = [v for v in self._voices
                        if v.kill_at is None or v.kill_at > abs0]
        if not self._voices:  # pragma: no cover - defensive
            self._create_initial_voice(self._current_query, abs0)

        # Fire every event that lands in this block.  This happens before the
        # block is consumed, so priming sees history ending with block t-1
        # and the new voice renders block t in sync with the old ones.
        due = [e for e in self._pending if e.at < block_end]
        self._pending = [e for e in self._pending if e.at >= block_end]
        for event in due:
            self._fire(event, abs0)

        # Advance every voice by exactly this block.  All voices stay frame
        # aligned, so their outputs can be mixed with per-sample gains.
        rendered = []
        for voice in self._voices:
            voice_out = np.column_stack([
                voice.convs[0].process_block(x),
                voice.convs[1].process_block(x),
            ])
            rendered.append((voice, voice_out))
        self._history.append(np.array(x, copy=True))

        t = abs0 + np.arange(n)
        # Equal-power normalization across the voices alive at each sample.
        # A single ramp has gains (cos, sin) summing to unit power; when a
        # third voice enters mid-ramp or a stale voice is still decaying its
        # residual below 1, the raw weights no longer sum to one and the
        # output would dip.  Dividing by the per-sample total power keeps the
        # level constant regardless of how many filters overlap.
        power = np.zeros(n)
        raw = []
        for voice, voice_out in rendered:
            g = voice.gain_at(t)
            raw.append(g)
            power += g ** 2
        alive = np.zeros(n, dtype=int)
        for (voice, voice_out), g in zip(rendered, raw):
            if not np.any(g > 1e-9):
                continue
            alive += (g > 1e-9).astype(int)
        norm = 1.0 / np.sqrt(np.maximum(power, 1e-12))
        out = np.zeros((n, 2))
        for (voice, voice_out), g in zip(rendered, raw):
            if not np.any(g > 1e-9):
                continue
            out += voice_out * (g * norm)[:, None]

        peak_alive = int(alive.max()) if n else 0
        self.alive_history.append(peak_alive)
        self.max_alive_filters = max(self.max_alive_filters, peak_alive)
        # Exact sample count with more voices than the cap.  The block-level
        # peak cannot tell a 32-sample de-click overlap from a 1024-sample
        # parallel one; this can.
        self.alive_over_cap_samples += int(
            np.count_nonzero(alive > self.handover_budgets.max_alive_filters))
        self._fade_pos = max(
            (v.remaining_fade(block_end) for v in self._voices), default=0)

        if self.apply_distance_gain:
            self._distance_gain += 0.2 * (
                self._target_gain - self._distance_gain)
            out *= self._distance_gain

        self._record_block(peak_alive)
        self._block_count += 1
        return out

    # ------------------------------------------------------------------
    def _record_block(self, alive: int) -> None:
        q = self._current_query
        geo = getattr(self, "_last_geometry", None)
        if geo is not None:
            src, lis, yaw, pitch, roll, rel = geo
            distance = rel.distance_m
        else:
            src = lis = yaw = pitch = roll = None
            distance = getattr(self, "_last_distance", None)
        self.direction_log.append(DirectionRecord(
            block_index=self._block_count,
            time_seconds=self._block_count * self.block_size / self.sample_rate,
            world_source=tuple(src) if src is not None else (),
            listener_position=tuple(lis) if lis is not None else (),
            yaw_deg=yaw if yaw is not None else 0.0,
            pitch_deg=pitch if pitch is not None else 0.0,
            roll_deg=roll if roll is not None else 0.0,
            azimuth_deg=q.azimuth_deg,
            elevation_deg=q.elevation_deg,
            distance_m=distance if distance is not None else 0.0,
            distance_gain=self._distance_gain,
            hrtf_source=q.source,
            crossfading=self._fade_pos > 0,
            handover_mode=self._last_mode,
            alive_filters=int(alive),
        ))

    # ------------------------------------------------------------------
    def force_hrir(self, hrir: np.ndarray) -> None:
        """Replace the running voice with one exact HRIR (latency probes)."""
        if self._current_query is None:
            raise RuntimeError("set_direction must be called first")
        q = self._current_query
        self.reset()
        self._current_query = HRTFQuery(
            azimuth_deg=q.azimuth_deg, elevation_deg=q.elevation_deg,
            hrir=np.asarray(hrir, dtype=float),
            corner_indices=q.corner_indices, weights=q.weights,
            source=q.source)
        self._create_initial_voice(self._current_query)

    def _clear_runtime_state(self) -> None:
        self._voices = []
        self._pending = []
        self._banks = [[], []]
        self._active = 0
        self._fade_pos = 0
        self._block_count = 0
        self._last_handover_at = None
        self._prev_event_sample = None
        self._last_mode = ""
        self._distance_gain = 1.0
        self._target_gain = 1.0
        self._last_distance = None
        self.max_alive_filters = 0
        self.alive_over_cap_samples = 0
        self.alive_history = []
        self.handover_log = []
        self.direction_log = []
        self._history = deque(maxlen=self.history_blocks)

    def reset(self) -> None:
        """Rebuild a clean single voice on the current direction."""
        query = self._current_query
        self._clear_runtime_state()
        if query is not None:
            self._current_query = query
            self._create_initial_voice(query)
        if hasattr(self, "_last_geometry"):
            del self._last_geometry

    # ------------------------------------------------------------------
    def state_report(self) -> dict:
        """Snapshot of everything needed to audit the processing state."""
        bank_reports = []
        for bank in self._banks:
            if not bank:
                bank_reports.append(None)
            else:
                states = [c.get_state() for c in bank]
                bank_reports.append({
                    "left_write_index": states[0].write_index,
                    "right_write_index": states[1].write_index,
                    "left_input_ring_norm": float(
                        np.linalg.norm(states[0].input_ring)),
                    "right_input_ring_norm": float(
                        np.linalg.norm(states[1].input_ring)),
                    "left_spectrum_ring_norm": float(
                        np.linalg.norm(states[0].spectrum_ring)),
                    "right_spectrum_ring_norm": float(
                        np.linalg.norm(states[1].spectrum_ring)),
                })
        return {
            "block_size": self.block_size,
            "sample_rate": self.sample_rate,
            "algorithm_latency_samples": self.algorithm_latency_samples,
            "algorithm_latency_seconds": self.algorithm_latency_seconds,
            "crossfade_blocks": self.crossfade_blocks,
            "crossfade_seconds": (
                self.crossfade_blocks * self.block_size / self.sample_rate),
            "parallel_fade_samples": self.parallel_fade_samples,
            "rebuild_ramp_samples": self.rebuild_ramp_samples,
            "active_bank": self._active,
            "fade_samples_remaining": self._fade_pos,
            "blocks_processed": self._block_count,
            "handover_mode": self.handover_mode,
            "last_handover_mode": self._last_mode,
            "max_alive_filters": self.max_alive_filters,
            "alive_over_cap_samples": self.alive_over_cap_samples,
            "active_voices": len(self._voices),
            "history_blocks_retained": self.history_blocks,
            "required_history_samples": self.required_history_samples,
            "switch_condition": self.switch_condition.as_dict(
                self.sample_rate),
            "handover_log": list(self.handover_log),
            "current_direction": {
                "azimuth_deg": self._current_query.azimuth_deg,
                "elevation_deg": self._current_query.elevation_deg,
                "hrtf_source": self._current_query.source,
                "corner_indices": self._current_query.corner_indices,
                "interpolation_weights": self._current_query.weights,
            } if self._current_query else None,
            "banks": bank_reports,
        }
