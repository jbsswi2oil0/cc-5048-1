"""Binaural renderer: live HRTF selection, interpolation and crossfading.

Direction changes never swap filters abruptly. A new HRIR pair is loaded into
an idle convolver bank and the two banks are mixed with an equal-power ramp
over a fixed number of blocks, so continuous input audio produces no clicks.
Spatialization always comes from two independently convolved channels --
there is no stereo-balance shortcut anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .convolution import PartitionedConvolver
from .coordinates import relative_direction
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


class BinauralRenderer:
    def __init__(self, hrtf: HRTFBase, block_size: int = 512,
                 crossfade_ms: float = 5.0,
                 reference_distance_m: float = 1.0,
                 apply_distance_gain: bool = True):
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError("block_size must be a positive power of two")
        self.hrtf = hrtf
        self.sample_rate = hrtf.sample_rate
        self.block_size = int(block_size)
        requested = max(
            1, int(round(crossfade_ms * 1e-3 * self.sample_rate / block_size)))
        # At least two blocks so the equal-power ramp always spans a real
        # transition interval, even for large block sizes where a fixed
        # millisecond duration would otherwise fit inside one block.
        self.crossfade_blocks = max(2, requested)
        self.reference_distance_m = float(reference_distance_m)
        self.apply_distance_gain = bool(apply_distance_gain)

        self._banks = [[], []]  # two banks, each with left/right convolver
        self._active = 0
        self._fade_pos = 0  # samples remaining; 0 means settled on active
        self._current_query: Optional[HRTFQuery] = None
        self._distance_gain = 1.0
        self._target_gain = 1.0
        self._last_distance = None
        self._block_count = 0
        self.direction_log: list[DirectionRecord] = []

    # ------------------------------------------------------------------
    @property
    def algorithm_latency_samples(self) -> int:
        return self.block_size

    @property
    def algorithm_latency_seconds(self) -> float:
        return self.block_size / self.sample_rate

    def active_hrir(self) -> np.ndarray:
        if self._current_query is None:
            raise RuntimeError("set_direction must be called before processing")
        return self._current_query.hrir

    # ------------------------------------------------------------------
    def _build_bank(self, hrir: np.ndarray):
        return [
            PartitionedConvolver(hrir[0], self.block_size),
            PartitionedConvolver(hrir[1], self.block_size),
        ]

    def set_direction(self, azimuth_deg: float, elevation_deg: float,
                      distance_m: Optional[float] = None) -> HRTFQuery:
        """Select/interpolate HRIRs for a head-relative direction.

        Re-selecting the identical direction is a no-op, so callers that
        configure the initial pose never trigger a spurious crossfade.
        """
        query = self.hrtf.query(azimuth_deg, elevation_deg)

        if self._current_query is None:
            self._banks[self._active] = self._build_bank(query.hrir)
            self._banks[1 - self._active] = self._build_bank(query.hrir)
        elif (abs(query.azimuth_deg - self._current_query.azimuth_deg) > 1e-12
              or abs(query.elevation_deg
                     - self._current_query.elevation_deg) > 1e-12):
            idle = 1 - self._active
            self._banks[idle] = self._build_bank(query.hrir)
            self._fade_pos = self.crossfade_blocks * self.block_size
        self._current_query = query

        if distance_m is not None:
            self._target_gain = self.reference_distance_m / max(
                distance_m, 1e-6)
            self._last_distance = float(distance_m)
        return query

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
    def process_block(self, mono_block: np.ndarray) -> np.ndarray:
        """Render one mono block to a (block_size, 2) stereo float64 block."""
        x = np.asarray(mono_block, dtype=np.float64).reshape(-1)
        if x.size != self.block_size:
            raise ValueError(
                f"expected {self.block_size} samples, got {x.size}")
        if self._current_query is None:
            raise RuntimeError("set_direction must be called before processing")

        a = self._active
        b = 1 - a
        out_a = np.column_stack([
            self._banks[a][0].process_block(x),
            self._banks[a][1].process_block(x),
        ])
        out_b = np.column_stack([
            self._banks[b][0].process_block(x),
            self._banks[b][1].process_block(x),
        ])

        n = self.block_size
        if self._fade_pos > 0:
            remaining = self._fade_pos
            total = self.crossfade_blocks * self.block_size
            start = total - remaining
            t = (start + np.arange(n)) / total
            t = np.clip(t, 0.0, 1.0)
            gain_old = np.cos(0.5 * np.pi * t)[:, None]
            gain_new = np.sin(0.5 * np.pi * t)[:, None]
            out = out_a * gain_old + out_b * gain_new
            if remaining <= n:
                self._active = b
                self._fade_pos = 0
            else:
                self._fade_pos = remaining - n
        else:
            out = out_a

        # Smooth distance gain with a one-pole to avoid zipper noise.
        if self.apply_distance_gain:
            self._distance_gain += 0.2 * (
                self._target_gain - self._distance_gain)
            out *= self._distance_gain

        self._record_block()
        self._block_count += 1
        return out

    # ------------------------------------------------------------------
    def _record_block(self) -> None:
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
        ))

    def reset(self) -> None:
        # Rebuild both banks from the currently selected HRIR so convolver
        # state, ring buffers and filter partitions stay consistent; simply
        # zeroing convolvers while keeping stale banks would mix old state
        # with the next set_direction call.
        if self._current_query is not None:
            hrir = self._current_query.hrir
            self._banks = [self._build_bank(hrir), self._build_bank(hrir)]
            self._active = 0
        for bank in self._banks:
            for conv in bank:
                conv.reset()
        self._block_count = 0
        self._fade_pos = 0
        self._distance_gain = 1.0
        self._target_gain = 1.0
        self._last_distance = None
        if hasattr(self, "_last_geometry"):
            del self._last_geometry
        self.direction_log.clear()

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
            "active_bank": self._active,
            "fade_samples_remaining": self._fade_pos,
            "blocks_processed": self._block_count,
            "current_direction": {
                "azimuth_deg": self._current_query.azimuth_deg,
                "elevation_deg": self._current_query.elevation_deg,
                "hrtf_source": self._current_query.source,
                "corner_indices": self._current_query.corner_indices,
                "interpolation_weights": self._current_query.weights,
            } if self._current_query else None,
            "banks": bank_reports,
        }
