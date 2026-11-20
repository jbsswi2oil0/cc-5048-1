"""Uniformly partitioned overlap-save convolution.

The mono input is partitioned into blocks of ``block_size`` (B) samples and
each HRIR into matching filter partitions. Every block costs one FFT pair
per partition and produces exactly B output samples; the real-time callback
and the offline renderer share this exact primitive, its state and its
latency.

Two-layer structure
-------------------
1. An exact zero-latency uniformly partitioned overlap-save core. Each FFT
   frame of length 2B holds two consecutive input blocks with the current
   block in the second half::

       frame_t = [x_{t-1} | x_t].

   Filter partition k occupies samples ``[kB, (k+1)B)``. Circular wrap can
   only contaminate a frame's first half, so the second half is exact linear
   convolution and is returned. A frequency-domain delay line of P+1 frame
   spectra combines the partitions::

       core_t = H_0 frame_t + H_1 frame_{t-1} + ... + H_P frame_{t-P}.

   Slots are addressed by absolute block number (never the cyclic write
   pointer), so long HRIRs cannot alias onto a freshly written frame.

2. An explicit one-block output delay line. The core emits each block's
   result in the same block that receives the input; holding the result for
   B samples makes the streaming processor causal and gives it a fixed,
   measurable look-ahead latency of exactly B samples.

Measurable quantities:

* ``algorithm_latency_samples`` -- the B-sample output delay;
* :meth:`PartitionedConvolver.get_state` -- delay-line snapshot for audits;
* :func:`measure_pipeline_latency` -- empirical impulse timing, equal to the
  documented block latency for a unit-impulse HRIR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConvolverState:
    input_ring: np.ndarray
    spectrum_ring: np.ndarray
    output_delay: np.ndarray
    write_index: int
    blocks_seen: int


class PartitionedConvolver:
    """One channel of uniformly partitioned overlap-save convolution."""

    def __init__(self, impulse_response: np.ndarray, block_size: int):
        block_size = int(block_size)
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError("block_size must be a positive power of two")
        h = np.asarray(impulse_response, dtype=np.float64).reshape(-1)
        self.block_size = block_size
        self.nfft = 2 * block_size
        self.n_partitions = int(np.ceil(h.size / block_size))

        hp = np.zeros((self.n_partitions, self.nfft))
        for p in range(self.n_partitions):
            chunk = h[p * block_size:(p + 1) * block_size]
            hp[p, :chunk.size] = chunk
        self.filter_spectra = np.fft.rfft(hp, n=self.nfft, axis=1)

        self.input_ring = np.zeros(self.nfft)
        slots = 1
        while slots < self.n_partitions + 1:
            slots *= 2
        self.ring_slots = max(2, slots)
        self.spectrum_ring = np.zeros(
            (self.ring_slots, self.nfft // 2 + 1), dtype=np.complex128)
        self.write_index = 0
        self._blocks_seen = 0
        self.output_delay = np.zeros(self.block_size)
        self.algorithm_latency_samples = block_size

    # ------------------------------------------------------------------
    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Consume exactly B input samples, return B output samples."""
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        if x.size != self.block_size:
            raise ValueError(
                f"expected {self.block_size} samples, got {x.size}")

        self.input_ring[: self.block_size] = self.input_ring[self.block_size:]
        self.input_ring[self.block_size:] = x
        x_spectrum = np.fft.rfft(self.input_ring)

        p = self.write_index
        self.spectrum_ring[p] = x_spectrum
        block_number = self._blocks_seen
        acc = np.zeros_like(x_spectrum)
        for k in range(self.n_partitions):
            slot = (block_number - k) % self.ring_slots
            acc += self.spectrum_ring[slot] * self.filter_spectra[k]
        self.write_index = (p + 1) % self.ring_slots
        self._blocks_seen += 1

        core = np.fft.irfft(acc, n=self.nfft)[self.block_size:].copy()
        out = self.output_delay
        self.output_delay = core
        return out.copy()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.input_ring.fill(0.0)
        self.spectrum_ring.fill(0.0)
        self.output_delay.fill(0.0)
        self.write_index = 0
        self._blocks_seen = 0

    def get_state(self) -> ConvolverState:
        return ConvolverState(
            input_ring=self.input_ring.copy(),
            spectrum_ring=self.spectrum_ring.copy(),
            output_delay=self.output_delay.copy(),
            write_index=self.write_index,
            blocks_seen=self._blocks_seen,
        )


def measure_pipeline_latency(renderer, sample_rate: int) -> dict:
    """Measure block-processing latency and total acoustic arrival.

    * ``block_latency_samples`` -- overlap-save latency measured with a
      unit-impulse HRIR (no filter group delay), must equal ``block_size``;
    * ``total_arrival_samples`` -- where the actual selected HRIR's first
      peak emerges (block latency plus the HRIR's own onset/ITD delay).
    """
    b = renderer.block_size
    n_blocks = max(8, (renderer.active_hrir().shape[-1] // b) + 4)

    def impulse_run(hrir: np.ndarray):
        renderer.reset()
        renderer.force_hrir(hrir)
        left_peak = right_peak = -1
        for k in range(n_blocks):
            block = np.zeros(b)
            if k == 0:
                block[0] = 1.0
            out = renderer.process_block(block)
            # Block k emits absolute samples [k*b, (k+1)*b); with one-block
            # latency the impulse HRIR first peaks in block 1.
            if left_peak < 0 and np.max(np.abs(out[:, 0])) > 1e-9:
                left_peak = k * b + int(np.argmax(np.abs(out[:, 0])))
            if right_peak < 0 and np.max(np.abs(out[:, 1])) > 1e-9:
                right_peak = k * b + int(np.argmax(np.abs(out[:, 1])))
        return left_peak, right_peak

    unit_hrir = np.zeros((2, renderer.hrtf.hrir_length))
    unit_hrir[:, 0] = 1.0
    block_l, block_r = impulse_run(unit_hrir)
    total_l, total_r = impulse_run(renderer.active_hrir())

    return {
        "block_size_samples": b,
        "block_latency_samples_left": block_l,
        "block_latency_samples_right": block_r,
        "block_latency_seconds": block_l / sample_rate,
        "documented_algorithm_latency_samples": b,
        "matches_documentation": block_l == b and block_r == b,
        "total_arrival_samples_left": total_l,
        "total_arrival_samples_right": total_r,
        "total_arrival_seconds": total_l / sample_rate,
    }
