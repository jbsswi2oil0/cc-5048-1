"""Offline render and export path.

Produces, for one scripted trajectory:

* ``binaural_<name>.wav``        -- the real-convolution stereo result;
* ``hrirs/<name>_<direction>.wav`` -- every distinct HRIR pair that was used;
* ``direction_log_<name>.csv``   -- per-block azimuth/elevation/pose record;
* ``manifest_<name>.json``       -- full audit manifest (latency, blocks,
                                    filter provenance, checksums).

The offline renderer runs the *same* :class:`BinauralRenderer` block loop as
the live callback, so exports are bit-identical in algorithm to playback.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .renderer import BinauralRenderer


def render_offline(renderer: BinauralRenderer, mono: np.ndarray,
                   trajectory: list[dict]) -> np.ndarray:
    """Render a mono signal along a trajectory of per-block directions.

    Each trajectory item is a dict with keys ``azimuth_deg``,
    ``elevation_deg`` and optionally ``distance_m``; one item is consumed per
    processed block. A constant direction is repeated as needed.
    """
    b = renderer.block_size
    n_blocks = int(np.ceil(mono.size / b))
    padded = np.concatenate([mono, np.zeros(n_blocks * b - mono.size)])
    out = np.zeros((n_blocks * b, 2))
    used_hrirs: dict[tuple, np.ndarray] = {}

    for k in range(n_blocks):
        step = trajectory[min(k, len(trajectory) - 1)]
        query = renderer.set_direction(
            step["azimuth_deg"], step["elevation_deg"],
            step.get("distance_m"))
        used_hrirs.setdefault(
            (round(query.azimuth_deg, 3), round(query.elevation_deg, 3)),
            query.hrir.copy())
        block = padded[k * b:(k + 1) * b]
        out[k * b:(k + 1) * b] = renderer.process_block(block)

    # trim back to the original signal length plus algorithmic flush
    flush = b
    return out[: mono.size + flush], used_hrirs


def static_trajectory(azimuth_deg: float, elevation_deg: float,
                      n_blocks: int,
                      distance_m: float | None = None) -> list[dict]:
    step = {"azimuth_deg": azimuth_deg, "elevation_deg": elevation_deg}
    if distance_m is not None:
        step["distance_m"] = distance_m
    return [dict(step) for _ in range(n_blocks)]


def sweep_trajectory(az_start: float, az_end: float,
                     elevation_deg: float, n_blocks: int) -> list[dict]:
    return [
        {"azimuth_deg": float(a), "elevation_deg": float(elevation_deg)}
        for a in np.linspace(az_start, az_end, n_blocks)
    ]


def export_session(out_dir: str | Path, name: str,
                   renderer: BinauralRenderer, stereo: np.ndarray,
                   used_hrirs: dict[tuple, np.ndarray],
                   signal_name: str) -> Path:
    out_dir = Path(out_dir)
    hrir_dir = out_dir / "hrirs"
    hrir_dir.mkdir(parents=True, exist_ok=True)

    wav_path = out_dir / f"binaural_{name}.wav"
    peak = np.max(np.abs(stereo))
    int_scale = 1.0 if peak <= 1.0 else 1.0 / peak
    sf.write(wav_path, (stereo * int_scale).astype(np.float32),
             renderer.sample_rate, subtype="FLOAT")
    # Hash the on-disk artifact (container included) so a reviewer can verify
    # the exported WAV bit-for-bit against the manifest.
    output_sha = hashlib.sha256(wav_path.read_bytes()).hexdigest()

    hrir_index = []
    for (az, el), hrir in sorted(used_hrirs.items()):
        fname = f"{name}_az{az:+.1f}_el{el:+.1f}.wav"
        hrir_path = hrir_dir / fname
        sf.write(hrir_path, hrir.T.astype(np.float32),
                 renderer.sample_rate, subtype="FLOAT")
        hrir_index.append({
            "file": f"hrirs/{fname}",
            "azimuth_deg": az,
            "elevation_deg": el,
            "sha256": hashlib.sha256(hrir_path.read_bytes()).hexdigest(),
        })

    log_path = out_dir / f"direction_log_{name}.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(asdict(renderer.direction_log[0]).keys()))
        writer.writeheader()
        for rec in renderer.direction_log:
            row = asdict(rec)
            row["world_source"] = json.dumps(row["world_source"])
            row["listener_position"] = json.dumps(row["listener_position"])
            writer.writerow(row)

    manifest = {
        "session": name,
        "signal": signal_name,
        "sample_rate": renderer.sample_rate,
        "block_size_samples": renderer.block_size,
        "algorithm_latency_samples": renderer.algorithm_latency_samples,
        "algorithm_latency_seconds": renderer.algorithm_latency_seconds,
        "crossfade_seconds": (
            renderer.crossfade_blocks * renderer.block_size
            / renderer.sample_rate),
        "blocks_processed": renderer._block_count,
        "hrtf_database": renderer.hrtf.grid_record(),
        "hrirs_used": hrir_index,
        "output_file": wav_path.name,
        "output_sha256": output_sha,
        "output_peak_linear": float(peak),
    }
    manifest_path = out_dir / f"manifest_{name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
