"""Command-line entry point for the binaural exhibit.

Modes
-----
gui       interactive drag-and-listen window (needs a display + PortAudio)
export    offline render of the verification scenarios into exports/
selftest  numerical checks: symmetry, behind-cue, latency, crossfade, I/O
signals   write the three source signals as WAV files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .convolution import measure_pipeline_latency
from .gui import BinauralGUI, HeadlessGUI
from .hrtf import load_hrtf
from .offline import (
    export_session,
    render_offline,
    static_trajectory,
    sweep_trajectory,
)
from .realtime import RealtimeBinauralPlayer, audio_thread_io_audit
from .renderer import BinauralRenderer
from .signals import build_default_library

BLOCK_SIZE = 512


def build_renderer(args) -> BinauralRenderer:
    hrtf = load_hrtf(sofa_path=args.sofa)
    return BinauralRenderer(
        hrtf, block_size=args.block_size,
        crossfade_ms=args.crossfade_ms)


def cmd_gui(args) -> None:
    renderer = build_renderer(args)
    library = build_default_library(renderer.sample_rate)
    signal = library[args.signal]
    renderer.set_direction(0.0, 0.0, 1.2)
    player = RealtimeBinauralPlayer(renderer, signal, loop=True)
    gui = BinauralGUI(player)
    print(f"backend: {player.backend}; HRTF: {renderer.hrtf.source_name}")
    if player.backend == "simulated":
        print("PortAudio unavailable - running headless scenario instead.")
        headless = HeadlessGUI(player)
        headless.scenario([
            ("preset", "front"), ("preset", "left"),
            ("preset", "right"), ("preset", "back"),
            ("yaw", 90.0), ("source", (1.2, 0.5, 0.0)),
        ], blocks_per_move=4)
        print(json.dumps(renderer.state_report(), indent=2))
        return
    gui.run()


def cmd_export(args) -> None:
    renderer = build_renderer(args)
    library = build_default_library(renderer.sample_rate)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "static_front": static_trajectory(0, 0, 12, distance_m=1.2),
        "static_left": static_trajectory(90, 0, 12, distance_m=1.2),
        "static_right": static_trajectory(-90, 0, 12, distance_m=1.2),
        "static_behind": static_trajectory(180, 0, 12, distance_m=1.2),
        "azimuth_sweep": sweep_trajectory(-180, 180, 0, 60),
    }
    for name, trajectory in scenarios.items():
        renderer.reset()
        renderer.set_direction(trajectory[0]["azimuth_deg"],
                               trajectory[0]["elevation_deg"], 1.2)
        stereo, hrirs = render_offline(
            renderer, library[args.signal], trajectory)
        manifest = export_session(
            out_dir, f"{args.signal}_{name}", renderer, stereo, hrirs,
            args.signal)
        print(f"exported {name}: {manifest}")

    latency = measure_pipeline_latency(renderer, renderer.sample_rate)
    (out_dir / "latency_check.json").write_text(json.dumps(latency, indent=2))
    print("latency:", latency)


def cmd_selftest(args) -> None:
    from .selftest import run_all_checks

    report = run_all_checks(args.block_size)
    print(json.dumps(report, indent=2, default=str))
    failures = [k for k, v in report["checks"].items() if not v["passed"]]
    raise SystemExit(1 if failures else 0)


def cmd_signals(args) -> None:
    import soundfile as sf

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    library = build_default_library()
    for name, x in library.items():
        path = out_dir / f"source_{name}.wav"
        sf.write(path, x.astype(np.float32), 48000)
        print(path)
    print("audio callback audit:", audio_thread_io_audit())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sofa", help="optional measured SOFA HRIR file")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        choices=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--crossfade-ms", type=float, default=5.0)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_gui = sub.add_parser("gui")
    p_gui.add_argument("--signal", choices=["impulse", "broadband", "speech"],
                       default="speech")
    p_gui.set_defaults(func=cmd_gui)

    p_exp = sub.add_parser("export")
    p_exp.add_argument("--signal", choices=["impulse", "broadband", "speech"],
                       default="broadband")
    p_exp.add_argument("--out", default="exports")
    p_exp.set_defaults(func=cmd_export)

    p_test = sub.add_parser("selftest")
    p_test.set_defaults(func=cmd_selftest)

    p_sig = sub.add_parser("signals")
    p_sig.add_argument("--out", default="recordings")
    p_sig.set_defaults(func=cmd_signals)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
