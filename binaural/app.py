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


def cmd_measure(args) -> None:
    """Calibrate the switch condition from measurement, then prove it routes.

    The measured condition is fed back into fresh renderers so the same
    numbers that were reported are the ones that decide the playback path.
    """
    from .experiment import aligned_compare, render_turn_scenario
    from .handover import (
        PARALLEL_TAIL,
        STATE_REBUILD,
        HandoverBudgets,
        calibrate_switch_condition,
    )

    budgets = HandoverBudgets(
        max_alive_filters=args.max_alive,
        max_itd_drift_samples=args.max_itd_drift,
        max_transient_step=args.max_transient)
    # The scenarios below build a ProceduralHRTF internally; measure against
    # that same model so the reported history length matches what was rendered.
    from .hrtf import ProceduralHRTF

    hrtf = ProceduralHRTF()
    sample_rate = hrtf.sample_rate
    calib = calibrate_switch_condition(
        block_size=args.block_size, sample_rate=sample_rate,
        budgets=budgets, fade_ms=args.crossfade_ms,
        delays=tuple(args.delays))
    condition = calib["condition"]

    threshold = condition.threshold_samples
    # One delay below and one at/above the threshold: the condition must send
    # them down different paths, and the log must show that it did.  Each is
    # also rendered with both modes forced so the audible cost is side by
    # side.  The bass tone is used here because the ITD drift and the
    # handover step are both meaningful on it.
    probes = {}
    for name, delay in (("below", max(0, threshold // 2)),
                        ("at", threshold)):
        runs = {}
        for mode in ("auto", PARALLEL_TAIL, STATE_REBUILD):
            r = render_turn_scenario(
                block_size=args.block_size, sample_rate=sample_rate,
                second_turn_delay_samples=delay,
                mode=mode, fade_ms=args.crossfade_ms, measure="bass")
            runs[mode] = {
                "mode": r.mode,
                "itd_drift_samples": r.itd_drift_samples,
                "transient_peak": r.transient_peak,
                "max_alive_filters": r.max_alive_filters,
                "alive_over_cap_samples": r.alive_over_cap_samples,
                "modes_logged": [e["mode"] for e in r.handover_log],
                "second_turn_mode": (
                    r.handover_log[-1]["mode"] if r.handover_log else None),
            }
        probes[name] = {
            "delay_samples": int(delay),
            "signal": "bass",
            "note": ("itd_drift is only defined for the pulse train; see "
                     "measurements for per-delay overlap drift"),
            "resolved_second_turn_mode": runs["auto"]["second_turn_mode"],
            "modes": runs,
        }

    # Half-block shift: audio and motion move together, so after re-aligning
    # by the same offset the two renders must agree except at the filter cut.
    shift = args.block_size // 2
    base = render_turn_scenario(
        block_size=args.block_size, sample_rate=sample_rate,
        second_turn_delay_samples=threshold,
        mode=STATE_REBUILD, fade_ms=args.crossfade_ms, measure="pulse")
    moved = render_turn_scenario(
        block_size=args.block_size, sample_rate=sample_rate,
        second_turn_delay_samples=threshold,
        mode=STATE_REBUILD, fade_ms=args.crossfade_ms, measure="pulse",
        shift_samples=shift)
    aligned = aligned_compare(base.stereo[:-shift] if shift else base.stereo,
                              moved.stereo[shift:])

    report = {
        "switch_condition": condition.as_dict(sample_rate),
        "measurements": calib["measurements"],
        "routing_probes": probes,
        "half_block_shift": {
            "shift_samples": int(shift),
            **aligned,
        },
        "history": {
            "required_history_samples": (
                (int(np.ceil(hrtf.hrir_length / args.block_size)) + 1)
                * args.block_size),
            "rule": "P+1 blocks of raw input, P=ceil(hrir_length/B)",
        },
    }
    print(json.dumps(report, indent=2, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=float))


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

    p_meas = sub.add_parser("measure")
    p_meas.add_argument("--out", default=None,
                        help="optional path for the JSON report")
    p_meas.add_argument("--max-alive", type=int, default=2)
    p_meas.add_argument("--max-itd-drift", type=float, default=0.5,
                        help="ITD drift budget, samples")
    p_meas.add_argument("--max-transient", type=float, default=0.5,
                        help="handover step budget, absolute")
    p_meas.add_argument("--delays", type=int, nargs="+",
                        default=[0, 128, 256, 384, 512, 768, 896, 1024,
                                 1280, 1536],
                        help="second-turn delays to probe, samples")
    p_meas.set_defaults(func=cmd_measure)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
