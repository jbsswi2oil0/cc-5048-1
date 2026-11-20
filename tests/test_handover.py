"""The two handover implementations, the condition that routes them, and the
measurements the exhibit promises."""

import numpy as np
import pytest

from binaural.experiment import (
    aligned_compare,
    bass_tone,
    impulse_train,
    measure_transient,
    pulse_locations,
    render_turn_scenario,
)
from binaural.gui import BinauralGUI
from binaural.handover import (
    AUTO,
    PARALLEL_TAIL,
    STATE_REBUILD,
    HandoverBudgets,
    SwitchCondition,
    calibrate_switch_condition,
    decide_mode,
    history_blocks,
    required_history_samples,
    select_handover,
)
from binaural.hrtf import ProceduralHRTF
from binaural.realtime import RealtimeBinauralPlayer
from binaural.renderer import BinauralRenderer


# ----------------------------------------------------------------------
# History rule
# ----------------------------------------------------------------------
def test_history_rule_is_p_plus_one_blocks():
    # 128-sample HRIR, B=512 -> P=1 -> need 2 blocks.
    assert required_history_samples(512, 128) == 1024
    assert history_blocks(512, 128) == 2
    # 128-sample HRIR, B=64 -> P=2 -> need 3 blocks.
    assert required_history_samples(64, 128) == 192


def test_history_is_exactly_enough_to_rebuild_a_voice():
    """A rebuilt voice fed the required history then a block matches the
    running voice exactly; one block less does not."""
    from binaural.convolution import PartitionedConvolver

    hrtf = ProceduralHRTF()
    b = 64
    hrir = hrtf.query(90, 0).hrir[0]  # 128 samples, P=2
    rng = np.random.default_rng(1)
    x = rng.standard_normal(b * 12)

    running = PartitionedConvolver(hrir, b)
    for k in range(10):
        running.process_block(x[k * b:(k + 1) * b])

    n_hist = history_blocks(b, hrir.size)
    rebuilt = PartitionedConvolver(hrir, b)
    for k in range(10 - n_hist, 10):  # exactly the retained blocks
        rebuilt.process_block(x[k * b:(k + 1) * b])
    a = running.process_block(x[10 * b:11 * b])
    c = rebuilt.process_block(x[10 * b:11 * b])
    assert np.max(np.abs(a - c)) < 1e-12

    short = PartitionedConvolver(hrir, b)
    for k in range(10 - (n_hist - 1), 10):
        short.process_block(x[k * b:(k + 1) * b])
    d = short.process_block(x[10 * b:11 * b])
    assert np.max(np.abs(a - d)) > 1e-6


# ----------------------------------------------------------------------
# Condition and routing
# ----------------------------------------------------------------------
def test_condition_chooses_by_threshold():
    cond = SwitchCondition(
        threshold_samples=1024, binding_constraint="alive_filter_cap",
        budgets={}, required_delay_samples={}, source="test")
    assert cond.choose(0) == STATE_REBUILD
    assert cond.choose(1023) == STATE_REBUILD
    assert cond.choose(1024) == PARALLEL_TAIL
    assert decide_mode(512, cond) == STATE_REBUILD
    assert decide_mode(4096, cond) == PARALLEL_TAIL
    assert decide_mode(0, cond, forced_mode=PARALLEL_TAIL) == PARALLEL_TAIL


def test_select_handover_reports_the_binding_constraint():
    cond = select_handover(fade_samples=1024, delta_itd_samples=34.0,
                           budgets=HandoverBudgets())
    # The alive cap needs the whole ramp; drift needs much less.
    assert cond.binding_constraint == "alive_filter_cap"
    assert cond.threshold_samples == 1024


def test_calibrated_condition_is_measured_and_self_consistent():
    res = calibrate_switch_condition(
        block_size=512,
        delays=(128, 512, 1024),
        budgets=HandoverBudgets(
            max_alive_filters=2, max_itd_drift_samples=0.5,
            max_transient_step=0.5))
    cond = res["condition"]
    assert cond.source == "measured"
    # Three filters only overlap for delays shorter than one ramp (1024).
    by_delay = {r["second_turn_delay_samples"]: r
                for r in res["measurements"]}
    assert by_delay[128]["max_alive_filters"] == 3
    assert by_delay[1024]["max_alive_filters"] == 2
    assert cond.threshold_samples == 1024
    assert cond.binding_constraint == "alive_filter_cap"


# ----------------------------------------------------------------------
# Mode difference and alive counts
# ----------------------------------------------------------------------
def test_parallel_tail_needs_a_third_filter_state_rebuild_does_not():
    d = 256
    par = render_turn_scenario(block_size=512, second_turn_delay_samples=d,
                               mode=PARALLEL_TAIL)
    reb = render_turn_scenario(block_size=512, second_turn_delay_samples=d,
                               mode=STATE_REBUILD)
    assert par.max_alive_filters == 3
    assert par.alive_over_cap_samples > 0
    assert reb.max_alive_filters == 2
    # The rebuild's only cap breach is the 32-sample de-click ramp.
    assert reb.alive_over_cap_samples == 0


def test_overlap_creates_measurable_itd_drift_only_for_parallel():
    d = 256
    # Pulses must land inside the overlap window (applied .. applied+fade), so
    # use a train spaced one block apart rather than the default 80 ms.
    train = impulse_train(sample_rate=48000, spacing_s=256 / 48000,
                          duration_s=1.0)
    par = render_turn_scenario(block_size=512, second_turn_delay_samples=d,
                               mode=PARALLEL_TAIL, measure="pulse",
                               signal=train)
    reb = render_turn_scenario(block_size=512, second_turn_delay_samples=d,
                               mode=STATE_REBUILD, measure="pulse",
                               signal=train)
    assert par.overlap_itd_drift_samples > 10.0
    assert reb.overlap_itd_drift_samples == pytest.approx(0.0, abs=1e-6)


def test_auto_routing_follows_the_threshold():
    below = render_turn_scenario(block_size=512, second_turn_delay_samples=256,
                                 mode=AUTO)
    at = render_turn_scenario(block_size=512, second_turn_delay_samples=1024,
                              mode=AUTO)
    assert below.handover_log[-1]["mode"] == STATE_REBUILD
    assert at.handover_log[-1]["mode"] == PARALLEL_TAIL


# ----------------------------------------------------------------------
# Test signals
# ----------------------------------------------------------------------
def test_impulse_train_pulses_are_isolated():
    x = impulse_train(spacing_s=0.05, duration_s=0.5)
    locs = pulse_locations(x)
    assert np.all(np.diff(locs) >= int(0.05 * 48000) - 1)
    assert x[locs[0]] == pytest.approx(0.8)


def test_bass_tone_is_smooth_and_sustained():
    x = bass_tone(duration_s=0.4)
    step = np.max(np.abs(np.diff(x)))
    assert step < 0.01 * np.max(np.abs(x)) + 1e-3
    assert np.max(np.abs(x)) == pytest.approx(0.6, rel=1e-3)


def test_transient_measure_is_zero_for_a_pure_tone():
    # A smooth sine has tiny sample-to-sample steps.
    assert measure_transient(bass_tone(duration_s=0.2)[:, None].repeat(
        2, axis=1)) < 0.01


# ----------------------------------------------------------------------
# Half-block shift alignment
# ----------------------------------------------------------------------
@pytest.mark.parametrize("measure", ["pulse", "bass"])
def test_half_block_shift_is_a_pure_relabeling(measure):
    """Audio and motion move together; after re-aligning by the same shift the
    render is identical, so the only possible audible difference is the cut."""
    shift = 256
    base = render_turn_scenario(block_size=512, second_turn_delay_samples=512,
                                mode=STATE_REBUILD, measure=measure)
    moved = render_turn_scenario(block_size=512, second_turn_delay_samples=512,
                                 mode=STATE_REBUILD, measure=measure,
                                 shift_samples=shift)
    cmp = aligned_compare(base.stereo[:-shift], moved.stereo[shift:])
    assert cmp["max_abs_diff"] < 1e-9
    assert cmp["max_abs_a"] > 0.1  # there was real signal to compare


# ----------------------------------------------------------------------
# Renderer scheduling used by the UI
# ----------------------------------------------------------------------
def make_armed_renderer():
    hrtf = ProceduralHRTF()
    r = BinauralRenderer(hrtf, block_size=256, apply_distance_gain=False)
    r.set_direction(0, 0, 1.2)
    return r


def test_pending_events_can_be_cancelled_by_tag():
    r = make_armed_renderer()
    r.schedule_direction(90, 0, 1.2, at_sample=512, tag="a")
    r.schedule_direction(-90, 0, 1.2, at_sample=1024, tag="b")
    assert len(r.pending_events()) == 2
    assert r.cancel_pending(tags=("b",)) == 1
    remaining = r.pending_events()
    assert len(remaining) == 1 and remaining[0]["tag"] == "a"


def test_handover_log_records_tags():
    r = make_armed_renderer()
    r.schedule_direction(90, 0, 1.2, at_sample=256, tag="second_turn")
    for _ in range(4):
        r.process_block(np.zeros(256))
    assert r.handover_log[-1]["tag"] == "second_turn"


# ----------------------------------------------------------------------
# GUI: draggable second turn and two-ear difference
# ----------------------------------------------------------------------
def build_gui():
    hrtf = ProceduralHRTF()
    renderer = BinauralRenderer(hrtf, block_size=256)
    renderer._current_query = None
    renderer.set_direction(0, 0, 1.2)
    player = RealtimeBinauralPlayer(
        renderer, np.sin(np.arange(24000) * 0.01), loop=True)
    return BinauralGUI(player), player, renderer


def test_gui_second_turn_delay_rearms_the_scheduled_event():
    gui, _player, renderer = build_gui()
    gui.start_turn_gesture(delay_samples=256)
    first = [e for e in renderer.pending_events()
             if e["tag"] == "second_turn"]
    assert first and first[0]["at"] == 256

    gui.set_second_turn_delay(1024)
    second = [e for e in renderer.pending_events()
              if e["tag"] == "second_turn"]
    assert len(second) == 1 and second[0]["at"] == 1024


def test_gui_difference_only_plays_l_minus_r():
    gui, player, renderer = build_gui()
    gui.set_difference_only(True)
    assert player.difference_only
    # Drive the difference path and its mirror through the renderer.
    stereo = np.array([[0.6, 0.2], [0.1, 0.4], [-0.3, 0.3]])
    d = player._to_output(stereo)
    assert np.allclose(d[:, 0], 0.5 * (stereo[:, 0] - stereo[:, 1]))
    assert np.allclose(d[:, 0], d[:, 1])

    gui.set_difference_only(False)
    assert np.allclose(player._to_output(stereo), stereo)


def test_headless_scenario_still_runs_with_gesture_armed():
    gui, _player, _renderer = build_gui()
    gui.start_turn_gesture(delay_samples=256)
    out = gui.player.simulate_blocks(8)
    assert out.shape == (8 * 256, 2)
    assert np.isfinite(out).all()
