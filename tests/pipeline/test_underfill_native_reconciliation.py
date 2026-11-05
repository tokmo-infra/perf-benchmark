"""Independent native standard-metrics verification of stop/horizon proofs."""
from dataclasses import replace

from perf_benchmark.plugins.metrics.standard import StandardMetrics
from tests.metrics.test_maxtpm_billing import session, trial
from tests.pipeline.test_underfill_stop import SECOND, complete, occupancy, terminal
from perf_benchmark.pipeline.execution.underfill import UnderfillStop, PROOF_WAIT_NS


def native(facts, *, end):
    backend = StandardMetrics({"maxtpm": True, "maxtpm_start_percent": 55}).create_backend(seed=0)
    backend.open()
    try:
        backend.consume((*facts, replace(session(0, end), id="s-0")))
        return backend.finalize_trial(trial(20)).diagnostics
    finally:
        backend.close()


def first_eleven(control):
    facts = [occupancy(control, 0, 1)]
    for i in range(11):
        facts.append(complete(control, i, i + 1))
    facts.append(occupancy(control, 20, 0))
    return facts


def test_partial_first_pass_cancelled_old_response_uses_session_end_bound():
    control = UnderfillStop("trial", 4, 20, 1)
    facts = first_eleven(control)
    control.begin_turn("t-11", 0, 11, 12 * SECOND)
    assert control.poll(81 * SECOND) is not None
    # The final cancelled Turn has an OLD real response timestamp, not the wall
    # clock. Eight untouched dataset Turns guarantee native's partial-pass branch.
    control.request_stop(81 * SECOND)
    cancelled = replace(terminal(11, 13, status="cancelled"), turn_completed_ns=81 * SECOND)
    control.observe(cancelled)
    facts.append(cancelled)
    diagnostics = native(facts, end=81)
    assert diagnostics["maxtpm_start_ns"] == 11 * SECOND
    assert diagnostics["maxtpm_end_ns"] == 20 * SECOND
    assert diagnostics["maxtpm_underfill_confirmed_ns"] == 80 * SECOND
    control.reconcile(diagnostics)
    assert control.snapshot()["native_window_matches"] is True
    assert control.snapshot()["result_at_risk"] is False


def test_full_first_pass_cleanup_clock_cannot_extend_terminal_horizon():
    control = UnderfillStop("trial", 4, 20, 1)
    facts = first_eleven(control)
    for i in range(11, 20):
        control.begin_turn(f"t-{i}", 0, i, 12 * SECOND)
    assert control.poll(81 * SECOND) is None
    assert control.poll(81 * SECOND + PROOF_WAIT_NS) is None
    assert control.hold_reason is None
    assert control.proof_wait_reason == "first_pass_horizon_unprovable_long_tail"
    assert control.proof_wait_long_tail
    # Counterfactual finalization: all tail responses were completed at 13s,
    # even if their session cleanup only finished at 141s. Native stops at 13s+1ns.
    facts.extend(terminal(i, 13, status="cancelled") for i in range(11, 20))
    diagnostics = native(facts, end=141)
    assert diagnostics["maxtpm_start_ns"] == 11 * SECOND
    assert diagnostics["maxtpm_end_ns"] == 13 * SECOND + 1
    assert diagnostics["maxtpm_underfill_confirmed_ns"] is None
    assert control.requested_ns is None
    assert control.snapshot()["result_at_risk"] is True


def test_unreturned_old_response_would_reorder_55_so_cannot_be_timed_out():
    control = UnderfillStop("trial", 4, 20, 1)
    control.begin_turn("t-19", 0, 19, 0)
    facts = first_eleven(control)
    assert control.poll(81 * SECOND) is None
    assert control.poll(81 * SECOND + PROOF_WAIT_NS) is None
    assert control.hold_reason is None
    assert control.proof_wait_reason == "completion_prefix_unprovable_long_tail"
    assert control.proof_wait_long_tail
    # This completion was unavailable to Python, not absent from native time.
    # Cancelling it as though it completed "now" would have forged start=11s.
    facts.append(terminal(19, 0.5, status="cancelled"))
    diagnostics = native(facts, end=141)
    assert diagnostics["maxtpm_start_ns"] == 10 * SECOND
    assert diagnostics["maxtpm_end_ns"] == 20 * SECOND
    assert control.start_ns is None
    assert control.requested_ns is None


def test_predispatch_queue_floor_then_exact_native_underfill_reconciles():
    control = UnderfillStop("trial", 4, 20, 1)
    control.begin_turn("t-19", 0, 19, 0)
    facts = first_eleven(control)
    assert control.poll(81 * SECOND) is None
    control.advance_undispatched_turn("t-19", 81 * SECOND + 1)
    assert control.poll(82 * SECOND) is not None
    control.request_stop(82 * SECOND)
    cancelled = replace(terminal(19, 82, status="cancelled"),
                        response_completed_ns=None, turn_completed_ns=82 * SECOND)
    control.observe(cancelled)
    facts.append(cancelled)
    diagnostics = native(facts, end=82)
    control.reconcile(diagnostics)
    assert control.start_ns == diagnostics["maxtpm_start_ns"] == 11 * SECOND
    assert diagnostics["maxtpm_end_ns"] == 20 * SECOND
    assert diagnostics["maxtpm_underfill_confirmed_ns"] == 80 * SECOND
    assert control.native_window_matches is True
    assert control.hold_reason is None
