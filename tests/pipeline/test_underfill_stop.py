"""Deterministic CPU proofs and native-metrics differentials; no HTTP or models."""
from dataclasses import fields, replace
import random
import unittest

from perf_benchmark.pipeline.execution.underfill import UnderfillDecision, UnderfillStop
from perf_benchmark.pipeline.results.measurement import AttemptFact, OccupancyFact, TurnFact

SECOND = 1_000_000_000


def terminal(index, seconds, *, status="succeeded", session_ordinal=0, http_status=200,
             cls=TurnFact):
    # Defaults are deliberately synthetic; no dataset total or manifest is real.
    values = {f.name: None for f in fields(TurnFact)}
    values.update(
        trial_id="trial", id=f"t-{index}", session_id=f"s-{session_ordinal}",
        ordinal=index, status=status, usage={}, metadata={}, response_samples=(),
        turn_started_ns=0, turn_completed_ns=int((seconds + 100) * SECOND),
        response_completed_ns=int(seconds * SECOND), http_status=http_status,
        attempts=1, output_event_count=0, output_chars=0,
    )
    return cls(**values)


def complete(control, index, seconds, *, floor=0, **kwargs):
    fact = terminal(index, seconds, **kwargs)
    control.begin_turn(fact.id, 0, index, floor)
    control.observe(fact)
    return fact


def occupancy(control, seconds, value, *, submit=True):
    fact = OccupancyFact("trial", "in_flight", int(seconds * SECOND), value)
    control.capture_occupancy(fact.at_ns, fact.value)
    if submit:
        control.observe(fact)
    return fact


def started(*, target=2048):
    control = UnderfillStop("trial", 3072, 20, target)
    occupancy(control, 0, target)
    for i in range(11):
        complete(control, i, i + 1)
    return control


class UnderfillProofTests(unittest.TestCase):
    def test_selected_denominator_decimal_ceil_not_admitted_or_attempts(self):
        for count, expected in ((20, 11), (21, 12), (228770, 125824), (77001, 42351)):
            self.assertEqual(UnderfillStop("trial", 3072, count, 2048).threshold, expected)
        control = UnderfillStop("trial", 3072, 20, 2048)
        control.observe(terminal(0, 1, cls=AttemptFact))
        self.assertEqual(control.received_turns, 0)
        self.assertEqual(control.threshold, 11)

    def test_configured_percentages_use_exact_decimal_ceil_and_zero_is_rejected(self):
        for percent, expected in ((40, 8), (53, 11), (12.5, 3), (99.9, 20)):
            self.assertEqual(UnderfillStop("trial", 30, 20, 10, percent).threshold, expected)
        with self.assertRaises(ValueError):
            UnderfillStop("trial", 30, 20, 10, 0)

    def test_final_turns_sort_response_time_not_delivery_or_lifecycle_time(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2048)
        for i in range(12):
            control.begin_turn(f"t-{i}", 0, i, 0)
        # 11 arrivals appear to put threshold at 12s, but t-0 at 1s is pending.
        for i in reversed(range(1, 12)):
            control.observe(terminal(i, i + 1))
        self.assertIsNone(control.poll(100 * SECOND))
        self.assertIsNone(control.start_ns)
        self.assertEqual(control.progress_watermark_ns, -1)
        control.observe(terminal(0, 1))
        control.poll(100 * SECOND)
        self.assertEqual(control.start_ns, 11 * SECOND)

    def test_response_none_uses_native_lifecycle_fallback(self):
        control = UnderfillStop("trial", 3072, 1, 2048)
        occupancy(control, 0, 2048)
        control.begin_turn("t-0", 0, 0, 0)
        control.observe(replace(terminal(0, 1), response_completed_ns=None))
        control.poll(102 * SECOND)
        self.assertEqual(control.start_ns, 101 * SECOND)

    def test_pre55_underfill_does_not_spend_grace_before_start(self):
        control = started()
        occupancy(control, 1, 2047)
        complete(control, 19, 71, floor=12 * SECOND)
        self.assertIsNone(control.poll(71 * SECOND))  # strict watermark is 71s - 1ns
        decision = control.poll(71 * SECOND + 1)
        self.assertEqual(decision, UnderfillDecision(11 * SECOND, 11 * SECOND, 71 * SECOND))

    def test_recovery_before60_resets_then_exact60_recovery_preserves(self):
        control = started()
        occupancy(control, 20, 2047)
        occupancy(control, 79.999, 2048)
        occupancy(control, 90, 2047)
        occupancy(control, 150, 2048)
        complete(control, 19, 150, floor=12 * SECOND)
        self.assertIsNone(control.poll(150 * SECOND))
        decision = control.poll(150 * SECOND + 1)
        self.assertEqual(decision.end_ns, 90 * SECOND)
        self.assertEqual(decision.confirmed_ns, 150 * SECOND)

    def test_equal_timestamp_final_value_no_zero_duration_recovery(self):
        control = started()
        occupancy(control, 20, 2047)
        occupancy(control, 30, 2048)
        occupancy(control, 30, 2047)
        complete(control, 19, 80, floor=12 * SECOND)
        decision = control.poll(80 * SECOND + 1)
        self.assertEqual(decision.end_ns, 20 * SECOND)
        control = started()
        occupancy(control, 20, 2047)
        occupancy(control, 20, 2048)
        self.assertIsNone(control.poll(100 * SECOND))

    def test_captured_but_not_submitted_recovery_blocks_watermark(self):
        control = started()
        occupancy(control, 20, 2047)
        recovery = occupancy(control, 79.999, 2048, submit=False)
        self.assertIsNone(control.poll(200 * SECOND))
        self.assertLess(control.occupancy_watermark_ns, recovery.at_ns)
        self.assertEqual(len(control.pending_occupancy), 1)
        control.observe(recovery)
        self.assertIsNone(control.poll(201 * SECOND))
        self.assertIsNone(control.hold_reason)

    def test_reordered_occupancy_handoff_holds_instead_of_inventing_tie_order(self):
        control = started()
        occupancy(control, 20, 2047, submit=False)
        recovery = occupancy(control, 20, 2048, submit=False)
        control.observe(recovery)
        self.assertIn("handoff_order", control.hold_reason)
        self.assertIsNone(control.poll(200 * SECOND))

    def test_cancelled_never_manufactures55_native_conflict_is_hold(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2047)
        for i in range(10):
            complete(control, i, i + 1)
        complete(control, 10, 11, status="cancelled")
        self.assertIsNone(control.poll(1000 * SECOND))
        self.assertIsNone(control.start_ns)
        self.assertIn("cancelled_before", control.hold_reason)
        self.assertTrue(control.snapshot()["result_at_risk"])

    def test_failed_turn_counts_http500_does_not_cancel_or_hold(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2048)
        for i in range(11):
            complete(control, i, i + 1, status="failed", http_status=500)
        self.assertIsNone(control.poll(20 * SECOND))
        self.assertEqual(control.start_ns, 11 * SECOND)
        self.assertIsNone(control.hold_reason)
        self.assertIsNone(control.requested_ns)

    def test_firstpass_terminal_horizon_cannot_be_extended_by_wall_clock(self):
        control = started()
        occupancy(control, 20, 2047)
        for i in range(11, 20):
            control.begin_turn(f"t-{i}", 0, i, 0)
        self.assertIsNone(control.poll(1000 * SECOND))
        self.assertIsNone(control.decision)
        # Progress sealed now, but all unfinished dataset Turns could have
        # completed at 13s: still no proof the native first pass reaches 80s.
        control = started()
        occupancy(control, 20, 2047)
        for i in range(11, 20):
            control.begin_turn(f"t-{i}", 0, i, 12 * SECOND)
        self.assertIsNone(control.poll(1000 * SECOND))
        self.assertEqual(control.decision.confirmed_ns, 80 * SECOND)
        for i in range(11, 20):
            control.observe(terminal(i, 13))
        self.assertIsNone(control.poll(1001 * SECOND))
        self.assertIsNone(control.requested_ns)

    def test_long_old_turn_conservatively_waits_natural_end(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2047)
        control.begin_turn("old", 0, 99, 0)
        for i in range(11):
            complete(control, i, i + 1)
        self.assertIsNone(control.poll(10_000 * SECOND))
        self.assertIsNone(control.start_ns)

    def test_bad_clock_and_identity_never_produce_action(self):
        control = started()
        control.observe(replace(terminal(90, 100), trial_id="other-trial"))
        self.assertIsNone(control.poll(1000 * SECOND))
        self.assertIn("identity", control.hold_reason)
        control = started()
        control.poll(30 * SECOND)
        control.begin_turn("late", 0, 99, 1)
        self.assertIn("fence", control.hold_reason)

    def test_missing_start_occupancy_is_hold_not_low_concurrency(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        for i in range(11):
            complete(control, i, i + 1)
        occupancy(control, 12, 2047)
        self.assertIsNone(control.poll(100 * SECOND))
        self.assertEqual(control.hold_reason, "occupancy_not_observed_at_start")

    def test_stop_is_once_and_reconciles_without_mutating_native_result(self):
        control = started()
        occupancy(control, 20, 2047)
        complete(control, 19, 80, floor=12 * SECOND)
        control.begin_turn("t-11", 0, 11, 21 * SECOND)
        self.assertIsNotNone(control.poll(80 * SECOND + 1))
        control.request_stop(81 * SECOND)
        self.assertIsNone(control.poll(100 * SECOND))
        with self.assertRaises(RuntimeError):
            control.request_stop(100 * SECOND)
        control.observe(terminal(11, 82, status="cancelled"))
        self.assertIsNone(control.hold_reason)
        native = {"maxtpm_start_ns": 11 * SECOND, "maxtpm_end_ns": 20 * SECOND,
                  "maxtpm_underfill_confirmed_ns": 80 * SECOND}
        before = dict(native)
        control.reconcile(native)
        self.assertTrue(control.native_window_matches)
        self.assertEqual(native, before)
        control.reconcile({**native, "maxtpm_end_ns": 21 * SECOND})
        self.assertTrue(control.snapshot()["result_at_risk"])

    def test_random_native_concurrency_differential(self):
        # Read the unchanged native function, without importing DuckDB/plugins.
        import ast
        from itertools import groupby
        from pathlib import Path
        source = (Path(__file__).resolve().parents[2] / "perf_benchmark" / "plugins"
                  / "metrics" / "standard" / "maxtpm.py")
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "_concurrency_end")
        namespace = {"groupby": groupby, "MINUTE_NS": 60 * SECOND}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        _concurrency_end = namespace["_concurrency_end"]
        rng = random.Random(550060)
        for _ in range(500):
            control = started()
            rows = [(0, 2048)]
            stamp = 0
            for _ in range(35):
                stamp += rng.randint(0, 30)
                value = rng.choice((0, 2047, 2048))
                occupancy(control, stamp, value)
                rows.append((stamp * SECOND, value))
            horizon = (stamp + 100) * SECOND
            complete(control, 19, horizon / SECOND, floor=12 * SECOND)
            expected_end, confirmed, reason = _concurrency_end(rows, 11 * SECOND, horizon, 2048)
            actual = control.poll(horizon + 1)
            if reason == "concurrency_under_target_60s":
                self.assertEqual((actual.end_ns, actual.confirmed_ns), (expected_end, confirmed))
            else:
                self.assertIsNone(actual)




class UnderfillAuditRegressionTests(unittest.TestCase):
    def test_returned_long_tail_advances_floor_before_release_not_progress(self):
        from perf_benchmark.pipeline.contracts.executor import AttemptResult
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2047)
        control.begin_turn("old", 0, 99, 0)
        for i in range(11):
            complete(control, i, i + 1)
        self.assertIsNone(control.poll(12 * SECOND))
        result = AttemptResult("old", 1, status="failed")
        result.request.http_status = 500
        result.request.response_completed_ns = 80 * SECOND
        control.complete_attempt("old", result, 80 * SECOND + 1)
        # No final TurnFact for old yet: no extra progress, but the prefix is sealed.
        decision = control.poll(81 * SECOND)
        self.assertEqual(control.received_turns, 11)
        self.assertEqual(decision, UnderfillDecision(11 * SECOND, 11 * SECOND, 71 * SECOND))
        self.assertEqual(control.active["old"][0], 80 * SECOND)

    def test_native_pending_long_tail_waits_without_invalidating_later_proof(self):
        from perf_benchmark.pipeline.execution.underfill import PROOF_WAIT_NS
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2047)
        control.begin_turn("native-pending", 0, 99, 0)
        for i in range(11):
            complete(control, i, i + 1)
        self.assertIsNone(control.poll(12 * SECOND))
        self.assertIsNone(control.hold_reason)
        self.assertIsNone(control.poll(12 * SECOND + PROOF_WAIT_NS))
        self.assertIsNone(control.hold_reason)
        self.assertEqual(control.proof_wait_reason, "completion_prefix_unprovable_long_tail")
        self.assertTrue(control.proof_wait_long_tail)
        self.assertTrue(control.snapshot()["result_at_risk"])
        self.assertIsNone(control.requested_ns)
        self.assertIsNone(control.start_ns)

    def test_partial_first_pass_uses_native_session_bound_without_new_completion(self):
        control = started()
        occupancy(control, 20, 2047)
        # One active tail, eight unstarted Turns: cancelling cannot complete the
        # selected pass. Its session terminal timestamp bounds native observation.
        control.begin_turn("t-11", 0, 11, 12 * SECOND)
        decision = control.poll(81 * SECOND)
        self.assertEqual(decision, UnderfillDecision(11 * SECOND, 20 * SECOND, 80 * SECOND))
        self.assertEqual(control.latest_completion_ns, 11 * SECOND)
        control.request_stop(81 * SECOND)
        control.observe(terminal(11, 81, status="cancelled"))
        self.assertIsNone(control.hold_reason)

    def test_last_wave_old_results_cannot_extend_first_pass_after_long_proof_wait(self):
        from perf_benchmark.pipeline.execution.underfill import PROOF_WAIT_NS
        control = started()
        occupancy(control, 20, 2047)
        for i in range(11, 20):
            control.begin_turn(f"t-{i}", 0, i, 12 * SECOND)
        self.assertIsNone(control.poll(81 * SECOND))
        self.assertIsNotNone(control.decision)  # occupancy alone is insufficient
        self.assertIsNone(control.hold_reason)
        self.assertIsNone(control.poll(81 * SECOND + PROOF_WAIT_NS))
        self.assertIsNone(control.hold_reason)
        self.assertEqual(control.proof_wait_reason, "first_pass_horizon_unprovable_long_tail")
        self.assertTrue(control.proof_wait_long_tail)
        self.assertIsNone(control.requested_ns)

    def test_old_pre_transport_queue_does_not_pin_completion_prefix(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2048)
        control.begin_turn("queued", 0, 19, 0)
        for i in range(11):
            complete(control, i, i + 1)
        self.assertIsNone(control.poll(12 * SECOND))
        self.assertIsNone(control.start_ns)
        control.advance_undispatched_turn("queued", 13 * SECOND)
        control.poll(14 * SECOND)
        self.assertEqual(control.start_ns, 11 * SECOND)
        self.assertEqual(control.received_turns, 11)  # Queue work is not progress.
        self.assertIsNone(control.hold_reason)

    def test_delayed_native_completion_after_long_wait_still_sets_correct_threshold(self):
        from perf_benchmark.pipeline.execution.underfill import PROOF_WAIT_NS
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2048)
        control.begin_turn("t-19", 0, 19, 0)
        for i in range(11):
            complete(control, i, i + 1)
        control.poll(12 * SECOND)
        control.poll(12 * SECOND + PROOF_WAIT_NS)
        self.assertIsNone(control.start_ns)
        self.assertIsNone(control.hold_reason)
        control.observe(terminal(19, 0.5))
        control.poll(100 * SECOND)
        self.assertEqual(control.start_ns, 10 * SECOND)  # Not the false 11 s.
        self.assertIsNone(control.hold_reason)

    def test_last_wave_admission_during_ack_invalidates_partial_horizon(self):
        control = started()
        occupancy(control, 20, 2047)
        for i in range(11, 19):
            control.begin_turn(f"t-{i}", 0, i, 12 * SECOND)
        self.assertIsNotNone(control.poll(81 * SECOND))
        # Simulate admission during flush. Even though admission normally uses
        # the current clock, use an earlier still-legal floor for the adversarial
        # case: no proof based solely on there once having been unstarted work.
        control.begin_turn("t-19", 0, 19, 13 * SECOND)
        self.assertFalse(control.ready_to_stop(81 * SECOND))
        with self.assertRaises(RuntimeError):
            control.request_stop(81 * SECOND)

    def test_post_confirmation_admission_is_a_valid_terminal_horizon(self):
        control = started()
        occupancy(control, 20, 2047)
        for i in range(11, 19):
            control.begin_turn(f"t-{i}", 0, i, 12 * SECOND)
        self.assertIsNotNone(control.poll(81 * SECOND))
        control.begin_turn("t-19", 0, 19, 81 * SECOND)
        self.assertTrue(control.ready_to_stop(81 * SECOND))
        control.request_stop(81 * SECOND)

    def test_pruning_before_start_keeps_predecessor_and_future_changes(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        occupancy(control, 0, 2048)
        occupancy(control, 1, 2047)
        occupancy(control, 2, 2048)
        control.begin_turn("t-0", 0, 0, 3 * SECOND)
        control.poll(4 * SECOND)
        self.assertEqual(list(control.occupancy), [(2 * SECOND, False)])
        occupancy(control, 5, 2047)
        control.observe(terminal(0, 6))
        for i in range(1, 11):
            complete(control, i, i + 6, floor=4 * SECOND)
        complete(control, 19, 100, floor=4 * SECOND)
        actual = control.poll(101 * SECOND)
        self.assertEqual(actual, UnderfillDecision(16 * SECOND, 16 * SECOND, 76 * SECOND))

    def test_large_same_side_history_is_constant_space_including_ties(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        for i in range(100_000):
            occupancy(control, i / 1000, i % 2048)
        self.assertEqual(len(control.occupancy), 1)
        self.assertEqual(control.peak_occupancy_segments, 1)
        self.assertEqual(control.captured, 100_000)
        self.assertIsNone(control.hold_reason)
        # A same-ns recovery immediately undone never creates a duration.
        stamp = 100
        occupancy(control, stamp, 2048)
        occupancy(control, stamp, 2047)
        self.assertEqual(list(control.occupancy), [(0, True)])

    def test_long_spell_capacity_holds_instead_of_truncating_proof(self):
        from perf_benchmark.pipeline.execution.underfill import MAX_OCCUPANCY_SEGMENTS
        control = UnderfillStop("trial", 3072, 20, 2048)
        for i in range(MAX_OCCUPANCY_SEGMENTS + 10):
            occupancy(control, i * 61, 2047 if i % 2 else 2048)
        self.assertEqual(control.hold_reason, "occupancy_replay_capacity_unprovable")
        self.assertLessEqual(control.peak_occupancy_segments, MAX_OCCUPANCY_SEGMENTS)
        self.assertIsNone(control.poll(1_000_000 * SECOND))
        self.assertIsNone(control.decision)
        self.assertFalse(control.occupancy)

    def test_large_pre55_short_spells_do_not_exhaust_replay_and_ties_remain_exact(self):
        control = UnderfillStop("trial", 3072, 20, 2048)
        for i in range(100_000):
            occupancy(control, i / 1000, 2048 if i % 2 == 0 else 2047)
        self.assertLessEqual(control.peak_occupancy_segments, 3)
        self.assertGreater(control.compacted_short_spells, 49_000)
        self.assertIsNone(control.hold_reason)
        # The last low begins at 99.999s. Undoing recovery at the same timestamp
        # must preserve that exact origin, even across a long raw history.
        occupancy(control, 110, 2048)
        occupancy(control, 110, 2047)
        for i in range(11):
            complete(control, i, i + 1)
        complete(control, 19, 161, floor=12 * SECOND)
        decision = control.poll(162 * SECOND)
        self.assertEqual(decision, UnderfillDecision(11 * SECOND,
                                                    99_999_000_000, 159_999_000_000))

    def test_random_interleaved_polls_and_ties_match_native_exactly(self):
        from perf_benchmark.plugins.metrics.standard.maxtpm import _concurrency_end
        rng = random.Random(819255)
        for _ in range(300):
            control = UnderfillStop("trial", 3072, 20, 2048)
            rows = []
            def row(stamp, value):
                occupancy(control, stamp, value)
                rows.append((int(stamp * SECOND), value))
            row(0, 2048)
            # Prefix polling prunes old history while progress is still below K.
            for i in range(10):
                row(i + 1, rng.randrange(2045, 2051))
                row(i + 1, rng.randrange(2045, 2051))
                complete(control, i, i + 1, floor=i * SECOND + 1)
                control.poll((i + 1) * SECOND + 1)
            complete(control, 10, 11, floor=10 * SECOND + 1)
            control.begin_turn("tail", 0, 99, 12 * SECOND)
            stamp = 12
            for _ in range(60):
                stamp += rng.randint(0, 4)
                row(stamp, rng.randrange(2045, 2051))
                control.poll(stamp * SECOND)
            horizon = (stamp + 61) * SECOND
            actual = control.poll(horizon + 1)
            end, confirmed, reason = _concurrency_end(rows, 11 * SECOND, horizon, 2048)
            self.assertIsNone(control.hold_reason)
            if reason == "concurrency_under_target_60s":
                self.assertEqual(actual, UnderfillDecision(11 * SECOND, end, confirmed))
            else:
                self.assertIsNone(actual)

    def test_compaction_matches_native_for_arbitrary_55_starts(self):
        from perf_benchmark.plugins.metrics.standard.maxtpm import _concurrency_end
        rng = random.Random(60421)
        for _ in range(500):
            control = UnderfillStop("trial", 3072, 20, 2048)
            rows, stamp = [], 0
            for _ in range(rng.randrange(1, 120)):
                stamp += rng.choice((0, 0, 1, 20, 59, 60, 61, 90))
                value = rng.choice((0, 2047, 2048, 2049))
                item = occupancy(control, stamp, value)
                rows.append((item.at_ns, value))
            start = rng.randrange(max(2, stamp + 2))
            for i in range(11):
                complete(control, i, start)
            horizon = max(stamp, start) + 100
            complete(control, 19, horizon)
            actual = control.poll(horizon * SECOND + 1)
            end, confirmed, reason = _concurrency_end(rows, start * SECOND,
                                                    horizon * SECOND, 2048)
            if reason == "occupancy_not_observed_at_maxtpm_start":
                self.assertIsNone(actual)
                self.assertEqual(control.hold_reason, "occupancy_not_observed_at_start")
            elif confirmed is not None:
                self.assertEqual(actual, UnderfillDecision(start * SECOND, end, confirmed))
            else:
                self.assertIsNone(actual)
