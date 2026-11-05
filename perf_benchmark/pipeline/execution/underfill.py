"""Engine-owned stop proof, never a replacement for native first-pass metrics.

Returned HTTP timestamps may tighten a producer floor, but a pending native
result cannot: it could already have completed before Python observes it. Such
uncertainty keeps waiting for evidence, never a timeout masquerading as a
completion watermark. Undispatched work can safely advance its bound on the
engine loop; submitted native requests cannot.
"""
from collections import deque
from dataclasses import dataclass
import heapq

from perf_benchmark.pipeline.results.measurement import AttemptFact, OccupancyFact, TurnFact

MINUTE_NS = 60_000_000_000
START_PERCENT = 55
# Diagnostic long-tail threshold, not a workload/proof-expiry timeout.
PROOF_WAIT_NS = MINUTE_NS
MAX_OCCUPANCY_SEGMENTS = 8192


@dataclass(frozen=True, slots=True)
class UnderfillDecision:
    start_ns: int
    end_ns: int
    confirmed_ns: int


class UnderfillStop:
    def __init__(self, trial_id: str, session_count: int, turn_count: int, target: int, start_percent=START_PERCENT):
        if any(type(n) is not int or n <= 0 for n in (session_count, turn_count, target)):
            raise ValueError("underfill stop requires positive selected counts and target")
        self.trial_id, self.session_count, self.turn_count, self.target = (
            trial_id, session_count, turn_count, target,
        )
        from perf_benchmark.pipeline.maxtpm import parse_start_percent
        from decimal import Decimal
        self.start_percent = parse_start_percent(start_percent)
        if self.start_percent == 0:
            raise ValueError("underfill stop does not support a zero-percent start")
        numerator, denominator = (Decimal(str(self.start_percent)) / 100).as_integer_ratio()
        self.threshold = (turn_count * numerator + denominator - 1) // denominator
        self.active = {}  # id -> (earliest possible completion, session ordinal, turn ordinal)
        self.finished = set()
        self.turns = []  # completion time, session ordinal, turn ordinal, id
        self.pending_occupancy = {}  # capture sequence -> (native timestamp, value)
        # Only predicate transitions matter to _concurrency_end. Preserve their
        # exact ns and final value at ties; raw absolute values still go to metrics.
        self.occupancy = deque()  # timestamp, below target
        self.last_capture_ns = self.last_occupancy_low = None
        self.peak_occupancy_segments = 0
        self.compacted_short_spells = 0
        self.captured = self.submitted = self.received_turns = self.sealed_turns = 0
        self.latest_completion_ns = -1
        self.progress_watermark_ns = self.occupancy_watermark_ns = -1
        self.start_ns = self.under_since_ns = self.value = None
        self.decision: UnderfillDecision | None = None
        self.requested_ns = self.hold_reason = self.native_window_matches = None
        self.proof_wait_reason = self.proof_wait_since_ns = None
        self.proof_wait_long_tail = False
        self.closed = False

    def hold(self, reason: str):
        # Freeze proof counts on uncertainty. Do not retain an unbounded replay
        # after disabling the action, and never alter the authoritative facts.
        self.hold_reason = self.hold_reason or reason
        self.occupancy.clear()
        self.turns.clear()

    def begin_turn(self, turn_id: str, session_ordinal: int, turn_ordinal: int, at_ns: int):
        if self.hold_reason:
            return
        if (turn_id in self.active or turn_id in self.finished
                or not 0 <= session_ordinal < self.session_count
                or at_ns <= self.progress_watermark_ns):
            self.hold("turn_producer_fence_invalid")
            return
        self.active[turn_id] = (at_ns, session_ordinal, turn_ordinal)

    def advance_undispatched_turn(self, turn_id: str, at_ns: int):
        """Only the engine may call this before ANY transport invocation.

        No response timestamp exists yet: an eventual response (or no-attempt
        lifecycle fallback) must be at or after this bound. Never use it for a
        pending native result, which may already have completed unobserved.
        """
        if self.hold_reason or self.requested_ns is not None:
            return
        producer = self.active.get(turn_id)
        if producer is None:
            return  # Final Turn already handed off; no producer remains.
        if type(at_ns) is not int or at_ns < producer[0]:
            self.hold("undispatched_producer_clock_invalid")
            return
        self.active[turn_id] = (at_ns, *producer[1:])

    def complete_attempt(self, turn_id: str, result, ready_ns: int):
        """Tighten, never release, a final-Turn producer's lower bound.

        This is not 55% progress. Only the final TurnFact counts. Opt-in forbids
        retries, so its returned response timestamp cannot be replaced later.
        """
        if self.hold_reason or self.requested_ns is not None:
            return
        producer = self.active.get(turn_id)
        stamp = result.request.response_completed_ns
        if (producer is None or result.turn_id != turn_id or result.attempt != 1
                or (stamp is not None and (type(stamp) is not int
                                          or not producer[0] <= stamp <= ready_ns))):
            self.hold("returned_attempt_completion_fence_invalid")
            return
        self.active[turn_id] = (stamp if stamp is not None else ready_ns, *producer[1:])

    def abandon_turn(self, turn_id: str):
        # Only the no-Attempt SchedulingStopped path may abandon a registration.
        self.active.pop(turn_id, None)

    def capture_occupancy(self, at_ns: int, value: int):
        if self.hold_reason:
            return
        if (at_ns <= self.occupancy_watermark_ns or value < 0
                or (self.last_capture_ns is not None and at_ns < self.last_capture_ns)):
            self.hold("occupancy_producer_fence_invalid")
            return
        self.last_capture_ns = at_ns
        self.captured += 1
        self.pending_occupancy[self.captured] = (at_ns, value)

    def observe(self, fact):
        if self.hold_reason:
            return
        if fact.trial_id != self.trial_id:
            self.hold("fact_trial_identity_mismatch")
            return
        # AttemptFact subclasses TurnFact; attempts MUST NOT advance 55%.
        if isinstance(fact, AttemptFact):
            return
        if isinstance(fact, OccupancyFact):
            if fact.kind != "in_flight":
                return
            sequence = self.submitted + 1
            if self.pending_occupancy.get(sequence) != (fact.at_ns, fact.value):
                self.hold("occupancy_capture_handoff_order_mismatch")
                return
            self.pending_occupancy.pop(sequence)
            self.submitted = sequence
            if self.decision is not None:
                return  # later captures cannot erase an already sealed spell
            low = fact.value < self.target
            # Commit compaction only AFTER the recovery's timestamp group is
            # final. A same-ns low/high/low must NOT split a continuous spell.
            # A CLOSED low spell shorter than 60s cannot qualify for ANY future
            # 55% start (clipping at start only shortens it). Remove the pair,
            # retaining its real preceding high and all potentially qualifying
            # spell endpoints. This changes no native start/end/confirmation.
            if (len(self.occupancy) >= 3
                    and fact.at_ns > self.occupancy[-1][0]
                    and not self.occupancy[-1][1] and self.occupancy[-2][1]
                    and self.occupancy[-1][0] - self.occupancy[-2][0] < MINUTE_NS):
                self.occupancy.pop()
                self.occupancy.pop()
                self.compacted_short_spells += 1
            if self.occupancy and self.occupancy[-1][0] == fact.at_ns:
                self.occupancy.pop()  # zero-duration intermediate state
                prior = self.occupancy[-1][1] if self.occupancy else self.value
            else:
                prior = self.last_occupancy_low
            if prior is None or low != prior:
                if len(self.occupancy) == MAX_OCCUPANCY_SEGMENTS:
                    self.hold("occupancy_replay_capacity_unprovable")
                    return
                self.occupancy.append((fact.at_ns, low))
            self.last_occupancy_low = low
            self.peak_occupancy_segments = max(self.peak_occupancy_segments, len(self.occupancy))
            return
        if not isinstance(fact, TurnFact):
            return
        producer = self.active.pop(fact.id, None)
        if producer is None or fact.id in self.finished:
            self.hold("final_turn_without_producer_fence")
            return
        floor, session_ordinal, turn_ordinal = producer
        self.finished.add(fact.id)
        stamp = (fact.response_completed_ns if fact.response_completed_ns is not None
                 else fact.turn_completed_ns)
        if (type(stamp) is not int or stamp < floor
                or stamp <= self.progress_watermark_ns or fact.ordinal != turn_ordinal):
            self.hold("final_turn_completion_violates_watermark")
            return
        if fact.status not in {"succeeded", "failed", "cancelled"}:
            self.hold("nonterminal_turn_fact")
            return
        self.received_turns += 1
        self.latest_completion_ns = max(self.latest_completion_ns, stamp)
        if fact.status == "cancelled":
            if self.requested_ns is None:
                self.hold("cancelled_before_controlled_stop_native_progress_at_risk")
            return
        if self.start_ns is None:
            heapq.heappush(self.turns, (stamp, session_ordinal, turn_ordinal, fact.id))

    def _wait_for_proof(self, reason: str, now_ns: int):
        if reason != self.proof_wait_reason:
            self.proof_wait_reason, self.proof_wait_since_ns = reason, now_ns
            self.proof_wait_long_tail = False
        elif now_ns - self.proof_wait_since_ns >= PROOF_WAIT_NS:
            # A slow producer is not invalid evidence. Keep accepting its real
            # completion instead of permanently discarding an otherwise valid
            # run after an arbitrary observer deadline. Safety TTFT is separate.
            self.proof_wait_long_tail = True

    def ready_to_stop(self, now_ns: int) -> bool:
        """Recheck in the SAME non-await segment as native cancellation.

        Complete first pass: its terminal max must reach confirmation. Partial
        first pass: at least one selected Turn will remain unstarted after all
        active Turns settle, so native uses session/trial bounds. An active Turn
        guarantees a session finishes after this stop. Merely being unfinished
        in Python is NOT proof when cancellation could finish all selected Turns.
        """
        if (self.decision is None or self.hold_reason or self.requested_ns is not None
                or now_ns < self.decision.confirmed_ns
                or self.received_turns >= self.turn_count):
            return False
        horizon = self.decision.confirmed_ns
        return (self.latest_completion_ns >= horizon
                or any(p[0] >= horizon for p in self.active.values())
                or (bool(self.active)
                    and self.received_turns + len(self.active) < self.turn_count))

    def poll(self, now_ns: int) -> UnderfillDecision | None:
        if self.closed or self.hold_reason or self.requested_ns is not None:
            return None
        # Strictly below the clock/floor: a future producer can use the same ns.
        watermark = min(now_ns - 1, min((p[0] - 1 for p in self.active.values()),
                                       default=now_ns - 1))
        if watermark < self.progress_watermark_ns:
            self.hold("completion_watermark_regressed")
            return None
        self.progress_watermark_ns = watermark
        if self.start_ns is None:
            while self.turns and self.turns[0][0] <= watermark:
                stamp, *_ = heapq.heappop(self.turns)
                self.sealed_turns += 1
                if self.sealed_turns == self.threshold:
                    self.start_ns = stamp
                    self.turns.clear()  # ranks beyond K cannot change the start
                    break
        if self.start_ns is None:
            # Future Kth completion cannot be earlier than this sealed prefix.
            # Retain its predecessor state and every later threshold transition.
            while len(self.occupancy) > 1 and self.occupancy[1][0] <= watermark:
                self.occupancy.popleft()
            if self.received_turns >= self.threshold:
                self._wait_for_proof("completion_prefix_unprovable_long_tail", now_ns)
            return None
        if self.received_turns >= self.turn_count:
            return None  # natural first pass already finished
        if self.proof_wait_reason == "completion_prefix_unprovable_long_tail":
            self.proof_wait_reason = self.proof_wait_since_ns = None
            self.proof_wait_long_tail = False
        watermark = min(now_ns - 1, min((p[0] - 1 for p in self.pending_occupancy.values()),
                                       default=now_ns - 1))
        if watermark < self.occupancy_watermark_ns:
            self.hold("occupancy_watermark_regressed")
            return None
        self.occupancy_watermark_ns = watermark
        while self.occupancy and self.occupancy[0][0] <= watermark:
            stamp, low = self.occupancy.popleft()
            if stamp <= self.start_ns:
                self.value = low
                self.under_since_ns = self.start_ns if low else None
                continue
            if self.value is None:
                self.hold("occupancy_not_observed_at_start")
                return None
            self._confirm(stamp)  # recovery at exactly +60s cannot erase the spell
            self.value = low
            if not low:
                self.under_since_ns = None
            elif self.under_since_ns is None:
                self.under_since_ns = stamp
        if watermark >= self.start_ns and self.value is None:
            self.hold("occupancy_not_observed_at_start")
            return None
        self._confirm(watermark)
        if self.decision is not None:
            if self.ready_to_stop(now_ns):
                return self.decision
            self._wait_for_proof("first_pass_horizon_unprovable_long_tail", now_ns)
        return None

    def _confirm(self, horizon: int):
        if (self.decision is None and self.start_ns is not None
                and self.under_since_ns is not None
                and horizon - self.under_since_ns >= MINUTE_NS):
            self.decision = UnderfillDecision(
                self.start_ns, self.under_since_ns, self.under_since_ns + MINUTE_NS,
            )

    def request_stop(self, at_ns: int):
        if not self.ready_to_stop(at_ns):
            raise RuntimeError("underfill stop has no unused proof")
        self.requested_ns = at_ns

    def reconcile(self, native_diagnostics):
        if self.requested_ns is not None:
            self.native_window_matches = (
                native_diagnostics.get("maxtpm_start_ns") == self.decision.start_ns
                and native_diagnostics.get("maxtpm_end_ns") == self.decision.end_ns
                and native_diagnostics.get("maxtpm_underfill_confirmed_ns")
                == self.decision.confirmed_ns
            )

    def snapshot(self):
        return {
            "enabled": True, "target_concurrency": self.target,
            "dataset_sessions": self.session_count, "dataset_turns": self.turn_count,
            "start_percent": self.start_percent, "start_turns": self.threshold,
            "eligible_sealed_turns": self.sealed_turns, "received_final_turns": self.received_turns,
            "start_ns": self.start_ns, "under_since_ns": self.under_since_ns,
            "end_ns": self.decision.end_ns if self.decision else None,
            "confirmed_ns": self.decision.confirmed_ns if self.decision else None,
            "progress_watermark_ns": self.progress_watermark_ns,
            "occupancy_watermark_ns": self.occupancy_watermark_ns,
            "pending_final_turns": len(self.active),
            "captured_occupancy": self.captured, "submitted_occupancy": self.submitted,
            "pending_occupancy": len(self.pending_occupancy),
            "occupancy_segments": len(self.occupancy),
            "peak_occupancy_segments": self.peak_occupancy_segments,
            "compacted_short_spells": self.compacted_short_spells,
            "occupancy_segment_limit": MAX_OCCUPANCY_SEGMENTS,
            "proof_wait_reason": self.proof_wait_reason,
            "proof_wait_since_ns": self.proof_wait_since_ns,
            "proof_wait_warning_ns": PROOF_WAIT_NS,
            "proof_wait_long_tail": self.proof_wait_long_tail,
            "proof_wait_is_workload_timeout": False,
            "proof_counts_frozen_on_hold": bool(self.hold_reason),
            "stop_requested_ns": self.requested_ns, "hold_reason": self.hold_reason,
            "closed": self.closed, "native_window_matches": self.native_window_matches,
            "result_at_risk": bool(self.hold_reason) or self.native_window_matches is not True,
            "limitation": "pending_transport_has_no_live_completion_fence; "
                          "unproven_stop_waits_for_actual_evidence; "
                          "native_finalize_and_exporter_still_required",
        }
