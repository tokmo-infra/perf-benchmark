"""Committed, replaceable 10-second buckets; formulas stay in Standard Metrics.

Late results update their request-start bucket. Token rates use completion time.
ITL uses the original event intervals in DuckDB, never quantiles of request means.
"""

from time import monotonic

from perf_benchmark.core.models import MetricWindowMeasurement
from perf_benchmark.pipeline.contracts.results import MetricSeries
from perf_benchmark.pipeline.results.measurement import AttemptFact, OccupancyFact, TurnFact

BUCKET_NS = 10_000_000_000
STATISTICS = frozenset(("am", "time_wm", "p50", "p90", "p95", "p99", "min", "max"))
DISTRIBUTIONS = frozenset((
    "ttft", "tpot", "itl", "e2e", "input_tokens", "output_tokens", "prefill_throughput",
    "decode_throughput", "output_tokens_per_event_estimate", "speculative_acceptance_proxy",
))


class Timeline:
    def __init__(self):
        self.dirty = set()
        self.last_occupancy = {}
        self.next_publish = monotonic() + 1

    def observe(self, facts):
        for fact in facts:
            if isinstance(fact, TurnFact) and not isinstance(fact, AttemptFact):
                for stamp in (fact.request_ns, fact.response_completed_ns):
                    if stamp is not None:
                        self.dirty.add((fact.trial_id, stamp // BUCKET_NS * BUCKET_NS))
            elif isinstance(fact, OccupancyFact) and fact.kind == "in_flight":
                bucket = fact.at_ns // BUCKET_NS * BUCKET_NS
                previous = self.last_occupancy.get(fact.trial_id, (fact.at_ns, fact.value))[0] // BUCKET_NS * BUCKET_NS
                self.dirty.update((fact.trial_id, t) for t in range(previous, bucket + 1, BUCKET_NS))
                self.last_occupancy[fact.trial_id] = (fact.at_ns, fact.value)

    def emit(self, backend, *, force=False, final_trial=None):
        if not force and monotonic() < self.next_publish:
            return ()
        self.next_publish = monotonic() + 1
        selected = sorted(self.dirty)
        if not force:
            selected = selected[:4]  # Bound the number of buckets per publication.
        if not selected:
            return ()
        backend._flush_pending(force=True)
        output = []
        for trial, start in selected:
            end, name = start + BUCKET_NS, f"timeline:{start}"
            succeeded = backend._one(
                "SELECT count(*) FROM turns WHERE trial_id=? AND request_ns>=? AND request_ns<? "
                "AND status='succeeded'", [trial, start, end],
            )[0]
            points = [p for p in backend._distribution_points(trial, name, start, end, succeeded, 0, request_only=True, statistics=STATISTICS)
                      if p.metric in DISTRIBUTIONS and p.statistic in STATISTICS]
            where = "trial_id=? AND response_completed_ns>=? AND response_completed_ns<?"
            params = [trial, start, end]
            completed = backend._one(f"SELECT count(*) FROM turns WHERE {where} AND status='succeeded'", params)[0]
            prompt = backend._usage_totals(where, params, "prompt", completed)
            completion = backend._usage_totals(where, params, "completion", completed)
            throughput = []
            backend._add_token_scalars(throughput, trial, name, BUCKET_NS / 1e9,
                                       prompt, completion, completed, (0, 0), (0, 0), 0)
            points.extend(p for p in throughput if p.metric in {
                "workload_prompt", "workload_completion", "workload_total",
            })
            observed = self.last_occupancy.get(trial)
            covered = observed is not None and (observed[0] >= end or
                      (final_trial == trial and observed[1] == 0))
            points.append(backend._point(
                trial, name, "actual_in_flight_concurrency", "value",
                backend._time_weighted(trial, "in_flight", start, end) if covered else None,
                "available" if covered else "unavailable",
                None if covered else "occupancy_not_observed" if observed is None else "bucket_in_progress",
                "scheduler",
            ))
            output.append(MetricSeries(trial, MetricWindowMeasurement(
                name, start, end, BUCKET_NS / 1e9, "available", None,
            ), tuple(points)))
        self.dirty.difference_update(selected)
        return tuple(output)

    def release(self, trial):
        self.dirty = {key for key in self.dirty if key[0] != trial}
        self.last_occupancy.pop(trial, None)
