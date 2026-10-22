"""Build immutable published records for reader/Saver fixtures."""
from dataclasses import replace
from types import MappingProxyType

from perf_benchmark.pipeline.results.saver import _snapshot
from perf_benchmark.pipeline.contracts.results import ScalarMetricRecord


def frozen_record(kind, *args, **kwargs):
    record = _snapshot(kind, kind(*args, **kwargs))
    return replace(record, metrics=MappingProxyType({name: ScalarMetricRecord(**value)
                                                   for name, value in record.metrics.items()}))
