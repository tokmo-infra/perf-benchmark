"""Lossless metric-set definitions and explicit final-Turn patches.

Only structure is shared: every numeric value remains a saved Metrics result.
"""
from collections import OrderedDict
import math

from perf_benchmark.plugins.saver.sqlite.encoding import _json_dump

_LAYOUTS = OrderedDict()


def encode_metrics(metrics):
    layout, values = [], []
    for name, metric in sorted(metrics.items()):
        if len(metric.statistics) == 1:
            name_of_statistic, value = next(iter(metric.statistics.items()))
            keys = (name_of_statistic,)
        else:
            keys = tuple(sorted(metric.statistics))
            numbers = [metric.statistics[key] for key in keys]
            value = None if all(number is None for number in numbers) else numbers
        layout.append((name, metric.unit, metric.source, metric.formula_version,
                       keys, metric.availability, metric.unavailable_reason))
        values.append(value)
    key = tuple(layout)
    payload = _LAYOUTS.get(key)
    if payload is None:
        payload = _json_dump(key)
        # At most 128 small layouts; arbitrary large plugin schemas bypass it.
        if len(payload.encode()) <= 32768:
            if len(_LAYOUTS) >= 128:
                _LAYOUTS.popitem(last=False)
            _LAYOUTS[key] = payload
    return payload, values


def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a == b == 0:
        return math.copysign(1, a) == math.copysign(1, b)
    return a == b


def metric_patch(layout, base_layout):
    previous = {row[0]: (i, row) for i, row in enumerate(base_layout)}
    indices = []
    for definition in layout:
        old = previous.get(definition[0])
        indices.append(old[0] if old is not None and old[1] == definition else None)
    names = {row[0] for row in layout}
    return indices, [i for i, row in enumerate(base_layout) if row[0] not in names]


def metric_delta(values, base_values, patch):
    indices, removed = patch
    changes = [[i, value] for i, (old, value) in enumerate(zip(indices, values))
               if old is None or not same_value(base_values[old], value)]
    return changes, removed


def restore_metrics(layout, values, *, base_layout=(), base_values=(), removed=None):
    if removed is not None:
        changes = dict(values)
        if len(changes) != len(values) or any(type(i) is not int or not 0 <= i < len(layout) for i in changes):
            raise ValueError('invalid metric patch layout')
        names = {row[0] for row in layout}
        if removed != [i for i, row in enumerate(base_layout) if row[0] not in names]:
            raise ValueError('invalid metric removal layout')
        previous = {row[0]: (row, value) for row, value in zip(base_layout, base_values)}
        values = []
        for index, definition in enumerate(layout):
            if index in changes:
                values.append(changes[index])
            else:
                old = previous.get(definition[0])
                if old is None or old[0] != definition:
                    raise ValueError('missing saved metric value')
                values.append(old[1])
    if len(values) != len(layout):
        raise ValueError('invalid metric statistics layout')
    output = {}
    for definition, numbers in zip(layout, values):
        name, unit, source, formula, keys, availability, reason = definition
        if numbers is None:
            numbers = [None] * len(keys)
        elif len(keys) == 1:
            numbers = [numbers]
        if not isinstance(numbers, list) or len(numbers) != len(keys):
            raise ValueError('invalid metric statistics layout')
        output[name] = dict(unit=unit, source=source, formula_version=formula,
                           availability=availability, unavailable_reason=reason,
                           statistics=dict(zip(keys, numbers)))
    return output
