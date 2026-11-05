"""Shared maxTPM start selection used by metrics and optional stop control."""

import math


def parse_start_percent(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value < 100 or not math.isfinite(value)):
        raise ValueError("metrics maxtpm_start_percent must be finite and in [0, 100)")
    return float(value)


def resolve_start_percent(settings, session_count, target_concurrency):
    """Use selected sample count and request concurrency; explicit settings win."""
    if "maxtpm_start_percent" in settings:
        return settings["maxtpm_start_percent"], "configured"
    if (session_count is not None and target_concurrency is not None
            and session_count * 2 == target_concurrency * 3):
        return 53.0, "samples_1_5x_concurrency"
    return 40.0, "default"
