"""Shared validation for frozen facts, plugin configuration and request headers."""

import math
import re
from typing import Literal, overload


def random_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer")
    return value


@overload
def positive_int(value: object, name: str, *, optional: Literal[False] = False, allow_zero: bool = False) -> int: ...
@overload
def positive_int(value: object, name: str, *, optional: Literal[True], allow_zero: bool = False) -> int | None: ...
def positive_int(value, name, *, optional=False, allow_zero=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (not allow_zero and value == 0):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {bound} integer")
    return value


def positive_number(value, name, *, optional=False, allow_zero=False):
    if optional and value is None:
        return None
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0 and (allow_zero or value > 0))
    except OverflowError:
        valid = False
    if not valid:
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {bound}")
    return float(value)


def validate_extra_headers(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError("extra_headers must contain string keys and values")
    if any(
        re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", key) is None
        or re.search(r"[\x00-\x08\x0a-\x1f\x7f\ud800-\udfff]", item) is not None
        for key, item in value.items()
    ):
        raise ValueError("extra_headers contains an invalid header name or value")
    names = [key.lower() for key in value]
    if len(names) != len(set(names)):
        raise ValueError("extra_headers contains duplicate header names")
    reserved = set(names) & {"host", "content-type", "content-length"}
    if reserved:
        raise ValueError(f"extra_headers conflicts with managed headers: {', '.join(sorted(reserved))}")
    return value
