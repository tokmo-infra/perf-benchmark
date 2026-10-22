from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from perf_benchmark.core.validation import random_seed


def case_description(config):
    value = config.get("case")
    if not isinstance(value, str) or not value.strip() or any(ch in value for ch in "\r\n\t"):
        raise ValueError("case is required: describe this test on one nonempty line")
    return value.strip()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError("configuration root must be an object")
    value["seed"] = random_seed(value.get("seed", 0))
    value["case"] = case_description(value)
    return value, resolved.parent
