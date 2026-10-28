#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
  else
    PYTHON=python
  fi
fi
exec "$PYTHON" -m pytest -q \
  tests/normalization/test_open_swe_tool_ids.py::test_match_single_repairs_only_unambiguous_real_ids_and_preserves_source \
  tests/normalization/test_open_swe_tool_ids.py::test_match_single_rejects_ambiguous_parallel_results \
  'tests/normalization/test_open_swe_traces_e2e.py::test_complete_recorded_swe_replay_uses_natural_outputs[recorded]' \
  'tests/normalization/test_open_swe_traces_e2e.py::test_complete_recorded_swe_replay_uses_natural_outputs[anthropic-tools]' \
  tests/app/test_reasoning_validation.py::test_template_validation_negotiates_supported_effort \
  tests/runtime/test_native_executor.py::test_started_callback_failure_settles_native_before_release \
  tests/executor/test_rust_openai_chat.py::test_pending_attempt_settles_before_releasing_body \
  tests/executor/test_openai_chat.py::test_explicit_null_uses_server_default_without_output_cap \
  tests/executor/test_rust_openai_chat.py::test_null_output_cap_survives_native_http \
  tests/metrics/test_maxtpm_billing.py::test_maxtpm_missing_usage_does_not_drop_a_minute_or_request \
  tests/metrics/test_maxtpm_billing.py::test_maxtpm_latencies_use_exact_successful_completion_cohort_and_pooled_quantiles \
  tests/pipeline/test_progress.py::test_decode_preview_weights_receive_time_and_resets_each_trial \
  tests/exporter/test_tabular.py::test_summary_uses_global_distributions_and_places_parameters_first \
  tests/metrics/test_maxtpm_billing.py::test_billing_preserves_partial_cost_and_does_not_assume_missing_cache_zero \
  tests/metrics/test_maxtpm_billing.py::test_billing_distinguishes_no_observed_cost_from_observed_zero \
  tests/exporter/test_tabular.py::test_starred_income_stays_numeric_in_the_existing_four_columns \
  tests/exporter/test_tabular.py::test_append_matching_headers_and_preserve_original_on_mismatch \
  tests/pipeline/test_resources.py::test_run_and_trial_resources_close_once
