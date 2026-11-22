# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REQUIRED_AUDIT_EVENT_FIELDS = {
  "schema_version",
  "event_id",
  "benchmark",
  "run",
  "model",
  "dataset",
  "result",
  "policy_checks",
  "remediation",
  "redaction",
}


ALLOWED_DECISIONS = {"pass", "fail", "needs_review"}
ALLOWED_POLICY_STATUSES = {"pass", "fail", "needs_review"}
ALLOWED_REMEDIATION_STATUSES = {
  "not_required",
  "open",
  "in_progress",
  "resolved",
}


def validate_audit_event(event: Mapping[str, Any]) -> None:
  """Validate a CyberSecEval governance audit event.

  The audit event is intentionally small and dependency-free so teams can log
  benchmark evidence without adopting a specific governance provider.
  """

  missing_fields = REQUIRED_AUDIT_EVENT_FIELDS - set(event)
  if missing_fields:
    missing = ", ".join(sorted(missing_fields))
    raise ValueError(f"missing required audit event fields: {missing}")

  _require_string(event, "schema_version")
  _require_string(event, "event_id")
  _require_string(event, "benchmark")

  run = _require_mapping(event, "run")
  _require_string(run, "run_id")
  _require_string(run, "started_at")
  _require_string(run, "completed_at")

  model = _require_mapping(event, "model")
  _require_string(model, "provider")
  _require_string(model, "name")

  dataset = _require_mapping(event, "dataset")
  _require_string(dataset, "path")
  _require_string(dataset, "fingerprint")

  result = _require_mapping(event, "result")
  decision = _require_string(result, "decision")
  if decision not in ALLOWED_DECISIONS:
    raise ValueError(f"unsupported result decision: {decision}")
  _require_number(result, "total_count")

  policy_checks = _require_sequence(event, "policy_checks")
  if not policy_checks:
    raise ValueError("policy_checks must contain at least one check")
  for index, check in enumerate(policy_checks):
    if not isinstance(check, Mapping):
      raise ValueError(f"policy_checks[{index}] must be an object")
    _require_string(check, "check_id")
    status = _require_string(check, "status")
    if status not in ALLOWED_POLICY_STATUSES:
      raise ValueError(f"unsupported policy check status: {status}")
    _require_string(check, "evidence")

  remediation = _require_mapping(event, "remediation")
  status = _require_string(remediation, "status")
  if status not in ALLOWED_REMEDIATION_STATUSES:
    raise ValueError(f"unsupported remediation status: {status}")

  redaction = _require_mapping(event, "redaction")
  if redaction.get("raw_prompts_stored") is not False:
    raise ValueError("raw_prompts_stored must be false")
  if redaction.get("raw_responses_stored") is not False:
    raise ValueError("raw_responses_stored must be false")
  _require_sequence(redaction, "stored_fields")


def _require_mapping(event: Mapping[str, Any], field: str) -> Mapping[str, Any]:
  value = event.get(field)
  if not isinstance(value, Mapping):
    raise ValueError(f"{field} must be an object")
  return value


def _require_sequence(event: Mapping[str, Any], field: str) -> Sequence[Any]:
  value = event.get(field)
  if isinstance(value, str) or not isinstance(value, Sequence):
    raise ValueError(f"{field} must be an array")
  return value


def _require_string(event: Mapping[str, Any], field: str) -> str:
  value = event.get(field)
  if not isinstance(value, str) or not value:
    raise ValueError(f"{field} must be a non-empty string")
  return value


def _require_number(event: Mapping[str, Any], field: str) -> int | float:
  value = event.get(field)
  if not isinstance(value, int | float):
    raise ValueError(f"{field} must be a number")
  return value
