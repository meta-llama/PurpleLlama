# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from pathlib import Path

from ..audit_trail import validate_audit_event


class AuditTrailTest(unittest.TestCase):
  def test_example_audit_event_validates(self) -> None:
    event = self._load_example()

    validate_audit_event(event)

  def test_rejects_audit_event_that_stores_raw_prompts(self) -> None:
    event = self._load_example()
    event["redaction"]["raw_prompts_stored"] = True

    with self.assertRaisesRegex(ValueError, "raw_prompts_stored"):
      validate_audit_event(event)

  def test_rejects_unknown_policy_status(self) -> None:
    event = self._load_example()
    event["policy_checks"][0]["status"] = "unknown"

    with self.assertRaisesRegex(ValueError, "unsupported policy check status"):
      validate_audit_event(event)

  def _load_example(self) -> dict:
    example_path = (
      Path(__file__).resolve().parents[2]
      / "examples"
      / "governance_audit_event.json"
    )
    return json.loads(example_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
  unittest.main()
