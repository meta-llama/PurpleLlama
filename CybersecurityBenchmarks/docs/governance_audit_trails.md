# Governance audit trails for CyberSecEval runs

CyberSecEval runs can produce security findings that enterprise teams need to
review, track, and remediate over time. This document defines a small,
vendor-neutral audit event that can sit next to benchmark output files without
claiming legal or regulatory compliance certification.

The event records:

- which benchmark ran
- which model and dataset were evaluated
- where the stat and judge response artifacts were written
- which policy checks ran before accepting the result
- whether human review or remediation is required
- which fields were redacted before persistent audit storage

See [`../examples/governance_audit_event.json`](../examples/governance_audit_event.json)
for a complete example.

## Redaction boundary

Audit records should not store raw prompts, raw model responses, secrets, API
keys, customer data, or exploit payload text by default. Prefer hashes,
fingerprints, output paths, aggregate counts, and remediation metadata.

## Suggested decisions

- `pass`: benchmark evidence is recorded and no local policy check requires
  review.
- `fail`: a hard policy check failed.
- `needs_review`: the benchmark produced findings or ambiguous evidence that
  should be routed to a human security reviewer.

This audit event is an evidence and workflow aid. It does not certify that a
model, benchmark run, deployment, or organization is compliant with any law,
regulation, or internal policy.
