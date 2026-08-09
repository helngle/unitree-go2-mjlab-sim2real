# Repository research policy

## Mandatory Reference Gate

Before implementing, evaluating, or training any innovative observation or
preprocessing rule, reward, architecture, memory/estimator, loss, curriculum,
Teacher/Student interface, training mechanism, or acceptance mechanism:

1. Provide at least one directly relevant primary paper or public auditable
   GitHub implementation.
2. Map the referenced component's exact inputs, outputs, timing, and algorithm
   to the proposed project change. State every material deviation. For GitHub,
   record the repository URL, commit/tag, and license.
3. Do not present adjacent or merely related work as a direct precedent.
4. Obtain explicit user approval of the reference and deviations before code
   changes, smoke tests, GPU diagnostics, formal evaluations, or training.
5. Project logs and agent reasoning may motivate a literature search, but are
   not sufficient authorization for an innovative experiment.
6. If no direct reference exists, report
   `NO_DIRECT_REFERENCE_DO_NOT_IMPLEMENT` and stop at read-only analysis.

This gate does not prevent routine bug fixes or faithful implementation of an
already approved, referenced method. If classification is uncertain, treat the
change as innovative and apply the gate.
