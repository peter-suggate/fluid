# Adaptive volume implementation progress

Implementation follows [the production plan](adaptive-geometric-volume-plan.md).

## Copy and default cutover

The complete adaptive-mass method tree and its sparse-world adapter/device library
were copied literally. The new adaptive-volume method is installed as the default
in the UI, built-in adaptive scene profiles, harnesses, benchmarks and tests.
Original adaptive-mass remains available by explicit selection.

The [copy provenance](../artifacts/adaptive-volume-copy/README.md) records 136
byte-equal files before wiring edits and a runtime dependency audit finding no
executable route from the new method into the original method/adapter pair.
Internal names and filenames were intentionally retained for this first step.

The first complete canonical Dawn run passed 16/17 lanes. All behavior checks
and both frame-time checks passed. The page-budget lane's test reported success
at 29.58 seconds, but process completion exceeded its unchanged 30-second timeout.
See [initial gate receipt](../artifacts/adaptive-volume-copy/dawn-gate-initial.json).
The timeout is being investigated; this is not a passing full gate.

Copy field/topology parity and final integration verification are in progress.
No unit tests were added. Validation uses Dawn; TypeScript and import/install
checks are supplementary static checks. Existing static failures were observed,
including copied diagnostics with errors matching their original source.

## Numerical implementation

The production method currently still runs the copied CM12 algorithm. Independent
geometric interface and physical subface helpers are being prepared for immediate
integration into real pressure/presentation consumers. They do not yet constitute
geometric-volume transport, and the volume-authority cutover is not complete.

Remaining work includes coherent geometric transport/pressure coupling, bounded
large-step subcycling, geometric topology transfer, and source/solid volume
accounting. Their integration must preserve the functioning production method.
