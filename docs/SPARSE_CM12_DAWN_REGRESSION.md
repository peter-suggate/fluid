# Sparse CM12 Dawn regression gate

Run `npm run test:dawn:sparse-cm12` after significant simulation, topology,
presentation, boundary, or live-edit changes. Use `-- --list` for the matrix,
`-- --lane=<id>` to isolate a failure, and `-- --out=<path>` to retain receipts.
Unload browser GPU scenes first. The runner uses isolated processes and the
repository-wide WebGPU lease; never run another Dawn process concurrently.

## What belongs in the gate

Assert observable contracts: finite/nonnegative transported amounts, conserved
liquid, symmetry, authored resolution constraints actually taking effect,
successful live insertion, current renderable publication, and front travel.
Failure-latch and generation-lifetime lanes deliberately test corruption and
resource safety, which remain meaningful contracts.

Do not pin incidental page counts, exact intermediate rung ladders, allocation
slot identities, or duplicate UI/QA bookkeeping. Production defaults can change
brick sizes and storage layouts without changing physical behavior. Convert
coordinates using the published lattice instead of hard-coded B8 geometry.
Wait for the public live-edit publication boundary before checking its result.

The transported density field stores amount per full cell volume. It is not a
clamped occupancy mask: the conservative transport contract permits local
excess while preserving extensive volume. Check finite/nonnegative state and
conservation, rather than requiring every density or intermediate capacity
residual to lie within eight float32 epsilons of one.

## Baselines

Performance references are in
`benchmarks/results/sparse-cm12-dawn-regression-baselines.json`; numerical
behavior references are in `sparse-cm12-dawn-behavior-baseline.json` beside it.
Keep shader execution ceilings distinct from process timeouts, which also
include compilation and readback. Any revised reference needs the measured
receipt and an explanation. Assertion removals must explain which observable
contract remains; a solver fault is not an obsolete test assumption.
