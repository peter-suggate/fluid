# Sparse CM12 PCM full-refresh oracle

## Purpose

`full-refresh-oracle` is a QA-only construction mode for separating pressure
membership/closure errors from arithmetic-order errors. It is not a runtime
fallback and must never be selected by a dirty ratio, GPU readback, fault, or
scene property.

The oracle globally re-evaluates current pressure cells and rows each frame,
then invokes every pressure coefficient, RHS, solve, and projection kernel
through the same PCM1 stable-ID rank-select order as the local scheduler. A
local run and an oracle run therefore use identical pressure equations and
floating-point invocation order; only the membership/theta producer coverage
differs.

## Construction API

Add one immutable QA option to resident construction:

```ts
type SparseCM12PressureMembershipQA = "local" | "full-refresh-oracle";
```

Production constructors expose only `"local"`. The Dawn regression harness
may request `"full-refresh-oracle"` explicitly. The chosen value fixes the
encoded pipeline sequence for the resident lifetime; it is not stored in a
host variable that changes from frame to frame.

## Oracle frame sequence

The oracle retains the ordinary PCM begin/finalize/repair sequence and the
topology retirement producers. Between begin and finalize it encodes:

1. Topology cell retirement writes `false` for every old rung stable ID.
2. `classifyPressureCells` visits the complete current accepted-cell traversal
   and publishes current `pressureDensity >= isovalue` candidates.
3. PCM repairs dirty leaves/ancestors and publishes canonical cell rank.
4. Topology row retirement visits incidence closure for old rungs and publishes
   current false membership/theta zero.
5. `classifyRows` visits the complete current accepted-row traversal, computes
   theta with the normal equations, and publishes current row candidates.
6. PCM repairs dirty leaves/ancestors and publishes canonical row rank.
7. Coefficient, RHS, MGPCG, projection, and diagnostics execute unchanged via
   `pressureCellInvocation` / `pressureRowInvocation`.

The local temporal cell/row producers are not encoded in oracle mode. This
keeps the oracle independent of the closure under test. Topology retirement is
still required because a full current accepted traversal cannot publish false
for stable IDs removed from that traversal.

## Fail-closed contract

- PCM header, generation, capacity, candidate conflict, closure, and count-tree
  faults retain their existing behavior: indirect `x=0` and no solve work.
- Oracle mode never falls back to legacy append/swap lists or a second solver.
- The mode is identified in the QA receipt, so an oracle artifact cannot be
  mistaken for a production-performance receipt.
- No oracle timing is admissible for the non-pressure performance gate.

## Comparison receipt

The regression runner constructs independent local and oracle solvers from the
same scene/options and compares after every step:

- PCM cell/row active-bit SHA-256, popcount, generation, phase, and faults;
- accepted topology generation, rung histogram, prepared/committed counts;
- density, velocity, pressure, and divergence byte hashes;
- pressure cell/row counts, theta hash, coefficient hash, RHS hash, and the
  existing residual/iteration receipt.

The first mismatch is reported as `{step, domain, field, local, oracle}`.
Equal PCM bitsets with a theta/coefficient mismatch identifies stale row
authority; unequal bitsets identifies missing membership closure. Equal
membership/theta/coefficient hashes with a later physical mismatch identifies
an arithmetic/order violation outside classification.

## Integration touchpoints

- `webgpu-sparse-cm12-resident.ts`: immutable create option and the two fixed
  pressure-topology schedules; no per-frame branch on mutable state.
- `webgpu-sparse-cm12-resident.wgsl.ts`: reuse existing global classifiers and
  PCM producers; add only QA hash publication if coefficient/theta hashes are
  not already available.
- `webgpu-adaptive-mass-solver.ts`: pass the QA construction option through.
- `run-sparse-cm12-temporal-regressions.ts`: paired local/oracle lane and
  first-mismatch receipt. The normal dam and performance lanes stay local.

