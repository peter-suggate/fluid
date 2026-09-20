# GPU support requests and the finite-phi blocker

## Implemented support mechanism

`UniformPageSupport` now feeds `UniformPageGeneration` in the same command buffer.
It scans only accepted resident pages, retains every nonzero conservative volume
and every phi below the positive interface band (including all deep negative phi),
measures velocity displacement on the GPU, and emits signed neighboring page
requests. External source records seed remote regions without a world-sized scan.

The caller supplies spacing, timestep, conservative extra displacement allowances,
and the compound operator's read reach. The default reach covers elementary phi
advection/redistancing only; contact continuation, agreement, extension, pressure,
correction, sources and presentation dependencies still need their complete support
contract before production retirement is legal. Source commands must already cover
their complete interface-band footprint. Start velocity alone does not certify RK2
midpoint velocity or later force/projection updates; allowances and subsequent
validation remain required.

The output is a complete desired set. Duplicate requests are merged by the generation
planner. The request buffer has a separate fixed capacity; duplicates can exhaust
it even when the unique set would fit. That is a reported failure, never truncation.
The producer records request overflow, coordinate overflow and nonfinite-field faults.
All refuse candidate publication. The serial request emitter and topology planner
still need large-pool timing measurements before deployment.

Dawn/Metal tests pass for:

- GPU-only classify → request → allocate → initialize → publish in one submission.
- A mass-bearing page with ambient phi, and deep negative phi with zero V.
- A 27-page signed frontier crossing the x=0 seam.
- Current GPU velocity increasing support to 45 pages; insufficient pool capacity
  leaves accepted membership unchanged.
- Remote-source pool exhaustion preserving the entire accepted field pool.
- Excessive reach and NaN refusing publication.
- Emptying, retiring, then initializing a remote source without host allocation.

This component is not wired into production fluid operators. The UI remains on the
all-resident cutover. No new scene speedup or memory reduction is claimed.

## Rejected direct phi cap

A controlled experiment capped initial phi, advection, redistancing and surface-volume
correction to ±16 hMax while leaving the dense backing and all operators otherwise
unchanged. The unbounded solver was the control. The cap is wider than the geometric
read bound for an **exact SDF**, but the evolved field does not satisfy that assumption.

Garden hose fields matched through frame 9. At frame 10, near-interface advected phi
still matched exactly, but final phi differed by 1.238 cells. At frame 12 the volume
field L1 difference was 8.196% of reference mass, and the maximum near-interface phi
difference was 3.503 cells. These are field differences, not measurements of lost
mass. This failed the predeclared 0.5% volume-field / 0.1-cell interface comparison.
The diagnostic run printed all samples with threshold assertions disabled; its TAP
success is **not** an acceptance pass.

The cap was removed from runtime code and was not promoted. Evidence is retained:

- `uniform-finite-phi-diagnostic.txt`: per-frame comparison capture.
- `uniform-finite-phi-rejected.patch`: experimental runtime changes.
- `uniform-finite-phi-probe.ts.txt`: probe source; copy into `tests/` after applying
  the patch to reproduce. `PHI_BAND_DIAGNOSTIC=1` captures the first 12 frames without
  threshold assertions; omit it for the 64-frame acceptance test, which fails.

The first observed discrepancy is between the near-interface advection and final-phi
checkpoints, consistent with the redistance/correction stage depending on changed
far-air values. It does not isolate a specific Newton query or distinguish every
redistance effect from correction. Further stage capture is required for that claim.

The next numerical step is to reconstruct a valid finite distance band from the
interface, including its solid-contact treatment, and explicitly validate support.
Increasing a guessed cap is not a substitute for that contract. Allocation and
support generation are now available independently for that implementation.
