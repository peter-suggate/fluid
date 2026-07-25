# Data Model Collapse — one authority, no intermediate state

**Written 2026-07-25**, from the engine-collapse measurements based on
e0441ee. Measured: **98.03 ms/advance free-run** (160.90 probe-attributed
across 7 engine segments). Companion: `docs/STAGE_COLLAPSE_PLAN.md`.
This document records three cross-cutting censuses of the pipeline's data
model and the deletion plan they imply.

## Implementation status

The first implementation attempt was reverted after the two-second minimal
dam smoke exposed an incomplete power publication and invalid velocity. The
safe, independently verifiable parts have since been reapplied with stricter
surface gates:

- **D1, consumer slice:** fused fine transport now resolves direct face-band
  identities through the authoritative O(1) identity table. It no longer
  performs a Morton-key binary search, and it validates the owner and power
  publications against their own epochs rather than comparing unrelated
  generation counters.
- **D2, SPGrid slice:** level state, worklists, ranked directories,
  neighbours, and transfer records use compact per-level prefix arenas while
  preserving the original at-most-50% hash load. The representative 16³,
  1024-row plan uses 18,762 state slots instead of repeating the 16,384-slot
  finest bound at all six levels (1,876,200 state bytes, about 42× smaller
  than the former 78,643,200-byte state arena).
- **D2, geometry slice:** power-face publication materializes each physical
  polygon once and reuses it for area, centroid, and quadrature derivation.
- **Work reduction:** staged Galerkin CG uses a workgroup-uniform active flag
  to stop its fixed encoded tail after convergence; no extra stage, buffer,
  or dispatch was introduced.
- **Regression authority:** the full two-second minimal dam gate now measures
  front-surface terracing and enclosed front/back surface holes in both
  raster directions, in addition to its energy, component, publication, and
  finite-velocity checks.

Still outstanding are the engine-wide D1 mask authority, deletion of the
transient power graph and other D2 duplicate representations, and the full
D3 epoch/trust collapse. Those must land as separately gated migrations.
The reverted pass-broker role fencing, cross-domain epoch comparison,
default CPT extension, RAP reuse shortcut, and uncalibrated numerical band
changes are explicitly not part of this implementation.

## 0. Measurement notes (read before comparing profiles)

- The segmented queue-wall probe adds a **per-advance-constant ~63 ms**
  (189.28−125.95 pre-collapse; 160.91−98.03 post-collapse). It does not
  shrink with fewer boundaries — it concentrates into the biggest segments.
  Compare **free-run** numbers across builds; use `FLUID_ENGINE_SPLIT=fine`
  for like-for-like per-checkpoint attribution.
- The row-b engine's 73.55 ms is mostly this arithmetic (deflated ≈ 45 ms vs
  a deflated old-parts baseline ≈ 19 ms + the previously separate projection
  tail). The real regression inside it is §4a below.
- Acceptance gate: `tools/power-dam-performance-report.ts:253-256` requires
  phase id `pressure-system`, which only exists under
  `FLUID_ENGINE_SPLIT=fine`. Fix: add that env to `profile:power-dam-ui` or
  teach the gate the engine ids.

## 1. Census results (three independent audits)

**Topology representations: ~14 materialized encodings of the same octree,**
in 4 idioms (sorted-directory+binary-search, flat tables, CSR/link arenas,
bitmask+rank). Eight encode the same `(level,cell)→row` relation. ~45 WGSL
binary-search sites across 14 files; ~44 lookup-function definitions. The
target representation — per-level occupancy bitmasks + popcount rank —
already exists, working and generation-transactional, in
`lib/webgpu-octree-spgrid-vcycle.ts` and in CPU form in
`lib/sparse-brick-octree.ts`.

**GPU state: ~196 MB allocated; ~3.1 MB (1.6%) is authoritative physics**
(fine phi 1 MB, face normalVelocity DOFs ~0.75 MB irreducible fraction,
coarse phi 65 KB, leaf headers 196 KB, pressure 16 KB, solid SDF, small
persistent state: page remap/free list, residency hysteresis). The rest:
~67 MB derived projections, ~86 MB transactional twins, ~20 MB scratch.
Notables before the compact-arena slice: SPGrid used
`levelStride = nextPowerOfTwo(rowCapacity·16)` at every level, creating
>98% padding at coarse levels; the face band rebuilds a second 9.5 MB
power-face graph per frame from the committed one; velocity exists in five
representations; and the Poisson operator is assembled 2–3 times.

**Trust machinery: ~45–55 dispatches, 3–6.5 MB of copies, and ~30M
redundant per-thread validation loads per frame.** Worst: fused transport
re-ran a ~14-word publication check up to 8–10× per transported sample;
uncached `ownerAt` appears in 18-direction loops. ~15 generation counters
exist in four families; carry+commit double arenas exist because row indices
are not stable across frames, which is also why sorts and directory rebuilds
exist. History check: rollback fires only under injected faults; the worst
real bug (owner-page tombstone leak) passed the per-probe checks; the
per-probe machinery's false-firing caused the only production rejections;
and the escalating rebuild after rejection was the documented 30–100× stall.

## 2. The three cross-cutting deletions

### D1 — One topology authority: occupancy masks + rank

Authoritative set becomes four things: (1) per-level Morton-ordered 64-bit
occupancy words + per-word rank bases, A/B pair (~4 KB coarse + 512 B fine
at 16³); (2) the immutable catalog (+V2 stencil LUT); (3) rank-indexed value
arenas, faces at fixed row-rank×12 slots; (4) small persistent state
(page remap/free list, hysteresis, epoch words).

Everything else becomes O(1) functions: cell→row = rank; owner-of-point =
≤6 bit probes (deletes owner pages as authority); 18-bit descriptor = 18 bit
probes; deltas = mask XOR; iteration order = bit scan (reproduces exact
`(level,morton)` order, preserving determinism); worklists = rank compaction;
face identity carry = old-rank→new-rank remap from the A/B masks.

Deletes: frontier sort/merge pipeline, owner-page three-pass publisher, band
sorted pairs + identity-mark dedup apparatus, power-faces sorted directory +
delta identity merge, three extra coarse sample directories, summary bitonic
sort/merge (becomes a dense value mip over ranks), and fine sorted worklists.
The ~45 binary-search sites become ~2; catalog boundary resolution remains.

### D2 — State diet: derived-on-the-fly, delta commits

- SPGrid per-level strides (`R/8^ℓ`) + dirty-page commit instead of resident
  candidate twins: target **−117 MB** and deletion of the full-copy commit.
  Compact resident and candidate arena addressing is implemented; dirty-page
  candidate commit remains outstanding.
- Delete the transient power graph; sample the committed face graph.
- Compute face normals, centroids, and quadrature from geometry code +
  catalog in consumers (target −13.8 MB and three commit copies).
- Keep velocity face DOFs only and reconstruct in consumers (deletes
  capture/pack/commit passes repeated five times per frame).
- Assemble one operator (`leafEntries`) and consume it in all solvers.
- Inline single-consumer intermediates (seed scratch, transport outcomes
  chain, fine-to-coarse contributions).

### D3 — Minimal trust model

- Intra-frame kernels trust inputs unconditionally: WebGPU submission
  ordering prevents observable mid-frame mutation. Use one epoch word per
  domain, checked once per dispatch at most. `ownerAtCached` is the in-repo
  proof; `webgpu-octree.ts` already runs 23 owner sites unchecked.
- Run one validation reduction per domain per epoch before flip: capacity
  counts, `free==capacity−resident`, scalar reason-coded invariants, and
  generation pairing. Rollback means **refuse to flip** (the host already
  advances parity only after submit); delete rollback phi, transported-phi
  snapshot, and settle branch. Rejection must not escalate into a rebuild
  storm.
- Collapse ~15 generation counters into ~5 epoch words plus CPU pairing.
- Retain a `FLUID_DEEP_VALIDATE=1` lane for per-probe checks and readbacks;
  both historical hard bugs were diagnosed through deep validation.

## 3. Immediate fixes and constraints

1. **Pass-broker role fencing:** do not restore the reverted
   `computeForIndirectBuffer` behavior. It ended compute passes on each
   storage↔indirect role transition and recreated ~24 passes where the face
   band was designed around ~4. The current broker keeps dispatch-local
   usage legal and copies finalized storage-authored records to a distinct
   INDIRECT-only buffer at the semantic boundary.
2. Fix the performance gate label described in §0.
3. Do not narrow redistance/transport bands without a separately calibrated
   surface gate. Check band-phi extensions and phi-failure counters before
   changing the consumed generation's sampling collar.

## 4. Order-of-magnitude arithmetic

246 → 98 ms landed via engines, warm start, and parallelization. The censuses
say what remains is dominated by directory/sort/carry/commit maintenance
(~150–200 dispatches/generation, targeting ~15–25 under D1), ~30M validation
loads (D3 targets thousands), and duplicated graph/state rebuilds (D2). The
floor after D1–D3 is the physics: a few value sweeps plus the adaptive solve.

Sequence: §3 fixes → D3 probe hoisting and stable slots → D1 mask authority
per cluster (frontier, face band, fine) → D2 alongside each authority
migration. Every slice must pass the two-step diagnostic first and then the
full 500-step/two-second minimal dam surface gate before the next deletion.
