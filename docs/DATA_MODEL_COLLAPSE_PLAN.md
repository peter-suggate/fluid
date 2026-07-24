# Data Model Collapse — one authority, no intermediate state

**Written 2026-07-25**, on the uncommitted engine-collapse tree (base e0441ee).
Measured: **98.03 ms/advance free-run** (160.90 probe-attributed across 7
engine segments). Companion: `docs/STAGE_COLLAPSE_PLAN.md` (largely
implemented). This doc records three cross-cutting censuses of the pipeline's
*data model* and the deletion plan they imply.

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
  phase id `pressure-system`, which only exists under `FLUID_ENGINE_SPLIT=fine`.
  Fix: add that env to `profile:power-dam-ui` or teach the gate the engine ids.

## 1. Census results (three independent audits, current tree)

**Topology representations: ~14 materialized encodings of the same octree,**
in 4 idioms (sorted-directory+binary-search, flat tables, CSR/link arenas,
bitmask+rank). Eight encode the same `(level,cell)→row` relation. ~45 WGSL
binary-search sites across 14 files; ~44 lookup-function definitions. The
target representation — per-level occupancy bitmasks + popcount rank —
**already exists, working and generation-transactional, in
`lib/webgpu-octree-spgrid-vcycle.ts`** (:997-1004 rank lookup, :1441-1457
dirty-gated rebuild) and in CPU form in `lib/sparse-brick-octree.ts:266-279`.

**GPU state: ~196 MB allocated; ~3.1 MB (1.6%) is authoritative physics**
(fine phi 1 MB, face normalVelocity DOFs ~0.75 MB irreducible fraction,
coarse phi 65 KB, leaf headers 196 KB, pressure 16 KB, solid SDF, small
persistent state: page remap/free list, residency hysteresis). The rest:
~67 MB derived projections, ~86 MB transactional twins, ~20 MB scratch.
Notables: SPGrid uses `levelStride = nextPowerOfTwo(rowCapacity·16)` at ALL
levels (spgrid-vcycle.ts:554) → 62 MB committed + 62 MB candidate twins,
>98% padding at coarse levels; the face band rebuilds a **second** 9.5 MB
power-face graph per frame from the committed one (face-closest-point.ts:
1032-1038); velocity exists in 5 representations; the Poisson operator is
assembled 2-3 times.

**Trust machinery: ~45-55 dispatches, 3-6.5 MB of copies, and ~30M
redundant per-thread validation loads per frame.** Worst: fused transport
re-runs a ~14-word publication check up to 8-10× per transported sample
(fused-transport.ts:91,103-104 ≈ 26M loads/frame); uncached `ownerAt` in
18-direction loops (face-closest-point.ts:2422,2958,2964). ~15 generation
counters in 4 families; carry+commit double arenas exist **because row
indices are not stable across frames**, which is also why the sorts and
directory rebuilds exist. History check: rollback fires only under injected
faults; the worst real bug (owner-page tombstone leak) *passed* the
per-probe checks; the per-probe machinery's false-firing (Dawn predicate
miscompile, 903ce08) caused the only production rejections; the escalating
rebuild after rejection was the documented 30-100× stall.

## 2. The three cross-cutting deletions

### D1 — One topology authority: occupancy masks + rank
Authoritative set becomes four things: (1) per-level Morton-ordered 64-bit
occupancy words + per-word rank bases, A/B pair (~4 KB coarse + 512 B fine
at 16³); (2) the immutable catalog (+V2 stencil LUT); (3) rank-indexed value
arenas, faces at fixed row-rank×12 slots; (4) small persistent state
(page remap/free list, hysteresis, epoch words).
Everything else becomes O(1) functions: cell→row = rank; owner-of-point =
≤6 bit probes (deletes owner pages as authority); 18-bit descriptor = 18 bit
probes; deltas = mask XOR; iteration order = bit scan (reproduces the exact
(level,morton) sort order → determinism preserved); worklists = rank
compaction; face identity carry = old-rank→new-rank remap from the A/B masks.
Deletes: frontier sort/merge pipeline, owner-page 3-pass publisher,
band sorted pairs + identity-mark dedup apparatus, power-faces sorted
directory + delta identity merge, 3 extra coarse sample directories,
summary bitonic sort/merge (→ dense value-mip over ranks), fine sorted
worklists; ~45 binary-search sites → ~2 (catalog boundary resolve stays).

### D2 — State diet: derived-on-the-fly, delta commits
- SPGrid per-level strides (R/8^ℓ) + dirty-page commit instead of resident
  candidate twins: **−117 MB** and the full-copy commit pass.
- Delete the transient power graph; sample the committed face graph.
- Face normals/centroids/quadrature computed from geometryCode + catalog in
  consumers (−13.8 MB, −3 commit copies).
- Velocity: face DOFs only; reconstruct in consumers (kills capture/pack/
  commit passes ×5/frame).
- One operator assembly (leafEntries) consumed by all solvers.
- Single-consumer intermediates inlined (seed scratch, transport outcomes
  chain, fine-to-coarse contributions).

### D3 — Minimal trust model
- Intra-frame kernels trust inputs unconditionally (WebGPU submission
  ordering guarantees no mid-frame mutation is observable). One epoch word
  per domain, checked once per dispatch at most. `ownerAtCached` is the
  in-repo proof; webgpu-octree.ts already runs 23 ownerAt sites uncheck.
- One validation reduce per domain per epoch before flip: capacity counts,
  `free==capacity−resident`, scalar reason-coded invariants, generation
  pairing. Rollback = **refuse to flip** (host already advances parity only
  post-submit); delete rollbackPhi/transportedPhiSnapshot/settle branch.
  Rejection must be non-escalating (no rebuild storm).
- ~15 generation counters → ~5 epoch words + CPU pairing.
- `FLUID_DEEP_VALIDATE=1` lane re-enables per-probe checks and readbacks
  (both historical hard bugs were diagnosed through it — keep it).

## 3. Immediate fixes (this week, independent of D1-D3)

1. **Pass-broker fence regression (real, new, every frame):** the `run()`
   rewrite routes dispatches through `computeForIndirectBuffer`
   (webgpu-pass-broker.ts:25-45) which ends the compute pass on every
   storage↔indirect role transition of the shared indirect buffer — ~24
   passes per face-band encode where the design had ~4 (design comment at
   face-closest-point.ts:864-866). Per-dispatch usage scopes make the mixed
   pass legal; only same-dispatch bind+consume is invalid (already thrown).
   Narrow the guard to that case.
2. Gate label fix (§0). 3. Verify band-narrowing side effect: redistance
   band 23→16 / transport 16→11 may under-seed face-band phi (check
   bandPhiExtensions/phi-failure counters); if so restore the sampling
   collar for the consumed generation only.

## 4. Order-of-magnitude arithmetic

246 → 98 ms landed via engines/warm-start/parallelization. The censuses
say what remains is dominated by: directory/sort/carry/commit maintenance
(~150-200 dispatches/generation → ~15-25 under D1), ~30M validation loads
(D3 → ~thousands), and duplicated graph/state rebuilds (D2). The floor
after D1-D3 is the physics: a few value sweeps + adaptive solve — the
single-digit-ms regime the handoff targets. Sequence: §3 fixes → D3 items
1-2 (probe hoist, stable slots → carry deletion) → D1 (mask authority,
per-cluster: frontier first, face band second, fine third) → D2 alongside.
