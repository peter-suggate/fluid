# Fine-band 10x — diagnosis, experiment ladder, and gates

Date: 2026-07-30. Branch: `perf/structured-cutover` (uncommitted working tree).
Scope: the non-pressure ("fine band" and its feeding maintenance) slices of the
octree power-liquids advance. This document records what was measured in this
session, why the previous 10x attempts had to fail, the architecture that can
succeed, and a risk-ordered experiment ladder — **specification only, nothing
here is implemented**. Companion documents: `POWER_LIQUIDS_ULTIMATE_M1MAX.md`
(protocol + refutation log — normative), `POWER_LIQUIDS_TEMPORAL_COHERENCE_HANDOFF.md`
(existing coherence inventory), `POWER_LIQUIDS_5X_GPU_PLAN.md` (phase-packing
design, §6), and `POWER_LIQUIDS_10X_DISCOVERY_EXPERIMENTS.md` — the
oracle/ablation/roofline probes (X-1…X-9) that *locate* the 10x and choose
between the candidate architectures before this ladder's Wave 2 is built.
Run the discovery experiments first where the two documents disagree on
priority.

## 1. New evidence captured this session

### 1.1 Artifacts produced

- `artifacts/scene-size-overhead/large-current/` — **first per-stage xctrace
  counter capture of the `large` lane** (240 steps, label isolation +
  hardware counters). `report.html`, `summary.json`, `mini-dam.trace`.
  Attribution complete: 48.75 ms/advance attributed across 287 exact stages,
  0 composite. Untraced control printed by the same run: **54.05 ms/advance**;
  traced 106.22 (1.97x isolation distortion — per-stage numbers are a ranking,
  not a wall; the clean control is the only comparable number).
- Coarse row-delta census on the large lane (240 steps,
  `FLUID_OCTREE_ROW_DELTA_CENSUS=1`): wall 60.94 ms/advance with census
  overhead, terminal 14,470 active fine bricks.
- Note a wall discrepancy to re-establish before any A/B:
  `baseline-large.json` (Jul 29) 100.29; `large-band1-fixed.json` (Jul 30
  12:44) 35.76 at 5,997 bricks; this session's untraced control 54.05 and
  census run 60.94 at 14,470 bricks, all `--steps=240`. Brick count differs
  run-to-run because the dam evolves and cost drifts upward with step index
  (documented +46–63% over a 500-step run in the gpu-frame-profile skill).
  **Every experiment below must carry its own interleaved A/B/A/B control at
  fixed `--steps`; never compare against any wall in this or any other doc.**

### 1.2 Large-lane frame anatomy (48.75 ms attributed, label-isolated)

By family (this session's aggregation of `large-current/summary.json`):

| family | ms/adv | calls | biggest members |
|---|---:|---:|---|
| pressure + SPGrid solve | 20.28 | 1216 | §6.3 adjoint children 1.77 (165 calls × 11 µs), V-cycle candidate build level sets 1.54 @ 1% occ |
| octree topology | 7.40 | 14 | **resident grading closure 3.00 @ 9% occ**, **transfer accepted velocity 2.70 @ 0% occ** |
| fine transport | 5.13 | 20 | **advect fine phi rare 2.95 @ 7% occ**, common 1.22 @ 11%, finalize page delta 0.64 |
| fine topology/summary/volume | 4.12 | 54 | three full-band volume reductions ~1.5 total, reset/refinement 0.42 |
| fine JFA | 4.01 | 33 | floods 3.27 @ ~50% occ / 68% ALU; other 30 passes ≤ 0.25 each |
| air support | 3.47 | 8 | Section 5 march 1.66 @ 17%, publish identities 1.57 @ 36% |
| structured velocity | 2.74 | 50 | advect class 8 0.58, class 7 0.48, extrapolate 0.41 |
| misc/power/volume | 1.60 | 25 | classify changed-key dirty/support 0.66 |

Frame-wide: mean compute occupancy **9.0%**, ALU **12.7%** (contradiction with
"GPU busy 51%" is the whole story: busy ≠ utilized).

### 1.3 The census number that sizes everything

`FLUID_OCTREE_ROW_DELTA_CENSUS=1`, large lane, 240 steps, means over the run:

- rows current ≈ 1,484; **added ≈ 23.0, retired ≈ 15.6 per advance — membership
  changes ~2.6%/advance** (max added 1,147 on impact frames);
- but **dirty ≈ 1,417 and affected ≈ 1,417 — ~95% of all rows are classified
  dirty every advance**, and `meanAffected == meanDirty` exactly;
- 9 of 240 generations had zero affected rows.

The physical temporal coherence the 0.004 s frame promises is real (2.6%
membership churn). The pipeline's *dirty classification* discards it — the
dirty set is 60x larger than the membership delta. Whether that over-inclusion
is legitimate (coefficients/fractions genuinely change in every wet row) or
conservative slop is **the highest-information open question in this document**
(experiment M-2).

Fine-lane equivalent numbers could not be captured: `FLUID_WORKSET_CENSUS=1`
on the large bench emitted no `fine-transport-workset-census` lines (the
structured-dynamics census and row-delta census did emit). The plumbing in
`lib/webgpu-octree-fine-levelset-transport.ts:457-486` (`censusTick`) reads
`process.env` directly and should have fired; suspect the mapAsync lag-1
protocol never resolving under the bench's fence cadence. Fix and rerun as
M-1 before sizing E-7/E-9.

## 2. Why the previous 10x attempts failed — and what that proves

Three measured facts from the artifacts and the refutation log, which any
future attempt must respect:

1. **The frame is not dispatch/pass-overhead-bound.** One command buffer and
   one submit per advance, no mid-frame readbacks
   (`lib/physics-step-program.ts:123-132`); pass boundary ≈ 5 µs, all 170 ≈
   0.9 ms (`tools/webgpu-pass-encoder-isolation.ts:37-40`); deleting 231
   zero-work indirect dispatches measured as a 2 ms *regression*
   (refutation #9); ~1,150 of 2,144 dispatches already launch zero workgroups
   (converged solve tail) and are below noise.
2. **Naive megakernelization inverts the problem.** The persistent
   one-workgroup MGPCG wins small scenes (mini 107→58 ms/10) and **loses the
   large lane 87→122 ms** (`recheck-persistent-*.json` vs
   `recheck-hierarchical-*.json`): one workgroup is one GPU core. Fusing pass
   chains without restoring width trades launch overhead (which is ~nothing)
   for serialization (which is everything).
3. **Band shrinking, which changed no command structure at all, took the large
   lane 87 → 35.8 ms** (`large-band1-fixed.json`). Work removal is the only
   lever that has ever moved this wall.

Corollary: the 10x lives in exactly two places — (a) **kernel work removal**
(fewer samples touched, fewer words fetched per sample, fewer substeps/rounds
per frame), and (b) **launch shapes proportional to the delta, not the
capacity** (which is work removal seen from the launch side). Pass merging,
dispatch-count reduction, and encoder games are dead ends, measured twice.

## 3. Structural diagnosis of the non-pressure frame

The ~28.5 ms of non-pressure attributed time decomposes into three defect
classes. (File:line references are the authoritative map; all verified this
session.)

### D1. Every-frame rebuild of nearly-static structure

The physical change per 0.004 s frame is ~2.6% membership churn, yet:

- **`Octree resident grading closure` (3.0 ms, 9% occ)** re-runs the balance
  rounds over the **entire resident tile stream** every advance —
  `lib/webgpu-octree.ts:3070-3097` restores "the immutable active-residency
  stream for balance only" and dispatches `balanceRounds ×
  (coarseRefinementSizes + candidates)` over all resident tiles, even when the
  refinement delta is a handful of tiles. Grading closure is the transitive
  closure of a delta; it can be marched from the delta cone (E-3).
- **Fine transport worksets are rebuilt from scratch every advance** —
  classify → reduce → scan → publish → compact
  (`lib/webgpu-octree-fine-levelset-transport.ts:397-424`), acknowledged as a
  gap in `POWER_LIQUIDS_TEMPORAL_COHERENCE_HANDOFF.md:246-249`.
- **Air-support publication encoded twice per advance, identity/clear stages
  dispatch `ceil(domainVolume/256)`** — full domain, not band
  (`lib/webgpu-octree-air-velocity-support-gpu.ts:524`, encode sites
  `lib/webgpu-octree.ts:3250` and `:3479`). Sanctioned fix shape already
  specified: B2 in `POWER_LIQUIDS_ULTIMATE_M1MAX.md:549-634` (BLOCKED in its
  naive form; the two-part sanctioned reorder is spelled out there).
- The SPGrid candidate build runs twice per advance (handoff doc item 2, the
  `!preparedCaptureSource` skip that never fires,
  `lib/webgpu-octree-spgrid-vcycle.ts:1606` vs `lib/webgpu-octree.ts:3318`) —
  pressure-side, listed for completeness since ~2.5 ms of it is
  reclassifiable as pure maintenance.

### D2. Capacity-shaped and domain-shaped launches over sparse sets

On the large lane, capacity (81,920 logical bricks / 32,768 page capacity)
exceeds the active set (5–18k bricks) by 2–10x, and:

- dense-shaped fine-lane passes enumerated this session:
  `clearFinePageDelta`, `clearTopologyErrors`, `classifyDesiredPageIdentities`,
  `compactDesiredPageIdentities`, `assignDesiredPageIdentities`, transport's
  `clearStructuredFineDelta` + scan trio, summary reclamation — all
  `ceil(capacity/64 or /256)` (`lib/webgpu-octree-fine-levelset-topology.ts`,
  `-transport.ts`); the recurring topology mask scan/scatter walks the **full
  logical lattice**, 320 workgroups (`topology.ts:1131-1152`, `:1706-1720`).
- payload arenas are sized dense on large (5 channels × 2 slots × 81,920 × 64
  × 4 B ≈ 210 MB at 23% residency), so every capacity-shaped clear also pays
  4x the bytes it needs (`lib/webgpu-octree-fine-levelset-bricks.ts:83-119`,
  capacity plan `lib/webgpu-octree.ts:710-737`).
- two O(pages × producers) serial loops inside classification:
  `interfaceNeighborRadii` and `repairNeighbor`
  (`lib/webgpu-octree-fine-levelset-topology.ts:1978-1997`) — every desired
  page loops all recurring producer keys.
- `auditOctreeProductionSource`'s `"capacity-dispatch"` rule
  (`lib/webgpu-octree-work-accounting.ts:983,1068-1090`) already exists to
  flag these statically. Run it, enumerate, convert (E-6).

### D3. Narrow/serial kernels doing wide work, and wide kernels doing needless work

- **`Transfer accepted velocity to changed topology faces` (2.70 ms @ 0%
  occ).** Now consumes the class-4 compact list, but each *thread* performs a
  4-step backtrack search plus a **5³ = 125-cell `acceptedRowContaining`
  scan**, each probe a dependent page-table walk
  (`lib/webgpu-octree-structured-dynamics.ts:1290-1341`). A few hundred
  changed faces → a few workgroups → the whole machine waits ~2.7 ms on
  pointer chases. This is a widening candidate, not a gating candidate (E-2).
- **`Advect fine phi rare/common` (4.17 ms combined, 7–11% occ).** One 64-lane
  workgroup per 4³ brick; per-sample cost is ~160 fetched words: per substep
  ~9 air-owner record resolutions (~36 per 4-substep trace,
  `lib/webgpu-octree-fine-levelset-transport.wgsl.ts:69-89`), then a
  `sampleFine` 8-tap gather where **every tap re-validates the page directory**
  (directory load + 3 metadata loads + header checks,
  `transport.wgsl.ts:340-342`). Occupancy is register-bound at 9 workgroups/
  core (144/768 slots; threadgroup memory would allow 21,
  `lib/webgpu-octree-fine-levelset-transport.ts:378-392`). Two independent
  levers: fetch-width (E-5, sanctioned E2 form) and substep count (E-4).
- **JFA floods (3.27 ms @ ~50% occ, 68% ALU, 96% ALU-limiter).** The one
  genuinely healthy family — and it is burning its ALU on u32 division: ~96
  divisions/lane/dispatch from `physicalSampleQ`/`unpackBrick`/`localCoord`.
  The exact shift-addressing variant is **already implemented, documented as
  an integer identity, and off by default**
  (`FLUID_FINE_JFA_B4_ADDRESSING`,
  `lib/webgpu-octree-fine-levelset-redistance.ts:689-736`). E-1 is a flag
  flip plus the gates.
- **Three full-band volume reductions per advance** (~1.5 ms: measure,
  re-reduce, overlap partials) for a correction whose input drifts at
  0.004 s timescales (E-8).

## 4. The architecture that gets to 10x — persistent mirrors, delta repair

The insight to build around: **the authoritative sparse structures (page
table, worksets, halo tables, summaries, air-support identities, graded
octree) are functions of a surface that moves ≤ 1–2 fine cells per frame.
Rebuild-from-scratch is O(active set) per frame; repair-from-delta is
O(changed set) per frame — and the measured changed set is ~2.6%.**

The repo already contains every ingredient, each proven in one place and
absent everywhere else:

| ingredient | proven at | absent from |
|---|---|---|
| GPU fingerprint + fail-closed carry | `probeCandidateSkip`/`applyCandidateSkip`, `lib/webgpu-octree-spgrid-vcycle.ts:3073-3094` | entire fine lane, air support, grading |
| compact changed-key stream as sole producer | transport phase-mask delta → recurring topology (`transport.wgsl.ts:523-524`) | worksets, summaries consume it; grading/air/boundary do not |
| displacement-bounded work ladder | JFA even-prefix skip off `pageDelta[9]` (`redistance.ts:837-841`) | transport substeps, volume cadence |
| epoch-gated zero-delta reuse | `lib/webgpu-octree-topology-epoch.ts:243-247` | fine lane (own generation, never skips) |
| conditional-stage predicate hook | `lib/physics-step-program.ts:133-137` | **zero predicates declared** |
| narrow-work co-scheduling design | `POWER_LIQUIDS_5X_GPU_PLAN.md:185-218` (phase-packing) | unbuilt; sequenced last on purpose |

Wave 1 (E-1…E-6) removes work inside the existing architecture. Wave 2
(E-7…E-11) extends the delta-repair discipline to the families that ignore
it. Phase-packing (E-12) is the terminal occupancy multiplier once kernels
are lean. The honest arithmetic: Wave 1 ≈ 28.5 → ~14 attributed; Wave 2 is
what buys the rest — its ceiling is set by the M-1/M-2 census numbers, which
is why the measurement experiments come first.

**On the "cache-efficient mirror" (gather the band into a dense apron'd
structure each frame, run hot loops against it):** the general form is
already refuted for transport — the backtrace reach is 8+1 cells, so a staged
φ tile is 22³ = 42,592 B against the 16,384 B workgroup budget, and staged
traffic (~666 words/sample) exceeds what the kernel fetches today (~160)
(`POWER_LIQUIDS_ULTIMATE_M1MAX.md`, refutation #2). The *sanctioned* residue
of the idea is staging the **addressing**, not the fields (E2 there = E-5
here), plus keeping the JFA's per-workgroup 27-page halo pattern
(`redistance.ts:868-885`) as the template for any kernel that still resolves
pages per-tap. Any future full-mirror proposal must first produce a
words-per-sample budget that beats 160, including its own build+refresh
traffic amortized over the passes that read it.

## 5. Experiment ladder

Ordering is by information-per-risk, not expected milliseconds. M-* are
measurement-only (no behavior change, no gates). E-* carry gates. Every E-*
follows the protocol of `POWER_LIQUIDS_ULTIMATE_M1MAX.md` §"How to run and
verify": clean walls only from `npm run benchmark:power-dam-mini|-large|
-moving-interface|-ui`, **interleaved A/B/A/B at fixed `--steps`**, A/A pair
alongside any Gate A fingerprint claim, tripwires on
(`FLUID_TRIPWIRES=1`; topology rollback / unaccepted restriction rows /
0 pressure iterations on churn / 0xFFFFFFFF fine-band count / solver
non-convergence all fail the experiment regardless of wall).

### M-1. Fine-lane delta census (fix plumbing, then measure)

- **Question:** per advance on `large` and `mini`: transport pages by class
  (common/rare), changed keys, dirty pages, support pages, added/retired
  pages, `pageDelta[9]` measured displacement distribution.
- **How:** `FLUID_WORKSET_CENSUS=1` emitted nothing for
  `fine-transport-workset-census` on the large bench this session while the
  other censuses emitted; debug `censusTick`'s lag-1 mapAsync under bench
  fence cadence (`lib/webgpu-octree-fine-levelset-transport.ts:457-486`)
  first. Add a one-line histogram of `pageDelta[9]` per generation while
  there (it is already copied host-side for the census path).
- **Decides:** E-4's substep histogram, E-7's skip fraction, E-9's carry
  fraction, E-10's cadence. If dirty/support ≈ active every frame (the
  coarse census's 95% pattern repeated in the fine lane), Wave 2 pivots from
  "skip unchanged pages" to "fix the dirty classification" — same
  experiments, different first target.

### M-2. Explain the 95% dirty over-inclusion

- **Question:** of the ~1,417 dirty rows/advance, how many have any input
  word actually changed (the 4-word fingerprint of `probeCandidateSkip`
  already defines "changed")? Is `meanAffected == meanDirty` an aliasing of
  two counters or a real equality?
- **How:** extend the row-delta census drain
  (`lib/webgpu-octree.ts:3186-3230`) to also count
  fingerprint-unchanged-but-dirty rows; zero code in the hot path.
- **Decides:** whether D1 fixes attack classification (cheap) or genuinely
  incompressible change (then Wave 2's ceiling drops and E-12 rises in
  priority).

### E-1. JFA B4 shift addressing on by default

- **Change:** default `FLUID_FINE_JFA_B4_ADDRESSING=1`
  (`lib/webgpu-octree-fine-levelset-redistance.ts:268,969`; plan-compatibility
  predicate `fineLevelSetJFAB4AddressingRequested` already refuses non-B4
  plans). Update the pinned default in
  `tests/webgpu-octree-fine-levelset-bricks.test.ts:1084-1094` (named here per
  the test-pinning rule).
- **Hypothesis:** floods are 96% ALU-limiter with ~96 u32 divisions/lane;
  variant is a documented exhaustive integer identity → 20–40% off 3.27 ms of
  floods on large, similar share on mini/ocean where JFA is 25% of frame.
- **Gate A** (bit-exact): `npm run test:webgpu:minimal-power-dam-break` flag-off
  vs flag-on, `phiBitXor` + final field stats identical, A/A pair first.
  Script already drafted this session:
  interleaved `benchmark:power-dam-large --steps=240` × {0,1} × 2 rounds, then
  three fingerprint smokes (A/A then A/B).
- **Risk:** none identified; the file's own comment block is the proof
  sketch. If Gate A fails, the identity claim is wrong — stop, report, do not
  Gate-B it through.

### E-2. Widen the topology-face velocity transfer

- **Change:** `transferStructuredTopologyCandidate`
  (`lib/webgpu-octree-structured-dynamics.ts:1314-1341`): one workgroup per
  candidate face; lanes cooperatively evaluate the 4-step backtrack and the
  5³ neighborhood (125 lanes of `acceptedRowContaining`), reduce by the
  existing deterministic key `(distance, candidateRow)` — a min-reduction is
  order-independent, so the selected carrier is bit-identical; the single
  field evaluation for the selected carrier stays scalar and unmodified.
- **Hypothesis:** 2.70 ms @ 0% occ → bounded by (faces × 1 wg) at real
  occupancy; expect an order of magnitude on this stage.
- **Gate A** claimed (integer selection unchanged, one float path executed
  once either way) — but per refutation #12 (storage round-trips/FMA), any
  restructure that moves the *float evaluation* into different control flow
  must verify fingerprints, and fall back to Gate B with the envelope
  (retention ≈0.99 @ t=0.22/0.30, KE ≈0.45 @ t=0.38, floor circularity
  1.16→1.04, red gates not redder) if Gate A shows drift.
- **Watch for:** refutation #10 (divergent early-exit vs uniform sweep in
  workgroup memory) does not apply — operands are global-memory page walks —
  but keep the 5³ sweep uniform per lane anyway; no early exits.

### E-3. Grading closure from the refinement delta cone

- **Change:** replace the all-resident balance rounds
  (`lib/webgpu-octree.ts:3070-3097`) with a frontier march seeded by the
  refinement delta (the exact wet-frontier tile delta already exists two
  passes later, `:3110-3114`): iterate balance only over tiles reachable from
  the delta through 2:1 violations, to a fixed point (the air-support march,
  `lib/webgpu-octree-air-velocity-support-gpu.ts:718-721`, is the sanctioned
  fixed-point template — 12 wide waves then persistent tail).
- **Hypothesis:** 3.0 ms → O(delta cone); census says the cone is a few
  percent of resident on non-impact frames.
- **Gate B** (eligibility logic changes; identical *final grading* must be
  proven, not assumed): add a debug arm that runs both old and new closure
  and diffs the graded stream for N=500 steps before trusting the fast path;
  then the standard envelope. Tripwire: any topology rollback.
- **Risk:** correctness of "closure = delta cone" depends on the invariant
  that last frame's grading was already closed — a rejected/rolled-back
  generation violates it; the march must fail closed to full closure on
  rollback (`pageDelta` rollback list / epoch reject signals exist).

### E-4. Per-brick substep bound for fine transport

- **Change:** `state[1] = max(fineFactor, ceil(vmax·dt/h))` is planned from
  the **global** vmax (`planStructuredFineTransportSubsteps`,
  `transport.wgsl.ts:150`); the departure loops run that count for every
  sample (`:401-404`). Classify already touches every page
  (`classifyStructuredFineTransportBlocks:191`); have it also fold the
  brick+halo max face speed into 2 bits of substep class (1/2/4/global), and
  let the departure loop read the per-class count. Same numeric path,
  fewer iterations where the field is slow.
- **Hypothesis:** dam-break late phase is mostly calm pool; M-1's speed
  histogram sizes this. Expect the largest single transport win (family is
  5.13 ms).
- **Gate B** strictly (substep count changes the characteristic path — this
  is a legitimate reordering of float work, exactly the class Gate A
  forbids). Envelope + red-gate discipline; also check `volumeDrift 0` holds
  over 500 steps since interface bricks must keep the global count —
  restrict reduced counts to `REGULAR_COMMON` (fully submerged) pages first,
  which never contain the interface by construction (`:192`).
- **Prior art guard:** refutation #1 (band radius coupling) does not apply —
  no radius changes — but the topology producer consumes transport's
  phase-mask delta, so verify the delta stream is substep-invariant (it
  compares committed φ signs, not path history).

### E-5. Stage transport addressing (sanctioned E2 form)

- **Change:** exactly `POWER_LIQUIDS_ULTIMATE_M1MAX.md` E2 (`:1024-1046`) —
  per-workgroup staging of the page-directory window and air-owner record
  addresses (not fields): the brick's reachable page set is generation-fixed,
  so resolve once per workgroup into workgroup memory the way the JFA flood
  resolves `floodPageIds[27]` (`redistance.ts:868-885`), then `sampleFine`
  and `airOwner` index through it with shifts (extend E-1's B4 addressing to
  the transport WGSL while in there — same pinned 4³/64 constants,
  same integer-identity argument).
- **Hypothesis:** cuts the ~160 words/sample fetch and the per-tap
  validation chain; raises the 9/21 workgroups-per-core register plateau
  only if the staged window also shrinks live registers — measure both.
- **Gate A** target (addressing identity), with the E-2 caveat: any drift ⇒
  stop and re-derive, not Gate B.

### E-6. Capacity-shaped fine-lane launches → compacted/delta-shaped

- **Change:** the D2 list, in three tiers: (a) clears sized to
  active-not-capacity (worklist header already carries the active count);
  (b) identity-assignment and scan trios driven off the changed-key stream
  instead of capacity sweeps; (c) the two O(pages × producers) loops
  (`interfaceNeighborRadii`/`repairNeighbor`) bounded by a per-page producer
  bitmask. Run `auditOctreeProductionSource` first to enumerate every
  `"capacity-dispatch"` hit and fix the list, not examples.
- **Hypothesis:** the fine-topology/summary/misc tail (~5.7 ms across ~80
  small calls) is mostly this; also removes most of the 210 MB arena traffic
  on large.
- **Gate A** per item (launch shape only, same lanes execute) — the E4
  precedent in the handoff doc already validated the pattern on the transfer
  list. Items that change *which* lanes run (tier b/c) are Gate B.

### E-7. Fine-page unchanged-carry (the fine lane joins the coherence regime)

- **Change:** per-page fingerprint (committed φ generation word + phase-mask
  delta bit + halo-changed bit) gating: transport departure, JFA
  refresh/seed, summary base recompute, and the volume overlap partials all
  skip pages whose fingerprint is unchanged, via the
  `probeCandidateSkip`/`applyCandidateSkip` template (GPU-resident, fail
  closed on the accepted commit token — never host-side). This is the
  fine-lane analogue of what the epoch already grants SPGrid.
- **Sized by:** M-1. If unchanged-page fraction is ~75%+ off-impact (the
  2.6% membership churn suggests it), this is the single biggest Wave 2 item
  across four families at once.
- **Gate B** with the full 500-step gate list plus the band-sweep quality
  gates (`artifacts/band-0-vs-4-full-mini-2026-07-29/comparison.md` table);
  the failure mode is exactly refutation #1's "surface freezes and the frame
  gets faster" — require the per-generation tripwires and the
  disconnection/ceiling oracles from the wall-sticking memory note.

### E-8. Volume-correction cadence

- **Change:** the measure/re-reduce/overlap trio (~1.5 ms) runs every
  advance; run it every K advances with the drift bound as tripwire
  (volumeDrift already printed; K=4 at dt=0.004 bounds unmeasured drift to
  16 ms of physics).
- **Gate B**; cheap to trial, easy to revert; red gates decide.

### E-9. Transport workset carry across advances

- **Change:** the classify→reduce→scan→publish→compact rebuild
  (`transport.ts:397-424`) becomes delta-repair: pages enter/leave classes
  only via the changed-key stream; unchanged pages keep last frame's class
  records (they are keyed by physical id + generation already).
- **Gate A** for the shape (same classes, same pages, provably identical
  worksets on a frozen scene), **Gate B** for the recurring path.
  Depends on M-1 confirming class churn is small.

### E-10. Air-support B2, sanctioned form only

- Already fully specified and blocked-form-refuted in
  `POWER_LIQUIDS_ULTIMATE_M1MAX.md:549-634`: (1) move the second encode
  after candidate validation, (2) prepare stage zeroes the indirect args on
  a clean GPU verdict. Plus delta-shape the `ceil(domainVolume/256)`
  identity/clear stages (same E-6 discipline). ~3.5 ms on large, ~365 ms on
  ocean. **Do not re-derive; implement the doc's form or leave it.**

### E-11. Displacement-gated JFA ladder floor

- **Change:** `firstEnabledFloodPass` already skips an even prefix when
  `pageDelta[9] ≤ 2`; ceiling parity proved two stride-1 repairs are
  insufficient at displacement 1 (comment at `redistance.ts:832-836`), so
  the current floor is 4 passes. Trial the remaining headroom: skip
  seed+refresh for pages whose own partial reported zero displacement
  (per-page, not per-frame — partials already exist per page).
- **Gate B**; the wall-sticking free-fall oracles are the regression canary
  (that failure class came from exactly this kind of under-repair).

### E-12. Phase-packing (last, by design)

- The narrow-work co-scheduler of `POWER_LIQUIDS_5X_GPU_PLAN.md:185-218`:
  typed GPU task list + one wide dispatcher per dependency phase. It is the
  only fix for the residual "0.1–2% occupancy publication spans" that
  survive E-1…E-11, and the plan itself orders it after kernel-work
  reduction. Do not build it first; do not skip it last.

### Expected-value summary (attributed ms on large, isolation-ranked)

| wave | items | families touched | plausible attributed Δ |
|---|---|---|---:|
| measure | M-1, M-2 | — | 0 (sizes everything) |
| 1 | E-1, E-2, E-3, E-5, E-6 | JFA, topology, transport, tail | ~10–14 |
| 1.5 | E-8, E-10 | volume, air | ~3–4 |
| 2 | E-4, E-7, E-9, E-11 | transport, all fine | M-1-dependent; the 10x hinge |
| 3 | E-12 | residual narrow spans | occupancy multiplier |

Wall-clock claims must come only from clean interleaved lane runs; the
column above ranks, it does not promise.

## 6. Standing rules distilled for this effort

1. Score changes on measured kernel time or work removed — never on
   dispatch/pass counts (measured twice as a non-lever, once as a
   regression).
2. One-workgroup persistence is for small constructed capacities only; on
   large lanes width beats fusion.
3. Any cross-kernel fusion or de-fusion across a storage buffer is Gate B
   (FMA contraction changes trajectories at 6e-5 and diverges by step 502 —
   refutation #12); "the shaders read identical" is not evidence.
4. Fail-closed GPU-resident gating only; no host-side skips, no readback
   scheduling (`hostSchedulingUsesReadback = false` is contractual).
5. Uniform sweeps beat divergent early-exit searches on workgroup-memory
   operands (refutation #10).
6. A change that gets faster while any tripwire fires is not a speedup
   (a silent topology rollback once manufactured a fake 1.66x).
7. Interleave every A/B; the machine drifts ~5% and the lanes drift with
   step index.

## 7. Session artifacts and reproduction

- Large-lane counter capture: `node --import tsx
  tools/profile-mini-dam-xctrace.ts --counters --isolate-pass-labels
  --lane=large --steps=240 --out=artifacts/scene-size-overhead/large-current`
  (~8 min; report at `large-current/report.html`).
- Census: `FLUID_OCTREE_ROW_DELTA_CENSUS=1 FLUID_WORKSET_CENSUS=1 node
  --import tsx tools/benchmark-power-dam.ts --lane=large --steps=240`
  (fine-transport census currently silent — see M-1).
- E-1's ready-to-run A/B script (interleaved walls + A/A + A/B fingerprint
  smokes):

```sh
#!/bin/zsh
set -e
echo "=== interleaved large-lane walls (A=flag0, B=flag1) ==="
for round in 1 2; do
  for flag in 0 1; do
    echo "--- round $round flag=$flag ---"
    FLUID_FINE_JFA_B4_ADDRESSING=$flag node --import tsx \
      tools/benchmark-power-dam.ts --lane=large --steps=240 2>&1 \
      | grep -E "ms/advance|validation errors"
  done
done
echo "=== Gate A fingerprints (mini smoke): A/A first, then A/B ==="
for run in a0-first a0-second b1; do
  flag=0; [ "$run" = "b1" ] && flag=1
  FLUID_FINE_JFA_B4_ADDRESSING=$flag npm run -s test:webgpu:minimal-power-dam-break 2>&1 \
    | grep -iE "phiBitXor|fingerprint|field stats|PASS|FAIL|validation" \
    | tail -20 > /tmp/b4-smoke-$run.txt
done
diff /tmp/b4-smoke-a0-first.txt /tmp/b4-smoke-a0-second.txt && echo "A/A IDENTICAL"
diff /tmp/b4-smoke-a0-first.txt /tmp/b4-smoke-b1.txt && echo "A/B IDENTICAL (Gate A pass)"
```
