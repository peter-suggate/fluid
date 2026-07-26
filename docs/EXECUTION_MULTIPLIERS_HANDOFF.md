# Execution Multipliers Handoff — eliminating A/B/C/D

**Written 2026-07-25.** Companion to `docs/POWER_LIQUIDS_PAPER_REVIEW.md`.
Audience: an engineer taking over the performance work. This document is
prescriptive: what to change, why it is safe, how to prove each step
worked. It is deliberately light on code specifics — the principles are
the contract; the code will drift.

## 0. The measured ground truth (do not re-litigate)

Mini dam (16³ coarse, 64³ fine band), M1 Max, 2026-07-25:
**56.4 ms/advance; GPU timestamp sum 58.1 ≈ wall 58.2**, so the frame is
~95% shader execution, not launch overhead. Command-stream link taxes on
this stack (microbenchmark, scratchpad `dispatch-latency-bench.mjs`):
dependent dispatch 3.6 µs · pass boundary 8.5 µs · pass split by buffer
copy 64 µs · one submit per dispatch 40 µs. Non-solve execution is
~43 ms against an arithmetic floor of **~3–4 ms** (the fine field is
1 MB; transport interpolation ~1–1.5 ms, banded JFA ~0.5–1 ms, all other
sweeps ~0.5 ms). The gap is four multipliers that stack per stage:

- **A — thread-count inflation**: kernels sized by domain/capacity, not
  by content (transport: 262,144 lanes for a ~60k wet band; 307k threads
  for 4,096 coarse cells; volume sweeps full-domain ×5).
- **B — per-thread inflation**: 55–60% of hot shader text is defensive
  re-validation (~26M redundant loads/frame); the rare-case fallback sets
  the register ceiling for every lane.
- **C — schedule inflation**: iteration counts are compile-time worst
  case (solve: 20 cycles × 4 smooths × 2 = 320 smoother dispatches for a
  4,096-row system; 8×2 repair waves; 6+6 band-phi rounds; 8 coarse
  redistance passes; an 8-deep serialized merge ladder).
- **D — duplicated quantities**: the same physical quantity is
  materialized several times per frame (velocity ≥4 representations;
  a full-domain velocity cache re-deriving what transport just sampled;
  a summary mip; a second coarse distance field re-redistanced 8×).

A stage like transport is A×B; the solve is C on a chain; the face band
is A×B×C×D at once. Fixing one multiplier in one stage moves single
milliseconds; the target is to make each multiplier structurally
impossible, stage by stage.

## 1. Global rules (apply to every change below)

1. **One change per commit, predicted number first.** Before running,
   write down the expected ms and counter deltas. A result that beats
   the prediction is as suspicious as one that misses it.
2. **No accuracy-affecting default flips in a perf commit** (the af0658d
   lesson). Gates only tighten in optimization commits.
3. **Ship counters with every claim**: per-pass GPU ms
   (`profile:power-dam-mini:fine`), lanes-dispatched vs lanes-that-did-work,
   iterations-executed vs iterations-scheduled.
4. **Add the quiescent benchmark first** (see §6) — it is the single
   sharpest instrument for A and C and takes an afternoon.

---

## 2. Multiplier A — dispatch content, not capacity

**Principle.** Every dispatch's lane count must be proportional to the
set of elements that can change this frame. That set is computed on the
GPU (mask/prefix compaction) and fed through indirect dispatch args. No
kernel may iterate `capacity` or `dims³` unless it is provably the
active set.

**Why this is safe here.** The machinery already exists and is proven
in-repo: indirect dispatch is used widely; prefix-sum compaction gave
2.8–4.3× in the leaf solve; the fine brick worklist maintains a compact
resident list. What is missing is the *policy* that every kernel must
consume a compacted worklist, and the residency policy that keeps the
worklist small.

**Prescription.**
1. **Make "active set" a first-class object per domain.** Three of them:
   fine wet-band lanes, coarse active rows, and the cut-row subset
   (rows whose faces the interface crosses). Each is a compacted index
   list + count in an indirect-args slot, rebuilt by mask-scan when the
   topology epoch bumps, untouched otherwise.
2. **Convert consumers mechanically.** For each kernel: replace
   `lane = global_id` + capacity guard with `element = worklist[lane]` +
   indirect count. Do transport first (largest single win), then the
   volume reduces, then `buildPowerCoarseSelectorRows` and the
   restriction sweep.
3. **Split divergent rare cases into a second compacted dispatch.**
   Where a kernel handles a common case and a rare fallback (transport's
   air path is the canonical example), the first pass classifies and
   appends rare lanes to a worklist; a second, separately-compiled
   kernel processes them. This is also the main lever for B's register
   ceiling. The classify/evaluate pair already in the air sampler is the
   in-repo proof.
4. **Then fix residency itself.** At mini scale the band is nearly the
   domain because seeding pre-dilates to the full support width
   (audit gap #3). Adopt the paper's rule verbatim: allocate interface
   blocks + one block ring, let the CPT/redistance activate the rest on
   demand. Brick count must scale with interface area × fixed width.
   This is the step that makes A pay at production scale, and it is the
   one with real regression risk — it lands last, under the surface
   gates, with the band-underseed counters watched (the af0658d band
   narrowing regression was exactly this done carelessly).

**Acceptance.** New counter: Σ lanes dispatched per advance. Prediction:
transport 262k → ~60k lanes and 7.0 → ≤2.5 ms (with B untouched);
total lanes/advance drops ≥5×; quiescent-frame cost (§6) drops in
proportion to the active set, not the domain.

**Traps.** (a) Don't build worklists with sorts — mask + prefix rank
preserves `(level, morton)` order deterministically. (b) An empty delta
must cost zero *encoded* work too: keep the (0,1,1) indirect pattern —
at 3.6 µs/dispatch it is cheap — but never let a "delta" kernel fall
back to scanning capacity to decide it has nothing to do.

---

## 3. Multiplier B — validate once per epoch, not once per access

**Principle.** Within one queue submission, WebGPU ordering guarantees
that inputs cannot mutate under a kernel. Therefore a kernel may trust
any input that was validated once when it was published. Validation
budget: one reduce per domain per epoch, plus one epoch-word check per
dispatch. Everything denser than that is deleted or moved to the
`FLUID_DEEP_VALIDATE=1` lane.

**Why this is safe here.** The history audit is unambiguous: per-lane
rollback machinery has only ever fired on injected faults; the worst
real bug (tombstone leak) *passed* the per-lane checks; the only
production rejections came from the checking machinery itself
false-firing. Both historical hard bugs were found via the deep-validate
lane, which stays. `ownerAtCached` and the 23 unchecked owner sites in
the octree module are the in-repo proof that hoisting is sound.

**Prescription.**
1. **Hoist, don't delete, first.** For each hot kernel, list every check
   executed per lane or per loop iteration (publication headers,
   generation words, reciprocity, validFace, bounds already implied by
   the worklist). Move each to one of: (i) the worklist builder (a lane
   that made it into the active set is valid by construction — this is
   the deep synergy with A); (ii) a once-per-workgroup check whose
   result lanes read from workgroup memory; (iii) the per-epoch
   validation reduce that runs before the publication flip.
2. **Then shrink the hot kernel's register footprint.** With rare cases
   split out (A.3), recompile the common-case kernel without the
   fallback code and confirm occupancy rises (Metal capture, or simply
   the ms). The air-fallback being 4× the hot-path size is currently the
   register ceiling for all 262k transport lanes.
3. **Keep two things.** Scalar, reason-coded check style wherever a
   check survives (a vectorized rewrite once tripped a Dawn/Metal
   miscompile), and the fail-closed *publication* gate: candidate state
   is validated once, whole, before the epoch flip; rejection means the
   old epoch stays live — never a mid-frame per-lane rollback, and never
   an escalating rebuild (that escalation was the documented 30–100×
   stall).
4. **Collapse the counter families.** ~15 generation counters in four
   namespaces become ~5 epoch words. A consumer checks exactly one word,
   exactly once per dispatch.

**Acceptance.** Defensive-statement fraction of hot shader corpus
< 15% (census script exists from the shader-defense audit). Transport
alone: 7.0 → ~1.5–2 ms combined with A. The deep-validate lane still
catches an injected fault in CI (add one such injection test — it is
the license for all this deletion).

**Traps.** Do not hoist across the submission boundary: anything read
from a *previous* submission's output still deserves its one epoch-word
check. And do not delete the two-step Dawn diagnostic lane — trust is
being narrowed, not removed.

---

## 4. Multiplier C — schedules driven by state, not by worst case

**Principle.** No fixed iteration count in any encode path. Every loop
either (i) terminates by a GPU-resident convergence/fixed-point test,
(ii) derives its trip count from a measured bound (displacement, band
width, dirty count) written by the GPU last frame, or (iii) collapses
into a single dispatch because the whole problem fits one workgroup.
A fixed count may survive only with a comment stating the invariant
that makes it sufficient *and necessary*.

**Why this is safe here.** The repo already proved the key mechanism:
the staged Galerkin CG stops its encoded tail after convergence with a
workgroup-uniform active flag — no extra dispatch, no readback. Every
other schedule can use the same pattern or the indirect-zeroing variant
(a converged stage writes 0 into the next stage's indirect args).

**Prescription.**
1. **The solve (14.8 ms, 505 dispatches — biggest single item).**
   Three independent steps, in order:
   (a) residual-gate the cycle loop: encode the worst-case 20, let the
   active-flag/indirect-zero mechanism turn converged cycles into
   (0,1,1) no-ops — measured cost of a dead cycle is ~25 dispatches ×
   3.6 µs ≈ 0.1 ms, so a solve that converges in 4 cycles costs ~4/20
   of today's 14.8 plus ~1.5 ms of dead encode;
   (b) batch the coarse levels: every level whose row count fits one
   workgroup runs inside a single workgroup-resident dispatch that
   smooths/restricts/prolongs in shared memory (the disabled
   persistent-3 lane is the prototype — at 16³ this is *all* levels
   below the finest, collapsing ~20 dispatches per cycle to ~3);
   (c) longer term, the faithful first-order M1 preconditioner from the
   paper review replaces the Galerkin hierarchy entirely and deletes
   the RAP refresh — (a) and (b) are still worth doing first because
   they are mechanical and de-risk (c).
2. **JFA strides (fixed 10).** The stride schedule needs to cover the
   maximum distance the interface moved, not the band width. CFL bounds
   displacement to ~1–2 fine cells per step; warm-seed from last frame's
   closest points and run 2–3 strides + the repair pass. Keep the full
   log₂ schedule for cold start and epoch changes. Gate: the CPT
   differential bound harness (P1) must exist first; this is the one C
   item with accuracy exposure.
3. **Repair waves (8×2) and band-phi rounds (6+6).** Same mechanism as
   the solve: each wave writes a remaining-work count; the next wave's
   indirect args are zeroed when it hits 0. Steady state should execute
   1–2 waves. If the count *never* reaches zero, that is a seeding bug
   being papered over — surface it, don't schedule around it.
4. **The merge ladder (8 serialized merges in the summary path).**
   Do not optimize it — it is a D casualty (§5). If any part survives,
   a rank-indexed value mip replaces sort-and-merge outright.
5. **Add the schedule counter.** Every formerly-fixed loop reports
   `executed/scheduled` per advance. This is what makes C regressions
   visible forever after.

**Acceptance.** The quiescent benchmark (§6) is the real gate: a settled
tank must execute ~1 smoother round, 0 repair waves, 2 JFA strides.
Prediction for the dam in motion: solve 14.8 → 4–6 ms via (a)+(b) alone
at unchanged residual targets (verify with the existing relative-L2
check); face-band iterative stages halve.

**Traps.** Convergence tests must be GPU-resident (flag/indirect) —
never a readback in the hot path. And never loosen the residual target
in the same commit that gates the schedule (rule 2).

---

## 5. Multiplier D — one authority per quantity

**Principle.** Each physical quantity has exactly one authoritative
representation per frame. Everything else is either derived at the
point of consumption or is a cache that must name its consumers and
show a measured win over on-demand derivation. When a stage exists only
to keep representation #2 coherent with representation #1, the fix is
deletion, not optimization.

**Prescription — the four standing deletions, each independently
landable, each behind the surface gates:**
1. **Velocity (biggest and clearest).** Authority: face normal-velocity
   DOFs. The full-domain `publishFineVelocityCache` (3.15 ms, 262k
   lanes) re-derives at every fine sample what its consumers could
   sample on demand — and transport already builds exactly that
   interpolant internally. Enumerate the cache's consumers (fused air
   sampler, renderer), point them at the shared interpolant/CPT
   extension, delete the pass. If profiling later proves a consumer
   genuinely needs a cache, rebuild it *over the active set* (A) — that
   is a different, ~0.3 ms object.
2. **Distance.** One field: the fine CPT is authoritative; the coarse
   phi is its restriction plus a collar. Today the coarse field is
   restricted *and then re-redistanced with 8 fixed passes* (1.6 ms) —
   the second Eikonal solve is representation maintenance. Delete it;
   keep the restriction and the GFM consumers' collar. (This is A1 from
   the algorithmic plan; the band-underseed counters and still-water
   gate are its safety net.)
3. **Volume control.** Measurement-only: keep the fail-closed
   measurement and telemetry, drop the full-capacity correction sweeps
   (5 of the volume path's dispatches, ~1M+ full-capacity lane visits
   per advance at mini scale), and let the ≤1%/500-step drift gate
   arbitrate. If drift fails, the fix is transport/redistance
   conservation, not resurrecting the global offset.
4. **Summaries and the transient face graph.** The summary mip and the
   8-deep merge ladder (3.7 ms) maintain a derived overview of state
   whose physics consumers must be enumerated; anything only diagnostics
   reads moves behind a flag, anything physics reads becomes a
   rank-indexed mip over the active set. The face band's transient
   power graph duplicates the committed publication — sample the
   committed graph (already planned as D2).

**Acceptance.** GPU state census: authoritative fraction rises from
~1.6% toward ~20%+; per-pass profile loses the cache/summary/coarse-
redistance lines outright (−8 to −9 ms). Surface gates at original
thresholds; energy-ledger taps unchanged.

**Traps.** Deletion order matters: point consumers at the authority
*first*, measure parity (IoU + gates), then delete the duplicate in the
following commit. Never both in one commit — that is how af0658d became
unrevertable-in-parts.

---

## 6. The instrument to build first: the quiescent benchmark

Add a lane to the benchmark tool: run the mini dam to settle (~2 s),
then measure ms/advance over the next 60 steps of still water. Report
it beside the in-motion number, plus the counters from §2–§4
(lanes dispatched, iterations executed/scheduled, per-pass ms).

Its power: in still water the correct cost of A is near zero lanes, of
C is near zero iterations, of D is nothing to re-derive. Today's
pipeline will show ~55 ms because every multiplier is content-blind.
Every change in this document moves this number toward the floor, and
any regression in any multiplier shows up here first and unambiguously.
Target end state: quiescent ≤ 3 ms; dam-in-motion ≤ 12 ms with solve,
≤ 5–6 ms non-solve — which is the arithmetic floor plus margin, on the
scene this document measured.

## 7. Suggested order (risk-ranked, each step gated)

1. §6 quiescent benchmark + counters (pure instrumentation).
2. C.1a solve residual gating + C.3 wave gating (no numerics change at
   unchanged tolerances; ~−8 ms predicted).
3. D.3 volume measurement-only (assessed, small blast radius; ~−1.5 ms).
4. A.2+B.1/B.2 on transport (worklist + hoist + fallback split; ~−5 ms).
5. D.1 velocity cache deletion (~−3 ms) then D.4 summaries (~−3 ms).
6. C.1b coarse-level batching (~−3 ms), then C.2 warm JFA under the P1
   CPT bound (~−2 ms).
7. D.2 one distance field (~−1.5 ms), then A.4 interface-proportional
   residency (the scale-out lever; needs the surface gates at full
   strictness).
8. C.1c faithful M1 — merges into the paper-review track (P3).

Steps 2–6 are independent of the direct-CPT face-band work in flight
next door; coordinate only on §5 items touching `face-closest-point.ts`.
