# Losasso activity-scaled advance — implementation handoff

**Date:** 2026-08-05
**Baseline:** 47.80–48.00 ms/advance untraced, 50-step symmetric-expansion
steady benchmark (506.4 dispatches, 306 indirect, 39.14 passes per advance).
**Target:** cost proportional to *activity*, not *area* — ~5 ms/advance on a
calm surface, ~15–25 ms on the fully-moving guide lane.
**Predecessor:** `docs/losasso-10x-handoff.md` (SP0–SP6; SP1 + parts of
SP2–SP5 landed, closed out at 47.80 ms). This document is its successor and
absorbs the unfinished SP items where noted.
**Evidence base:** three code maps produced 2026-08-05 against this working
tree (solver policy, fine-band fidelity machinery, refine/coarsen lifecycle).
All `file:line` refs below were verified in that pass.

## The thesis

A fluid surface is calm, or moves slowly, with isolated regions of fast
activity. Today's advance charges every fine-band page, every pressure row,
every band face the full price every advance, whether or not anything within
reach of it moved. The predecessor plan made *reconstruction* conditional on
topology epochs (landed: grading quiesces, exact row-topology reuse, hierarchy
coefficient refresh). This plan makes *maintenance* conditional on measured
per-page activity:

> **Steady-state cost = small fixed core + (area-shaped work × activity
> fraction).**

Everything stays GPU-resident and massively parallel: activity classification
is a compact/dilate pass of the kind the band publisher already runs; skipped
work is zero-workgroup indirect dispatch, exactly the mechanism the solver
already uses to retire converged CG iterations. No host branches, no
readbacks, no per-advance encode divergence.

## Answers to the three motivating questions

**1. Temporal coherence — where does it already exist, where is it missing?**

The pressure solve is *already* temporally coherent, and better than the
folklore says. There is exactly one solve per advance
(`lib/webgpu-uniform-eulerian.ts:1859`, `:2007`); it is warm-started from the
previous advance's pressure (`lib/webgpu-octree-pipelined-mgpcg.ts:1689-1701`,
provenance chain documented at `:377-411`); convergence is checked GPU-side
each iteration and remaining indirect dispatch records are zeroed on success
(`:1742-1788`), including the 0-executed-iteration case on a settled seed
(`:1785`). The measured warm number is ~4–5 executed of 16 encoded. The
oft-quoted "243 MGPCG iterations per advance" is `mgpcgDispatchesPerAdvance` —
an *encode-shape* metric reproducible statically as `19 + 16·14 = 243`
(`webgpu-octree-pipelined-mgpcg.ts:912-931`) — not executed iterations. Do not
spend this program on "reducing solver iterations"; they are already few.

What has **no** temporal coherence is the fine-band maintenance layer:

| Consumer | Per-advance iteration domain | Ref |
|---|---|---|
| Fine transport (advect + commit) | ALL live band pages × 64 samples, twice; no activity check of any kind | `lib/webgpu-octree-losasso-fine-transport.ts:238-241`, prepare at `.wgsl.ts:75` |
| Volume ledger | ALL pages × ALL 64 samples, plus 3 full-reduction rounds | `lib/webgpu-octree-fine-levelset-volume.ts:478-487` |
| Coarse-phi restrict / ghost | ALL rows and ALL faces at **capacity** (4,096 rows encoded, 1,152 live) | `lib/webgpu-octree-losasso-coarse-phi.ts:194-223` |
| Extension band | full **page capacity** (`maximumResidentBricks`), plus 3 full-`faceCapacity` dilations | `lib/webgpu-octree-losasso-extension-band.ts:273-284` |
| Renderer sparse extract | ALL resident pages | `lib/webgpu-water-pipeline.ts:648-665` |
| Summary mips | delta-capable (warm mode exists) but cold/full still wins in practice | `lib/webgpu-octree-fine-levelset-summary-direct.ts:552-568` |
| Redistance | already delta-driven (dirty/support page lists) | `lib/webgpu-octree-fine-levelset-redistance.ts:664-665` |

Meanwhile every signal needed to classify activity already exists and is
already computed every advance:

- Per-page `PAGE_INTERFACE` / `PAGE_DIRTY` bits and the three-region changed-key
  stream (interface candidates / all-live keys / dirty keys), written by the
  transport commit kernel (`lib/webgpu-octree-losasso-fine-transport.wgsl.ts:133-145`,
  regions at `:144-145`).
- Measured maximum displacement, published in the delta header
  (`delta[7]`, `.wgsl.ts:153-155`) — today consumed only to size the band
  dilation radius.
- The structured (Power) lane's transport already has a *whole-generation*
  sleep with a per-page key snapshot (binding 16) and a displacement epsilon
  `FINE_LEVELSET_QUIESCENCE_DISPLACEMENT_EPSILON_CELLS = 1e-5`
  (`lib/webgpu-octree-fine-levelset-transport.ts:76-88`; decision kernel at
  `.wgsl.ts:190-256`). It is all-or-nothing (`activePages = sleeping ? 0 :
  pages`, `:245`) and **not wired to the Losasso transport at all**
  (`lib/webgpu-octree.ts:5998-6002` — only the structured transport has a
  governor).
- The recurring band publisher already computes dirty and support halos by
  separable Chebyshev dilation with derived radii
  (`lib/webgpu-octree-fine-levelset-topology.ts:2069`, radii at `:1251-1255`,
  `:2028-2032`) — exactly the "active region + conservative halo" geometry a
  per-page sleep needs.

The program is therefore mostly *wiring*, not invention: make the signals that
already exist drive the dispatch records that already exist.

**2. Do we need full fine-band factor-4 fidelity across the whole surface?**

Not temporally — but do not pursue *spatially* varying resolution. The fine
factor is one scalar for the whole domain, and at least nine independent
contracts pin it: the single-scalar `fineCellWidth`/`sampleDimensions` plan
(`lib/octree-fine-levelset-bricks.ts:166,191`) baked into every world↔fine
transform in transport, extension band, coarse phi and the renderer; brick
keys carrying no resolution field (`:198-220`); hard throws on any factor
outside {1,4,8} and any brick resolution ≠ 4 (`:159-164`, re-asserted in
topology, redistance, and consumer-sampling validation); band widths derived
once from the factor (`topology.ts:177-181`); 8-corner trilinear stencils
that hard-fail across an undefined resolution boundary
(`losasso-fine-transport.wgsl.ts:46-59`); the extension band's constant
`coarseSpan` per brick (`extension-band.wgsl.ts:135,142`); the renderer's
scalar core/halo handoff (`webgpu-water-pipeline.ts:631-633`); the summary
key-mapping factor branches (`summary-direct.ts:481-483,558-564`); and the
air-support kernel's explicit uniformity assert
(`webgpu-octree-air-velocity-support-gpu.ts:1973-1979`). Heterogeneous
resolution is a rewrite of the lattice identity.

The cheap equivalent is **temporal fidelity**: calm pages keep full spatial
resolution but are *maintained* less often — transport skipped when nothing in
reach moved (bit-identical carry), redistance/summary refresh at cadence,
volume kept by ledger carry. A calm page at factor 4 refreshed every 8th
advance costs less than an active page at factor 2 refreshed every advance,
and breaks no contract. AP1/AP2/AP4 below.

**3. Is the octree coarsening as much as it can?**

Structurally, yes — with two genuine gaps. Coarsening is a same-generation
rebuild: a dirty tile is reset to the largest fitting leaf and re-split from
scratch (`lib/webgpu-octree.ts:7974-7983`); there is no decay timer; the only
hysteresis is the spatial protection width (`:8179-8190`, deliberately — it
prevents split/carry flapping as the zero set crosses dyadic boundaries), and
the factor-1 temporal retention counter is deliberately disabled at factor 4
(`:7694`). Clean tiles are never touched. Refinement depth is a pure function
of distance-to-interface — no velocity, curvature, or activity term exists
anywhere in the criterion (`:8163-8212`) — and the interface shell is
*uniformly finest by design* (the Losasso premise; `:10051-10056`). On the
guide lane the 1,152 rows over 2,048 wet finest cells already mean the shell
is unit-sized and the interior is merged; there is little slack in the
criterion as stated.

The two real "fewer cells / less work" gaps are different in kind:

- **Capacity-shaped execution**, not topology: coarse-phi runs 4,096-row
  dispatches for 1,152 live rows; the extension band runs at maximum resident
  brick capacity. The cells are already few — the *dispatches* aren't.
  (Inherited SP4 territory; folded into AP2/AP3.)
- **Activity-blind shell refinement**: a flat, calm surface region keeps a
  unit-sized shell it arguably doesn't need, because depth cannot see
  calmness. Relaxing that is a numerics contract change (the fine band still
  carries the surface; ghost-distance coupling is what the coarse row
  actually feeds) — a legitimate but research-grade experiment, AP6, solo
  gate, never load-bearing for the headline target.

## The steady-state budget, honestly

The only clean recurring numbers are the 50-step wall (47.80–48.00 ms) and
the first-advance capture's *shares* (contended 3.7×; per-label absolute
times inflated). Recurring giants in the capture, cold labels excluded
(hierarchy publish 12.04 and full fine seeding 11.14 average 0.02
occurrences/advance in the 50-step audit):

| label | ms (contended) | shape |
|---|---|---|
| Fine transport direct axis-face sampling | 7.86 | area (all pages ×2) |
| Reduce resident fine overlap partials | 2.56 | area (all samples) |
| Extension band publish + dilations | ~2.9 | capacity |
| Finalize global fine page delta | 2.45 | area |
| V-cycle initialize ×17 + MGPCG drains | ~3.7 | encode-shape (mostly retired-but-encoded tail) |
| Coarse phi refresh ×2 + ghost ×3 | ~0.95 | capacity |
| Closed-form operator ×18, misc solve | ~1.0 | live |
| Summary/seed/band maintenance labels | ~2.5 | area/delta-capable |

With activity fraction `a` (fraction of live band pages inside the active
halo), the area-shaped block (~16 ms of contended share) scales to `~16·a`;
capacity-shaped becomes live-or-delta-sized; the encoded solve tail shortens
to executed+margin. At `a ≈ 0.1` — a calm surface with isolated splashes —
the arithmetic lands the attributed advance in the mid-single-digit
milliseconds, and inherited SP5 (single submission, no advance-path fences)
is what makes the wall track it. **On the fully-moving symmetric-expansion
guide lane `a` is large by construction; expect ~2×, not 10×, there.** The
10× claim lives on calm-dominated scenes, which is what real scenes are —
that is the point of the program, and AP0 exists to measure `a` before
anything is built on it.

## Work packages

Dependency spine: AP0 → AP1 → AP2/AP3 (parallel) → AP4/AP5 (renegotiations)
→ AP6 (research). Inherited-and-still-open from the predecessor: SP1's fused
sub-L0 D4 re-bless (code is default-on in encode-count terms; the env gate
must die), SP4 packing for surviving passes, SP5 single-submission. Do not
re-plan those here; land them under their existing descriptions.

### AP0 — Activity census and a calm lane (observability first)

Nothing below may be built until the activity distribution is measured.

1. Add per-advance counters to the smoke executor JSON line: live band pages,
   pages with `PAGE_DIRTY`, pages in the dirty halo, pages in the support
   halo, `delta[7]` max displacement, and executed solve iterations (already
   surfaced as `pressureIterationsUsed`, `tools/webgpu-smoke-executor.ts:2563`
   — note the freshness caveat at `:3277-3284`). All are readable from
   buffers that already exist; readback is diagnostics-cadence, never
   advance-path.
2. Add a **calm benchmark arm**: the same symmetric-expansion lane run past
   settling (long tail), plus one authored settled-tank lane. Record the
   activity-fraction time series for: symmetric-expansion early/late, mini
   dam late, ocean seiche. This is the ceiling of the whole program on one
   chart.
3. Re-answer SP3's open question with these counters: why does the summary
   warm path lose to cold — is `pageDelta[0]` genuinely large every advance
   (real churn), or is a generation bump forcing cold mode?

Noise rules apply: the x10 lane is ±5% within-arm; single-run 2–3% deltas
measure nothing; never compare across tripwire modes.

### AP1 — Per-page wake/sleep and activity-sized fine transport (the core)

Build the active-page worklist and make transport dispatch off it.

1. **Wake predicate, computed GPU-side per page:** a page is *awake* iff
   (a) its own measured displacement exceeds the sleep epsilon (per-page max
   sampled face speed × dt, in fine cells — the per-page analogue of the
   structured lane's `1e-5`-cell test, `fine-levelset-transport.wgsl.ts:234-235`),
   or (b) it lies within the dirty/support halo of any page whose phi changed
   last advance (`PAGE_DIRTY` dilated by the existing separable Chebyshev
   machinery, radii per `topology.ts:1251-1255`), or (c) it intersects the
   extension-band frontier of a face whose velocity classification changed.
   The halo radius must cover one advance of transport backtrace
   (`maximumBacktraceFineCells = 2·fineFactor` + interpolation support) — the
   same derivation the band planner already uses, not a new constant.
2. **Compact** awake pages into an active worklist + indirect dispatch record
   (the volume module's prepare-kernel pattern, `volume.ts:478`); publish
   `activeCount` beside the existing live count.
3. **Transport advect+commit dispatch off `activeCount`.** Sleeping pages are
   carried bit-identically: samples untouched, metadata generation restamped
   by a trivial restamp kernel so every consumer's generation check
   (`losasso-fine-transport.wgsl.ts:82,126`) still passes. The delta streams
   the commit kernel writes then shrink to the active set automatically,
   which is what makes AP2's warm paths genuinely small.
4. **Publication is never skipped.** Sleeping is "no work", never "no
   publish": the delta header, control words, and authority stamps must be
   written every advance, including the all-asleep advance. This is the
   dry-identity / gen-91 lesson (see Verification) and it is the single most
   dangerous part of this program.

Two contract tiers, landed separately:
- **Tier A (bit-identical):** wake epsilon = 0 — sleep only pages with
  *exactly* zero displacement and no halo contact. Gates must be
  bit-identical. This lands the machinery and its tripwires safely; on truly
  still regions it already wins.
- **Tier B (contract change, own gate):** epsilon = the structured lane's
  1e-5 cells. Numeric deltas bounded by epsilon per advance; D4/dry-identity
  gates judge it. Precedent exists (`FLUID_FINE_TRANSPORT_QUIESCENCE` A/B).

**D4 symmetry hazard (implementer must verify):** the guide lane holds
topology exactly symmetric while velocity carries ~3e-8 asymmetry. A
thresholded wake test can wake an asymmetric page set. Tier A is immune
(exact-zero test on symmetric quantities). For Tier B, either quantize the
displacement before comparison or demonstrate with the D4 gate that an
asymmetric wake set cannot produce phi divergence beyond tolerance (sleeping
vs transported differ by < epsilon by construction). If the gate flickers,
symmetrize the wake mask (OR with its D4 images) rather than loosening the
gate.

### AP2 — Activity-scaled derived state

Everything downstream of transport re-derives state for the whole area; make
each consume the active set. Order by capture share:

1. **Volume ledger (absorbs SP3.2):** cache per-page volume partials;
   sleeping pages contribute their cached partial, only active pages
   re-reduce. Steady advances keep the ledger by carry (transport commit
   already measures per-page change); the full capacity-shaped exact
   reduction runs at cadence (`FLUID_FINE_VOLUME_CADENCE` hook exists,
   `volume.ts:16,40-51`) and on epoch change. Correction rounds 2–3 become
   convergence-predicated zero-dispatch records. **Warning:** the
   capacity-sized scratch is the rounding barrier D4 exactness leans on
   (canonical-folds memory); any change to summation *membership or order*
   is SP6-class and lands solo with the dry-identity oracles run explicitly.
   Cached-partial reuse that preserves the exact fold order is Tier A;
   anything else is not.
2. **Coarse-phi restrict / ghost as deltas:** derive a dirty-row list (rows
   whose 9-gather support intersects an active page — a row-level dilation of
   the page mask through the existing leaf↔brick bounds) and a dirty-face
   list; dispatch restrict/ghost off those counts instead of
   `rowCapacity`/`faceCapacity` (`coarse-phi.ts:194-223`). The refresh path
   is already field-only; this makes it delta-sized too. Retired faces stay
   zero-coefficient, never dropped (predecessor WP3 caveat).
3. **Summary mips:** with AP1 shrinking the changed-key stream and AP0
   explaining warm-vs-cold, make warm the steady case — touched pages and
   ancestor chains only (`summary-direct.ts:552-568` already selects on
   `pageDelta[0]`).
4. **Renderer extract:** consume the active mask for re-extraction; calm
   pages keep last frame's extraction (render-side only, no sim contract).

### AP3 — Extension band retention

The band is rebuilt from scratch at full capacity every advance
(`extension-band.ts:273-284`) for a W=7 structure that is *static* wherever
the surface and wet-face classification didn't move. Retain the band across
advances; rebuild only faces whose supporting region is awake (the wake mask
dilated by the band width), with a full rebuild on topology epoch change.
Minimum honest first step: live-size the publish dispatch (worklist count,
not `maximumResidentBricks`) and delta-size the three dilations. The
coverage contract (transport band ⊆ extension band after one step of motion,
including corners — the topology-growth handoff's corner failure) must be
asserted with the existing `control[13]` latch, not assumed.

### AP4 — Temporal fidelity cadence (the "not full fidelity" axis, done right)

For pages that are awake-but-slow (displacement ≪ 1 fine cell/advance, no
interface sign traffic):

1. **Maintenance cadence:** redistance refresh and summary refresh at 1/N
   cadence with staggered phase across pages (avoid a thundering-herd frame).
   Transport itself stays every-advance for awake pages — semi-Lagrangian
   correctness is not cadence-tolerant, but *derived-state* refresh is.
2. **Activity-aware band radius:** the recurring dilation radius is driven by
   the *global* max displacement (`delta[7]` → `topology.ts:2028-2032`). Use
   a per-tile displacement maximum instead, so calm surface carries the
   minimum band (fewer resident pages at all — the true "fewer cells" win at
   the band level) while splash regions keep the CFL-derived radius. The
   residency floor and the throw at `topology.ts:299-307` guard the lower
   bound.

Both are contract renegotiations (they change which pages exist / how fresh
derived state is); each lands solo behind its own gate run, guided by AP0's
census, never speculatively.

### AP5 — Encoded solve tail from temporal prediction

The executed solve is small; the *encoded* tail is not: 16 encoded iterations
× (4 outer dispatches + apply + 9-dispatch fused V-cycle) with retired
iterations surviving as zero-workgroup dispatches and `stopped()`-guarded
V-cycle passes (`vcycle-gpu.ts:23,45-57,203`) — pass boundaries and encode
work with no physics. `selectOctreeFactorOneEncodedSolveTail` already
implements shorten-to-`previous+2` from a step-adjacent converged observation
(`lib/octree-solve-tail-policy.ts:197-262`) and is wired to nothing
(`:178-186`; call sites only in its test). Wire it for Losasso: encoded tail
= clamp(last executed + 2, floor, 16), keyed to same-topology-epoch, falling
back to the full envelope on epoch change or nonconvergence.

This deliberately renegotiates the stated policy that no prior-step
observation shapes the next command graph (`octree-solve-tail-policy.ts:2-6`)
— the renegotiation is *encode-shape only, physics-identical*: the executed
iteration sequence is unchanged for any solve that would have converged
within the shortened tail, and the nonconvergence path (publish best iterate
+ `ERROR_NONCONVERGENCE`, `pipelined-mgpcg.ts:1993-2021`) already handles the
mispredicted case loudly. State this in the PR; land solo. Optionally also
try the dormant rank-one temporal predictor
(`FLUID_OCTREE_PRESSURE_TEMPORAL_PREDICTOR`) on the calm lane — measure, don't
assume; the warm-start comment already claims no unclaimed win at steady
state.

### AP6 — Calm-shell coarsening (research tier; never load-bearing)

The only true topology-side coarsening slack: allow size-2 interface owners
where the surface is *flat and calm* — re-introducing, in activity-gated
form, exactly the clause Losasso's shader transform deletes
(`webgpu-octree.ts:10057-10066`). The fine band still carries the surface at
factor-4 resolution; the coarse row's job is ghost-distance coupling
(`coarse-phi.wgsl.ts:161-213`), which is first-order in row size. The
in-source warning stands: splitting policy changes moved the mini-dam
frontier 1,248→1,500 rows (+20%) and exhausted the solve tail
(`:8197-8200`) — this experiment moves it the other way but through the same
coupled machinery. Prereqs: AP0 census with per-size leaf histogram
(`FLUID_OCTREE_TOPOLOGY_CENSUS=1` exists), a flatness/calmness evidence term
threaded into `pressureRefinementEvidence` from the fine summary + wake mask,
and its own solo gate including wall-contact scenes. If it fights the gates,
drop it — the headline target does not depend on it.

## Verification contract

- **Gates per land:** symmetric-expansion D4 (exact topology/volume/diagonal;
  velocity/pressure/RHS within blessed tolerances), dry-identity / still-scene
  oracles (class-4 zero-RHS), and the 50-step wall benchmark plus the new
  AP0 calm arm. Fresh dated capture per retained change; compare label
  tables, never walls across tripwire modes.
- **Sleep must be loud — the standing hazard.** The large-lane forensics
  (dry-identity memory, `docs/losasso-topology-growth-handoff.md` gen-91)
  both reduce to one failure family: a "nothing to do" path that silently
  stops publishing, and fail-closed machinery that validly accepts the empty
  result forever. Every AP therefore ships with tripwires, on by default in
  test lanes:
  - `activeCount == 0` while the displacement header is nonzero ⇒ fatal.
  - Sleeping pages restamped every advance; a consumer generation-check miss
    on a slept page ⇒ fatal, not fallback.
  - The all-asleep advance still publishes delta header + authority stamps;
    assert via the existing frontier attempted-vs-accepted gap word.
  - AP3 band coverage: any in-band sample lacking velocity coverage latches
    `control[13]` ⇒ fatal in test lanes.
- **Tier discipline:** Tier A (bit-identical scheduling/sizing) and Tier B
  (epsilon/cadence contract changes) never share a land. SP6-class summation
  changes (AP2's ledger) land solo with the oracles run explicitly.
- **Noise discipline:** ±5% within-arm on the x10 lane; the mini-dam
  nondeterminism rules (dam-248) apply — judge by gate counters, use
  signature comparison for bisection.

## What NOT to do

- **Don't build spatially-varying fine resolution.** Nine contracts pin the
  scalar factor; the win is temporal cadence at uniform resolution. If a
  future program truly needs mixed resolution, it is a lattice-identity
  redesign, not a patch.
- **Don't optimize the solver's executed iterations.** They are 4–5 and
  warm-started with a 0-iteration settled case. The solve costs encode-shape
  (AP5) and V-cycle constant factors (inherited SP1), nothing else.
- **Don't micro-optimize passes AP1–AP3 shrink to delta size** — same rule as
  the predecessor's warning about its own v1 draft. Fusing a full-area pass
  that should run on 10% of pages is wasted motion.
- **Don't invent new halo constants.** Every wake/halo radius derives from
  the band planner's existing CFL-shaped quantities; a free constant here is
  the implicit-coverage-contract failure mode the corner-splash bug already
  demonstrated.
- **Don't let the sleep path near the publication path without tripwires.**
  A quiescence mechanism that can silently classify the world as
  needs-no-work is the gen-91 bug with better marketing.
- **Don't judge the calm-surface win on the moving guide lane.** The guide
  lane exists for correctness (D4); the activity win is measured on the AP0
  calm arm. Conversely, don't bless numerics on the calm lane — correctness
  gates stay on the lanes that move.
- Don't baseline against the contended 08-05 capture; recapture quiet
  (predecessor SP0 rules stand).
