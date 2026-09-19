# Tile-first Uniform Geometric: the 4h grid as a real compute grid

Read-only design study, 2026-09-19, working tree `codex/main-unpushed-20260918`.
No GPU started, no repo file changed, git worktree untouched.

Scope: the project owner's proposal that the 32³ grid of 4h tiles becomes a real
compute grid covering the whole domain, with fine h cells used only where the
surface needs them. Target scene `cm12-figure-7`: 128³ cells, h = 0.05 m,
dt = 1/30 s, a radius-20-cell ball (33 510 cells, 1.6%) falling 4.5 m into a
closed empty tank and spreading into a sub-cell sheet. D ≈ 6 fine cells/step
(VERIFIED: free-fall from `ball([0,90,0],20)` at `cm12-paper-scenes.ts:548`,
`CM12_CELL_SIZE_M=0.05` at `:37`, v ≈ 9.4 m/s ⇒ v·dt/h ≈ 6.3).

Every claim is tagged VERIFIED (file:line) or INFERRED.

---

## Verdict in one page

1. **The two-level sampler works and is the load-bearing idea.** Defining the
   air-side velocity and phi by the 4h level outside the fine set removes the
   2D part of the exact mask's reach. The live set drops from ~4 tiles of
   dilation to ~2 (phi) and ~1 (velocity advection/project). On figure 7 that is
   4021 → 2038 live tiles in the ball regime: a 2× smaller live set than the
   exact mask, 16× smaller than dense.
2. **The owner's literal proposal — V transported on 4h cells — fails on this
   exact scene**, and fails on the feature the figure exists to show. §3A gives
   the counter-example: the sheet is 2 fine cells deep = 0.5 coarse cells, phi is
   positive through it so the phi-derived capacity γ is identically 0, and
   phi-proportional reconstruction therefore has nowhere to put the tile's V.
   It must fall back to uniform spreading, converting a 0.5-coarse-cell sheet
   into a 4-cell-thick fog in one step. Separately it makes the Sec. 3.7
   overfill trigger 64× coarser. Recommend option **B**, fine transport on a
   live set, no coarse V at all.
3. **Hybrid transport (B) is exactly conservative under a much weaker condition
   than a dependency closure**: not "F closed under donor/receiver alternation",
   but simply **V = 0 strictly outside the built set**. §3B proves the
   double-count the brief suspected, and shows the V = 0 condition removes it.
   Transport's own reach is then ceil(D)+1 ≈ 7 cells ≈ 2 tiles, not 4.
4. **The frame is pressure and this design does not touch it.** Pressure is 69%
   of the frame on mini64 (VERIFIED, `docs/benchmarks/uniform-pressure-tolerance-2026-09-19.json`
   via `uniform-geometric-work-map-candidates.md`, wall 78.3 ms / pressure 53.9 ms).
   Everything designed here lives in the other 31%, so its whole-frame ceiling is
   ~28% — a 1.4× frame, at best. On figure 7 specifically the liquid occupancy is
   **1.6%**, not mini64's 36%, so candidate C1 (skip pressure tiles with no baked
   liquid) has ≥ 97% of tiles skippable and dominates everything below. Say this
   to the owner plainly before building any of this.
5. **The design avoids adaptive-volume's seam problem only by never solving
   anything at 4h.** The coarse level is (a) the *definition* of a non-physical
   far field and (b) a *scheduling predicate*. The moment anything dynamical runs
   at 4h beside a fine region you have a flux seam at a place where the two
   discretisations disagree — which is the wall adaptive-volume hit. Option A
   crosses that line; option B does not.

---

## Q1. Two-level sampling replaces the halo

### What already exists

VERIFIED, `webgpu-uniform-velocity-extrapolation.ts:216-262`: the extension
builds hierarchy levels by repeated halving until the shortest axis is 1. At
128³ that is 7 levels: 64³, 32³, 16³, 8³, 4³, 2³, 1³. **Level index 1 (0-based)
is exactly the 32³ = 4h tile grid.** Each level owns a `.down` texture (written
by restriction) and an `.up` texture (written by prolongation). The up chain
starts at the coarsest `.down` and fills each finer `.up` in turn
(`:246-262`), so **`hierarchyLevels[1].up` is a complete 4h MAC velocity field
defined over the entire domain, already computed every step.** No new field is
needed for the velocity half of the sampler.

VERIFIED, `webgpu-uniform-velocity-extrapolation.wgsl.ts:387-415`
(`hierarchyComponentSample`): the transfer is staggered — the two transverse
axes are cell-centred, the component axis is face-centred at
`(target[c]+1)·src/tgt − 1`, i.e. the texel at coarse index t on the component
axis is the **positive face of coarse cell t**, the same convention the fine
texture uses (`sampledFaceVelocity`, `webgpu-uniform-reference.wgsl.ts:246-249`).
So the 4h level is a genuine MAC field in the same convention; the coarse
sampler is the fine formula with `dims()` replaced by coarse dims and the
position divided by 4.

### The samplers

`sampleVelocityComponent(p, c)` today (VERIFIED `webgpu-uniform-reference.wgsl.ts:260-269`):
`q = clamp(p − offset, lower, dims−1)` with `offset = (0.5,0.5,0.5)` except
`offset[c] = 1.0`; 8 taps of the padded transport shell at `base+o+1`.

Proposed two-level form:

```
fn sampleVelocityComponent(p:vec3f, c:u32) -> f32 {
  if (TWO_LEVEL && !fineResident(tileOf(p))) { return coarseVelocityComponent(p, c); }
  ... existing fine path ...
}
fn coarseVelocityComponent(p:vec3f, c:u32) -> f32 {
  let cd = vec3f(coarseDims());          // dims()/4
  var offset = vec3f(0.5); offset[c] = 1.0;
  var lower = vec3f(0.0);  lower[c] = -1.0;
  let q = clamp(p*0.25 - offset, lower, cd - vec3f(1));
  ... 8 taps of uvCoarseVelocityIn, zero outside [0,cd) on the component axis ...
}
```

Tile-class lookup: reuse the shipped shape exactly. `uvSharpenTileIndex`
(`uniform-volume.wgsl.ts:250-253`) already maps a cell id to a word in the third
conditioning plane behind an 8-word header; a second map word range at
`2N + 8 + tileCount` holds the residency class. One `atomicLoad` per lookup,
which is the same cost the sharpening map already pays 32 times per step.

**Boundary convention mismatch to fix.** The fine path reads the *padded*
transport shell, so faces outside the domain read the shell's zero ring; the
coarse path would have to clamp. Use the `sampledFaceVelocity` rule explicitly:
return 0 when `p[c] < 0 || p[c] >= dims[c]`, applied at the coarse level too.
Otherwise a closed-wall tile samples a replicated interior velocity and the
released-wall term (`uvReleasedWalls`, `uniform-volume.wgsl.ts:75-97`) sees a
different wall speed than the dense schedule.

### Continuity across the fine/coarse boundary

Three separate questions, and they have different answers.

**(a) Is the cascade 4h→2h→h the same as direct 4h→h?** Almost. On a nested
grid, linear interpolation of linear interpolants is exact *within one coarse
interval*, so the cascade reproduces the 4h piecewise-linear interpolant exactly
everywhere except where the two bracketing 2h samples straddle a 4h knot. There
the cascade is a convex blend of two adjacent linear pieces, differing from the
direct interpolant by at most ≈ 1/8 of the local second difference of the 4h
field. INFERRED (algebra above; the code is `hierarchyComponentSample` twice).
Also VERIFIED: the known-mask renormalisation
(`webgpu-uniform-velocity-extrapolation.wgsl.ts:407-414`) is a **no-op once all
8 taps are known** — `weightSum = 1` and the result is plain trilinear. After
the dense 32³ prolong every coarse face is known (INFERRED: the up-chain fills
from the 1³ level where at least one component is known whenever any liquid
exists), so far-air sampling never exercises the renormalisation.
**Conclusion: use direct 4h sampling. Drop the 2h level outside the fine set.**
Nothing in far air is physical; an O(h²·∂²v) smoothing difference is below the
noise of a field that is itself an interpolated fill.

**(b) Is the sampled field continuous across the class boundary?** No, and it
cannot be made so. Fine values in the outermost fine tiles are prolongations of
coarser values, but trilinear interpolation *of samples of* a trilinear function
is not that function. The jump is the same O(h²·∂²v) size as (a). It matters
only if it accumulates. It does not, provided the far field is **defined** at 4h
rather than advected on the fine lattice — which is exactly what Q2 buys.
This is the structural pairing: two-level velocity without two-level phi would
let seam kinks accumulate in the far-air phi step after step.

**(c) Does the jump ever touch something physical?** Only if the class boundary
is closer to phi = 0 than the deepest consumer's reach. Consumers and their
reaches, all VERIFIED:

| Consumer | Reach from phi = 0 | Source |
|---|---|---|
| `pressureLiquid` | 0 cells (sign at the cell) | `webgpu-uniform-reference.wgsl.ts:241` |
| `surfaceOccupancy` (gravity, viscosity gate) | **2 cells** (`0.5 − phi/(4h)` > 0 ⟺ phi < 2h) | `:167-171`, `applyVelocityForces:877-885` |
| sharpening admission band | 2.1 cells (`tuning.y` default, `method.ts:68`) | `uniform-volume.wgsl.ts:281` |
| `uvRedistancePhi` closest-point search | ±4 cells, band 4·max(h) | `uniform-volume.wgsl.ts:127-132` |
| `uvTrace` departure | D ≈ 6 cells | `:51-59` |
| `divergenceAt` one-cell negative halo | 1 cell around liquid | `:1016-1033` |

So a **2-tile (8-cell) reach for phi** puts the class boundary at |phi| ≥ 8h,
twice the redistance band and four times the occupancy reach. A **1-tile reach
for velocity advection and projection** suffices (Q4). The exact mask's ~4-tile
reach exists because the backtrace must land on valid fine data; with a coarse
fallback the departure point is always valid, so that term vanishes.

### Which extension passes shrink

VERIFIED pass inventory from `webgpu-uniform-velocity-extrapolation.ts:343-400`
plus `buildExtrapolationAuthority` (`webgpu-uniform-reference.wgsl.ts:583-584`):

| Pass | Dispatch domain today | After |
|---|---|---|
| `buildExtrapolationAuthority` | dense 128³ (2.10 M) | fine tiles |
| `seedActiveFront` | dense 128³ | fine tiles |
| `updateActiveFront` ×8 | **already indirect** over the active front | unchanged |
| `prepareActiveDispatch` ×9 | 1 thread | unchanged |
| `resolveConvergedFront` | dense 128³ | fine tiles |
| restrict base→64³ | 64³ = 262 k | **keep dense** (see below) |
| restrict 64³→32³ … 2³→1³ (6) | 37 449 total | dense |
| prolong 1³→2³ … 16³→32³ (5) | 4 681 total | dense |
| prolong 32³→64³ | 262 k | fine-tile footprint at 2h |
| prolong 64³→128³ | dense 128³ | fine tiles |
| `packTransportShell` | dense 128³ | fine tiles |

**Keep every restrict dense.** The restrict direction is what carries the FIM
band's known values *up* into the coarse levels; if it were skipped outside the
fine set it would leave stale `knownMask` bits in the coarse textures
(`restrictKnownVelocity` writes `primaryOut` unconditionally,
`:474`), and clearing them would cost as much as computing them. Its total is
262 k + 37 k ≈ 300 k thread-instances ≈ 14% of **one** dense base pass, against
the five dense base passes (authority, seed, resolve, final prolong, pack =
10.5 M) it lets you delete.

**Residual dense cost after the change**: 300 k (restricts) + 37 k (prolongs at
32³ and coarser) ≈ 337 k thread-instances, versus ≈ 11.1 M today. ≈ 3%.
The 32³ `.up` level must stay dense — it *is* the coarse sampler's texture.

---

## Q2. Coarse phi as far field and predictor

### Storage

VERIFIED: phi is a dense vertex field, (n+1)³ (`uvAdvectPhi`,
`uniform-volume.wgsl.ts:119-123`, dispatched at `vertices` dims,
`webgpu-uniform-reference.ts:1226-1228`). Tile corners are vertices at indices
(4i,4j,4k) — a 33³ subset of the 129³ lattice. For rung 2 (sparse work, dense
storage) read them strided from the existing texture: **no new storage**, as the
brief says. For rung 3 a 33³ r32float own-texture is 143 KB.

### Coarse advection each step

`uvAdvectPhiCoarse`: 33³ = 35 937 traces (1.7% of the 2.10 M fine vertex
traces), each a `uvTrace` at 4h using the coarse velocity sampler, then
`uvPhiCoarse` at the departure point. Wall continuation and released-wall terms
(below) evaluated at 4h. Cost is ~1.7% of the current `uvAdvectPhi` — free.

**This gives the "no stale far field" property.** Because far-air phi is
*re-derived* every step from a field that covers the whole domain, a tile that
leaves the fine set carries no state forward: its fine vertices simply stop
being read (the sampler is region-keyed), and when it is re-promoted,
`uvAdvectPhi` — the first phi pass of the step — writes its fine vertices from a
two-level departure sample before anything reads them. **No promotion pass, no
demotion pass, no staleness.** (INFERRED from the encode order at
`webgpu-uniform-reference.ts:1226-1228`: advect is first, and the residency
classification must be encoded before it.)

### Prediction — where the idea is weakest

**A coarse-advected phi is not a conservative predictor.** Semi-Lagrangian
level-set advection *erodes* thin and small features; at 4h a 5-coarse-cell-radius
ball resampled with a 1.5-coarse-cell displacement per step will shrink and
round. A predictor that erodes can under-predict the band and drop live tiles.
Do not use coarse phi as the sole admission test.

What coarse phi *is* safe for is **rejection with an explicit Lipschitz margin**:

> tile T is far ⟺ min over its 8 corners of φ_c ≥ r_tile + reach, where
> r_tile = ½·√3·4h ≈ 3.46h is the largest distance from any point of T to its
> nearest corner.

This is sound only if φ is 1-Lipschitz. VERIFIED it is not guaranteed to be:
`uvRedistancePhi` restores metric distance only inside `band = 4·max(h)`
(`uniform-volume.wgsl.ts:127-133`); outside, advection by a compressive air
velocity can steepen the field. Two mitigations, in order of preference:

1. **Classify from the fine band, dilate per tile.** One classify pass exactly
   like `uvClassifySharpenTiles` (one `uvPhi` per cell, workgroup OR, one word
   per tile) gives the seed set with no Lipschitz assumption. Then dilate by a
   **per-tile** reach: each tile publishes `max |v|·dt/h` over its own 64 cells
   (a second workgroup reduction in the same pass), and the dilation radius is
   `ceil(perTileMax) + margin`. That replaces the global max-speed dilation the
   brief wants replaced, without trusting an advected SDF. For figure 7 the
   ball's own tiles have max speed ≈ 6 cells, the tank's far air ≈ 0 — so the
   ball's wake dilates and the rest does not.
2. Use the coarse-phi rejection only as a *cheap pre-filter* that runs before the
   fine classify, with a doubled margin. Saves the fine classify's 2.10 M `uvPhi`
   calls in tiles the coarse test already rejects.

**Margins to size, all VERIFIED as constraints:**
- interface moving within a tile: 4h (a tile is 4 cells).
- velocity variation inside a tile: use the per-tile max, not the tile-centre value.
- redistance search: ±4 cells (`:132` clamps `q` to `p ± 4`).
- sharpening band: 2.1 cells.
- `surfaceOccupancy` gravity gate: 2 cells.
- `uvClosedWallPhi` (`:101-118`): traces **one cell inside** each incident closed
  plane, so a boundary tile's continuation source is in the same tile. No extra
  reach. It fires only when `continued < 0`, i.e. only at a wet wall, which the
  band classification already admits.
- `uvReleasedWalls` (`:75-97`): evaluated at every vertex, but the term
  `dt·away − inward·(p[axis]−plane)·h` is only positive within `dt·away/h` ≤ D ≈ 6
  cells of a released plane. For figure 7 the **ceiling is released** and the
  floor is not (VERIFIED: `released = inward·acceleration > 0.5|g|`; floor
  inward = +1 with `cellGravity.w < 0` gives −|g|, ceiling inward = −1 gives
  +|g|; `top:"closed"` at `cm12-paper-scenes.ts:227` so `boundary.w` ambient is
  false). The ball starts 38 cells below the ceiling, so this is inert here, but
  a design that evaluates it only at 4h vertices in non-fine tiles changes a
  linear ramp's sampling — low risk, flag it.

### What is written into a demoted tile's fine vertices

Nothing, and that is the point. The two consumers of the *dense* fields are:

- **`uvPublish`** (`uniform-volume.wgsl.ts:333-339`), which writes the renderer's
  surface texture as `0.5 − uvPhi(centre)/h` and `uvOpen(id)` per fine cell.
  VERIFIED that this must stay dense: the prior audit's Q3(2) is right — clipping
  it leaves stale surface outside the live set. **Keep `uvPublish` dense but make
  its `uvPhi` two-level.** Far cells then cost 8 coarse taps instead of 8 fine
  taps — same cost, but it needs no fine phi to exist, which is what rung 3 wants.
- **Field overlays**, which read the same dense texture. Same answer.

`reduceDiagnostics` (`webgpu-uniform-reference.wgsl.ts:1558`) also stays dense:
it sums `surfaceOccupancy` into `reductions[0]` and raw V into `reductions[3]`,
and clipping it would under-count the very conservation number the method is
judged on.

---

## Q3. V transport on 4h cells

### Option A — all transport at 4h. **Reject.**

Mechanically it is attractive: coarse V is the exact sum over the tile
(conservative by construction), the edge arena drops from 80 B × 2 097 152 =
**167.8 MB** to 80 B × 32 768 = **2.6 MB** (VERIFIED `UNIFORM_VOLUME_EDGE_BYTES = 80`,
`uniform-volume.wgsl.ts:19`), and the twelve dense coupling passes plus four
scratch clears (VERIFIED `webgpu-uniform-reference.ts:1229-1234`) become 64×
cheaper. Four objections, in increasing severity.

**A1. Diffusion.** D ≈ 6 fine cells = 1.5 coarse cells. A trilinear gather at 4h
spreads a coarse cell's content over 8 coarse cells = up to 512 fine cells, in
one step. Fine transport spreads over 8 fine cells. CM12's entire claim is that V
is *less* diffusive than the level set; 4h transport gives it up.

**A2. The reconstruction has no capacity in exactly the case that matters.**
VERIFIED chain: `uvGather` writes `gammaOut = uvTarget(id)`
(`uniform-volume.wgsl.ts:220`), and `uvTarget` (`:222-234`) returns
`fraction · uvOpen(id)` where `fraction` is a phi-derived plane-box fill — so
**γ = 0 wherever every phi sample in the cell is positive.** Figure 7's sheet is
thinner than a cell, so every vertex phi through it is positive
(a sheet of thickness t < h between two vertex planes leaves both planes at
distance ≥ (h−t)/2 > 0). Therefore in the sheet's tile, Σγ = 0 and
"distribute V in proportion to γ" is 0/0. The fallback is uniform spreading,
which takes a 0.5-coarse-cell sheet and lays it out as V ≈ 0.125 through a full
4-cell tile. **One step of that destroys the feature the figure exists to show.**
This is the counter-example; it is not marginal and it is not fixable by a
relay at 4h, because a 4h relay moves V between *tiles*, and the problem is the
distribution *inside* one tile.

**A3. It guts Sec. 3.7.** VERIFIED: `volumeCorrectionDivergence` is
`min(0.5·max(0, V − open), open)/dt` (`webgpu-uniform-reference.wgsl.ts:1036`),
applied per fine cell and **only at `pressureLiquid` cells**
(`webgpu-uniform-pressure-multigrid.wgsl.ts:174-176`). With phi-proportional
reconstruction, V_fine = V_c·γ/Σγ ≤ γ ≤ open whenever V_c ≤ Σγ — so the overfill
divergence is **identically zero** except when a whole 4h tile is over-full. The
trigger becomes 64× coarser and, when it fires, 64× larger. That is worse
physically: local compression at impact — the moment figure 7 is about — would
be silently absorbed instead of relieved. Note this matters *more* at the
defaults, because liquid capacity balancing is **off**
(`uniform-volume-method.ts:24-25`), so Sec. 3.7 is the only remaining relief.

**A4. Sub-tile geometry disappears.** `uvTrace` walks every crossed half-cell so
a characteristic cannot tunnel a thin wall (VERIFIED, `:49-59`). At 4h the walk
is in coarse cells and a 1-cell-thick internal solid is invisible. Figure 7's
only solids are the domain walls, so this is not a blocker *here* — but it makes
option A scene-dependent, which the "judge it on the real scene" rule turns into
"judge it on every scene". Sources are the same story: `dropSource`
(`webgpu-uniform-reference.wgsl.ts:145-160`) is an 8-sub-sample coverage per fine
cell; at 4h a nozzle becomes a 0.2 m blob. Figure 7 has no per-step source
(VERIFIED: the ball is an initial condition, `uniform-volume-initial.ts`), so
this too is a portability objection, not a figure-7 objection.

**What survives from option A:** the *coarse V field itself*, as a cheap
diagnostic and as a tile-class predicate ("this tile holds V"). Computing
V_c = Σ over the tile is one workgroup reduction in the classify pass and is
worth having. It just must not be the transport authority.

### Option B — fine transport on a live set. **Recommend.**

The brief asks whether the donor normalisation creates or destroys volume at the
class boundary. It **creates** it. Proof, from the code:

VERIFIED `uvNormalizeDonors` (`uniform-volume.wgsl.ts:166-171`):
`w[i][k] /= S[donor_k]` with `S[d] = Σ_i w[i][d]` accumulated by `uvSumDonors`
(`:148-152`). VERIFIED `uvGather` (`:214-221`): `V'(i) = Σ_k w[i][k]·V(d_k)`.
Therefore

    Σ_i V'(i) = Σ_d V(d) · ( Σ_i w[i][d] ) = Σ_d V(d) · 1 = Σ_d V(d)

for every donor d that has at least one receiver — conservation comes precisely
from the column sum being normalised to 1.

Now build rows only for receivers in a set F, and leave cells outside F with
their old V (identity). The column sums are then over F only, so after
`uvNormalizeDonors`, `Σ_{i∈F} w[i][d] = 1` for **every** donor d that any
receiver in F samples — including donors **outside** F. Those donors therefore
give away *all* of their V to F, and simultaneously keep all of it under the
identity rule. Total after = Σ_{d ∈ D(F)} V(d) + Σ_{i ∉ F} V(i), which exceeds
the true total by exactly

    Σ_{d ∈ D(F) \ F} V(d).

**V is created, once per step, at every liquid boundary crossing.** The brief's
instinct was right, and the air-side is harmless only because V(d) = 0 there.

**The simplest conservative coupling is not a closure — it is a predicate on V.**
Make the built set satisfy

    V(d) = 0 for every d ∉ F.

Then the created amount is identically zero and the gather is *exactly* the
dense result restricted to F. There is a second, independent requirement: cells
outside F must not need to *receive*, or the front stalls (liquid that should
have moved into a newly wetted cell is instead redistributed back inside F,
because the donor's column still normalises to 1). That requires F to contain the
forward image of the liquid. Combining:

    F ⊇ ( {V ≠ 0} ∪ {band} ) ⊕ (ceil(D) + 1) cells  ≈  ⊕ 2 tiles.

Both conditions are then met and transport is exactly conservative and exactly
positioned, with **no dependency closure, no marking passes, no alternation** —
the thing the prior audit rejected as too complex (its §Q5) is not needed,
because the closure's only purpose was to get the *weights* right at zero-V
donors, and a zero-V donor contributes `w·0 = 0` to every gather regardless of
what its weight is. (The weights of *other* rows do change, through S[d], but
only for columns whose V is 0 — the same induction the prior audit ran for C3 at
`uniform-geometric-work-map-candidates.md:133-149`.)

Caveat, VERIFIED: `uvAddDonor` accumulates through a float compare-exchange loop
(`:27-32`), so column sums are order-dependent and the dense schedule is already
not bit-reproducible. The claim is **summand-identical**, never bit-identical.
Any A/B must compare drift statistics, not bits.

Two bookkeeping items for option B:
- `uvGather` also writes `gammaOut = uvTarget(id)` (`:220`). γ is read by
  `uvPrepareSharpen`/`uvProposeSharpen` (`:278`, `:299-300`). Sharpening's
  admission set is the band, which is inside F by construction — but γ must
  still be *written* in every tile sharpening reads. Since F ⊇ band ⊕ 2, it is.
  Outside F, `uvGather` must write V unchanged and γ = 0 (or skip both and let
  the sharpening map exclude the tile; the latter is what
  `uvCommitSharpen:324-326` already does for its own map).
- The four `clearBuffer(this.conditioningScratch)` calls
  (`webgpu-uniform-reference.ts:1230,1232`) clear all 3N words when transport
  uses the first N. Already flagged in the prior audit; with a live set they
  should clear only live tiles' words, which needs the arena indexed by tile.

### How the current method actually treats "V where phi > 0" (context for both)

VERIFIED and worth stating because it changes what "sheet survival" means:

- `uvPrepareSharpen` (`:275-286`) sets `relay = phi > 0 && desired ≤ 1e-6`, and
  the cell's *need* becomes `dose·max(1.0 − own, 0)` instead of
  `dose·max(desired − own, 0)`. So a positive-phi, zero-capacity cell is both a
  surplus and a large need — it is a **conduit**, not a store.
- `uvProposeSharpen` (`:287-305`) only permits flux `inwardA/inwardB`, i.e. down
  the phi gradient toward the interface.
- Both are gated on `admitted = uvOpen > 0.99999 && abs(phi) < 2.1h` (`:281`).

So: V sitting in a sub-cell sheet **is actively relayed back toward phi < 0** if
it is within 2.1 cells of the interface, and simply sits there (advected only) if
it is further out. And `uvPublish` renders `0.5 − phi/h` — **phi only**. A
sub-cell sheet with phi > 0 everywhere is therefore *conserved in V, invisible in
the render, and absent from the pressure system* (`pressureLiquid(p) ⟺ phi < 0`,
`:241`). That is a pre-existing property of the method at HEAD, not something
this design introduces, but it means the only honest "sheet survival" observable
is `info.volumeCellSum` / `rawVolumeDrift`
(VERIFIED `webgpu-uniform-reference.ts:1621-1623`) plus the spatial distribution
of V — **not** the rendered surface.

---

## Q4. Velocity advection

**Confirmed, and the two-level sampler makes it stronger.**

VERIFIED chain (independently re-derived from source, agreeing with the prior
audit's §Q4):

1. `project` writes `v[axis] = 0` when neither `id` nor `id + e_axis` is
   `pressureLiquid` and the face is open
   (`webgpu-uniform-reference.wgsl.ts:1096-1103`).
2. The only reader of a velocity between advection and projection is
   `divergenceAt`, called from `mgBuildFinestRhs` **only at `pressureLiquid`
   cells** (`webgpu-uniform-pressure-multigrid.wgsl.ts:174-176`), and it reads
   `faceVelocity(q)` and `faceVelocity(q − e_axis)` — a one-cell negative halo
   (`:1016-1033`).
3. MacCormack is not the default (`uniform-volume-method.ts:9`); with it,
   `reverseAdvection`/`correctAdvection` add a dense read of the predicted field
   and this result does not hold.

So the advected velocity of a cell is dead unless the cell or its +x/+y/+z
neighbour is `pressureLiquid`. The live set for `semiLagrangianAdvection` and
`project` is **`pressureLiquid` dilated by one cell — one tile of reach — and no
coarse compute at all**, because every backtrace that leaves the set is answered
by the coarse sampler rather than by fine data that had to be kept alive.
`applyVelocityForces` confirms the physics agrees: gravity is applied only where
`surfaceOccupancy > 1e-5` on either side, i.e. within 2 cells of phi = 0
(`:877-885`), so far-air velocity receives no body force and is pure extension
output.

**Bulk-liquid interior tiles still need fine work.** They are `pressureLiquid`,
so their velocity is a genuine unknown, `project` writes a real gradient there,
and the pressure hierarchy's finest level is fine and dense. There is no version
of this design in which a deep-liquid tile is coarse. For figure 7 that is cheap
— the liquid is 1.6% of cells — but it is the reason "bulk liquid" is a class
for phi/transport purposes only.

**What must stay on the trivial path** in a skipped tile (VERIFIED, each a
separate write in the same kernels):
- `carryBoundaryVelocity(id)` (`:914`), meaningful only on the `id[axis] == 0`
  planes.
- `textureStore(volumeOut, id, volume(id))` — the V copy (`:925`, and again in
  `project:1105`).
- `textureStore(pressureOut, id, vec4f(0.0))` — the pressure clear (`:925`).
- the `min(v, faceVelocity(id))` wall-retention arms at `id[axis] == d−1`
  (`:920-922`), which read the *old* velocity.
- `project`'s special `id[axis] == 0` and `id[axis] == d−1` arms
  (`:1068-1093`), and its `cellOpenFraction < 1` arm (`:1094-1095`).
- inflow/drop cells — `scanExternalActiveSources` (`:1477`) already computes the
  exact test.

---

## Q5. Work model for cm12-figure-7

128³ cells = 2 097 152; 32³ = 32 768 tiles. Counts below are geometric estimates
(INFERRED; a tile intersects a ball of radius R if its nearest point is within R,
approximated as a ball of radius R + ½√3·4h = R + 3.46 cells).

### Falling-ball regime (r = 20 cells)

| Set | Tiles | % of 32 768 | Fine cells | % of dense |
|---|---:|---:|---:|---:|
| liquid interior (φ < −4 cells) | 130 | 0.4% | 8 320 | 0.4% |
| band only (\|φ\| ≲ 4 cells) | ~715 | 2.2% | 45 760 | 2.2% |
| seed (band ∪ liquid) | ~845 | 2.6% | 54 080 | 2.6% |
| **⊕1 tile** (advection, project) | ~1 356 | 4.1% | 86 784 | 4.1% |
| **⊕2 tiles** (phi, transport) | ~2 038 | 6.2% | 130 432 | 6.2% |
| exact mask ⊕4 tiles | ~4 021 | 12.3% | 257 344 | 12.3% |

### Floor-sheet regime (~2 cells deep over the 128×128 floor)

| Set | Tiles | % | Fine cells | % of dense |
|---|---:|---:|---:|---:|
| seed (bottom tile layer) | 1 024 | 3.1% | 65 536 | 3.1% |
| ⊕1 tile | 2 048 | 6.3% | 131 072 | 6.3% |
| ⊕2 tiles | 3 072 | 9.4% | 196 608 | 9.4% |
| exact mask ⊕4 tiles | 5 120 | 15.6% | 327 680 | 15.6% |

### Per stage

Coarse cost is 32 768 thread-instances = **1.6% of one dense pass**, per coarse
pass. With at most 3 coarse passes per step (coarse phi advect, coarse classify,
coarse reduce) that is < 5% of one dense pass — free, as the brief says.

| Stage | Dense now | Two-level (ball / sheet) | Exact mask ⊕4 |
|---|---|---|---|
| extension (5 dense base passes + 2×262 k) | 11.1 M | 0.34 M dense + 5 × fine | 0.34 M + 5 × fine |
| phi advect + redistance (vertex) | 4.2 M | 36 k coarse + 2 × fine⊕2 | 2 × fine⊕4 |
| transport (12 passes + 4 clears) | 25.2 M | 12 × fine⊕2 (1.56 M / 2.36 M) | 12 × fine⊕4 (3.09 M) |
| sharpening (33 passes) | already tiled on the band | unchanged | unchanged |
| SL advection + project | 4.2 M | 2 × fine⊕1 (0.17 M / 0.26 M) | 2 × fine⊕4 |
| `uvPublish`, `reduceDiagnostics` | 4.2 M | stays dense, cheaper per thread | stays dense |
| pressure (~2350 passes, 7 levels) | **untouched** | untouched | untouched |

**Whole-frame estimate.** Taking the prior audit's measured split (pressure 69%,
extension 10%, transport 5.5%, sharpening 2%, phi 0.8%, remainder ~13% of which
2–4 ms is SL+project), the shrinkable part is ≈ 26% of the frame and it shrinks
by ~10×, so the ceiling is a **~23% frame reduction, 1.3×**. INFERRED, and it is
measured on mini64 not figure 7; on figure 7 the liquid fraction is 1.6% instead
of 36%, so the shrinkable stages shrink further *and* pressure's share is likely
higher. Do not sell this as a 10× frame.

### Dispatch form per pass

The repo has both forms already: direct dispatch with a per-thread tile-flag
early exit (`uvSharpenTileActive`, `uniform-volume.wgsl.ts:254-257`) and
indirect dispatch (`dispatchWorkgroupsIndirect` in the balancing path,
`webgpu-uniform-reference.ts:1240`, and throughout the extrapolator). The
relevant standing facts: indirect dispatch costs ~15–25 µs against ~3–6 µs
direct, and a zero-workgroup indirect dispatch is not free; the balancing A/B
favoured direct early exit. Against that, a direct dispatch at 128³ launches
32 768 workgroups of which 94–97% exit immediately.

Recommendation:
- **Direct + early exit** for passes encoded once per step: the classify passes,
  `uvPublish`, `reduceDiagnostics`, `uvGather`, `semiLagrangianAdvection`,
  `project`, `packTransportShell`, the extension seed/resolve/authority.
- **Compacted indirect over a tile list** for passes encoded many times against
  one unchanged list, where the ~20 µs setup amortises: the 12 transport coupling
  passes, the 32 sharpening passes (already tiled; a list would replace the
  per-thread flag), and — if it is ever built — the 168 level-0 pressure
  smoother passes. The compaction itself is one pass plus an atomic counter,
  exactly what `uvClassifySharpenTiles` already does at `:270-272`.
- The kernel change for a list is three lines: `let t = tileList[wg.x]` then
  `id = tileOrigin(t) + local`. `uvCell`/`linearIndex`
  (`webgpu-uniform-reference.wgsl.ts:630-631`) already do the decode arithmetic.

### Memory that could later be released (sizing only)

Per-fine-cell payloads at 128³, computed from the allocation shapes (INFERRED
totals, VERIFIED unit sizes):

| Payload | Bytes/cell | 128³ |
|---|---:|---:|
| nine-donor edge arena (`UNIFORM_VOLUME_EDGE_BYTES = 80`) | 80 | 167.8 MB |
| velocity A–D (rgba32f ×4) | 64 | 134.2 MB |
| FIM scratch (values/distances A,B + resolved ×2, rgba32f ×6) | 96 | 201.3 MB |
| transport shell ×2 (rgba32f, 130³) | — | 70.3 MB |
| volume A/B + gamma A/B (r32f ×4) | 16 | 33.6 MB |
| vertex phi ×2 (r32f, 129³) | — | 17.2 MB |
| **fine-resident subtotal** | | **≈ 624 MB** |
| pressure hierarchy (haloed, 7 levels) | | ≈ 90 MB — stays |
| extension hierarchy (≤ 64³) | | 9.6 MB — stays |

At the two-level design's ~6% tile occupancy, the fine-resident subtotal would
fall to ≈ 37 MB plus a 32 768-entry u32 directory (128 KB). Total ≈ 724 MB →
≈ 140 MB, a ~5× reduction. This is rung-3 sizing only; nothing here designs
sparse storage, and inactive slots in a preallocated pool release nothing.

---

## Q6. Ranking, first experiments, and where it fails

### Ranked by (win × simplicity)

| # | Piece | Win | Simplicity | Physical risk |
|---|---|---|---|---|
| 1 | **Two-level velocity sampler, numerics only** (no schedule change) | 0 (it is the gate) | High — one binding + one branch | Low: far air is non-physical |
| 2 | **Transport live set ⊕2 tiles** (option B) | ~5% frame; 168 → ~10 MB arena later | High — one classify, six gated passes | Medium: the V = 0 predicate must hold |
| 3 | **SL advection + project live set ⊕1 tile** | 2–4 ms (~3%) | High — one shared classify | Low (prior audit proved the dead-value chain) |
| 4 | **Extension live set + coarse 32³ fill** | ~9% frame | Medium — 5 passes + restrict/prolong bookkeeping | Medium: stale `knownMask` if a skipped write is missed |
| 5 | **Coarse phi far field** | ~1% frame; structural enabler for 1–4 | Medium — new coarse advect + two-level `uvPhi` | Medium-high: SL erosion, Lipschitz margin |
| 6 | Coarse V as a *diagnostic/predicate* only | 0 | High | None |
| — | **Coarse V as transport authority (option A)** | large on paper | High | **Unacceptable — §3A2/A3** |
| — | Pressure tile map (C1, not this study) | **5–12% on mini64, far more on fig 7** | Medium | Medium |

### Smallest live A/B for each, on cm12-figure-7

All follow the shipped pattern: a `select` param in
`lib/methods/uniform/uniform-volume-method.ts` (exactly like `sharpeningWorkMap`
at `:30-32`), threaded through `uniformReferenceSolverOptions` into a
pipeline-overridable `override` constant (like `UV_SHARPEN_TILE_WORK`,
`uniform-volume.wgsl.ts:247`) so the dense control is the same module with the
override false and the lookups fold away.

**E1 — `twoLevelVelocitySampler` (do this first).**
Change: expose `hierarchyLevels[1].up` from
`webgpu-uniform-velocity-extrapolation.ts` (add a getter beside
`hierarchyLevelCount` at `:299`); bind it into the main shader's group layouts in
`webgpu-uniform-reference.ts`; add `coarseVelocityComponent` and the branch in
`sampleVelocityComponent` (`webgpu-uniform-reference.wgsl.ts:260`). Gate on the
*existing* sharpening tile map dilated by 2 so no new classify is needed.
**Nothing shrinks** — this A/B is purely about whether substituting the 4h field
in far air perturbs the run.
Observable: `info.rawVolumeDrift` and `volumeCellSum` over 180 frames; the phi
surface at the impact frame; V's spatial extent at the sheet frame. Stage ms
should be *unchanged or slightly worse*. If drift or the surface moves, stop —
every other item depends on this being inert.

**E2 — `transportWorkMap`.** Add `uvClassifyTransportTiles` beside
`uvClassifySharpenTiles` (same file, same word-plane convention; it needs a
second map word range and a `V ≠ 0 || band` workgroup OR plus a 2-tile
dilation). Gate `uvBuildEdges`, `uvSumDonors`, `uvFallback`, `uvNormalizeRows`,
`uvNormalizeDonors`, `uvGather` (all in `uniform-volume.wgsl.ts:137-221`); the
encode sites are `webgpu-uniform-reference.ts:1229-1234, 1253`.
Observable: `volumeCellSum` must track the dense arm to float noise; transport
stage ms from the existing `UNIFORM_VOLUME_PHASE.coupling` seam; and a deliberate
negative control — run it with a **1-tile** dilation and confirm the drift turns
positive (the double-count of §3B), which proves the mechanism rather than the
number.

**E3 — `advectionWorkMap`.** One shared classify for
`semiLagrangianAdvection` + `project` on `pressureLiquid ⊕ 1 tile`, with the
trivial-path writes listed in Q4. Requires E1 (otherwise the reach is D, not 1).
Observable: `UNIFORM_ADVANCE_PHASE.advectionCorrection` and
`pressureProjection` seam ms; max speed (`reductions[2]`); divergence residual
from the multigrid's fine-residual measure.

**E4 — `extensionWorkMap`.** Gate authority/seed/resolve/pack/2h-prolong/h-prolong
on the same map ⊕1, keep every restrict and every level ≤ 32³ dense.
Observable: `uniformFIMExecutedPasses` and terminal-face count must be unchanged;
`narrow-band-front` and `hierarchy-fill` seam ms.

**E5 — `coarsePhiFarField`.** Last, because it is the riskiest and the smallest
direct win. Observable: phi contour overlay against the dense arm at frames
around impact; `reductions[0]` (the `surfaceOccupancy` sum) as a scalar surface
proxy; sheet extent.

### Where this is likely to fail — stated plainly

1. **Option A on figure 7.** §3A2. γ = 0 through a sub-cell sheet, so
   phi-proportional reconstruction is 0/0 and the fallback smears the sheet over
   a 4-cell tile in one step. This is the strongest single argument in the study
   and it is against the owner's literal proposal.
2. **The frame is pressure.** ~69% (mini64, measured). Nothing here touches it.
   Figure 7's 1.6% liquid occupancy makes the pressure tile map the single
   largest available win on this scene, by a wide margin.
3. **A 2-tile reach is not obviously enough for phi in a fast regime.** At impact
   the ball's velocity reverses over one step; the per-tile max-speed dilation
   must use the *pre-advection* velocity, which is a lower bound on the
   post-impact spread. If the front outruns the mask, liquid arrives in a
   non-resident tile and the band classification misses it next step.
4. **The class boundary is a discontinuity.** O(h²·∂²v) in velocity and phi. It
   is harmless only while the boundary sits ≥ 8 cells from phi = 0. Every stage
   that widens its own reach (a longer trace, a wider redistance, MacCormack)
   eats that margin silently.
5. **Boundary-plane kernels are a shell of 6 144 tiles = 18.75%.** The floor
   alone is 1 024. `carryBoundaryVelocity`, `project`'s `id[axis] == 0/d−1` arms,
   `uvReleasedWalls`, `uvClosedWallPhi`. The arguments in Q2 say they do *not*
   force residency, but this is the likeliest place to be wrong, and being wrong
   costs a fifth of the domain.
6. **The late regime erases the win.** Once the sheet covers the floor and
   splashes, the ⊕2 live set is 9.4% and climbing. Measure at t = 3 s, not only
   at t = 0.5 s.
7. **Nothing here is bit-exact**, by construction (float CAS column sums,
   `uvAddDonor:27-32`) and by intent (the brief says exactness is not required).
   Every gate must therefore be a drift/statistics gate, and the "dense control"
   arm must be run in the same session — a single-run 2–3% difference measures
   nothing.
8. **Two-level phi and two-level velocity must land together.** Either one alone
   accumulates seam error in the far field step after step. That makes the
   minimum shippable increment larger than E1 alone suggests.

---

### Files that would change

- `lib/methods/uniform/uniform-volume.wgsl.ts` — classify entries, tile-map word
  ranges (`UNIFORM_VOLUME_SHARPEN_TILE_MAP_WORD = 8` at `:18` is the pattern),
  gated transport/phi kernels, `uvPhi` two-level branch.
- `lib/methods/uniform/webgpu-uniform-reference.wgsl.ts` — `sampleVelocityComponent`
  (`:260`), `pressurePhi`/`surfaceValue` (`:163-165`, `:234-238`) if `uvPhi` goes
  two-level, `semiLagrangianAdvection` (`:913`), `project` (`:1060`).
- `lib/methods/uniform/webgpu-uniform-reference.ts` — bind-group layouts for the
  coarse texture, `encodeGeometricVolume` (`:1223-1276`), the advance encode
  order (`:1479-1560`), new pipeline-override compilations (`:840-860`).
- `lib/methods/uniform/webgpu-uniform-velocity-extrapolation.ts` — expose the 32³
  `.up` texture; gate authority/seed/resolve/pack/2h+h prolongs (`:343-400`).
- `lib/methods/uniform/uniform-volume-method.ts` — one `select` param per
  experiment, mirroring `sharpeningWorkMap` at `:30-32` and `:66`.
- `lib/methods/uniform/uniform-volume-stages.ts` / `uniform-volume-pipeline.ts` —
  stage/seam registration for any new timed phase.
