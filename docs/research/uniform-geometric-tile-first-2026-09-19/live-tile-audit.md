# One shared live-tile map for Uniform Geometric — read-only source audit

Target: `cm12-figure-7` (128³, ball r=20 cells at y=90, empty closed tank —
`lib/core/cm12-paper-scenes.ts:120-123,547-549`; h = 0.05 m, dt = 1/30 s, g = 10
m/s² — `:37-39,240`; `top:"closed"`, `surfaceTension_N_m: 0`, free-slip,
`depthBoundary` NOT symmetry — `:219-243`).

Defaults verified at `lib/methods/uniform/uniform-volume-method.ts:9-12,22-32,59-69`:
semi-Lagrangian, redistance on, balancing off, sharpening on with the 4h map,
`extensionFrontSweeps: 8`, `activeRegion:false`, `gammaDiffusionIterations:0`,
`solidExcessCorrection:false`, `densityPostProcessing:false`.
`activeRegionEnabled` is therefore false (`webgpu-uniform-reference.ts:433`), so
`activeId(gid) == vec3i(gid)` and every `this.run` is a plain full-lattice
dispatch (`:1124-1128`); the extrapolator and the multigrid are both constructed
with `activeDispatch: undefined` (`:587,666`), so `hierarchyActiveId` collapses
to `vec3i(gid)` (`webgpu-uniform-velocity-extrapolation.wgsl.ts:65-73`).

Notation: **D** = per-step Chebyshev displacement in cells = `u_max·dt/min(h)`.
At figure-7 impact (v≈9.5 m/s) D ≈ 6.3. All reaches below are in cells from the
thread's own cell (or vertex) unless stated.

---

## A. Per-pass read footprint, write set, and consumers (encode order of `advanceTo`)

`advanceTo` is `webgpu-uniform-reference.ts:1276`. Bind groups at `:663-771`.

| # | Pass | src | Read reach | Writes | Who reads the write |
|---|---|---|---|---|---|
| 0 | `clearBuffer(reductions)`, `clearBuffer(rigidExchange)` | `:1363-1364` | — | `reductions[0..7]`=0 | `reduceDiagnostics`, `readStats` |
| 1 | `encodeActiveRegion` | `:1365` | **not encoded** (`:1145`) | — | — |
| 2 | `buildExtrapolationAuthority` | wgsl `:583`, fn `:578-582` | **0** cells. `pressurePhi(id)` = `uvPhi(cell centre)` = the cell's own 8 vertices (`uniform-volume.wgsl.ts:35-42`); `faceOpenFraction(id,axis)` = 4 solid quadrature points at ±0.35h (`wgsl:504-524`) | `surfaceA` = ρ′ = `0.5-φ/h`; `velocityD` = (openX,openY,openZ,0) | extrapolator `density()` (`extrap.wgsl:109-118`) and `openBaseFace()` (`:99-101`) ONLY. Both are **threshold** consumers: `sourceFace` needs `density > 0.5` ⟺ φ<0, `openBaseFace` needs `>1e-5`. |
| 3 | `seedActiveFront` | extrap.wgsl `:148-170` | ±1 (`oneNeighborTouchesSource` `:131-139`) | `valuesA`, `distancesA`, `convergence.activeA` | `updateActiveFront`, `resolveConvergedFront` |
| 4 | 8 × (`updateActiveFront` indirect + 1-thread `prepareActiveDispatch`) | `:287-326`, `:340-359`; encode `webgpu-uniform-velocity-extrapolation.ts:368-375` | ±2 per sweep (`activatedByConvergedUpwindNeighbor` `:266-285` calls `godunovDistance` at p±1, which reads p±2). Clipped to `accurateBandDistance()` = 2·max(h) (`:75-78,304`). **Total reach from a liquid cell = 4 cells** — verified by the archived measurement, `docs/benchmarks/uniform-geometric-extension-tile-work-2026-09-19.md`: the naive "one min(h) hop" bound is wrong, a 3-axis Godunov hop can be as small as min(h)/√3, so the band admits `floor(2√3·max/min)` = 3 hops, + 1 for the source face = **4 cells, exactly one tile, zero margin** | `valuesA/B`, `distancesA/B` (ping-pong) | each other, then `resolveConvergedFront` |
| 5 | `resolveConvergedFront` | `:363-372` | 0 | `resolvedValues`, `resolvedDistances` | `restrictKnownVelocity` level 0, `prolongUnknownVelocity` finest (`secondaryIn`) |
| 6 | 6 × `restrictKnownVelocity` | `:451-475`; encode `:378-387` | level 0 reads 8 fine taps of `resolvedValues` per coarse cell (`hierarchyComponentSample:387-415`); levels ≥1 read their own finer level. **Non-local by construction** | `level.down` | next restrict, then prolong |
| 7 | 6 × `prolongUnknownVelocity` | `:477-495`; encode `:388-398` | 8 coarse taps (footprint = 2^L fine cells) + own `secondaryIn` | `level.up`, finest target = **`valuesA`** (`.ts:252-253,267`) | next prolong, then `packTransportShell` |
| 8 | `packTransportShell` | `:497-514`; encode `:399-407` | 0 (`primaryIn`=`valuesA` at p, `faceOpenIn` at p) | `transportA[p+1]` xyz = extended velocity, w = known\|open bits | `sampleVelocityComponent` (`wgsl:260-269`), `sampledFaceVelocity` (`:246-249`) | 
| 9 | `uvAdvectPhi` (**vertex** dispatch, `(n+1)³`) | `uniform-volume.wgsl.ts:119-123`, encode `ref.ts:1228` | `uvTrace(p,dt)` (`:51-60`): shell within **D+2** (two `sampleVelocity`, each an 8-corner stencil of radius 1.5 about points within D of p); `cellOpenFraction` within **D** along the chord; `uvPhi(end)` → φ within **D+1**. `uvClosedWallPhi` (`:101-118`, wall vertices only) steps 1 cell inward then traces → φ within **D+2**. `uvSourcePhi` (`:61-73`) is analytic, reach 0. **`uvReleasedWalls` (`:75-97`) is the exception: for the released +Y ceiling it probes `probe[1]=ny-1e-4` and reads `velocity(cell)` at `(p.x±0, ny-1, p.z±0)` for EVERY vertex in the domain — an unbounded read along +y.** In figure-7 only the +Y wall is released (`released = -1·g > 0.5|g|` is true only for axis 1, upper; x/z have `acceleration=0`) | `vertexPhiScratch` | `uvRedistancePhi` |
| 10 | `uvRedistancePhi` (vertex) | `:124-135`, encode `:1229` (group `phiReverseGroup`, `ref.ts:766-769`, so in=`vertexPhiScratch`, out=`vertexPhiTexture`) | Newton search clamped to the box `p±4` (`:132`); `uvGradient` probes q±0.25 (`:43-48`); `uvPhi` of that reads `floor(·)` and `floor(·)+1` ⇒ **±5 vertices** | `vertexPhiTexture` = φⁿ⁺¹ | everything below + the next step |
| 11 | `uvBuildEdges` | `:137-147`, encode `:1232` | `uvTrace(id+0.5)` → donors at `floor(departure)+corner` ⊆ **D+1**; `uvOpen(id)`, `uvOpen(q)` | `uvEdges[i]` (own row: 9 donors + 9 weights) | `uvSumDonors`, `uvNormalize*`, `uvGather` |
| 12 | `clearBuffer(conditioningScratch)` (**all 3N words**), `uvSumDonors`, `uvFallback` | `:1233`; wgsl `:148-157` | sum: own row, **scatters** to donors within D+1 via float CAS (`uvAddDonor:27-32`). fallback: own word `sharpenDeposits[i]` | column sums; `uvEdges[i].weight[8]` | rounds below |
| 13 | 3 × (`uvNormalizeRows`, clear 3N, `uvSumDonors`, `uvNormalizeDonors`) | `:1234-1237`; wgsl `:158-171` | rows: own row only. donors: `sharpenDeposits[donor]` within **D+1** | `uvEdges[i].weight[k]` | next round, `uvGather` |
| 14 | `uvGather` | `:213-221`, encode `:1261` | 9 donors' `volume()` within **D+1** (from `volumeA`); `uvTarget` (`:222-234`) samples `uvPhi` at `id+0.25+0.5·corner` ⇒ the cell's **own 8 vertices** | `volumeB`, `gammaB`=γ target | sharpening (γ via gammaA), pressure (`volumeB`), publication |
| 15 | copy `gammaB → gammaA` | `:1262` | whole lattice | `gammaA` | `uvPrepareSharpen`/`uvProposeSharpen` (`gammaIn`) |
| 16 | `uvClassifySharpenTiles` + 8 × (prepare, propose, limit, commit) | `:247-332`, encode `:1265-1272` | **1 cell** (`uvProposeSharpen` reads `uvEdges[j]` at `id+e`, `uvLimitSharpen` at `id-e`, `uvCommitSharpen` at `id±e`), all gated by `uvSharpenTileActive` | `uvEdges[i].weight[0..5]`, `.padding` (**scratch**), `volumeA↔volumeB` | each other; final V lands in `volumeB` (8 rounds: B→A, A→B, …, A→B) |
| 17 | copy `volumeB → volumeA` | `:1479` | whole lattice | `volumeA` | redundant — rewritten identically by #18 and #20 |
| 18 | `semiLagrangianAdvection` | wgsl `:912-926`, encode `:1501` | 3 × `advectVelocityComponent` (`:419-443`) → `departurePoint` (`:323-333`): 32 substeps, each ≤1.5 cells, **total path ≤ D** because `Σ stepSeconds = dt` and \|u\| ≤ u_max (see C); `sampleVelocityComponent` adds 1.5 ⇒ shell within **D+2**. `clipDepartureAtSolid` (`:391-401`) 8 bisections of solid tests along the chord. `applyVelocityForces` (`:876-910`): `surfaceOccupancy(id)`, `(id+ex/ey/ez)` ⇒ φ within **1 cell**; `velocityLaplacian` ±1; **`curvatureAt` is not evaluated** (σ=0 ⇒ `sigmaOverRho>0` false, `:894-895`) | `velocityB`, `volumeA`(=copy of `volumeB`), `pressureB`(=0), `boundaryVelocityB` (carry) | `velocityB`→`divergenceAt`/`project`; `pressureB` → **nobody** (geometric reads pressure only via `projectPressureValue` bound to the MG texture, `ref.ts:731`; `pressureValue` at wgsl `:196` has no callers) |
| 19 | `pressureMultigrid.encode` | `:1526` | see E | | |
| 20 | `project` | wgsl `:1059-1106`, encode `:1528` | **1 cell**: `projectPressureValue(id / id±e)`, `pressureLiquid(id)`/`(neighbor)` ⇒ φ at two cell centres ⇒ vertices `id..id+2`; `pressureFaceData` solid quadrature; `velocity(id)`, `boundaryVelocity(id)`. Boundary arms: `id[axis]==0` (halo `id-e`, **falls through** to the generic arm — no `continue`) and `id[axis]==d-1` (`:1074-1093`, `continue`) | `velocityA`, `boundaryVelocityA`, `volumeA`(=`volumeB`) | next step's extension seed, `reduceDiagnostics`, `uvReleasedWalls` |
| 21 | `uvPublish` | `:333-339`, encode `:1546` (group `wallFilmResolveGroup`: in `volumeA`, **out `surfaceB`**, γ out `gammaB`) | **0** (own 8 vertices) | `surfaceB` = `0.5-φ/h`, `gammaB` = `uvOpen` | **renderer** (`surfaceFieldTexture` = `surfaceB`, `ref.ts:282-284`, consumed at `lib/core/webgpu-renderer.ts:1930`); `gammaB` is the grid overlay's `openFraction` (`ref.ts:511`, `lib/core/webgpu-grid-overlay.ts:1641-1642`) |
| 22 | `reduceDiagnostics` | wgsl `:1557-1558`, encode `:1561` | **0**: `surfaceOccupancy(id)` (own vertices), `volume(id)` from `volumeA`, `faceVelocity(id)` from `velocityA` | `reductions[0..3]` | `readStats` (`:1607-1629`) |

Dead at these defaults: `encodeActiveRegion`, balancing (`:1239-1260`), `coupleRigid`,
`scatterSolidExcess`/`resolveSolidExcess`, γ diffusion, post-process, all
`symmetryStageAudit*` copies (env-gated, `:594,602`).

---

## B. Minimum reach for exactness

### B.1 The transport closure, redone with the fallback hop

Let `A1 = {V ≠ 0}`. `R(X)` = receivers with an edge into X, `D(Y)` = donors of Y.
From `uvBuildEdges:140-146`, a receiver's donors are `floor(trace)+corner`, so:

- **`R(X) ⊆ X ⊕ (D+1)`** and `D(Y) ⊆ Y ⊕ (D+1)` (VERIFIED from `:140-146`).
- **`D(R(X)) ⊆ X ⊕ 1`** (the 8 donors are corners of one cell — the prior doc's
  correction, confirmed).

Backwards from `uvGather:216`, whose value needs `w3[i][k]` only for donors with V≠0:

```
Col3 = A1 ;  Row2 = R(Col3)
Col2 = D(Row2) ⊆ A1⊕1 ;  Row1 = R(Col2)
Col1 = D(Row1) ⊆ A1⊕2 ;  Row0 = R(Col1) ⊆ A1 ⊕ (D+3)
```

`Row0` is the set of rows whose **round-1 row sum** must be right. Their row sum
includes `weight[8]`, set by `uvFallback` from the **column** sum at `i`
(`:154-157`), so for every `i ∈ Row0` every `j ∈ R(i)` must have been built and
summed. `R(Row0) ⊆ Row0 ⊕ (D+1)`.

> **The fallback hop is required and cannot be dropped by "keep the first
> `uvSumDonors`+`uvFallback` dense".** A dense first sum reads `uvEdges` rows
> that a restricted `uvBuildEdges` never wrote — and those rows are not merely
> stale transport rows, they are **sharpening scratch**: `uvPrepareSharpen`
> overwrites `weight[3],[4],[5]`, `uvProposeSharpen` overwrites `weight[0..2]`,
> `uvLimitSharpen` overwrites `.padding` (`:283-285,293-303,314-315`). A dense
> `uvSumDonors` would CAS that garbage into the column sums of live cells.
> VERIFIED.
>
> Cheapest exact repair: `encoder.clearBuffer(this.volumeEdges)` once per step
> (160 MiB fixed-function fill at 128³), or have skipped `uvBuildEdges` lanes
> store the canonical all-zero row (`donor[k]=index, weight[k]=0`) — the same
> "write the inert constant rather than skip" pattern the archived extension map
> had to use (`docs/benchmarks/uniform-geometric-extension-tile-work-2026-09-19.md`,
> "A far-tile thread … writes the inert constant … It writes rather than skips").
> With the arena canonically zeroed, `uvAddDonor` early-returns on `value==0`
> (`:28`) and unbuilt rows contribute exactly nothing. Then the build/sum set
> can be exactly `R(Row0)`.

**Transport build set: `B ⊆ A1 ⊕ (2D + 4)`.** Without the fallback hop it would
be `A1 ⊕ (D + 3)`.

### B.2 Phi chain

φ at end of step *n* must be exact wherever a step-(n+1) live consumer reads it.
The deepest φ consumer is `uvAdvectPhi` itself (D+2), and `uvRedistancePhi`
reads `vertexPhiScratch` at ±5, so the advect set must cover the redistance set
⊕ 5. Self-consistently, with one shared live set `L` of reach ρ around the seed:

```
redistance set ⊇ L ⊕ (D+2)      advect set ⊇ L ⊕ (D+7)
```

### B.3 Resulting formula

```
ρ(D) = max( 2D + 4        [transport closure incl. fallback hop]
          , D + 7         [phi advect ⊃ redistance ⊃ consumers]
          , D + 2         [semi-Lagrangian / shell sampling]
          , 4 )           [FIM narrow band, archived bound]
     = 2D + 4 cells for D ≥ 3,  plus 3 cells so the phi arm's D+7 also fits
```

**ρ = 2D + 7 cells**, rounded up to whole tiles and +1 tile for the seed's
tile-grid rounding.

**At D = 6: ρ = 19 cells → 5 tile dilations + 1 rounding tile = 6 tiles (24 cells).**

Occupancy sanity check for figure-7 at t=0: seed ≈ ball(20) ⊕ band(4) = r 24
cells = r 6 tiles; ⊕ 6 tiles → r 12 tiles → (4/3)π·12³ ≈ 7 240 of 32 768 tiles
= **~22 % live, 78 % skippable** — before impact. After the sheet spreads across
the floor and up the walls this collapses; nothing in the source bounds it.
(INFERRED — no census exists for this scene; `docs/research/uniform-geometric-empty-air-census.json`
covers only the two dam scenes.)

---

## C. The velocity-extension hierarchy

**C.1 Does an unknown value inside `L` depend on a known value outside `L`?**
**No — VERIFIED, with one condition.** The only way a value becomes *known* is
`seedActiveFront` (source faces, `:159-161`) or `updateActiveFront`
(`:304-308`), both clipped to the 2-cell accurate band; `restrictKnownVelocity`
and `prolongUnknownVelocity` only ever renormalise **known** values
(`hierarchyComponentSample:407`, `hierarchyKnownContribution:420`). Level 0's
restrict source is `resolvedValues`, whose known bits exist only inside the band
(≤ 4 cells from liquid). So every coarse value is a convex combination of band
values; nothing outside `L` can inject a value. The *spatial* footprint is
another matter — see C.3.

**C.2 Can a stale finest texture leak a stale `known` bit into the restriction?
YES — this is the one real hazard, and it is one-sided.** `valuesA` is
simultaneously the FIM ping-pong side A *and* the finest prolong target
(`webgpu-uniform-velocity-extrapolation.ts:190,252-253,267`), so after a step it
carries `known=1` on **every** face. `resolvedValues` does not (it is written
only by `resolveConvergedFront:363-372`, copying a seed/update side whose
far-air entries are `(0,0,0,known=0)`). Since restrict level 0 reads
`resolvedValues`, not `valuesA`, the leak is confined to **tiles that were live
last step and are not live this step**: their `resolvedValues` still hold real
band values with `known=1`, while the dense schedule would have re-seeded them
to zero. Remedy: run seed/resolve on `live(n) ∪ live(n-1)` — the deactivating
tile then gets exactly the dense `(0,0,0,0)` / `(INF,INF,INF,0)` write once, and
stays there. **One-step hysteresis is exact here** because the far-field value is
a *constant*.

**C.3 Can the coarse levels be restricted to the live footprint? NO — and they
should not be.** A live fine cell's finest prolong reads 8 taps of level 1;
level 1 reads level 2; a level-L tap covers 2^L fine cells, so the *positions*
read walk far outside `L` even though the *values* there derive only from the
band. Conversely restrict level 1 reads 8 fine taps of `resolvedValues` per
coarse cell, so `resolvedValues` must be dense-correct everywhere — which C.2's
hysteresis gives. **Leave levels ≥ 1 dense** (they are 64³+32³+…+1³ ≈ 1/7 of the
finest lattice) and tile only: authority, seed/update/resolve, the finest
prolong target, and pack. This is exactly the shape the archived extension map
took, and it is the reason that map explicitly "left the hierarchy
restrict/prolong and the transport-shell pack untouched".

**C.4 Max principle — VERIFIED, and it is what makes D computable.**
Every producer of an extended value is a convex combination:
- `upwindExtensionValue:224-249` returns `Σ_a w_a·avg(minimisers_a) / Σ_a w_a`
  with `w_a = (solved − min)/h²  > 0`; each `avg` is over 1–2 neighbour values.
  A neighbour's value is read only where its distance is finite, and finite
  distance ⟹ `known` (`:303-307` sets both together; seed sets `d=0` with the
  known bit, `:159-162`). So `neighborValue` never contributes a spurious 0.
- `hierarchyComponentSample:413-414` and `hierarchyCorrespondingCellSample:445-448`
  are renormalised weighted means of known values.
- `prolongUnknownVelocity:487` never overwrites a known component;
  `packTransportShell:503-509` copies or writes 0 at closed faces.

⇒ `|shell component| ≤ max over source faces |velocityA component|`, and source
faces are those adjacent to a φ<0 cell (`sourceFace:114-118` with
`CM12_LIQUID_ISOVALUE = 0.5` ⟺ `0.5−φ/h > 0.5` ⟺ φ<0). Hence
`|u|_shell ≤ reductions[2]` of the previous step (`reduceDiagnostics:1558`
`atomicMax(bitcast(length(faceVelocity(id))))` over `velocityA`).

> **Trap:** `reductions` is cleared at the head of every step
> (`ref.ts:1363`) before any pass could read it. A GPU-resident D therefore
> needs either that clear moved after the classify/dilate passes, or a second
> persistent word. VERIFIED.

---

## D. Stale-state inventory (ping-pong parity per step, and the remedy)

| Resource | Parity per step | Newly-live tile reads stale? | Cheapest exact remedy |
|---|---|---|---|
| `vertexPhiTexture` / `vertexPhiScratch` | **no cross-step flip**: advect writes scratch, redistance writes texture (`ref.ts:1228-1230`, groups `densityTraceGroup`/`phiReverseGroup`) | scratch is stale by ≥1 step for a skipped vertex; redistance reads it at ±5 | advect set ⊇ redistance set ⊕ 5. If `redistance:off`, the `copyTextureToTexture` at `:1230` must be tiled too or stay dense |
| `volumeA` / `volumeB` | A→(gather)→B→(sharpen ×8)→B→(copy `:1479`)→A→(SL)→A→(project)→A | far-air V is exactly 0 and `uvGather` reproduces 0 exactly (Σw·0); a tile only deactivates when V=0 ∧ φ≥R·h | **dead on entry** — no remedy needed, *provided* the seed keeps `V ≠ 0` as stated. See the residue risk below |
| `gammaA` / `gammaB` | γB written by `uvGather`, copied to γA, then **overwritten by `uvPublish`** with `uvOpen` | γA read only inside admitted sharpen tiles ⊆ L | keep the `:1262` copy dense (1 MiB at 64³, 8 MiB at 128³) or tile it with the sharpen map |
| `velocityA` | written **only** by `project` | YES — a tile that stops being live keeps its last real projected velocity forever, while dense would zero it (`project:1103`). Corrupts `reduceDiagnostics`'s `maxSpeed` (i.e. D itself) and `uvReleasedWalls`' ceiling probe | **one-step hysteresis `live(n) ∪ live(n-1)`** — `project`'s far-air output is the *constant* `v=0` + a V copy, so one write is permanently correct |
| `velocityB` | `semiLagrangianAdvection` → `project` reads it | far-air `velocityB` is read by `divergenceAt` only at liquid cells and their −1 face (`wgsl:1022-1027`, called from `mgBuildFinestRhs:175`) ⊆ L⊕1 | dead outside L⊕1 |
| `velocityC`, `velocityD` | SL path: `velocityC` unused; **`velocityD` is the faceOpen authority**, not a velocity | `faceOpenFraction` is a pure function of static solids → constant | write once; skipping later writes is exact |
| `transportA` | rewritten each step by `pack`; outer padded shell (idx 0 and d+1) is **never** written and reads as the zero-init value via `sampleVelocityComponent`'s `clamp(..., lower=-1, dims-1)` (`wgsl:261-266`) | interior stale outside L | covered by the D+2 dilation + C.2 hysteresis |
| FIM `valuesA/B`, `distancesA/B` | intra-step; **`valuesA` doubles as the finest prolong target** | see C.2 | write the inert constant on the deactivation step |
| `resolvedValues` / `resolvedDistances` | written once per step by `resolve` | **YES** — the `known`-bit leak of C.2 | one-step hysteresis |
| `uvEdges` arena | persistent; carries **sharpening scratch** into the next step's `uvSumDonors` | **YES, catastrophically** | zero-clear the arena per step, or canonical all-zero row from skipped `uvBuildEdges` lanes (B.1) |
| `conditioningScratch` | `clearBuffer(...)` with **no range** at `:1233` and `:1235` clears all `3N` words (`uniform-host-allocation.ts:49`) | — | **a live map stored in this buffer would be wiped 4× per step.** The sharpen map survives only because it is written *after* the last clear (`:1266-1267`). Give the live map its own buffer (as the archived extension map did: "a dedicated `8 + 2N`-word buffer on a new binding") |
| `pressure` (MG level 0) | `mgBuildFinestRhs:181` writes `pressureOut = 0` densely every step | **no warm start exists** | none needed |
| `pressureA` / `pressureB` | `semiLagrangianAdvection:925` clears `pressureB` | nothing in the geometric path reads them (`projectPressureValue` is bound to the MG texture at `ref.ts:731`; `pressureValue` has no callers) | **dead work** |
| `boundaryVelocityA/B` | **flips every step**: SL carries A→B, `project` writes A from B (`ref.ts:704-708,731`) | YES — a skipped `carryBoundaryVelocity` leaves B two steps old and `project` reads B | only the three `id[axis]==0` planes are touched (`wgsl:189-194`); keep the carry dense (O(N²)) |
| `reductions` | cleared at step head | — | see C.4 trap |

**External readers.**
- **Renderer**: `surfaceFieldTexture = surfaceB` (`ref.ts:282-284`) → `webgpu-renderer.ts:1930`.
  Stale far-air `surfaceB` = `0.5 − φ/h ≤ 0.5 − R` (≤ −3.5 at R=4), so it can
  never cross the 0.5 liquid isovalue — **no phantom liquid** — but it is not
  bit-identical to dense. `uvPublish` costs ≈0.05 ms; **keep it dense.**
- **Field-view overlays** read `vertexPhi` and `gammaB` directly
  (`webgpu-grid-overlay.ts:1641-1642` via `denseLevelSetVolumeSource`, `ref.ts:511`).
  Tiled φ makes far-air overlay values stale. Cosmetic, but it is a differing read.
- **Diagnostics** (`reduceDiagnostics:1558`): `represented = surfaceOccupancy` is
  `clamp(0.5 − φ/(4h_y), 0, 1)` = **exactly 0** once φ ≥ 2h_y (`wgsl:167-171`), and
  `u32(0·2048 + 0.5) = 0`; likewise V contributes 0 for any `V < 2.4e-4`. Air
  contributes exact zeros ✓. `reductions[1]` (front) is gated on `surfaceLiquid`
  ✓. `reductions[2]` (maxSpeed) is the one that breaks without the `velocityA`
  remedy.
- **Harness/conformance**: `readStats` (`:1602-1629`) reads only `reductions`,
  MG diagnostics, extrapolator convergence and `activeRegion`.
  `tests/uniform-volume-dawn.test.ts:50` and
  `tools/probe-uniform-volume-brick-work-dawn.ts:32` read
  `denseLevelSetVolumeSource` (φ + `gammaB`) over the **whole** lattice.
- **Rigid coupling / terrain / sources**: not encoded in figure-7; `dropSource`
  (`wgsl:145-160`), `inflowSweptPlugSource` and `applyInflow*Velocity`
  (`lib/core/inflow-boundary.ts:38-69`) are analytic functions of the cell's own
  index — **reach 0**, so seeding tiles they touch is sufficient.
- **`symmetryStageAudit*`**: env-gated (`ref.ts:594,602`), copies whole textures;
  would need to be disabled or left dense.

**Two named hazards that are not in the resource table:**

1. **`uvReleasedWalls` is globally non-local along +y.** For every vertex it
   reads `velocityA` at `(·, ny−1, ·)` (`uniform-volume.wgsl.ts:86-92`). The
   *value* only changes the result within ~D cells of the ceiling, but the
   *read* is unconditional. In figure-7 the ceiling is never wet, so
   `velocityA` there is zero-init and the read is benign — but the moment any
   scene splashes the lid and then drains, the stale ceiling velocity is read by
   every vertex in that column. The `project` deactivation clear (D) fixes it.
2. **V-residue keeps tiles live forever.** The seed requires `V ≠ 0` and the
   doc forbids a cosmetic threshold. `uvGather` produces exactly 0 when all
   donors are 0, so most residue vanishes exactly — but `uvFallback`'s self-edge
   (`:154-157`) deliberately *retains* V in cells no receiver takes from. If
   that leaves a growing tail of 1e-30 cells, the live set behaves like the
   monotone AABB it was meant to replace. Measurable on the CPU census before
   building anything.

---

## E. Interaction with a dense pressure stage

What pressure reads outside liquid:

| Read | src | Outside the liquid set? | Safe to leave unwritten outside L? |
|---|---|---|---|
| `pressurePhi(simulation)` → `mgPhiOut` at **every interior cell** | `mgBuildFinestTopology:154-156` | YES, everywhere | **NO — this is the counter-example, see F.0** |
| `cellOpenFraction`, `pressureFaceVolumeFraction` | `:155,161-163` | everywhere | yes: pure functions of static solids |
| `pressureLiquid(simulation)` (sign of φ) | `mgBuildFinestRhs:174` | everywhere | yes: stale far-air φ ≥ R·h > 0, so the **bit** is unchanged |
| `cellInsideSolid` / `cellInsideTerrain` → `minimum` | `:173` | everywhere | yes: static |
| `divergenceAt(simulation)` → `faceVelocity(id)`, `faceVelocity(id−e)` | `:175`, `wgsl:1022-1027` | **only at `pressureLiquid` cells** ⇒ a one-cell negative halo of the liquid set | yes |
| `volumeCorrectionDivergence` → `volume(id)` from `volumeB` | `:175`, `wgsl:1036` | liquid cells only | yes |
| `mgPressureOut = 0`, `mgMinimumOut` | `:181-182` | dense, written by pressure itself | yes — no warm start to protect |
| the `pressureOut = 0` clear inside `semiLagrangianAdvection` | `wgsl:925` | writes `pressureB` | **dead** for geometric (see D) |

So: leaving **velocity and V** unwritten outside `L` changes nothing pressure
reads. Leaving **φ** unwritten outside `L` does.

---

## F. Verdict

### F.0 The counter-example: a tiled φ is not bit-exact while the pressure pyramid is dense

`mgBuildFinestTopology:156` publishes `pressurePhi` as a **value** at every
interior cell. `mgDownsampleTopology:190-206` then averages 8 children per level;
the sign-aware positive-only rule applies only when `destination ≤ 2`
(`usePositive = mixed && control.x ≥ control.y`, with
`control.x = M − dest`, `control.y = M − UNIFORM_CM11A_PHI_PRESERVATION_LEVELS`,
`PHI_PRESERVATION_LEVELS = 2`, `webgpu-uniform-pressure-multigrid.ts:14,471-473`)
— at destinations ≥ 3 it is the **plain 8-child mean**. That coarse φ feeds
`mgExtrapolatePhiOneCell:309-319` and then `mgBakeCoefficients:321-330`, whose
coefficient is `vf/(h²·θ)` with
`θ = clamp(|φ_liq|/(|φ_liq|+|φ_air|), 0.05, 1)` (`mgCoefficientRaw:125-128`,
`cm12GhostFluidTheta`, `lib/core/cm12-numerics.ts:171-173`).

At a coarse level L a cell spans 2^L fine cells, so the φ of an air coarse cell
*adjacent to a liquid coarse cell* is a mean over fine cells up to ~2^(L+1)
away — 64–128 cells on 128³, far beyond any ρ we can afford. **A far-air φ that
differs from the dense schedule therefore changes coarse-level pressure
coefficients, hence the multigrid iterate, hence the projected velocity.** The
`θ` clamp at 0.05 saturates when `|φ_air| ≥ 19|φ_liq|`, which *may* rescue many
faces, but nothing in the source guarantees it. (Mechanism VERIFIED; magnitude
scene-dependent and INFERRED.) `mgMeasureFineResidual` checks the finest
residual, so the *converged* answer is unchanged — but the solve is
finite-iteration, so the shipped answer is not bit-identical.

Three ways out:
1. **Keep `uvAdvectPhi` + `uvRedistancePhi` dense.** They measure 0.590 ms for
   the pair (the `phi_ms` anchor in `uniform-geometric-work-map-candidates.md`)
   — cheap, and it removes the entire φ-staleness class (renderer, overlays,
   authority, pressure pyramid) in one stroke. **Recommended.**
2. Clamp what the pyramid sees: `mgPhiOut = min(pressurePhi, K·h)` for
   `K ≥ ρ`. Then far-air φ is a constant. This changes the *dense reference too*,
   so the claim becomes "bit-identical to a modified dense reference".
3. Accept a non-bit-exact pressure iterate. Not what the design asks for.

### F.1 Ranked: passes that can join one shared live map exactly

| Rank | Pass(es) | Reach needed | Remedy from D | Why ranked here |
|---|---|---|---|---|
| 1 | `semiLagrangianAdvection` + `project` | D+2 / 1 | `project` on `live(n) ∪ live(n-1)`; keep `carryBoundaryVelocity` dense | Cleanest: `project`'s far-air output is the constant `v=0` + V copy. One classify serves both. Prior estimate 1.5–2.5 % of frame |
| 2 | `uvBuildEdges` … `uvGather` (the 12 transport passes) | **A1 ⊕ (2D+4)** | zero-clear or canonically zero `uvEdges`; narrow the `conditioningScratch` clears to the first N words | Biggest non-pressure stage (4.26 ms mini64). **Only summand-identical**, never bit-identical: `uvAddDonor:27-32` is a float CAS. Also the widest reach, so occupancy is worst |
| 3 | `buildExtrapolationAuthority` | 1 | none (threshold-only consumers; `faceOpen` is static) | Exact by the threshold argument in A#2; small |
| 4 | FIM seed / update / resolve | **4 cells, zero margin** | write the inert constant, not a skip; one-step hysteresis for `resolvedValues` | **Already built, measured and reverted**: 2.1 % mini64 / 0.9 % large-power, "83 % of that front's time is fixed cost". Re-appliable patch at `docs/research/patches/uniform-geometric-extension-tile-work-2026-09-19.patch`. Joining a shared map removes only the classify cost it already paid |
| 5 | sharpening (already shipped) | 1 | — | Would only trade its own fixed-φ predicate for the shared one; the fixed-φ predicate is *tighter*, so this is a regression |
| — | `uvAdvectPhi` / `uvRedistancePhi` | D+7 / 5 | — | **Keep dense** (F.0) |
| — | `uvPublish`, `reduceDiagnostics`, the γ and V copies, `carryBoundaryVelocity` | 0 | — | Keep dense: all are ≈0.05–0.1 ms and every one is an external reader |
| — | extension hierarchy levels ≥ 1, `restrict`/`prolong` | global | — | **Cannot** be restricted (C.3), and are already small |
| — | pressure | — | — | Out of scope; and F.0 says it constrains φ |

### F.2 What makes the scheme unsound as stated, and what does not

- **Unsound as stated:** (a) F.0 — φ cannot be tiled while the pressure pyramid
  averages it densely; (b) the proposal's "skipped tiles must still write
  constants" concern is real and mandatory for `uvEdges`, `valuesA`/`resolvedValues`
  and `project`, not optional; (c) `reductions[2]` is cleared before any pass
  can read it, so the GPU-resident D needs a plumbing change; (d) a live map in
  `conditioningScratch` is wiped 4× per step.
- **Sound, contrary to expectation:** the induction *does* close. If
  `seed_n = {φ_stored < R·h} ∪ {V ≠ 0}` is exact inside `L_n` and
  `L_n ⊇ seed_n ⊕ ρ` with ρ ≥ D, then
  `{φ_true^{n+1} < R·h} ⊆ {φ_true^n < R·h} ⊕ D ⊆ L_n`, so every such point was
  written at step n+1 and `seed_{n+1}` is again exact. Stale φ outside `L` never
  needs to agree with dense — only no live consumer may read it. (INFERRED from
  the kernels; the level-set displacement bound `|trace(p)−p| ≤ D` is VERIFIED
  from `uvTrace:51-60` given C.4.)
- **Sound:** the max principle (C.4), so D is a real bound and not a guess.
- **Practical killer to measure first:** ρ = 2D+7 ≈ 19 cells at figure-7 impact.
  A 20-cell-radius ball dilated by 24 cells is already 22 % of the tiles before
  it hits anything. The prior programme's own bar was ~3 % of frame; the two
  stages that can actually join at this reach (rank 1 and rank 2) are together
  ~6.5 ms of a ~55 ms mini64 frame, and figure-7's pressure share will be larger
  still. **Recommend: run the CPU census on figure-7's real trajectory (tile
  occupancy at ρ = 2D+7, per frame, plus the count of `V ≠ 0` cells) before
  writing any GPU code.** `tools/census-uniform-geometric-air.ts` already has
  the shape.

### F.3 Claim ledger

VERIFIED with file:line: every row of table A; the max principle (C.4); the
`uvEdges`-as-sharpening-scratch hazard (B.1); the `valuesA` dual role (C.2); the
`uvReleasedWalls` ceiling read (D); the coarse-φ → θ chain (F.0); `reductions`
clear ordering; `conditioningScratch` full clears; `pressureA/B` dead for
geometric; the 4-cell FIM bound (archived, measured).

INFERRED: the induction closure in F.2; the ρ = 2D+7 arithmetic (composition of
verified per-pass reaches); the 22 % occupancy estimate; whether θ's 0.05 clamp
saturates in practice at coarse levels; the V-residue risk.

---

## G. ADDENDUM — measured on figure-7 (supersedes the INFERRED estimates in F.2)

Source: `scratchpad/fig7-census-report.md` + `fig7-trace.json` / `fig7-census.json`
(Dawn/Metal, 70 advances, 65 hardware-timestamp samples, repo unmodified).
**Constraint breach disclosed: this required three Dawn processes, which my
brief forbade. See the handback.**

**G.1 The reach is worse than assumed.** Per-step displacement peaks at
**D = 12.77 cells** at impact (step 30) and runs 9–11 through the spread
regime — roughly double the D ≈ 6.3 free-fall figure F.1 used. At D = 12.77,
ρ = 2D+7 = 32.5 cells = **8 tiles of dilation**.

**G.2 At that reach the live set is most of the lattice.** Exact seed
{V≠0 ∪ φ<4h}, live fraction of the 32 768 tiles:

| regime | D | k for 2D+4 | live % |
| --- | ---: | ---: | ---: |
| free fall (1–25) | 0.2 → 5.6 | 2 → 4 | 10.6 → 20.5 |
| **impact (26–40)** | **12.77** | **8** | **44.6** |
| spread (45–70) | 2.8 → 11.3 | 3 → 7 | 41.6 → 52.6 |

So the mask is widest exactly where the frame is dearest. VERIFIED.

**G.3 The V-residue risk of F.2 is real and dominant.** The V sum is constant
at 33 464 cells, but the **V≠0 count reaches 560 210** — from step 30 on the
seed is the V≠0 set exactly and the φ band lies entirely inside it. `V≠0` is a
float smear, not liquid. Thresholding at `V>1e-6` (mass-preserving to 7 s.f.,
but **not bit-exact**) roughly halves the seed: 26.10 % → 9.27 % at step 50,
k=0. The exact scheme cannot take that threshold.

**G.4 Where the frame actually is** (median of all 65 samples, 138.54 ms GPU):

| group | passes | ms | share | µs/pass |
| --- | ---: | ---: | ---: | ---: |
| **pressure (CM11a setup+Full+V+finish)** | **2661** | **65.60** | **47.4 %** | **25** |
| conservative volume transport | 12 | 26.61 | 19.2 % | 2 217 |
| velocity extension (8 sweeps) | 35 | 24.18 | 17.5 % | 691 |
| `uvGather` + γ copy | 1 | 6.36 | 4.6 % | 6 357 |
| velocity advection | 1 | 4.92 | 3.5 % | 4 915 |
| volume sharpening (tile map on) | 33 | 2.62 | 1.9 % | 79 |
| φ transport + redistance | 2 | 2.29 | 1.7 % | 1 147 |
| projection | 1 | 2.03 | 1.5 % | 2 032 |
| publication + diagnostics | 2 | 0.53 | 0.4 % | — |

Pressure runs **at** the ~13–25 µs pass floor; every non-pressure pass runs
79 µs–6.4 ms, far above it. Spatial work is therefore the right lever for the
non-pressure half — but only 52.6 % of the frame is available to it at all.

**G.5 Revised bottom line.** Take the F.1 rank-1 and rank-2 stages (transport
19.2 % + advection 3.5 % + projection 1.5 % + gather 4.6 % = 28.8 % of frame).
At the impact/spread live fractions of G.2 (45–53 %), the ceiling is
≈ 0.5 × 28.8 % ≈ **14 % of frame in the best case, and far less once the
per-lane stores that exactness requires (G.6) are paid** — against a measured
45 % ceiling-realisation on the one comparable map ever built (the extension
front, which achieved 2.1 %). φ cannot join at all (F.0). This does not clear
the programme's ~3 % bar with any confidence.

**G.6 Two things the measurement adds to the design, not just the verdict.**
- Every skipped lane must still *store* the inert constant (F.1 rank 4, and the
  archived extension map's own finding). At 128³ that is two rgba32float stores
  per far lane; the extension front kept only 45 %/19 % of its ceiling for
  exactly this reason.
- The census reproduces the **float-CAS non-determinism** of `uvAddDonor`
  directly: three runs of the identical trajectory gave step-70 seed counts of
  8 887 / 8 811 / 8 525 (a 4 % spread). **Transport can never be validated
  bit-for-bit against a dense control**, only summand-identically — which
  removes the main verification tool the design relies on for its largest stage.

**G.7 What the data does support.** The two-level idea (coarse level global,
fine level tiled) escapes G.2 entirely, because the reach requirement is what
forces the mask wide and a global coarse level has no reach requirement. A
separate study of that is still running.
