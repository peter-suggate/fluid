# Sparse CM12 as the paper's Algorithm 1 on fixed bricks

Status: analysis, 2026-08-23. Reference: `docs/papers/massConservingLiquids.txt`
(Chentanez & Müller 2012, "Mass-Conserving Eulerian Liquid Simulation").
Source of truth for "what runs": the production branch of `encode()` in
`lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts:4592-5360` (working tree at
4997cac0 + staged changes). Timings are the prior session's stage-cost medians
(mini64 / ocean-seiche, 4 samples, 65.5 µs Dawn ticks, bimodal lane — read as
proportions, not predictions).

## 0. The framing

We are implementing one paper on one data structure. The paper's time step is four
steps (Algorithm 1): velocity extrapolation; density advection + sharpening; velocity
advection + forces; incompressibility. Every one of those is a **map over cells or
faces** with one of three access shapes — a local 6/26-neighbour stencil, a backward
trace ending in a trilinear gather, or a trace ending in a trilinear scatter — plus the
pressure solve.

The data structure is a sparse, adaptively-refined brick lattice whose **topology is
fixed for the whole frame**. On a dense grid the maps need nothing from the grid but
arithmetic. On our lattice they need exactly five services, and the whole difference
between "the paper" and "our 18 stages / ~500 launches" is how those five are supplied
and how much else has accreted around them:

| service | what a kernel asks | used by |
|---|---|---|
| **T1 enumerate** | the k-th active cell / face / brick (and only those) | every pass |
| **T2 neighbours** | cell → 6 face-neighbour cells and the 6 faces between (2:1 seams: 1 or 4 across a face) | diffusion, sharpening δ±, pressure coefficients/RHS, collocation, activity census, extrapolation sweeps |
| **T3 point → corners** | arbitrary point → owning cell at its brick's rung + the 8 trilinear corner cells and weights | every semi-Lagrangian trace (density, γ, velocity faces, sharpening ∇ρ trace, solid-excess trace, tracers, D4 fold, presentation coarse gather) |
| **T4 face → cells** | face → its two cells, area, open fraction | velocity advection, body forces, coefficient build, projection |
| **T5 classify** | per cell: liquid (ρ′>0.5) / air / solid (V=0) / in-band (≤K cells from liquid) / over-full (ρ′>1) | extrapolation domain, sharpening domain, pressure membership, presentation |

T1–T4 are **pure functions of the topology**; on a topologically static scene (ocean:
0 commits in 24 frames) they are constants. T5 is the only per-frame fact, and it is a
*per-cell classification*, not a journal.

Everything in the pipeline that is not (a) one of the paper's maps, (b) one of T1–T5,
(c) adaptivity (measure / plan / transfer / recompile bricks), or (d) QA is noise. The
tables below classify every sweep that way.

Legend for the classification column used throughout:
**P** paper physics · **T** topology-service supply (re-derivation of T1–T5) ·
**B** bookkeeping (dirty/root/producer tracking, authority protocol, seal/begin/finalize
singletons, indirect copies) · **A** adaptivity (not in the paper, genuinely needed) ·
**Q** QA/diagnostic/presentation-only.

## 1. What is really needed: the paper as kernels

Fields: ρ, γ (cells); u (faces); V, V^f, u_s (solid fraction, face fractions, solid
velocity — static unless bodies move). Δt = 1/30 s, CFL up to ~25 in the paper.

| id | paper step | kernel (one map) | domain | access shape | services | reads → writes | scatter? |
|---|---|---|---|---|---|---|---|
| K1 | §3.3 velocity extrapolation | extend u from liquid faces into the air band, K sweeps (paper: JRW07 near, grid hierarchy far) | band faces/cells | local stencil, K sweeps | T1 T2 T5 | u, liquid mask → u_ext | no |
| K2 | §3.4 steps 1–3 | backward RK trace from cell centre through u_ext; trilinear gather γ; **scatter** γ_i·w⁻ into β at the 8 corners; cache the 8 corner ids + weights | active cells | trace + gather + scatter | T1 T3 | u_ext, γ → γ_adv, β, corner cache | yes (β) |
| K3 | §3.4 steps 4–5 | ρ_i^{n+1} = Σ_l γ_i/max(1,β_l)·w⁻_li·ρ_l^n using the cached corners; γ likewise | active cells | local (cached corners) | T1 | ρ, γ, β, cache → ρ', γ' | no |
| K4 | §3.4 steps 6–7 | for β_j<1: forward RK trace, **scatter** ρ_j(1−β_j)w⁺ and (1−β_j)w⁺·γ_j into corners | cells with β<1 | trace + scatter | T1 T3 | ρ, γ, β, u_ext → ρ', γ' | yes |
| K5 | §3.4 step 8 | γ-diffusion, 1–7 iterations, dimension by dimension: neighbours exchange ρ_j(γ_j−γ_i)/2γ_j and equalise γ | active cells | 6-stencil, iterated | T1 T2 | ρ', γ' → ρ'', γ'' | no (pair flow is symmetric; both cells compute it) |
| K6 | §3.5 eq. 4–17 | δ±, |∇ρ|±, w_i (needs max_j|ρ_i−ρ_j| over neighbours), ∆ρ_i with the eq. 17 clamps; ρ_i += ∆ρ_i | cells with ε<ρ≤0.5 (+1-ring) | 6-stencil | T1 T2 T5 | ρ'' → ρ''', ∆ρ | no |
| K7 | §3.5 Alg. 2 | TraceAlongField from cell centre along ∇ρ until the 0.5 contour or D∆x (forward Euler substeps, stop at solids); **scatter** −∆ρ_i trilinearly with solid weights zeroed | cells with ∆ρ≠0 | trace + scatter | T1 T3 T5 | ρ''', ∇ρ, ∆ρ, V → ρ⁴ | yes |
| K8 | §3.6 | for V_i<1 and ρ_i>V_i: d=ρ_i−V_i; trace along solid-SDF gradient S∆x; **scatter** d; ρ_i −= d | partial-solid cells | trace + scatter | T1 T3 | ρ⁴, V, solid SDF → ρ⁵ | yes |
| K9 | Alg. 1 step 3 | per face: backward trace through u_ext, trilinear sample u_ext; += f·Δt | active faces | trace + gather | T1 T3 T4 | u_ext, f → u* | no |
| K10 | §3.7 | ρ′=ρ/V; extrapolate ρ′ into V=0 neighbours; φ=−(ρ′−0.5)∆x; member ⇔ ρ′>0.5; RHS = Σ_faces u*·A·V^f + min(λ(ρ′−1),η)/∆x | active cells | 6-stencil | T1 T2 T4 T5 | ρ⁵, V, V^f, u* → member, φ, rhs | no |
| K11 | §3.7 / [CM11a] | per face: ghost-fluid θ from φ of both cells, coefficient = θ·A·V^f/∆x | member faces | face → 2 cells | T1 T4 | φ, V^f, member → θ, coef | no |
| — | §3.7 | pressure solve (out of scope) | | | | | |
| K12 | §3.7 | u ← u* − θ-scaled ∇p per face; collocate u to cell centres for next frame's traces; divergence diagnostic | member faces, active cells | face → 2 cells; 6-stencil | T1 T2 T4 | p, θ, u* → u, u_c | no |
| K13 | §3.8 | ρ post-process (γ″=2min(ρ,0.5) blurred), iso-surface at 0.5 for rendering | active cells | 6-stencil / page | T1 T2 | ρ⁵ → presentation | no |
| **A1** | — | measure per brick: interface present, motion, detail (activity) | bricks | reduce over cells | T1 T2 T5 | ρ, u, history → score | no |
| **A2** | — | plan rung per brick with 2:1 grading, hysteresis | bricks | 6/26-brick stencil | — | score, history → rung plan | no |
| **A3** | — | transfer ρ, γ, u to the new rung (restrict / prolong), reconstruct faces | changed bricks | local | T2 T4 | fields → fields′ | no |
| **A4** | — | recompile T1–T5 for changed bricks + neighbours | changed bricks | local | — | topology delta → lattice image | no |

Fourteen maps and one solve. Nine of them are pure local stencils or cached gathers;
four need an arbitrary-point lookup (K2, K4, K7, K8, K9 — five with K9); four need
atomics (K2's β, K4, K7, K8). Under a fixed topology K1–K13 are **pure functions of the
fields and the lattice image**; the adaptivity maps A1–A4 are the only place where the
lattice itself is an output. That is the whole frame.

Order note: the paper advects density before velocity; our encoder traces faces
(K9, inside `face-preparation`) before density (K2–K4). Both read u_ext and the
pre-advection scalars, so the order is immaterial; keep whatever the digests are
blessed on.

## 2. What runs today, sweep by sweep

One row per encoded launch group, production branch only (`legacyHostAuthorityOracleForQA`
and QA layouts excluded). "×n" is the launch count in that row. The K column names the
paper kernel the row serves; blank means it serves none.

### 2.1 transport-velocity-extension — 3.867 / 6.226 ms — K1

| sweep | ×n | class | domain / access | notes |
|---|---|---|---|---|
| begin / publish body / publish boundary / seal frame control (FCA) | 4 singletons + 1 copy | B | — | translates host inputs into indirect families; never reads fields |
| rigid voxelization; moving-solid roots; 2 bypass noops | 3 | B/P | solid cells list (L) | only the roots dispatch does physics, and only with moving bodies |
| begin VEX execution; initialize candidates | 2 | B/T | blast list (L) | the blast was sealed by last frame's presentation stage |
| `advanceVelocityExtensionCandidates{1..8}` | 8 | P + T | arrival-ordered blast list (L); per edge: pressure CSR neighbour (C), `rowAccepted` 4–10 loads, `cellActive` 5 loads, stamp, 4 velocity loads (+ `atomicExchange` on cached corners) | **this is K1**; 8 depth-specialised pipelines doing the same Jacobi sweep; adjacency is the *pressure* edge CSR, not T2 |
| commit; finalize | 2 | P/B | blast list | writes the per-cell accepted-velocity cache (Level-3 plane input) |
| TPA begin / clear / compile packets from FSM / finalize + 3 copies | 4 + 3 | B | packet capacity (L) | re-derives T1 for the three transport families from last frame's masks |

Ocean receipt: rootCount = blastCount = 491,140 of 537,220 cells (maximumDepth 0) — the
blast IS the accepted set; the root/blast machinery selects nothing there. Body stub on
mini: 0.92 → 0.13 ms (launches free, loop bodies are the cost).

### 2.2 face-preparation — 0.983 / 3.670 ms — K9 (minus forces)

| sweep | ×n | class | domain / access | notes |
|---|---|---|---|---|
| clear retired face-velocity support | 1 | T | retired dirty bricks (L) | |
| `publishSparseCM12FaceVelocitySupport` | 1 (brickCount) | T | all active TEI leaves → dense finest-grid 4-float lattice (33 MB/frame ocean) | rasterises u_ext so that T3 becomes arithmetic; measured necessary: TEI-at-every-corner was 8.6–8.9 ms vs 1.6–3.5 |
| `prepareSparseCM12DirtyBrickFaceRows` | 1 | P + T | dirty-brick list (L) → template row range → `rowAccepted`; RK2 1–16 substeps, 8 corners per sample from the dense cache (T) | **K9's trace**; the support test (two side samples) is the band gate |

### 2.3 conservative-transport — 3.998 / 9.896 ms — K2 K3 K4

| sweep | ×n | class | domain / access | notes |
|---|---|---|---|---|
| begin DCA; begin TPM | 2 | B | — | |
| `traceGammaAndBeta` | 1 | P + T | trace family packets (P); RK2 1–16 substeps; each corner `cm12TeiOwnerAtFine` (T3 via TEI, ~10 loads/4 levels at HEAD); β fixed-point atomics (A); publishes sharpening mask | **K2** + writes the 56 B/cell departure cache (= the paper's "A not stored explicitly") |
| `scatterDensityDeficit` | 1 | P + T | deficit packets (P); forward RK2; 8-corner TEI; atomics | **K4** |
| `gatherConservativeDensity` | 1 | P + B | packets (P); reloads cached 8 corners (S); writes ρ', γ'; publishes SURFACE / DENSITY-CHANGED masks; `tra1MarkScalarCellClosure` | **K3** + mask publication |
| seal TPM; `compileSparseCM12VexRootMasks`; seal DCA sources; `compileSparseCM12DynamicTRA`; seal targets + 1 copy | 5 + 1 | B (+T) | gather packets; VEX-root compile still walks cell incidence/row terms (C) | turns "density changed" into VEX roots and gamma row masks |

Ablation (ocean fast lane, 7.7 ms): trace 4.6 / closures 1.8 / tile floor 1.3 /
stencil 0.5 / launch 0.13. The trace is a **dependent-chain latency** (constant-velocity
arm removes 2 of 3 ms with owner work unchanged) — a cheaper T3 shortens a level inside a
sample, not the levels between RK2 samples.

### 2.4 tracer-advection — 0 / 0 ms — Q

Seed + advance over the tracer array; each lookup `ownerCellAt` (T3, legacy). Absent
when the view is off.

### 2.5 gamma-diffusion — 1.114 / 2.032 ms — K5 (2 iterations)

| sweep | ×n | class | domain / access | notes |
|---|---|---|---|---|
| `clearGammaReceipts` | 1 (×2) | B | all accepted cells | receipt clear for the fixed-point scatter |
| `scatterSparseCM12DynamicGamma{Snapshot,Refinement}Rows` | 1 (×2) | P + T | DCA touched-row packets → 6 axis/side masks → ITR stable row; per row negative×positive terms in stable order; paired antisymmetric fixed-point receipts (A) | **K5**, one iteration each; the pair flow is computed once per row and scattered to both cells |
| `finalizeGamma{Snapshot,Refinement}` | 1 (×2) | P | all accepted cells (S) | apply receipts |
| `clearSparseCM12DynamicRows` | 1 | B | DCA rows | |

7 launches for 2 Jacobi iterations of a symmetric 6-stencil. The paper does this in
place, dimension by dimension; a gather form (each cell sums its 6 incident flows, both
sides compute the identical flow) needs no receipts, no clears, no row masks.

### 2.6 surface-sharpening — 1.770 / 3.146 ms — K6 K7 K8

| sweep | ×n | class | domain / access | notes |
|---|---|---|---|---|
| native clears of receipts and ∆ρ | 2 clearBuffer | B | template cell count | |
| `prepareSharpeningField` | 1 | P + T | TPM sharpening-mask packets (P); 6-stencil via incidence (C) for δ±/max-diff; `sharpeningStats` incidence + opposite-side term walks | **K6** |
| `scatterSharpeningMass` | 1 | P + T | same packets; ∇ρ trace, default 7 substeps, TEI point owners (T3) + 8-corner stencil; fixed-point atomics | **K7** |
| `finalizeSharpening` | 1 | P | all accepted cells | apply receipts (was 1.90 ms before the FSM rewrite removed its CAS fan-out; arithmetic 0.13) |
| `clearSolidExcess` / `scatterSolidExcess` / `finalizeSolidExcess` / bypass noop | 4 | P + T | FCA solid-cell family (L); two incidence sweeps (C); atomics | **K8** |
| FSM begin / publish / seal | 3 | B | packed.brickCount | CHANGED / NONEXACT / BULK / FLIP ballots — this is the one place T5 is computed as ballots today |
| `compileSparseCM12FinalScalarVexRoots` | 1 | B + T | packed.brickCount; one-ring via incidence (C) | VEX roots from scalar change |

### 2.7 symmetry-authority — 0.066 / 0 ms — Q (scene oracle)

preserve / commit D4 + bypass noop + publish scalar output (4). Eight `ownerCellAt` per
cell (T3 legacy). Not in the paper; a symmetric-scene oracle. Bypass when the scene is
not D4.

### 2.8 body-forces — 0.197 / 0.197 ms — K9 (forces half)

`forceFaces` over all accepted rows (1): `rowAccepted` per row (T), `+= open·Δt·g`,
mirrored to both parity banks. The paper adds forces in the same pass as the velocity
advection.

### 2.9 pressure-topology — 2.228 / 3.736 ms — K10 K11 (+ solver cache)

~55 dispatches / ~20 copies, in ten phases (from the encoder at :4795–5057):

| phase | sweeps | ×n | class | what it is in paper terms |
|---|---|---|---|---|
| epoch open | begin canonical cells/rows, begin PCF, plan membership epoch, begin PTR, capture consumer generations | 6 | B | nothing — generation/journal protocol |
| seed PCF | seed previous brick / aggregate-edge / hierarchy-node / hierarchy-edge leaves | 4 + copies | B | preconditioner cache reuse bookkeeping |
| seed PTR | seed previous brick leaves, row leaves, finalize brick frontier | 3 + copies | B | journal |
| brick repair | repair brick leaves; reduce tree levels 1..N−1; finalize plan; repair changed bricks | ~5 + copies | A/B | topology-delta selector |
| cell membership | classify bootstrap cells; classify dirty cells; finalize frontier; repair canonical cell leaves; finalize cells; finalize cell execution | 6 + copies | **P + T + B** | **K10's membership** (ρ′>0.5, V=0 extrapolation, `pressureCellSubmerged` nested incidence walk) over a 2-ring dilation of changed cells — on ocean that is everything |
| row membership | repair row leaves; reduce tree; finalize row plan; repair topology rows; classify bootstrap rows; classify dirty rows; finalize frontier; repair canonical row leaves; finalize rows; finalize row execution | ~10 + copies | **P + T + B** | **K11's θ / row membership**; two term walks + 4 atomicAdds per row over ~1.18 M temporal rows |
| fine PCF | finalize PCF frontier; repair fine cache; finalize fine cache | 3 + copies | P | **K11's coefficients / diagonal** |
| aggregate PCF | repair brick + aggregate-edge worksets; finalize plan; repair aggregate edges; repair brick diagonals; finalize | 6 + copies | solver | preconditioner level 1 (legitimately pressure-specific) |
| hierarchy PCF | repair hierarchy node/edge worksets; finalize plan; repair diagonals; repair edges; finalize PCF | 6 + copies | solver | preconditioner level 2 |
| commit + RHS | commit PTR brick states; finalize bounded PTR; reopen journal; `preparePressure` | 4 | B + **P** | **K10's RHS** — incidence divergence gather + `pressureCellSubmerged` again |

Of ~55 launches, the paper's content is three maps (membership, θ/coefficients, RHS)
plus the two preconditioner levels. Everything else is the incremental-repair
transaction that keeps them consistent under *sparse* membership change — and on ocean
membership change is dense (the band rolls every frame).

### 2.10 pressure-rhs — 0.393 / 0.721 ms — solver setup

`initializePCG` + restrict/refine ladder + reductions (14). Solver-owned; listed only
because `preparePressure` (K10's RHS) lives at the *end of pressure-topology*, not here.

### 2.11 velocity-projection — 1.901 / 5.112 ms — K12

| sweep | ×n | class | domain / access | notes |
|---|---|---|---|---|
| advance activity clock; begin incremental activity | 2 | B | — | |
| FPA: begin; seed bootstrap; seed previous leaves; mark from pressure cells; mark from dirty bricks; finalize frontier; repair leaves; finalize plan; execute; verify; finalize execution (+3 copies) | 11 + 3 | **B + T + P** | private leaf/frontier catalogue, 6-level rank-tree descent per row, `pcmCellContains` per term | selects "rows with θ>0 touching a member cell" then runs **K12's face update** — on ocean the selection is every pressure row whenever the solve ran |
| `collocateAndDiagnose` | 1 | P + T + B | all accepted cells; incidence walk with `rowAccepted` per incidence (C), `pressureCellSubmerged` again for ρ<1, Kahan fold; records VEX roots + activity marks by CAS | **K12's collocation** + root bookkeeping |
| D4 preserve / commit / bypass | 3 | Q | eight `ownerCellAt` per cell | scene oracle |
| publish frame face output | 1 | B | — | |

### 2.12 projection-diagnostics — 0.197 / 0.524 ms — K12's diagnostic

measure (all accepted cells; incidence scan only to ask "touches a mixed seam?") +
reduce (2). The seam test is a static property of the topology (T2 flag).

### 2.13 activity-measurement — 0.655 / 1.966 ms — A1

mark scalar-dirty bricks; mark topology-dirty bricks; finalize worklist (+copy);
`measureBrickActivity` (per dirty brick: per cell incidence + row-term scans (C) for
surface crossings / exposed sides / deformation / swept support; fixed-point
workgroup reductions); age history; finalize census (6 + 1 copy). The per-cell facts it
needs are T5 + u — the same facts FSM already balloted.

### 2.14 resolution-planning — 0.197 / 1.180 ms — A2 (+ A4 prep)

plan rung; activate swept receivers; retire unsupported empties; log₂(B) 2:1 closure
sweeps (3 for B8, 4 for B16); validate; schedule; allocate pages; synthesize cells;
defer publication; clear shadow rows; begin shadow topology; build leaf worklist;
build structure worklist; finalize (+4 copies) ≈ 16. Brick-level; 6/26-neighbour
lookups via the hash directory (T). Genuinely adaptive — the compiler's input.

### 2.15 candidate-transfer — 0.393 / 0.393 ms — A3 A4

13 sub-seams, ~25 launches: field transfer; face reconstruction; face validation;
effects preflight (9: begin, census ×2, semantic preflight ×2, finalize ×2 …); IBO
construction + validation (4); TEI compilation; authorize (2); PTR publication; VEX
publication (2); effects seal (2); state publication (3); image replay (2). This is
the only stage whose *job* is to produce T1–T5 — and today it produces several partial
images (TEI for transport, IBO for boundary operators, shadow faces, PTR journal, VEX
effects) rather than one lattice image every kernel reads.

### 2.16 brick-retirement — 0 / 0 ms — A

mark post-topology activity; finalize worklist (+copy) (2 + 1).

### 2.17 presentation-publication — 1.114 / 3.015 ms — K13 + next frame's K1 plan

| sweep | ×n | class | notes |
|---|---|---|---|
| VEX plan: begin candidate; seal roots; seed roots; seal seed frontier; for depth 1..8 {prepare frontier; copy indirect; expand frontier; seal frontier}; finalize blast; copy blast | ~30 + 9 copies | **B + T** | root → blast BFS through global incidence; on ocean blast = accepted set; charged here, executed in stage 1 |
| FPL begin; initialize bricks; populate plan; import VEX blast; resolve closure; seal; finalize; build packet; finalize packet; execute; commit frame control | ~11 | P (K13) + B | page/packet based; dry pages already free (stub: 0.917 → 0.066 ms on long-dam) |

### 2.18 Launch and class totals

Counting the encoder's production branch by launch group above gives roughly 250
distinct launch sites; with the depth/level loops and indirect copies expanded the
measured receipt is **~505 non-pressure launches**, of which ~280 are
`@workgroup_size(1)` protocol singletons and ~60 are buffer copies feeding indirect
arguments. Physics launches (class P, the paper's fourteen maps): VEX 8–10, face 1,
transport 3, gamma 4, sharpening 5, forces 1, pressure setup 3, projection 2,
diagnostics 1, presentation ~2 ⇒ **~30**. Adaptivity (A): ~30. Everything else — about
**440 launches** — is T re-derivation, dirty/root/producer bookkeeping, authority
protocol, and QA.

By time on ocean (≈ 42 ms non-pressure in the receipt above; 53.5 ms at HEAD in the
bimodal lane): the five stages that are mostly T + B by launch count — VEX 6.2,
pressure-topology 3.7, projection 5.1, presentation 3.0, activity 2.0 — are ~20 ms;
transport + face-prep + sharpening + gamma (mostly P, with T inside every sample) are
~19 ms. The P-only floor measured by stub arms: transport trace 4.6 + stencil 0.5,
face trace ~1.6–3.5, sharpening physics ~1.3, VEX body ≤ 1, gamma < 1, K10–K12 maps
~1.5 ⇒ **~12–14 ms of the 42 is the paper; the rest is structure.**

## 3. The gap, by topology service

What each service costs today, who pays it, and what a fixed-for-the-frame lattice
image should make it cost. Load/level counts from the WGSL (`ta()` = atomicLoad on the
arena binding; a level is a load whose address depends on a previous load).

| service | today | loads / levels | paid by (per frame, ocean) | under one compiled lattice image | why the current tree hasn't closed it |
|---|---|---|---|---|---|
| **T1 enumerate** | arrival-ordered worklists per authority (VEX blast, transport packets ×3, DCA rows, FCA families, FPA rows via 6-level rank tree, pressure bootstrap/dirty lists, activity dirty bricks, shadow leaves, presentation pages); each compiled by its own begin/compile/seal/copy chain | 2–20 / 1–6 per item, plus ~300 launches to build the lists | every pass | tile-major dispatch: one workgroup per (brick, 64-tile); item id = arithmetic on (tile, lane); dynamic selection = one 64-bit ballot per brick per cause (FPL1/FSM already use this ABI) | each authority compiles its own list because each was designed to be *exact*; ocean flips 73 % of cells per frame so exactness selects nothing, and mini is launch-bound so the compile chain itself is the cost |
| **T2 neighbours** | pressure/incidence CSR: incidence begin/end → row → `rowAccepted` (4–10) → term begin/end → cell, coefficient; gamma pair CSR; pressure edge CSR for VEX | 10–16 / 4–5 per row | K5 (ITR rows), K6 stats, K8, K10 RHS + submerged (twice), K11 rows, K12 collocate, diagnostics, A1 census, VEX sweeps ×8, VEX-root compile, final-scalar one-ring | uniform tile bit: neighbours are `±1, ±r, ±r²` arithmetic (0 loads); seam/wall tiles: a compiled accepted-edge list, 2 loads / 1 level, template order preserved; `rowAccepted` a bitset | IBO1/ITR1 exist as transforms but only DCA gamma rows consume ITR; the resident image carries IRL + geometry, not a per-cell accepted-edge view, so every other consumer still walks the pressure CSR |
| **T3 point → corners** | `ownerCellAt`: LOD1 directory → brickSpan → acceptedBrickResolution → template cell range ×2 → brickActive → cellResolution → cellOpenVolume; transport uses TEI packet-staged owners; face-prep uses a 33 MB dense raster; sharpening/tracers/D4/presentation use the legacy chain | 10–12 / 4 per corner (≈140 loads / 6 levels per trilinear sample) | K2 ×≤16 substeps × 8 corners, K4, K7 ×7 substeps × 11, K8, K9 (via raster), tracers, D4 ×8, presentation | one widened brick descriptor (slot, rung, cellBase, origin, extent) staged per workgroup for the 27-neighbourhood: 1 shared-memory read + local arithmetic, then 8 plain field loads; one collocated velocity plane written by VEX commit + collocate so every sampler reads the same expression | three ownership representations coexist (LOD1, packed-u16, TEI2); the plane exists but only transport reads it; the face raster is still the only path that beat the gate — because the trace is latency-bound and an owner lookup deeper than ~1 level loses to a raster |
| **T4 face → cells** | row record + `rowAccepted` + template owner range; projection rows through FPA's private catalogue | 4–10 / 3 | K9 forces, K11, K12 faces, body forces, VEX per-edge | row tiles × `rowAccepted` bit; the two cells by arithmetic on (brick, rung, axis, local) | FPA's catalogue duplicates the row identity ITR already provides |
| **T5 classify** | computed in ≥6 places: TPM masks in transport, FSM ballots in sharpening, `pressureCellSubmerged` twice, PCM membership classify, FPA marks, activity marks, VEX roots (6 causes) — each via its own incidence walk + CAS | 3 M CAS spins / frame on ocean | every stage that selects | **one** per-cell classification written once per frame (at collocate, when ρ⁵ and u are final): liquid / air / band-depth / solid / over-full / changed, as 64-bit tile ballots per brick; every consumer reads the ballot | FSM (this tree) is the first instance of the right shape; it is consumed by transport packets and VEX roots only, and projection/pressure/activity still author their own |

The single sentence: **every consumer re-derives T1–T5 per visit because each was
built as a separate authority with its own exact dirty set; the fix is one lattice
image compiled at topology commit (T1–T4) plus one per-frame classification ballot
(T5), read by every map.** That is the compiled-topology direction — but pointed at
the paper's fourteen maps, not at making the dirty sets exact.

## 4. Drastic simplifications

Ordered by how much structure they delete per unit of physics risk. Each names what it
removes, what it keeps, and the receipt that says it is safe to try.

### S1. The frame is fourteen maps over one lattice image
Replace per-stage authorities (FCA, TPA, DCA, TPM, FSM, VEX roots/blast, FPA, PCM
frontier) with: a lattice image compiled in candidate-transfer (T1–T4) and one
classification ballot written in collocate (T5). Every kernel dispatches tile-major
over ballots. Launch budget: ~30 physics + ~30 adaptivity + ~10 pressure setup +
presentation ≈ 80 vs 505. Removes ~280 singletons and ~60 indirect copies.
*Evidence:* hydrostatic tank encodes 505 launches vs 506 on a dam break (change-blind);
ocean blast = accepted set; indirect launch ~15–25 µs vs direct 3–6 µs; body-stub arms
show bodies not launches cost on large scenes but launches dominate mini.
*Keeps:* the paper's maps, β/scatter atomics, the solve, adaptivity.

### S2. Velocity extrapolation as K band sweeps over the classification
K1 today: roots from 6 causes → BFS blast through global incidence (30 launches at
frame tail) → 8 depth-specialised sweeps over an arrival list with pressure-CSR
adjacency. Paper-minimal: band = cells within K of liquid (K = max trace reach,
ocean |v|Δt ≲ 3 cells; cap at the 8 the blast already uses), K Jacobi sweeps of one
pipeline over band tiles, T2 implicit neighbours. Deletes the root causes, blast
lists, plan-at-tail/execute-at-head split, and the 8 pipelines.
*Evidence:* ocean rootCount = blastCount = 491k, maximumDepth 0 — the BFS selects the
whole band every frame; VEX body stub 0.92 → 0.13 ms on mini means the sweeps are the
cost only where they do work. *Risk:* fixpoint identical if the sweep set is a superset
of the blast (it is) and the sweep order within a depth is unchanged — gate with the
digest lane.

### S3. One sampler for every trace
K2, K4, K7, K8, K9, tracers, D4, presentation all ask T3. Give them one function:
27-neighbourhood brick descriptors staged in workgroup memory (≤ 27 × 4 words, not the
20 KB halo that killed occupancy) + one collocated velocity plane. Delete the 33 MB
face raster **only after** the sampler beats it on the face stage gate (the raster is
the measured winner today because an arbitrary-point lookup deeper than ~1 level loses
to arithmetic on a latency-bound trace). *Evidence:* face-prep TEI-at-every-corner
8.6–8.9 ms vs raster 1.6–3.5; transport const-velocity arm −2.0 of 3.0 ms; halo
capacity 1 → 11.7 ms vs 20.6–23.0 at capacity 3. *Law to keep:* a kernel's
workgroup-memory + register budget is part of its contract; stage only what is reused
≫1× per fill.

### S4. Scalar conditioning as gather-form local maps
K5 (γ diffusion) and K6 (∆ρ) are symmetric 6-stencils. Today: 7 launches + DCA row
masks + fixed-point receipts + clears for two diffusion iterations; prepare/scatter/
finalize + receipts for sharpening; FSM + VEX-root compile after. Gather form: each
cell reads its 6 neighbours (T2), computes the 6 pair flows (both sides compute the
identical expression), applies them — no receipts, no clears, no row masks, 1 launch
per iteration. Only K7/K8 (trace-scatter) keep atomics. Deletes gamma's 5 bookkeeping
launches, sharpening's receipt clears, and the ITR row path for gamma.
*Risk:* bit-exactness — the paper's in-place dimension-by-dimension update and the
current receipt order are both sequential artefacts; a gather form is a *different*
(order-free) operator. It is a one-time blessed move with the reason named, not a
digest-exact conversion.

### S5. Pressure setup as three maps, repair only the preconditioner
K10 (membership / φ / RHS) and K11 (θ / coefficients) are per-cell and per-face maps
of T5 and ρ′. Today: a 55-launch PCM → PTR → PCF transaction designed for sparse
membership change. Proposal: rebuild membership + θ + RHS dense over band tiles every
frame (3 launches, `pressureCellSubmerged` computed once per cell as part of T5 and read
by RHS and collocate), keep PTR/PCF incremental **only** for the aggregate/hierarchy
preconditioner levels (O(bricks), genuinely worth caching). *Evidence:* the
membership classify already runs over a 2-ring dilation of changed cells = everything
on ocean; `classifyPressureRow` visits 1.18 M rows; pressure-topology is 3.7 ms for
three maps' worth of work. *Gate:* the hydrostatic lane (rows 0, iterations 0) must
stay quiet — dense rebuild of an unchanged band must write identical bits.

### S6. Projection + collocation + diagnostics + classification as two maps
K12 face update over row tiles × (`rowAccepted` ∧ θ>0) — the selection FPA's 11
launches compute is a property of K11's output, so K11 ballots it. Then one cell map:
collocate u_c, divergence max (workgroup partials), and **write the T5 ballots for the
next frame** (liquid / band / changed / over-full) — the one place the classification
is authored. Deletes FPA's catalogue, diagnostics' re-walk, VEX-root CAS, activity-mark
CAS, TPM/FSM publication passes (they become reads of this ballot).
*Evidence:* FPA on ocean selects every pressure row whenever the solve ran; finalize
CAS fan-out 1.77 of 1.90 ms; diagnostics 0.52 ms to ask a static question.

### S7. Adaptivity reads the ballot and compiles the image
A1 measures from the T5 ballot + u per tile and reduces to bricks (no per-cell
incidence census); A2 unchanged; A3 unchanged; A4 = candidate-transfer emits **one**
lattice image (brick descriptors, tile lists, rowAccepted bitset, seam-edge lists)
instead of TEI + IBO + shadow faces + PTR journal + VEX effects. Compile cost is
O(changed-brick surface): interior of a brick at rung r is template-identical; only
the six face layers depend on neighbours.

### What this keeps that looks deletable, and why
- The departure corner cache (56 B/cell): it *is* the paper's "A need not be stored
  explicitly" — K3 reuses K2's trace; removing it repeats the trace.
- The dense face raster: until S3's sampler wins the face gate.
- Fixed-point atomics for β, deficit, sharpening and solid-excess scatter: those are
  the paper's three scatter passes (plus solids); order-independence is the contract.
- Incremental preconditioner caches (aggregate / hierarchy PCF): O(bricks), reused
  every frame, legitimately pressure-specific.
- Activity history, 2:1 grading, transfer: the adaptivity the paper does not have.
- The hydrostatic "rows 0 / iterations 0" proportionality: S5/S6 must preserve it via
  the ballot (a band that did not change ballots no change), not via journals.

### What to measure
Ocean proportions across ≥ 3 interleaved captures per arm on an idle machine; per-stage
digests (bit-exact, or a blessed move with the reason named — S4 is one); launch count
per frame (505 → target ≤ 100); `acceptedCells / ballot population` so "static" is read
from the receipt; the hydrostatic quiet lane as the regression tripwire for S5/S6.

## 5. One-paragraph recap

The paper is fourteen maps (nine local stencils or cached gathers, five arbitrary-point
traces, four of them scattering) plus a solve, over five topology services of which four
are frame-constant. The resident runs those fourteen maps inside ~30 launches and
surrounds them with ~440 launches that re-derive the services per visit and maintain
per-stage exact dirty sets — on the reference large scene those sets select everything
and on the small scene the launches themselves are the cost. The simplification is not
a faster sweep; it is one compiled lattice image (T1–T4, rebuilt per topology commit),
one per-frame classification ballot (T5, written once at collocate), and the paper's
maps dispatched tile-major over them.
