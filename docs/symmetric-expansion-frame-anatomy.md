# Anatomy of one simulation frame — `symmetric-expansion`

**What this is.** A pass-by-pass account of exactly what happens inside one
`advanceTo()` of the `symmetric-expansion` scene: what each stage computes, which
part of Aanjaneya et al. 2017 it implements, and what the GPU actually does while
it runs. Every number is measured — from a Metal System Trace + GPU-counter
capture taken for this document — or read directly out of the code and the
solver's own end-of-run audit. Nothing here is estimated from a model of the
hardware except where explicitly labelled **[derived]**.

Captured 2026-08-04. Artifacts: `artifacts/xctrace-symmetric-expansion-2026-08-04/`
(`report.html`, `summary.json`, `mini-dam.trace`, `baseline.log`, `traced.log`).

---

## 0. Provenance and measurement integrity

| | |
|---|---|
| Machine | Apple M1 Max — 4 GPU partitions × 768 scheduler slots × 32 threads = **98,304 resident threads** (the profiler recovers this divisor from the counter quantisation; `slotConfidence` 0.89) |
| Backend | Dawn/Metal, `skip_validation,use_user_defined_labels_in_backend`, headless |
| Lane | `tools/profile-mini-dam-xctrace.ts --lane=symmetric-expansion --steps=240` |
| Untraced (shipping) wall | **237.75 ms/advance** over 240 advances |
| Traced (label-isolated) wall | **235.24 ms/advance** — 0.99× the untraced run |
| Representative advance | **236.68 ms** wall, **226.45 ms** GPU busy = **95.7 %** device utilisation, **10.23 ms** idle across 480 gaps |
| Attribution | 221 exact stages, **0 composite**, 100.0 % of interval time named |
| Counter coverage | 323 / 521 samples uncontended (65.6 %); WindowServer took 115 ms of GPU in the window |

Three caveats that matter when reading the numbers:

1. **Label isolation is free here.** The traced run splits the frame's 80
   shipping compute passes into 231 individually-labelled Metal encoders (plus
   233 tiny boundary blits totalling 0.62 ms). It measured **235.24 ms** against
   the clean baseline's **237.75 ms** — i.e. within machine drift. So per-pass
   attribution below describes the frame that actually ships, not a distorted
   one. The `dispatches` count is identical in both runs (486/advance); only the
   pass count changes.

2. **The fully-attributed decomposition is one advance.** The counter window
   held 11 anchor firings (11 advance boundaries, 10 complete advances). The
   report reduces the full 231-encoder decomposition to one complete
   representative advance. Section 2.3 gives the 10-advance distribution for
   every task above 1 ms so you can see which numbers are stable and which are
   not. **Two are not**: the pressure solve (43.5–61.5 ms) and structured
   velocity advection (1.6–24.3 ms).

3. **Occupancy/ALU/bandwidth are device-wide counters**, sampled only in windows
   where no other process had GPU work in flight (65.6 % coverage here). Per-task
   *time* is exact — it filters by process. Occupancy is directionally solid but
   should not be quoted to three digits.

---

## 1. The scene and its discretization

`createSymmetricExpansionScene()` (`lib/scenes.ts:599`): a 1.6 × 0.8 × 1.6 m
closed tank, free-slip walls, with one centred 0.8 × 0.4 × 0.8 m water body
(fill fraction 1/8) seeded as four bricks. It is the smallest dyadic domain that
leaves an equal one-brick gap to all four walls, which is why it is the D4
symmetry oracle: volume, velocity, pressure, topology and four-wall contact must
stay exactly symmetric under the dihedral group of the horizontal plane.

Resolved configuration in this lane (from `baseline.log`, `phase:"constructed"`
and `phase:"result"`):

| | |
|---|---|
| Coarse (simulation) lattice | **32 × 16 × 32** cells, Δx = 0.05 m |
| Maximum leaf size | 32 cells; interface refinement band 3; 1 grading layer |
| Fine (interface-tracking) factor | **4** → effective narrow-band lattice **128 × 64 × 128** |
| Live pressure rows | **2,091** of 16,384 dense cells (12.76 % — `compressionRatio`) |
| Power-diagram face slots | **7,756** |
| Published power descriptors | 2,100 — **2,074 "same-or-finer"**, **26 "same-or-coarser"** |
| Fine narrow band | **9,500** resident pages of 11,520 capacity; 1,836 interface pages; 4-page dilation ring |
| Fine page format | 4³ = 64 samples × 4 channels × 4 B = **1,024 B/page** (9.73 MB resident) |
| SPGrid V-cycle | **6 levels** (32×16×32 → 1×1×1), `planOctreeSPGridVCycle` |
| Section 5 air-support graph | 6,946 owner rows (2,091 liquid + 4,855 air) × 12 face slots = **83,352 face unknowns**, 26,120 seeded |
| Allocation | 155.4 MB authoritative + 27.2 MB scratch. Power authority 104.1 MB (incl. the 14.2 MB catalog), fine level set 49.9 MB |
| dt | fixed 0.004 s, one substep per advance |

The two descriptor counts are worth dwelling on, because they are the paper's
Section 6.1 encoding measured in the wild. The paper defines exactly two cases —
"cells with no coarser neighbour" (18-bit neighbour mask, 2¹⁸ possibilities) and
"cells with no finer neighbour" (3 bits of child position + 6 bits of coarse
neighbour direction, 2⁹ possibilities), plus one discriminating bit. This scene
publishes **2,074 of the first and 26 of the second**: it is very nearly a
uniform fine lattice inside the liquid with a thin transition collar, and the
adaptivity that pays for itself is not level-of-detail — it is that the 14,293
dry cells carry no rows at all.

The catalog those descriptors index (`lib/generated/octree-power-catalog.ts`) is
generated offline rather than populated on demand as the paper does:
**8,083 configurations, 1,608 distinct descriptors, 14.18 MB**, max face
incidence 30, max neighbour rows 36, max tetrahedra 68, 75 tetrahedron vertices,
worst float32 geometry error 5.7e-8. The paper reserves ~128 MB for its
on-demand table; ours is smaller because it stores only the configurations the
grading rule can actually produce.

Per-advance host-side command traffic (`gpuCommandAudit`, 240 advances):

| | per advance |
|---|---|
| Command buffers / `queue.submit` | **1.04** |
| Compute passes (shipping) | **80.0** |
| Dispatches | **486.0** — of which **314.0 indirect** |
| `copyBufferToBuffer` | 71.3 (36.6 kB) |
| `writeBuffer` | 31.0 (6.4 kB) |
| Bind groups created | 27.5 |

One advance is one command buffer and one submission. 65 % of dispatches take
their launch shape from GPU-authored indirect arguments; the host never reads
back a count to decide how much work to encode.

Physics result after 240 advances (0.96 s): 5 MGPCG iterations, relative
residual **5.86e-6**, volume drift **-1.14e-7**, 0 non-finite cells, 0 rejected
advances, 0 tripwires.

---

## 2. The frame at a glance

### 2.1 The eight bands of the advance

The advance is one fixed sequence declared in
`lib/physics-step-program.ts:OCTREE_STEP_PROGRAM` and encoded by
`WebGPUUniformEulerianSimulation.advanceTo` (`lib/webgpu-uniform-eulerian.ts:1791`).
Measured on the representative advance:

| Band | Passes | t (ms) | GPU ms | % |
|---|--:|---|--:|--:|
| **S1** `ready-topology-flip` — commit last tail's candidate epoch, rebuild Section 5 air support against it | 1–19 | 0.0 – 37.9 | **37.20** | 16.4 |
| **S2** `surface-transport` — fine + coarse level-set advection, narrow-band topology delta | 20–64 | 37.9 – 101.7 | **62.54** | 27.6 |
| **S3a** structured advection, forces, divergence RHS | 65–72 | 101.8 – 122.2 | **20.14** | 8.9 |
| **S3b** pressure solve (persistent MGPCG) | 73 | 122.2 – 165.7 | **43.51** | 19.2 |
| **S3c** structured projection, separation marking, cell-velocity reconstruction | 74–77 | 165.8 – 166.0 | **0.12** | 0.05 |
| **S3d** fine settlement — JFA redistance, volume correction, restriction, summaries | 78–153 | 166.1 – 187.0 | **18.55** | 8.2 |
| **S3e** Section 5 air support rebuilt on the settled fine generation | 154–163 | 187.0 – 220.6 | **33.14** | 14.7 |
| **S4** `inactive-topology-candidate` — build next epoch's topology/velocity/boundary | 164–231 | 220.6 – 236.6 | **13.01** | 5.8 |

(Bands sum to 228.21 ms against 226.29 ms of attributed interval time; the 1.9 ms
difference is 0.43 ms of genuinely overlapping encoders — Metal runs a few
independent encoders concurrently — plus band-edge rounding.)

`rigid-exchange` and `sparse-brick-world` do not appear: this scene has no rigid
bodies and the headless lane publishes no render world.

### 2.2 By subsystem, and against the paper's Table 2

Regrouping the same 226.29 ms by what the work *is*:

| Subsystem | ms | % | Paper Table 2 (Fig 2 / Fig 3 / Fig 8) |
|---|--:|--:|---|
| **Section 5 velocity extrapolation** (air support, run twice) | **67.40** | 29.8 | Velocity Extrapolation: **5 % / 5 % / 2.5 %** |
| **Fine level-set advection** (§5 multi-stepped semi-Lagrangian) | **50.06** | 22.1 | Level Set Advection: 32 % / 24 % / 17 % |
| **Pressure solve** (§4.3 MGPCG) | **43.51** | 19.2 | ─┐ |
| Structured advection / forces / divergence / projection (§4.1) | **19.98** | 8.8 | ─┼ Projection: 30 % / 31 % / 49 % |
| SPGrid hierarchy setup + commit (§4.2) | **6.80** | 3.0 | ─┘ |
| Fine narrow-band topology, pages, volume correction (§5) | **15.19** | 6.7 | (part of Grid Adaptation, 0 % / 8.6 % / 0 %) |
| **Fine narrow-band redistance** (JFA CPT; paper uses FMM) | **13.56** | 6.0 | Reinitialization: **25 % / 26 % / 23 %** |
| Next-epoch topology candidate (§6.1 / §6.3 / §4.2) | **7.86** | 3.5 | (Grid Adaptation) |
| Coarse level set on the octree (§5) | **0.94** | 0.4 | |
| Boundary conditions — free surface + solid (§4.1) | **0.72** | 0.3 | |

Two structural differences jump out and they are the headline of this document:

- **Velocity extrapolation is 29.8 % of our frame and 2.5–5 % of the paper's.**
  It is the single largest subsystem, larger than the pressure solve. The paper
  runs a serial CPU fast march over faces; we run a GPU frontier relaxation to a
  fixed point over 83,352 face unknowns — **twice per advance**.
- **Reinitialization is 6.0 % of our frame and 23–26 % of the paper's.** Replacing
  the paper's serial fast marching with a jump-flood closest-point transform is
  the one place where we are decisively ahead of the reference, both relatively
  and in efficiency (the JFA kernels are the only ones in the frame running at
  45–58 % occupancy and 71–74 % ALU).

Projection as a whole (solve + operators + hierarchy = 31.0 %) lands squarely
inside the paper's 30–49 % band, and our 5 PCG iterations sit inside the paper's
"satisfactory convergence within 6–10 iterations".

### 2.3 Which numbers are stable

Ten complete advances from the same trace, GPU ms per advance:

| Task | mean | min | max | representative |
|---|--:|--:|--:|--:|
| Octree persistent MGPCG — whole solve in one workgroup | **54.88** | 43.51 | 61.49 | 43.51 |
| Advect fine phi rare | 31.06 | 30.34 | 32.50 | 31.33 |
| March Section 5 sparse changed frontier — topology-commit | 30.89 | 29.20 | 32.52 | 31.27 |
| March Section 5 sparse changed frontier — settled-fine | 30.36 | 28.94 | 32.11 | 29.21 |
| Advect fine phi common | 18.00 | 16.24 | 18.83 | 18.25 |
| **Advect structured families** | **13.40** | **1.58** | **24.28** | 19.65 |
| Scatter recurring fine-band seed halos | 8.35 | 7.86 | 8.91 | 8.76 |
| Octree resident grading closure | 2.60 | 2.58 | 2.63 | 2.63 |
| SPGrid V-cycle — candidate build level sets | 2.09 | 2.04 | 2.25 | 2.06 |
| Compare topology-tile refinement signatures | 1.96 | 1.90 | 2.16 | 1.90 |
| Fine JFA — cooperative flood B→A stride 1 | 1.81 | 1.69 | 2.16 | 1.71 |
| SPGrid V-cycle — publish validated exact level deltas | 1.68 | 1.59 | 1.92 | 1.66 |
| SPGrid V-cycle — candidate link parent chains | 1.32 | 1.30 | 1.38 | 1.30 |
| **GPU busy total** | **229.65** | 210.59 | 248.41 | 226.45 |
| **Frame wall** | **240.51** | 222.46 | 261.47 | 236.68 |

Everything in the surface pipeline is flat to ±5 %. Two things are not:

- **The pressure solve swings 43.5 → 61.5 ms** — CG converges in 4–6 iterations
  depending on the step. The representative advance happens to be the cheapest of
  the ten; the honest per-advance figure for MGPCG is **54.9 ms (24 %)**, not
  43.5 ms (19 %).
- **Structured velocity advection steps from ~1.8 ms to ~20 ms between advance 3
  and advance 4 of the window**, and the frame wall steps up with it (222–232 ms
  → 237–261 ms). This is not noise; it is a switch. The code has exactly one
  switch that produces it: the deep-interior identity carry in `advect()`
  (`lib/webgpu-octree-structured-dynamics.ts:2233`), which skips the
  semi-Lagrangian trace for a face whose structural octree identity survived the
  epoch flip *and* whose two incident rows are deep interior. When topology
  churns, the carry misses and every face re-traces. **[Mechanism identified from
  the code; not independently instrumented in this capture.]**

---

## 3. Stage by stage

Timings below are the representative advance unless a mean is given. The full
ordered 231-pass timeline is Appendix A.

---

### S1 — `ready-topology-flip`: commit the epoch, rebuild the extended velocity field
**Passes 1–19 · 37.20 ms · 16.4 %**

#### Goal

The active topology epoch is immutable for a whole advance. Everything about the
*next* epoch — refined octree, power descriptors, Section 6.3 row coefficients,
structured velocity bank, boundary coefficients, SPGrid level sets — was built
at the *previous* advance's tail and cross-validated there. S1 is the single
seam where that candidate becomes accepted, atomically across seven independent
authorities, and where the Section 5 extended velocity field is rebuilt against
the newly accepted geometry so that everything downstream can sample velocity
anywhere, including in air.

If the candidate was poisoned, the gate retains the current epoch instead
(`encodeReadyCommitGate`, `lib/webgpu-octree.ts:3918`) and the step proceeds on
the old geometry. Zero rejections occurred over 240 advances here.

#### How the paper is realised

- **§4.2 (pyramid of sparse uniform grids).** The commit publishes each level's
  active/ghost cell sets. `SPGrid V-cycle · publish validated exact level deltas`
  is the moment the six-level pyramid the preconditioner walks becomes visible
  to the solver, as an exact *delta* against the previous generation rather than
  a rebuild.
- **§6.1 (topology encoding).** `Commit power descriptor publication` and
  `Commit power topology publication` flip the descriptor words and the
  catalog-resolved `PowerRowMetric` (19 stencil coefficients, face areas,
  normals, inverse distances) from the candidate bank to the accepted bank.
- **§5 (velocity extrapolation).** Passes 10–19. The paper: "*To extrapolate
  velocities outside the liquid region we first interpolate velocities from
  power faces to regular octree faces and use an approach similar to fast
  marching, this time operating on faces instead of cells and copying over the
  velocity value from the face closest to the free surface. … Subsequently, we
  interpolate back velocities from regular octree faces to all power faces.*"
  That is exactly the pass sequence:
  `Extrapolate structured ordinary faces and reconstruct support vectors` →
  `March … to a fixed point` → `Reconstruct Section 5 air-support vectors`.

#### Performance

| pass | ms | occ | ALU |
|---|--:|--:|--:|
| Commit gate + 7 authority commits (passes 1–8) | 0.06 | — | — |
| SPGrid V-cycle · publish validated exact level deltas | 1.66 | 0.1 % | 0.0 % |
| Publish structured air-support identities | 0.96 | 22.7 % | 10.3 % |
| Close sparse fine air-support demand | 0.89 | 5.4 % | 8.2 % |
| Canonicalize footprint air-support identities (4-pass radix sort) | 0.25 | 2.2 % | 1.3 % |
| Extrapolate ordinary faces + reconstruct support vectors | 0.48 | 7.6 % | 12.0 % |
| **March Section 5 sparse changed frontier to a fixed point** | **31.77** | **6.7 %** | **15.8 %** |
| Reconstruct Section 5 air-support vectors | 0.89 | 3.4 % | 8.0 % |

**The commit itself is free.** Eight authorities flip in 60 µs of single-workgroup
dispatches. The design goal — that an epoch flip cost nothing at the seam because
all the work happened a step earlier — is met exactly.

**The march is 85 % of the band.** Data layout: the air-support arena holds a
face-value bank `faceA` and a proposal bank `faceB`, each indexed
`row × 12 + 4·axis + quadrant` — 12 face slots per owner row (3 velocity axes ×
4 power-face quadrants), 6,946 rows, **83,352 slots**, plus an atomic
`faceFrontier` control block holding three per-axis queue pairs and a
generation-marked dedup word per face.

The march is a monotone frontier relaxation, not a fast march. Structure
(`lib/webgpu-octree-air-velocity-support-gpu.ts:1392`):

1. **12 wide prefix waves.** Each wave is four dispatches inside one compute
   pass: `expand` (256 lanes; each lane owns a packed `(axis, queue slot)` pair,
   walks up to 30 incident rows from the catalog adjacency, and appends 4
   quadrant destinations each with an `atomicExchange` dedup) → `relax` (256
   lanes; each surviving destination runs the 30 × 4 closest-face gather and
   writes `faceB`) → `commit` (256 lanes; `faceB ≠ faceA` → write back and
   enqueue) → `advance` (**one workgroup, one lane**, rotating queues and
   publishing the next wave's indirect args).
2. **Residual tail.** `marchAirSupportFacesChangedFrontier` — **3 workgroups**,
   one per velocity axis, 256 lanes each — loops to a fixed point inside the
   kernel using `storageBarrier` + `workgroupBarrier` for wave separation. It is
   the exact unbounded authority; the 12 prefix waves are just a fast path.
   `structuredAirSupportMarchDepth: 0` in the audit means the tail did no
   additional waves this run.

Measured 6.7 % mean occupancy, 15.8 % ALU, 5.1 GB/s read, 0.7 GB/s write,
placement 1.42× (moderately even across the four partitions). The waves *are*
launched wide — `dispatchFor(12 × 6946, 256)` is 326 workgroups, 83,456 threads,
85 % of the machine's residency — but occupancy time-averages to 6.7 % because
almost every lane early-outs on `local >= current` once the frontier collapses,
and the wave's cost is set by its slowest few workgroups. The four-dispatch
barrier chain runs 12 times inside one encoder, so there are ≥48 device-wide
serialisation points in this pass, each paying the full launch-and-drain latency
of a nearly-empty grid.

**This is the frame's clearest structural inefficiency.** 31.8 ms to fill ~57,000
face values (83,352 total minus 26,120 seeds) is **≈560 ns per face**, on a
machine that can sustain tens of billions of simple operations per millisecond.
It also runs a *second* time in S3e — the same 30 ms again — because the accepted
epoch's support (built here) and the settled fine generation's support (built
there) are different publications.

---

### S2 — `surface-transport`: move the free surface
**Passes 20–64 · 62.54 ms · 27.6 %**

#### Goal

Advect the 128 × 64 × 128 narrow-band level set and the coarse octree level set
forward by dt with the previous advance's projected, closest-point-extended
velocity; then rebuild the narrow band's page topology around the new interface
and hand the transported generation to the projection stage.

#### How the paper is realised

This band is the paper's Section 5 almost line for line.

- **Multi-stepped semi-Lagrangian.** The paper: "*we divide this time step by m,
  where m is the factor by which the fine level set resolution is higher than
  the effective resolution of the octree. Next, we take m steps (of forward
  Euler) backwards along the velocity field … After the mᵗʰ step we interpolate
  the level set, and assign its value back to the starting cell of the
  trajectory.*" `Plan GPU-resident fine transport substeps`
  (`webgpu-octree-fine-levelset-transport.wgsl.ts:200`) computes
  `required = max(max(1, segments), ceil(v_max·dt/h_fine))` with
  `segments = fineFactor = 4`. Measured
  `transportMaximumDisplacementFineCells: 1`, so **m = 4** — exactly the paper's
  rule, with a CFL-driven escape hatch that did not fire.
- **Velocity interpolation on the power dual mesh.** Each of the m sub-steps
  calls `transitionSample()`, which locates the owning power cell, and — if the
  cell is not the regular case — walks its catalog tetrahedra (up to 68), forms
  barycentric weights against the local Delaunay tetrahedralisation and blends
  four cell-centre velocities. That is §5's "*trilinear interpolation inside
  cubic cells and barycentric interpolation on tetrahedra*" plus §6.2's
  "*retrieval of local Delaunay tetrahedralizations*" from the byte-encoded
  vertex table, straight out of the generated catalog.
- **Beyond the paper.** The transport is BFECC, not the paper's first-order
  scheme: `Advect fine phi` (RK2-midpoint predictor) → `Reverse factor-1
  predicted phi` → `Apply bounded factor-1 MacCormack correction`, with the
  correction clamped to the donor cube at the same departure point. The paper's
  Limitations section names this as future work ("*we would like to explore
  improved advection schemes, such as FLIP or MacCormack*").
- **Dynamic topology.** The paper: "*we first allocate all pages corresponding to
  blocks that either contain the interface, or lie in the 1-ring of a block that
  contains the interface. After this step, cells that lie within the narrow band
  can be safely marked in parallel without any data hazards.*"
  `Scatter recurring fine-band seed halos` is exactly this dilation — with a
  **4-ring**, not a 1-ring, because the redistance band plus the maximum
  backtrace need the extra collar.

#### Performance

| pass | ms | occ | ALU | notes |
|---|--:|--:|--:|---|
| Fine seed maintenance + workset build (passes 20–27) | 0.44 | — | — | classify → block reduce → scan → publish → compact |
| **Advect fine phi common** | **18.61** | 12.2 % | 20.6 % | |
| **Advect fine phi rare** | **32.09** | 12.3 % | 26.1 % | |
| Status/delta reduction + commit (passes 30–36) | 0.19 | — | — | |
| **Scatter recurring fine-band seed halos** | **8.81** | **79.0 %** | 6.6 % | the frame's only well-fed kernel |
| Band rank/compact, page delta, gather/init payloads (passes 38–57) | 1.44 | — | — | `Finalize global fine page delta` alone is 1.06 ms |
| Fine-to-coarse restriction + coarse phi schedule (passes 58–64) | 0.68 | — | — | |

**Transport (50.7 ms, 22 % of the frame).** One workgroup of **64 lanes per fine
page**; each lane strides over the page's 64 samples, so one lane ≈ one fine
cell. Layout per page: `phi` (f32), `flags` (u32 with a `VALID` bit), `workA`/
`workB` (the two fast-marching channels) — the paper's exact 4-channel SPGrid.
Addressing is `(pageId << 6) | (lz<<4) | (ly<<2) | lx` when `b4FineAddressing` is
on, i.e. a pure shift, no division.

The kernel stages two workgroup-resident address windows before touching
anything: a 5³ = 125-entry page-id window and an air-owner window, so a
characteristic that leaves the workgroup's own page resolves its destination page
from LDS instead of re-walking the worklist directory. Phi values themselves are
never staged — the float evaluation path stays bit-exact.

Cost structure per advected sample: 4 substeps × `transitionSample` for the
predictor, the same again for the BFECC reverse pass, plus a bounded correction
that re-traces and gathers an 8-corner donor cube. `transitionSample` is the
expensive primitive — an air-owner lookup, then either the regular fast path or a
linear scan over up to 68 catalog tetrahedra with 4 velocity gathers each. The
audit reports **354,902 samples processed** per step, so 50.7 ms is **143 ns per
transported sample**, or roughly **7 ns per interpolation**.

Occupancy is 12.2 %, ALU 20.6–26.1 % — with 9,500 pages there is no shortage of
workgroups, so this is register/LDS-pressure and divergence limited, not
launch-limited. It is the one heavy kernel in the frame doing genuinely dense
arithmetic. **The `common`/`rare` split costs more than it saves here**: `rare`
(pages containing any positive or invalid sample — i.e. any page touching air)
is 32.1 ms against `common`'s 18.6 ms, and in a scene where the band is a shell
around a moving surface, most pages are `rare`.

**Halo scatter (8.81 ms at 79 % occupancy).** 256 lanes, one lane per
`(seed, halo offset)` pair over `1,836 interface pages × 9³ = 729` = **1.34 M
membership marks**, each an atomic into the desired-page mask, to produce
**9,500 unique pages**. That is **141× write amplification** — and it is the only
kernel in the frame that actually fills the machine, which is why it shows 79 %
occupancy at 6.6 % ALU: pure atomic/memory work, no arithmetic. Dropping the
dilation to the paper's 1-ring (27 volume) would cut it to ~0.33 ms, but the
4-ring is load-bearing for the redistance band; the fix is a hierarchical
dilation, not a smaller ring.

---

### S3a — structured advection, forces, divergence RHS
**Passes 65–72 · 20.14 ms · 8.9 %** (mean over 10 advances: 13.9 ms, range 2.1–24.8)

#### Goal

Advect the normal velocity component stored on every power-diagram face; apply
gravity and re-impose solid/free-slip constraints; assemble the divergence
right-hand side ρ·flux/dt for every liquid row.

#### How the paper is realised

- **§5 velocity advection.** "*we compute full velocity vectors at the centroid
  of each face and advect them using a standard semi-Lagrangian update,
  subsequently projecting the velocity onto the face normal direction.*"
  `advect()` does precisely that, with a second-order midpoint backtrace:
  sample at the face → sample at the half-step → depart from there → sample →
  `canonicalVelocityDot(transported, n)`. The cell-centre full velocity vectors
  come from the catalog's stored pseudoinverse columns — the paper's "*least
  squares fit based on all the face normal components*" [Feldman et al. 2005]
  precomputed per topological case rather than solved at runtime.
- **§4.1 divergence, eq (4).** `Fuse structured divergence RHS rows` evaluates
  `V_cell ∇·u = Σ_faces A_face (u·n)` per row, reading the face areas from the
  accepted `PowerRowMetric` bank. `Zero dry-identity divergence RHS rows` handles
  the disjoint dry-identity row class.

#### Performance

| pass | ms |
|---|--:|
| Prepare accepted structured dynamics worksets | 0.023 |
| Flatten structured boundary carry probes | 0.371 |
| **Advect structured families** | **19.650** |
| Commit advected structured families | 0.017 |
| Prepare worksets (again, for the force stage) | 0.022 |
| Force and constrain structured families | 0.020 |
| Fuse structured divergence RHS rows | 0.032 |
| Zero dry-identity divergence RHS rows | 0.006 |

**Everything except advection is essentially free.** Forces, constraints,
divergence and the identity zeroing together cost **58 µs** for 2,091 rows and
7,756 faces. The finite-volume operators of §4.1 are not a cost centre at this
scale — they are three storage reads and a fused multiply-add per face.

**Advection is 98 % of the band, at 0.7 % occupancy and 1.9 % ALU.**
`advectStructuredFamilies` is `@workgroup_size(64)`, one thread per face slot,
dispatched indirectly over the family union. With 7,756 face slots the launch is
**≤ 122 workgroups = 7,756 threads = 7.9 % of the machine's residency ceiling**
[derived]. Measured time-averaged occupancy is 0.7 %, an order of magnitude below
that ceiling: the kernel is a long divergent tail. Faces on the interface run
three full `characteristicSample` calls, each walking the local power/tetrahedral
mesh; deep-interior faces return after two loads via the identity carry. Within a
32-lane SIMD group, one interface face makes the other 31 lanes wait.

19.65 ms for 7,756 faces is **2.5 µs per face** — and since the deep-interior
carry means only interface-band faces actually trace (~670 interface rows ⇒
order 4,000 faces), the traced faces are costing **~5 µs each**. This kernel is
simultaneously the most starved (7.9 % ceiling) and the most divergent in the
frame.

The 1.6 ms ↔ 24.3 ms bimodality documented in §2.3 lives entirely here.

---

### S3b — the pressure solve
**Pass 73 · 43.51 ms (representative) / 54.88 ms (10-advance mean) · 19.2–24 %**

#### Goal

Solve L₂p = f for 2,091 unknowns to a relative residual of 5.86e-6.

#### How the paper is realised

This is Section 4.3 implemented literally. The paper defines a preconditioner
M whose action w = Mq is:

> (1) Starting with a zero initial guess p₀ = 0, execute k iterations of the
> damped Jacobi method on L₂p = q. This is only applied to a band (about 3
> voxels wide) around the boundaries and level transitions. …
> (2) Compute a correction δp = M₁r₁ by applying the first order accurate
> preconditioner M₁ from Setaluri et al. …
> (3) Repeat k additional iterations of the same Jacobi method … It was
> necessary to perform an adequate number (k ≈ 8) of boundary smoothing
> iterations.

`section43Correction` (`webgpu-octree-persistent-mgpcg.wgsl.ts:1350`) is exactly
that three-phase structure. The constants are the paper's:
**k = 8** (`OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH = 8`), band width
**3 layers** (`OCTREE_SECTION43_BOUNDARY_BAND_LAYERS = 3`), and the inner V-cycle
is the first-order SPGrid operator over the 6-level pyramid
(`vcycleCorrection`: pre-smooth + restrict down 5 levels, exact 1-cell bottom,
prolong + post-smooth back up), with a Chebyshev smoother of degree 2 or 4. The
outer loop is matrix-free CG on L₂ with an encoded budget of 10 iterations and a
hard ceiling of 16; it converged in **5**.

The full-operator apply `applyAllRows` is §6.3's hierarchical evaluation: 19
coefficients per row (centre + 6 face + 12 edge neighbours) fetched from the
catalog-resolved row template, with ghost contributions folded by the
`GhostValuePropagate`/`GhostValueAccumulate` pair the paper describes.

#### Performance

**The entire solve is one dispatch of one workgroup of 256 threads.**

```
readonly dispatchShape = [1, 1, 1] as const;          // webgpu-octree-persistent-mgpcg.ts:632
OCTREE_PERSISTENT_MGPCG_DEFAULT_LANES = 256;          // webgpu-octree-persistent-mgpcg.wgsl.ts:230
@compute @workgroup_size(256) fn persistentMGPCG(...)
```

Measured: **occupancy 0.3 %, ALU 0.4 %, 2.1 GB/s read, 0.3 GB/s write, placement
4.00× (pinned to exactly one of the four GPU partitions, three at literally
0.00 %).** 0.3 % of 98,304 threads is 295 — the counter is reporting the 256
lanes plus quantisation noise. **The measurement is an exact confirmation of the
structure: for 43–61 ms of every 237 ms frame, 99.7 % of this GPU is switched
off.**

Why it is written this way is legible from the code and is not an accident. The
kernel is *persistent*: state lives in a workgroup-private arena of 11 channels
(x, r, z, d, A·d, four compensated-f32 fields, two band u32 fields), every phase
is separated by `storageBarrier(); workgroupBarrier();`, and the compensated
summation tree emulates 128-lane virtual workgroups so the association order of
every f32 addition is fixed. That is what makes the solve **bitwise
reproducible** — which is precisely what the D4 symmetry gate on this scene
requires. A multi-workgroup solve would need device-wide barriers (WGSL has
none) and a deterministic cross-workgroup reduction.

The cost is latency, not throughput. Counting barrier-separated phases from the
code [derived]: per CG iteration ≈ 1 (x/r update) + ~60 (`section43Correction`:
7 pre-sweeps × 2, an `applyAllRows`, the inner RHS, ~25 in `vcycleCorrection`
across 6 levels, 8 post-sweeps × 2) + ~4 (merged reduction) + 1 (β) + 1
(`applyAllRows`) + ~2 (curvature) ≈ **68**; over 5 iterations plus setup,
**≈ 355 barrier-separated phases**. 43.5 ms / 355 ≈ **123 µs per phase**. Each
phase is one strided pass over ≤ 2,091 rows by 256 lanes — 8 rows per lane, each
a 19-entry gather through the row template — so ~155 dependent loads per lane
with **8 resident SIMD groups and therefore essentially no latency hiding**.
At a few hundred nanoseconds of uncovered memory latency per dependent miss the
arithmetic lands where it lands.

Put in absolute terms: **2,091 unknowns per 43.5 ms is 48,000 unknowns/second.**

The row-capacity gate that selects this executor is
`OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY = 65,536`
(`lib/webgpu-octree-section43-contract.ts:21`). Every scene at or below 65k
pressure rows takes this path.

---

### S3c — projection
**Passes 74–77 · 0.12 ms · 0.05 %**

#### Goal

Apply v ← v − dt·∇p/ρ on every power face, mark the lagged unilateral-contact
separation set from the solved pressure, and reconstruct cell-centre velocity
vectors for the next advance's samplers.

#### How the paper is realised

§4.1's discrete gradient: because the power diagram restores primal-dual
orthogonality, the pressure gradient across a face is a single difference along
the dual edge, scaled by the stored inverse distance. No T-junction special case
exists, which is the entire point of the power-diagram construction.
`Reconstruct projected structured rows` then applies the catalog's stored
pseudoinverse to rebuild full vectors at cell centres — the paper's least-squares
fit, precomputed.

#### Performance

| pass | ms |
|---|--:|
| Prepare accepted structured dynamics worksets | 0.023 |
| Mark structured overhead separation rows | 0.023 |
| Project structured families | 0.017 |
| Reconstruct projected structured rows | 0.054 |

**117 µs for the entire projection.** The reason the "projection" line of the
paper's Table 2 is 30–49 % and ours is dominated by the *solve* is visible here:
applying the gradient is 17 µs; solving for p is 43,510 µs. The ratio is 2,500:1.

---

### S3d — fine settlement: redistance, volume correction, restriction, summaries
**Passes 78–153 · 18.55 ms · 8.2 %**

#### Goal

The transported generation was published before the projection so that coarse phi
could be corrected before forces. Now that the projected velocity and its
closest-point extension exist, settle it: redistance the narrow band, correct
volume, restrict the fine level set onto the coarse octree, and refresh the
summary mip pyramid.

#### How the paper is realised

- **Reinitialization (§5, "fast marching and velocity extrapolation").** The
  paper runs a serial fast marching method on the narrow band and lists
  parallelising it as future work: "*We are presently reinitializing the signed
  distance with a serial Fast Marching method on the narrow band; alternative
  reinitialization schemes that admit parallelism certainly merit attention.*"
  **We took that future work.** `webgpu-octree-fine-levelset-redistance.ts`
  implements a jump-flooding closest-point transform: seed the interface, flood
  A↔B at strides 8, 4, 2, 1, 1, then resolve distance from the winning closest
  point. Warm generations reseed only dirty pages via a staged frontier; if the
  on-GPU resolve records show a collar miss, a fully on-GPU fallback replays the
  complete cold ladder (which is what happened on this advance — passes 99–115
  are the `complete support fallback` chain).
- **Volume correction.** Not in the paper. Three reduction/correction rounds
  (`Reduce resident fine overlap partials` → `Apply bounded global fine normal
  correction` → re-measure, twice) hold `volumeDrift` at -1.14e-7 over 240 steps.
- **Coarse correction (§5).** "*Subsequently, we use the fine level set to
  correct the coarse level set wherever we have valid ϕ-values.*"
  `Fine-to-coarse restriction · prepare/restrict/publish` then
  `Power coarse level set · persistent schedule`.

#### Performance

| group | ms | occ | ALU |
|---|--:|--:|--:|
| JFA frontier build + seed | 0.43 | 16.6 % | 16.4 % |
| JFA warm flood ladder (strides 8, 4, 2, 1, 1, 1) | 3.42 | 53 % | 72.6 % |
| JFA warm resolve | 0.24 | 16.6 % | 23.4 % |
| **JFA complete-support fallback ladder** | **9.06** | 35–58 % | 45–73 % |
| JFA finalize + commit distances | 0.07 | — | — |
| Volume measure/correct, 3 rounds (+ coarse volume) | 3.71 | 16.6 % | 63–65 % |
| Fine publication settle + restriction + coarse schedule | 0.85 | — | — |
| Fine-summary mip refresh | 0.79 | — | — |

**The JFA kernels are the healthiest code in the frame.** 45–58 % occupancy,
71–74 % ALU, 5–12 GB/s read. One voxel per lane over the resident support pages,
each stride pass a fixed 27-neighbour gather with no divergence. This is what the
rest of the frame should look like.

**The fallback ladder cost 9.06 ms — 49 % of the band — and it is conditional.**
The warm path (`cooperative flood` strides 8→1 plus two collar repairs) is 3.42
ms; when the on-GPU resolve records report an unresolved collar, `Fine JFA ·
prepare warm fallback` publishes a non-zero dispatch and the whole cold ladder
(refresh codes, reseed, 6 flood passes, 3 collar repairs, resolve) replays over
the full support. On this advance it fired. `redistanceUnresolvedCells: 0` and
`redistanceFallbackPages: 0` in the end-of-run audit, so the fallback is a
correctness guard that is paying full price on a step where — by the final
census — it changed nothing.

---

### S3e — Section 5 support, rebuilt on the settled generation
**Passes 154–163 · 33.14 ms · 14.7 %**

Byte-for-byte the same producer as S1 passes 10–19, invoked with
`site = "settled-fine"` instead of `"topology-commit"`
(`lib/webgpu-octree.ts:4176`). S1's rebuild targets the newly accepted epoch;
this one targets the fine generation that just settled, so that next advance's
transport has an extended velocity field consistent with the surface it will
sample.

| pass | ms | occ | ALU |
|---|--:|--:|--:|
| Identities, demand, canonicalisation, owner hashing (154–160) | 2.32 | 1–29 % | 0–12 % |
| Extrapolate ordinary faces + reconstruct support vectors | 0.47 | 2.8 % | 3.5 % |
| **March Section 5 sparse changed frontier to a fixed point** | **29.45** | 11.1 % | 25.8 % |
| Reconstruct Section 5 air-support vectors | 0.87 | 3.9 % | 7.8 % |

**Two full frontier marches per advance is 60.5 ms — 27 % of the frame.** Both
solve the same 83,352-unknown extrapolation problem over graphs that differ only
where the topology or the surface moved. The producer already has a
changed-frontier mode (`changedFrontier`) that seeds only altered faces and
carries settled ones; the measurement says that on this scene the "changed"
frontier is still most of the domain, or that the two sites cannot share carry
state.

---

### S4 — build the next epoch's candidate
**Passes 164–231 · 13.01 ms · 5.8 %**

#### Goal

From this advance's surface and projected velocity, build a complete, validated,
*inactive* candidate: refined octree, graded, with owner pages, a compact liquid
frontier, power descriptors and catalog-resolved rows, structured velocity and
boundary banks, and the SPGrid hierarchy — then cross-validate the whole thing in
one reduction. S1 of the next advance either commits it atomically or discards it.

#### How the paper is realised

- **§6.1 topology encoding.** `Resolve structurally dirty power descriptors`
  builds the 18-bit/9-bit descriptors; `Resolve affected power topology rows`
  looks each up in the 8,083-configuration catalog and writes the 19 stencil
  coefficients, face areas, normals and inverse distances into the candidate
  `PowerRowMetric` bank. This is the paper's "*retrieve stencil coefficients via
  a look-up table, indexed by a compact descriptor of the local topology*".
- **Grading.** `Octree resident grading closure` enforces §6.1's restriction that
  a cell and all its neighbours span only two levels — the invariant that makes
  the descriptor encoding well-defined at all.
- **§4.2 pyramid.** The `SPGrid V-cycle · candidate *` chain (21 phases) builds
  each level's active set, detects and inserts **ghost cells** with the paper's
  power-diagram amendment (edge neighbours that share a power face count as
  neighbours when deciding whether to spawn a ghost), counts/scans/writes the
  inter-level transfers, links parent chains, builds stencils and publishes
  spectral bounds for the Chebyshev smoother.
- **§4.1 boundary conditions.** `Resolve canonical free-surface boundary slots`
  (ghost-fluid θ) and `Resolve canonical solid boundary slots` (variational
  cut-cell apertures) — the paper's Brochu/Batty embedded-boundary treatment,
  requiring only "*liquid signed distance values at cell centres and solid signed
  distance values at cell vertices*".

#### Performance

| pass | ms |
|---|--:|
| Owner-page candidate prepare/commit | 0.098 |
| **Compare topology-tile refinement signatures** | **1.900** |
| Build structural tile delta + Octree reset and refinement | 0.521 |
| **Octree resident grading closure** | **2.630** |
| Wet-frontier delta, frontier classify/sort/merge (172–176) | 1.299 |
| Pressure-row candidate + row-delta publication (177–181) | 0.194 |
| Power descriptor + topology resolution (182–187) | 0.207 |
| Structured velocity candidate publication (188–196) | 0.603 |
| Boundary candidate (197–207) | 0.381 |
| **SPGrid candidate hierarchy, 24 phases (208–230)** | **5.180** |
| Validate complete inactive topology epoch | 0.012 |

**The catalog lookup is nearly free — 207 µs to resolve 2,100 descriptors and
write their metrics.** Section 6.1's central claim (that a lookup table beats
explicit meshing) is comprehensively vindicated: the topology encoding costs
0.09 % of the frame.

**The expensive parts are the delta machinery, not the physics.** `Compare
topology-tile refinement signatures` (1.90 ms), `Octree resident grading closure`
(2.63 ms), `SPGrid V-cycle · candidate build level sets` (2.06 ms) and
`candidate link parent chains` (1.30 ms) are all launched over the *provisioned*
tile/level domain rather than the changed set, and all measure 0.1–1.0 %
occupancy with placement 2.0–4.0× — i.e. one to two partitions of four doing
anything at all. Together they are 7.9 ms of the 13.0 ms band.

**The whole candidate rebuild runs every advance** (`quadtreeRebuildCadenceSteps:
1`, 240 rebuilds in 240 steps). The paper adapts topology per frame, not per
substep, and books 5 s of 58 s (8.6 %) for it on the rotating-paddle scene. Ours
is 5.8 % plus the 60 µs commit — comparable, and cheap for what it buys
(a fully validated, atomically-committed epoch with a rejection path).

---

## 4. Where the machine actually goes

Aggregating the counters over the representative advance:

| | |
|---|---|
| Frame wall | 236.68 ms |
| GPU busy | 226.45 ms (**95.7 %**) |
| GPU idle | 10.23 ms across 480 gaps (mean 21 µs, max 555 µs) |
| **Mean compute occupancy** | **11.7 %** (partitions: 11.8 / 9.6 / 12.1 / 10.2 %) |
| **Mean ALU utilisation** | **14.6 %** |
| **Mean read bandwidth** | **3.50 GB/s** |
| **Mean write bandwidth** | **0.82 GB/s** |

The device is busy 96 % of the time and using about **12 % of its threads and
15 % of its ALUs** to do it. Aggregate memory traffic is **4.3 GB/s** — on a part
whose observed streaming peak in this same profiler is ~78 GB/s, that is **5.5 %
of achievable bandwidth**.

For scale, the paper reports (Table 1, Intel Xeon E3-1241) a streaming-copy
reference of 14.45 GB/s and their optimised second-order Laplacian sustaining
**3.95 GB/s**. Our whole frame — on a GPU with an order of magnitude more
bandwidth — moves data at roughly the rate their single-socket CPU kernel did.
The bottleneck is not the algorithm's arithmetic intensity; it is that the work
is decomposed into launches too small and too serial to fill the machine.

Four distinguishable failure modes, all visible in the per-task counters:

1. **Thread-starved by construction.** MGPCG: 1 workgroup, 256 threads, 0.3 %
   occupancy, placement 4.00×, three partitions at 0.00 %. 43–61 ms.
   `Advect structured families`: ≤122 workgroups, 7.9 % ceiling, 0.7 % measured.
   19.7 ms (when the carry misses).
2. **Launched wide, drained narrow.** The Section 5 marches: 326 workgroups
   launched per wave, ~85 % residency ceiling, 6.7–11.1 % measured, because
   almost every lane early-outs and 12 barrier-separated waves each pay full
   launch-and-drain. 60.5 ms.
3. **Memory/atomic bound but well fed.** `Scatter recurring fine-band seed
   halos`: 79 % occupancy, 6.6 % ALU. 8.8 ms doing 1.34 M atomic marks to
   produce 9,500 pages.
4. **Actually healthy.** The JFA flood ladder: 45–58 % occupancy, 71–74 % ALU.
   Fine phi transport: 12 % occupancy but 21–26 % ALU, the only heavy kernel
   doing dense arithmetic.

The 10.23 ms of idle (4.3 % of the frame) is spread over 480 sub-25 µs
inter-encoder bubbles in the label-isolated run; the shipping frame has 80
passes rather than 231, so its bubble count is proportionally lower. This is not
where the time is.

---

## 5. Fidelity to the paper — a scorecard

| Paper element | Status here | Evidence |
|---|---|---|
| §4.1 power-diagram discretization, eq (3)/(4) | Implemented | 7,756 power faces; 2,074 same-or-finer + 26 same-or-coarser descriptors |
| §4.1 embedded free-surface + solid boundaries | Implemented | `Resolve canonical free-surface/solid boundary slots`, 0.13 ms |
| §4.2 pyramid of sparse uniform grids + ghost cells | Implemented, 6 levels | `SPGrid V-cycle · candidate detect/insert ghosts` |
| §4.2 ghost-cell amendment for power diagrams (edge neighbours) | Implemented | ghost detection consumes power-face adjacency |
| §4.3 M = Jᵏ(band) → M₁ V-cycle → Jᵏ(band), k ≈ 8 | Implemented exactly | k = 8, band 3 layers, 6-level first-order V-cycle |
| §4.3 6–10 PCG iterations | **5** | `octreeMGPCGDiagnostics.iterations` |
| §5 narrow band at 4× resolution, 4 SPGrid channels | Implemented | 128×64×128, 1,024 B/page = 64 samples × 4 channels |
| §5 multi-stepped semi-Lagrangian, m = fine/coarse factor | Implemented, **m = 4** | `Plan GPU-resident fine transport substeps` |
| §5 velocity interpolation (cube trilinear / tetra barycentric) | Implemented | `transitionSample` + catalog tetrahedra |
| §5 dynamic topology: allocate interface blocks + ring, then mark in parallel | Implemented with a **4-ring** | `Scatter recurring fine-band seed halos` |
| §5 fine→coarse level-set correction | Implemented | `Fine-to-coarse restriction` |
| §5 fast marching for redistancing | **Replaced** by parallel JFA CPT | the paper's own future work |
| §5 velocity extrapolation on faces | Implemented as a GPU frontier march | 83,352 face unknowns, run **twice** |
| §6.1 19-bit topology encoding, two cases | Implemented | 2,074 / 26 split |
| §6.2 local Delaunay tetrahedralization lookup | Implemented, **precomputed offline** | 8,083 configs, 68 max tetrahedra, 14.18 MB |
| §6.3 hierarchical operator, 19 stored coefficients | Implemented | catalog coefficients + `GhostValueAccumulate` |
| §5 first-order semi-Lagrangian for the level set | **Improved** to BFECC with bounded correction | the paper's future work |
| Volume conservation | **Added** (not in the paper) | drift -1.14e-7 over 240 steps |

The implementation is faithful. Where it departs it departs *upward* — the two
departures (parallel redistancing, BFECC advection) are both items the paper's
Limitations section names as desirable future work, and the parallel
redistancing is measurably a win: 6.0 % of our frame against 23–26 % of theirs.

---

## 6. Conclusions

**The physics is not the cost.** Every operator the paper spends its Section 4
and Section 6 on — the power-diagram Laplacian, the divergence, the gradient, the
catalog lookup, the boundary coefficients — costs a combined **1.08 ms**, 0.5 % of
the frame (0.16 ms of face/row operators, 0.72 ms of boundary resolution, 0.20 ms
of descriptor + catalog lookup). Section 6.1's central bet (compact topology descriptor + lookup table
instead of explicit meshing) pays off completely.

**Three items are 72 % of the frame**, and each is limited by decomposition, not
by arithmetic:

| | ms/advance (mean) | % | limiter |
|---|--:|--:|---|
| Section 5 frontier marches (two sites) | 61.3 | 27 % | 12 barrier-separated waves; wide launch, narrow drain; 3-workgroup residual tail |
| Pressure solve | 54.9 | 24 % | **one workgroup, 256 threads, 0.3 % occupancy**, ~355 serial barrier phases |
| Fine level-set advection | 49.1 | 22 % | dense but divergent; `rare` class is 63 % of it |

**The single most consequential number in this document is 0.3 %.** For 43–61 ms
of every 237 ms frame, the M1 Max runs 256 of its 98,304 threads, pinned to one
of four partitions, with the other three measuring 0.00 %. That is a deliberate,
documented trade — bitwise determinism, which the D4 symmetry gate on this exact
scene depends on, and which WGSL's lack of a device-wide barrier makes hard to
buy any other way. But it is a trade, and at 2,091 unknowns per 43.5 ms
(48,000 unknowns/second) it sets a hard ceiling on how large a scene this solver
can carry: the executor's gate admits up to 65,536 rows, and the cost is
essentially linear in rows against a fixed 256-lane budget.

**Second most consequential: 60.5 ms of every frame is spent solving the same
velocity-extrapolation problem twice.** Whether the two publications can share
carry state is an implementation question, not a numerical one — the paper needs
the extended field to exist once per surface update, and it books 2–5 % of its
frame for it.

**Where the headroom is**, ordered by measured ms and by how structural the
change would be:

1. *Widen the pressure solve* (≈55 ms). The determinism requirement is real, but
   it constrains the *reduction* order, not the stencil applies. A hybrid — wide
   multi-workgroup applies and smooths, with the compensated reductions kept in
   a single deterministic workgroup — would keep exactness where it matters.
2. *Halve the Section 5 marches* (≈30 ms) by sharing the extended field between
   the topology-commit and settled-fine sites, or by making the changed-frontier
   seeding actually sparse on this scene.
3. *Fix the wide-launch/narrow-drain shape of the march waves* (part of the
   remaining 30 ms): 12 host-encoded barrier waves over a provisioned-capacity
   grid, when the frontier is a few thousand faces, is the classic case for a
   persistent multi-workgroup kernel with an on-GPU queue.
4. *Give `Advect structured families` more threads* (≈20 ms when the carry
   misses). 7,756 faces at 64 lanes/workgroup cannot fill this machine; a
   per-face-per-sample decomposition, or splitting the traced and carried faces
   into separate dispatches to kill the divergence, would both help.
5. *Make the halo dilation hierarchical* (≈8 ms). 141× write amplification to
   produce 9,500 pages from 1,836 seeds.
6. *Make the JFA fallback ladder conditional in fact, not just in form* (≈9 ms
   on the advances where it fires and provably changes nothing).

Items 1–4 together account for **~95 ms of the 240.5 ms mean frame** (a 4×
widening of the solve, sharing one of the two marches, and de-divergencing the
structured advection). Recovering most of that lands the frame near **145 ms**
**[derived — arithmetic on the measured items, not an implemented result]** without
touching the discretization, the accuracy, or the symmetry guarantee — and that
frame would still be running at well under a quarter of this GPU's throughput.

---

## Appendix B — reproduction

```bash
node --import tsx tools/profile-mini-dam-xctrace.ts \
  --lane=symmetric-expansion --steps=240 \
  --out=artifacts/xctrace-symmetric-expansion-2026-08-04
```

~9 minutes: a clean untraced baseline over 240 advances, then a relaunch with
`FLUID_WEBGPU_DAWN_FEATURES=skip_validation,use_user_defined_labels_in_backend`
and an attached 3 s Metal System Trace + GPU-counter window in steady state.
Outputs `report.html` (interactive), `summary.json` (every number above),
`mini-dam.trace` (open in Instruments), and the raw NDJSON tables.

Solver-side facts (grid, rows, pages, iterations, command audit, volume drift)
come from `baseline.log`, records `phase:"constructed"`,
`phase:"final-performance-authority"` and `phase:"result"`.

The scene's own correctness gate is a *different* configuration — it runs at
`FLUID_OCTREE_GLOBAL_FINE_FACTOR=1` (coarse-only surface tracking, no fine narrow
band), so its frame does not contain bands S2/S3d at all:

```bash
npm run test:webgpu:symmetric-expansion          # factor 1, D4 symmetry gate
npm run test:webgpu:symmetric-expansion:fine     # factor 4, matches this profile
```

---

## Appendix A — the complete ordered frame

Every Metal compute encoder of the representative advance, in execution order.
`t` is milliseconds from the frame's first encoder; `ms` is that encoder's GPU
time; occupancy and ALU are device counters sampled during it (`—` = no
uncontended counter sample landed inside the encoder).

| # | t (ms) | ms | occ % | ALU % | pass |
|--:|-------:|---:|------:|------:|------|
| 1 | 0.00 | 0.006 | — | — | Open coupled topology ready-commit gate |
| 2 | 0.03 | 0.004 | — | — | Prepare exact accepted topology row commit |
| 3 | 0.05 | 0.013 | — | — | Commit accepted topology row identities and pressure seed |
| 4 | 0.09 | 0.008 | — | — | Publish ready owner-page generation |
| 5 | 0.12 | 0.010 | — | — | Commit power descriptor publication |
| 6 | 0.15 | 0.011 | — | — | Commit power topology publication |
| 7 | 0.19 | 0.005 | — | — | Accept structured publication |
| 8 | 0.22 | 0.005 | — | — | Accept structured boundary publication |
| 9 | 0.25 | 1.660 | 0.1 | 0.0 | SPGrid V-cycle - publish validated exact level deltas |
| 10 | 1.97 | 0.026 | — | — | Initialize structured air-support publication - topology-commit |
| 11 | 2.02 | 0.959 | 22.7 | 10.3 | Publish structured air-support identities - topology-commit |
| 12 | 3.03 | 0.885 | 5.4 | 8.2 | Close sparse fine air-support demand - topology-commit |
| 13 | 3.96 | 0.039 | — | — | Emit sparse fine air-support demand - topology-commit |
| 14 | 4.05 | 0.247 | 2.2 | 1.3 | Canonicalize footprint air-support identities - topology-commit |
| 15 | 4.34 | 0.029 | — | — | Publish canonical air-support candidates - topology-commit |
| 16 | 4.40 | 0.163 | — | — | Resolve and hash live adaptive air owners - topology-commit |
| 17 | 4.62 | 0.475 | 7.6 | 12.0 | Extrapolate structured ordinary faces and reconstruct support vectors - topology-commit |
| 18 | 5.14 | 31.766 | 6.7 | 15.8 | March Section 5 sparse changed frontier to a fixed point - topology-commit |
| 19 | 36.97 | 0.885 | 3.4 | 8.0 | Reconstruct Section 5 air-support vectors - topology-commit |
| 20 | 37.91 | 0.138 | — | — | Maintain compact octree fine seeds |
| 21 | 38.10 | 0.066 | — | — | Publish deterministic fine-seed brick residency |
| 22 | 38.21 | 0.161 | 0.0 | 0.0 | Plan GPU-resident fine transport substeps |
| 23 | 38.41 | 0.048 | — | — | Classify direct structured fine transport blocks |
| 24 | 38.51 | 0.018 | — | — | Reduce direct structured fine transport workset blocks |
| 25 | 38.55 | 0.024 | — | — | Scan direct structured fine transport workset groups |
| 26 | 38.60 | 0.007 | — | — | Publish direct structured fine transport worksets |
| 27 | 38.63 | 0.007 | — | — | Compact direct structured fine transport worksets |
| 28 | 38.66 | 18.612 | 12.2 | 20.6 | Advect fine phi common |
| 29 | 57.33 | 32.092 | 12.3 | 26.1 | Advect fine phi rare |
| 30 | 89.48 | 0.015 | — | — | Reduce structured fine transport status blocks |
| 31 | 89.53 | 0.014 | — | — | Publish structured fine transport status |
| 32 | 89.58 | 0.120 | — | — | Commit structured fine phi and phase delta |
| 33 | 89.74 | 0.004 | — | — | Clear structured fine transport delta header |
| 34 | 89.78 | 0.016 | — | — | Reduce structured fine transport delta blocks |
| 35 | 89.83 | 0.017 | — | — | Publish structured fine transport delta |
| 36 | 89.87 | 0.007 | — | — | Compact structured fine topology delta |
| 37 | 89.90 | 0.006 | — | — | Clear exact global fine page delta |
| 38 | 89.93 | 0.007 | — | — | Clear recurring fine-band identity mask |
| 39 | 89.96 | 0.079 | — | — | Publish recurring sparse fine band (compact seed classification and rank) |
| 40 | 90.08 | 8.811 | 79.0 | 6.6 | Scatter recurring fine-band seed halos |
| 41 | 99.46 | 0.149 | 14.3 | 11.4 | Scan and compact recurring fine-band logical identity |
| 42 | 99.62 | 0.040 | — | — | Classify exact global fine page delta |
| 43 | 99.68 | 0.008 | — | — | Compact sorted global fine changed keys |
| 44 | 99.69 | 0.007 | — | — | Prepare global fine page delta expansion |
| 45 | 99.79 | 0.048 | — | — | Classify exact changed-key dirty and support pages |
| 46 | 99.85 | 0.010 | — | — | Compact exact dirty and support pages |
| 47 | 99.87 | 1.060 | 0.1 | 0.2 | Finalize global fine page delta |
| 48 | 100.47 | 0.011 | — | — | Publish physical power-cell volumes |
| 49 | 100.94 | 0.007 | — | — | Publish exact post-redistance fine summary keys |
| 50 | 100.96 | 0.041 | — | — | Gather compact global fine flags/phi payloads |
| 51 | 101.01 | 0.043 | 32.6 | 22.7 | Gather compact global fine work A/B payloads |
| 52 | 101.06 | 0.018 | — | — | Snapshot exact global fine rollback pages |
| 53 | 101.10 | 0.045 | — | — | Initialize added global fine samples |
| 54 | 101.15 | 0.006 | — | — | Initialize added global fine work A/B samples |
| 55 | 101.16 | 0.074 | — | — | Gather all compact global fine adjacency |
| 56 | 101.27 | 0.007 | — | — | Finalize global fine publication |
| 57 | 101.29 | 0.008 | — | — | Fine-to-coarse restriction - prepare |
| 58 | 101.31 | 0.001 | 19.0 | 35.1 | Fine-to-coarse restriction - restrict |
| 59 | 101.32 | 0.026 | — | — | Fine-to-coarse restriction - publish |
| 60 | 101.36 | 0.016 | — | — | Fine JFA - publish support mask |
| 61 | 101.37 | 0.007 | 0.3 | 0.7 | Power coarse level set - publish exact dispatch |
| 62 | 101.39 | 0.006 | — | — | Fine JFA - reset recurring frontiers |
| 63 | 101.41 | 0.331 | — | — | Fine JFA - refresh transported closest-point codes |
| 64 | 101.42 | 0.306 | 59.1 | 73.9 | Power coarse level set - persistent schedule |
| 65 | 101.77 | 0.023 | — | — | Prepare accepted structured dynamics worksets |
| 66 | 101.82 | 0.371 | 46.8 | 67.6 | Flatten structured boundary carry probes |
| 67 | 102.24 | 19.650 | 0.7 | 1.9 | Advect structured families |
| 68 | 121.96 | 0.017 | — | — | Commit advected structured families |
| 69 | 122.00 | 0.022 | — | — | Prepare accepted structured dynamics worksets |
| 70 | 122.05 | 0.020 | — | — | Force and constrain structured families |
| 71 | 122.09 | 0.032 | — | — | Fuse structured divergence RHS rows |
| 72 | 122.16 | 0.006 | — | — | Zero dry-identity divergence RHS rows |
| 73 | 122.20 | 43.510 | 0.3 | 0.4 | Octree persistent MGPCG - whole solve in one workgroup |
| 74 | 165.78 | 0.023 | — | — | Prepare accepted structured dynamics worksets |
| 75 | 165.83 | 0.023 | — | — | Mark structured overhead separation rows |
| 76 | 165.88 | 0.017 | — | — | Project structured families |
| 77 | 165.93 | 0.054 | — | — | Reconstruct projected structured rows |
| 78 | 166.05 | 0.002 | — | — | Fine JFA - clear recurring frontier masks |
| 79 | 166.07 | 0.002 | — | — | Fine JFA - mark recurring dirty frontier |
| 80 | 166.10 | 0.001 | — | — | Fine JFA - build recurring frontier 6 |
| 81 | 166.13 | 0.001 | — | — | Fine JFA - build recurring frontier 5 |
| 82 | 166.15 | 0.001 | — | — | Fine JFA - build recurring frontier 4 |
| 83 | 166.17 | 0.002 | — | — | Fine JFA - build recurring frontier 3 |
| 84 | 166.20 | 0.001 | — | — | Fine JFA - build recurring frontier 2 |
| 85 | 166.22 | 0.002 | — | — | Fine JFA - build recurring frontier 1 |
| 86 | 166.25 | 0.001 | — | — | Fine JFA - build recurring frontier 0 |
| 87 | 166.27 | 0.005 | — | — | Fine JFA - finalize recurring frontier dispatches |
| 88 | 166.32 | 0.412 | 16.6 | 16.4 | Fine JFA - seed closest points |
| 89 | 166.79 | 0.002 | — | — | Fine JFA - cooperative flood A to B stride 8 |
| 90 | 166.81 | 0.002 | — | — | Fine JFA - cooperative flood B to A stride 4 |
| 91 | 166.84 | 0.853 | 53.3 | 72.6 | Fine JFA - cooperative flood A to B stride 2 |
| 92 | 167.67 | 0.005 | — | — | Reset global volume reduction |
| 93 | 167.72 | 0.853 | 53.2 | 72.6 | Fine JFA - cooperative flood B to A stride 1 |
| 94 | 168.62 | 0.853 | 53.6 | 72.6 | Fine JFA - cooperative flood A to B stride 1 |
| 95 | 169.51 | 0.009 | — | — | Prepare exact compact coarse volume dispatch |
| 96 | 169.55 | 0.011 | — | — | Reduce compact coarse volume partials |
| 97 | 169.58 | 0.853 | 53.2 | 72.6 | Fine JFA - cooperative flood B to A stride 1 |
| 98 | 169.58 | 0.057 | — | — | Finalize compact coarse volume |
| 99 | 169.67 | 0.042 | — | — | Prepare active global fine volume dispatch |
| 100 | 170.47 | 0.238 | 16.6 | 23.4 | Fine JFA - resolve A to B |
| 101 | 170.76 | 0.029 | — | — | Fine JFA - prepare warm fallback |
| 102 | 170.83 | 0.334 | 57.6 | 74.0 | Fine JFA - refresh full-support fallback closest-point codes |
| 103 | 171.20 | 0.411 | 16.6 | 17.4 | Fine JFA - seed full-support fallback closest points |
| 104 | 171.66 | 0.717 | 53.9 | 73.2 | Fine JFA - complete support fallback 1 stride 8 |
| 105 | 172.42 | 0.784 | 35.6 | 45.6 | Fine JFA - complete support fallback 2 stride 4 |
| 106 | 173.26 | 0.850 | 47.0 | 71.5 | Fine JFA - complete support fallback 3 stride 2 |
| 107 | 174.18 | 0.853 | 50.7 | 72.2 | Fine JFA - complete support fallback 4 stride 1 |
| 108 | 175.09 | 0.854 | 45.7 | 72.1 | Fine JFA - complete support fallback 5 stride 1 |
| 109 | 175.97 | 1.360 | 44.1 | 71.4 | Fine JFA - complete support fallback 6 stride 1 |
| 110 | 177.43 | 0.909 | 53.5 | 72.6 | Fine JFA - complete support fallback collar repair 3 |
| 111 | 178.38 | 0.854 | 53.4 | 72.7 | Fine JFA - complete support fallback collar repair 4 |
| 112 | 179.28 | 0.853 | 53.1 | 72.6 | Fine JFA - complete support fallback collar repair 5 |
| 113 | 180.19 | 0.250 | 15.7 | 21.2 | Fine JFA - resolve complete support fallback |
| 114 | 180.49 | 0.034 | — | — | Fine JFA - finalize |
| 115 | 180.55 | 0.034 | — | — | Fine JFA - commit distances |
| 116 | 180.70 | 1.140 | 16.6 | 63.4 | Reduce resident fine overlap partials |
| 117 | 181.90 | 0.040 | — | — | Finalize global fine volume |
| 118 | 181.99 | 0.023 | — | — | Apply bounded global fine normal correction |
| 119 | 182.03 | 1.140 | 16.6 | 65.3 | Re-reduce corrected global fine volume |
| 120 | 183.22 | 0.038 | — | — | Finalize first corrected global fine volume |
| 121 | 183.31 | 0.025 | — | — | Apply residual bounded global fine normal correction |
| 122 | 183.35 | 1.140 | 16.6 | 63.5 | Measure twice-corrected global fine volume |
| 123 | 184.55 | 0.038 | — | — | Finalize measured global fine volume |
| 124 | 184.61 | 0.009 | — | — | Finalize global fine publication |
| 125 | 184.65 | 0.020 | — | — | Settle deferred global fine publication |
| 126 | 184.69 | 0.014 | — | — | Settle deferred rejected fine work payload |
| 127 | 184.73 | 0.010 | — | — | Fine-to-coarse restriction - prepare |
| 128 | 184.76 | 0.269 | 19.0 | 35.1 | Fine-to-coarse restriction - restrict |
| 129 | 185.08 | 0.026 | — | — | Fine-to-coarse restriction - publish |
| 130 | 185.13 | 0.128 | 0.3 | 0.7 | Power coarse level set - publish exact dispatch |
| 131 | 185.29 | 0.369 | 59.1 | 73.9 | Power coarse level set - persistent schedule |
| 132 | 185.69 | 0.013 | — | — | Prepare direct fine-summary publication |
| 133 | 185.73 | 0.007 | — | — | Validate exact fine-summary delta |
| 134 | 185.76 | 0.011 | — | — | Validate fine-summary coarse rows |
| 135 | 185.79 | 0.014 | — | — | Retire direct coarse summary ranks |
| 136 | 185.83 | 0.010 | — | — | Remove direct fine-summary page references |
| 137 | 185.87 | 0.005 | — | — | Prepare sparse fine-summary directory-page reclamation |
| 138 | 185.90 | 0.001 | — | — | Reclaim empty fine-summary directory pages |
| 139 | 185.92 | 0.083 | — | — | Ensure sparse fine-summary directory pages |
| 140 | 186.03 | 0.014 | — | — | Ensure sparse coarse-summary directory pages |
| 141 | 186.07 | 0.039 | — | — | Ensure compact fine-summary ranks |
| 142 | 186.13 | 0.012 | 1.0 | 0.8 | Ensure compact coarse-summary ranks |
| 143 | 186.17 | 0.015 | — | — | Add direct fine-summary page references |
| 144 | 186.21 | 0.012 | — | — | Publish direct coarse summary ranks |
| 145 | 186.24 | 0.006 | — | — | Prepare active fine-summary mip dispatch |
| 146 | 186.27 | 0.130 | — | — | Recompute direct fine-summary bases |
| 147 | 186.43 | 0.039 | — | — | Recompute direct fine-summary parents level 1 |
| 148 | 186.49 | 0.033 | — | — | Recompute direct fine-summary parents level 2 |
| 149 | 186.55 | 0.034 | 17.7 | 22.0 | Recompute direct fine-summary parents level 3 |
| 150 | 186.61 | 0.037 | — | — | Recompute direct fine-summary parents level 4 |
| 151 | 186.67 | 0.041 | — | — | Recompute direct fine-summary parents level 5 |
| 152 | 186.71 | 0.005 | — | — | Stamp octree topology attempt generation |
| 153 | 186.74 | 0.229 | — | — | Publish direct fine-summary directory and active mip |
| 154 | 187.02 | 0.026 | — | — | Initialize structured air-support publication - settled-fine |
| 155 | 187.07 | 0.956 | 28.8 | 11.8 | Publish structured air-support identities - settled-fine |
| 156 | 188.08 | 0.911 | 6.1 | 9.3 | Close sparse fine air-support demand - settled-fine |
| 157 | 189.03 | 0.037 | — | — | Emit sparse fine air-support demand - settled-fine |
| 158 | 189.11 | 0.239 | 1.2 | 0.3 | Canonicalize footprint air-support identities - settled-fine |
| 159 | 189.40 | 0.027 | — | — | Publish canonical air-support candidates - settled-fine |
| 160 | 189.46 | 0.158 | 4.5 | 7.9 | Resolve and hash live adaptive air owners - settled-fine |
| 161 | 189.67 | 0.469 | 2.8 | 3.5 | Extrapolate structured ordinary faces and reconstruct support vectors - settled-fine |
| 162 | 190.20 | 29.448 | 11.1 | 25.8 | March Section 5 sparse changed frontier to a fixed point - settled-fine |
| 163 | 219.69 | 0.870 | 3.9 | 7.8 | Reconstruct Section 5 air-support vectors - settled-fine |
| 164 | 220.65 | 0.091 | 0.0 | 0.0 | Prepare inactive owner-page generation |
| 165 | 220.80 | 0.007 | — | — | Commit inactive owner-page candidate bank |
| 166 | 220.83 | 1.900 | 0.1 | 0.3 | Compare topology-tile refinement signatures |
| 167 | 222.78 | 0.024 | — | — | Build exact structural topology-tile delta |
| 168 | 222.83 | 0.497 | 0.1 | 0.6 | Octree reset and refinement |
| 169 | 223.37 | 2.630 | 1.0 | 3.0 | Octree resident grading closure |
| 170 | 226.07 | 0.080 | — | — | Build exact wet-frontier tile delta |
| 171 | 226.19 | 0.008 | — | — | Begin persistent octree leaf frontier |
| 172 | 226.25 | 0.749 | — | — | Classify exact dirty-tile frontier candidates |
| 173 | 227.04 | 0.077 | — | — | Sort dirty frontier candidates by level and Morton |
| 174 | 227.16 | 0.385 | 0.0 | 0.0 | Sorted old/new frontier merge |
| 175 | 227.61 | 0.027 | — | — | Inactive octree pressure-row candidate dirty-row deterministic scan |
| 176 | 227.66 | 0.112 | — | — | Inactive octree pressure-row candidate |
| 177 | 227.82 | 0.014 | — | — | Inactive octree pressure-row candidate row-delta one-ring publication |
| 178 | 227.87 | 0.027 | — | — | Inactive octree pressure-row candidate row-delta compact publication |
| 179 | 227.93 | 0.014 | — | — | Inactive octree pressure-row candidate row-delta validate publication |
| 180 | 227.97 | 0.011 | — | — | Prepare power descriptor control |
| 181 | 228.01 | 0.096 | — | — | Resolve structurally dirty power descriptors |
| 182 | 228.16 | 0.027 | — | — | Stage exact power descriptor carry and affected rows |
| 183 | 228.21 | 0.019 | 0.0 | 0.0 | Prepare power topology delta |
| 184 | 228.26 | 0.024 | — | — | Resolve affected power topology rows |
| 185 | 228.33 | 0.030 | — | — | Stage exact power topology carry and affected rows |
| 186 | 228.39 | 0.009 | — | — | Begin direct structured publication |
| 187 | 228.42 | 0.091 | — | — | Classify direct structured catalog slots |
| 188 | 228.56 | 0.090 | — | — | Prefix six structured velocity families |
| 189 | 228.71 | 0.039 | — | — | Scatter direct structured velocity slots |
| 190 | 228.79 | 0.031 | — | — | Publish direct Section 6.3 rows and worksets |
| 191 | 228.86 | 0.135 | — | — | Finalize direct structured publication |
| 192 | 229.04 | 0.041 | — | — | Reconstruct direct structured cell velocities |
| 193 | 229.13 | 0.137 | 3.7 | 2.8 | Transfer accepted velocity to changed topology faces |
| 194 | 229.31 | 0.039 | — | — | Reconstruct direct structured cell velocities |
| 195 | 229.38 | 0.008 | — | — | Prepare structured boundary candidate-ready transaction |
| 196 | 229.41 | 0.056 | — | — | Classify fine-over-coarse liquid rows |
| 197 | 229.51 | 0.116 | 7.8 | 40.5 | Resolve canonical free-surface boundary slots |
| 198 | 229.68 | 0.009 | — | — | Resolve canonical solid boundary slots |
| 199 | 229.71 | 0.010 | — | — | Commit canonical structured boundary slots |
| 200 | 229.75 | 0.122 | — | — | Rebuild symmetric structured boundary rows |
| 201 | 229.92 | 0.010 | — | — | Structured boundary worksets - count row classes |
| 202 | 229.95 | 0.012 | — | — | Structured boundary worksets - count family classes |
| 203 | 229.99 | 0.029 | 0.1 | 0.1 | Structured boundary worksets - scan blocks |
| 204 | 230.04 | 0.012 | — | — | Structured boundary worksets - scatter rows |
| 205 | 230.08 | 0.012 | — | — | Structured boundary worksets - scatter families |
| 206 | 230.12 | 0.013 | — | — | SPGrid V-cycle - select setup delta and capture changed L1 |
| 207 | 230.16 | 0.014 | — | — | SPGrid V-cycle - capture plan L1 delta |
| 208 | 230.70 | 0.036 | — | — | SPGrid V-cycle - capture reduce L1 delta |
| 209 | 230.75 | 0.013 | — | — | SPGrid V-cycle - build inactive exact level deltas |
| 210 | 230.83 | 0.005 | — | — | SPGrid V-cycle - candidate apply skip |
| 211 | 230.86 | 0.023 | — | — | SPGrid V-cycle - candidate prepare live schedules |
| 212 | 230.91 | 0.010 | — | — | SPGrid V-cycle - candidate commit changed L1 |
| 213 | 230.98 | 0.033 | — | — | SPGrid V-cycle - candidate clear levels |
| 214 | 231.03 | 2.060 | 0.7 | 0.3 | SPGrid V-cycle - candidate build level sets |
| 215 | 233.14 | 0.029 | — | — | SPGrid V-cycle - candidate detect ghosts |
| 216 | 233.19 | 0.045 | — | — | SPGrid V-cycle - candidate insert ghosts |
| 217 | 233.28 | 0.844 | 0.1 | 0.1 | SPGrid V-cycle - candidate build level deltas |
| 218 | 234.17 | 0.025 | — | — | SPGrid V-cycle - candidate count transfers |
| 219 | 234.22 | 0.042 | — | — | SPGrid V-cycle - candidate scan transfers |
| 220 | 234.30 | 0.053 | — | — | SPGrid V-cycle - candidate write transfers |
| 221 | 234.39 | 1.300 | 0.4 | 0.9 | SPGrid V-cycle - candidate link parent chains |
| 222 | 235.73 | 0.063 | 0.0 | 0.0 | SPGrid V-cycle - candidate mark brick occupancy |
| 223 | 235.82 | 0.011 | — | — | SPGrid V-cycle - candidate rank bricks |
| 224 | 235.85 | 0.051 | — | — | SPGrid V-cycle - candidate scatter ranked slots |
| 225 | 235.93 | 0.014 | — | — | SPGrid V-cycle - candidate mark page occupancy |
| 226 | 235.97 | 0.011 | — | — | SPGrid V-cycle - candidate compact pages |
| 227 | 236.00 | 0.019 | — | — | SPGrid V-cycle - candidate link page neighbours |
| 228 | 236.05 | 0.138 | 1.1 | 2.1 | SPGrid V-cycle - candidate build stencils |
| 229 | 236.24 | 0.274 | — | — | SPGrid V-cycle - candidate publish spectral bounds |
| 230 | 236.56 | 0.012 | — | — | SPGrid V-cycle - candidate validate hierarchy |
| 231 | 236.60 | 0.012 | 0.0 | 0.0 | Validate complete inactive topology epoch |
