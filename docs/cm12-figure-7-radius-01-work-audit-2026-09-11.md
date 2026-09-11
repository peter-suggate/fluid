# Figure 7, radius 0.1 m: adaptive-mass work audit

This is a code audit and CPU production-initialization census of the current
dirty working tree on 2026-09-11 (HEAD `7a6b36e61f4c9cc19d7b8a6266fa87e0e7093225`),
plus a reading of the user's existing browser timing panel. At the time of this
baseline audit, no solver code was changed and no new Dawn timing run was
performed: another test held the WebGPU lease when inspected, and the user's
paused browser continued rendering. Subsequent implementation and isolated
Dawn measurements are recorded in the
[compact tile execution document](sparse-cm12-compact-tile-execution-proposal-2026-09-11.md).

Reproduce the exact reset census without a GPU:

```sh
node --import tsx tools/census-cm12-figure-7-radius-01.ts
```

The machine-readable output is
[`cpu-census.json`](../artifacts/cm12-figure-7-radius-01/cpu-census.json).
It includes the complete scene, resolved method values, 64 wet-cell coordinates
and densities, and all 36 initial brick coordinates/resolutions.

## Scene and counting conventions

The open browser URL confirms Figure 7 with sphere radius **0.1 m**, center
**(0, 4.5, 0) m**, B8, coarse-first, scene timestep, and pressure relative
tolerance **0.194**. The other relevant controls show P8, pressure budget 128,
one gamma diffusion round, seven sharpening substeps with D=2.1, eight capacity
repair rounds, topology cadence 1 and budget 64. Markers and column-height
reconstruction are off. The authored box is **6.4³ m**, finest spacing **0.05 m**,
**128³ = 2,097,152** logical cells, gravity **10 m/s²**, timestep **1/30 s**,
zero tank fill and zero rigid bodies.

Sources: `lib/core/cm12-paper-scenes.ts` (`cm12Domain`, `createCm12Figure7`),
`lib/core/initial-fluid.ts` (`initialLiquidFractionAtCell`), and
`lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts` (atlas construction).

There are four different counts, which must not be conflated:

* **Nonzero density:** cells actually carrying some liquid.
* **Accepted:** current cell/row topology, including represented air.
* **Capacity:** all-rung templates and reserved growth pages; mostly unused.
* **Dispatched versus useful:** a GPU invocation can reject an inactive slot,
  a dry row or an unset mask. Counts below describe the dispatch domain unless
  explicitly called useful arithmetic. A kernel can make many accesses per item.

All exact reset counts refer to construction, before the first transport and
conditioning. Later masks, pressure counts and topology deltas depend on the
evolving state; the scene definition alone cannot give exact future counts.

## Exact reset census

The radius is two finest cells. Geometric volume is 0.004188790204786391 m³,
equivalent to **33.510321638291124** full finest cells. Initialization is eight
point samples per cell at offsets ±0.4h, not an analytic sphere/cube integral.
It produces this exact histogram:

| Density | Cell count | Integrated finest-cell volumes |
|---|---:|---:|
| 1 | 8 | 8 |
| 0.625 | 24 | 15 |
| 0.25 | 24 | 6 |
| 0.125 | 8 | 1 |
| Total nonzero | **64** | **30** |

The sampled mass is **0.00375 m³**. Its difference from geometric volume is
initial quadrature error, not simulation mass loss. Wet indices are the 4×4×4
box x=62..65, y=88..91, z=62..65. **32** cells exceed rho=0.5 and seed velocity
extension. Pressure uses its own density/submergence predicate after conditioning;
32 is not a claim about the first completed pressure solve.

The four liquid-containing bricks have coordinates `(7,11,7)`, `(8,11,7)`,
`(7,11,8)`, `(8,11,8)`. Each allocates 8³ cells. The initial air-support rule
continues the same resolution for `ceil((8+1)/8)=2` bricks along each face-normal
column. The deduplicated result is **32 additional dry B8 bricks**, hence:

| Representation | Exact count |
|---|---:|
| Liquid-containing bricks / dry support bricks | 4 / 32 |
| Active bricks, all at B8 | **36** |
| Accepted cells | **18,432** |
| Accepted cells with zero density | **18,368** |
| Intra-brick rows | 48,384 = 36 × 3 × 7 × 8² |
| Between-brick rows | 3,840 |
| Sparse-air boundary rows | 6,144 |
| Mixed-seam rows | 0 |
| Total accepted rows | **58,368** |
| Row terms / cell incidences | **110,592** |
| Rows incident to any nonzero-density cell | **240** |

Thus accepted cells are **288×** the nonzero-cell count, and **614.4×** the
integrated full-cell-equivalent volume. This is local sparsity, but coarse
allocation granularity plus a deliberately generous air band.

Sources: `sparse-brick-atlas.ts` (`matchedAirSupportLayerCount`,
`atlasWithInitialAirSupport`, `sparseCM12InitialActiveBrickKeys`) and
`sparse-atlas-composite-projection.ts` (`buildSparseAtlasCompositeGrid`).

## Exact capacity amplification

All 36 initial leaves are mutable, and the small catalogue fits all-rung backing.
The template packer creates **21,060 cells** = 36×(1+8+64+512), and
**69,780 rows** across valid rung/seam variants.

The growth demand is computed from **initiallyActiveBrickKeys.size**, not the
number of wet bricks:

```
max(1, 12 × 36 − inactiveAuthoredBricks) = 432 pages
```

Here inactiveAuthoredBricks=0. The compact curved-volume cap is 1,024, so all
432 requested pages are reserved. Each growth page has **512 cell slots** and
**1,728 face-row slots** = 3×9×8². Consequently:

* Leaf capacity = 36+432 = **468**.
* Cell capacity = 21,060+432×512 = **242,244**.
* Row capacity = 69,780+432×1,728 = **816,276**.
* Signed-world hash capacity = nextPowerOfTwo(2×468) = **1,024 entries**.
* Stable packet storage = 468×64 = **29,952 packet addresses**.
* B8 direct execution packet domain = 468×8 = **3,744 packets**.
* Initial accepted packet domain = 36×8 = **288 packets** of 64 cells.
* Mass-departure cache capacity = 56×242,244 = **13,565,664 bytes**.

The user panel reports **156.8 MiB** overall allocation; that is a rounded UI
measurement, not an exact allocation census from this CPU tool.

Source: `webgpu-sparse-cm12-resident.ts` construction around lines 4090–4220;
`sparseCM12TopologyPagePoolPlan`; `sparse-cm12-world-directory.ts`;
`sparse-cm12-transport-packet-authority.ts`.

## Every advance stage, in execution order

The following is the actual encoder, not just the diagram descriptions.
`webgpu-sparse-cm12-resident.ts` around lines 6200–6940 is authoritative.
Some diagram prose still describes pressure aggregates/hierarchies that the
current production encoder explicitly no longer computes.

| Stage | Work and exact reset-sized execution domain |
|---|---|
| **1. Transport velocity extension** | Seal frame/body/boundary authority; copy indirect arguments; moving-solid work is bypassed with zero bodies. Compile/cache accepted packet addresses. Initialize masks and run **8** extension sweeps. Stable compact schedule: **288 packets ×64 lanes ×9 passes =165,888 lane invocations**. A schedule-rebuild frame intentionally uses the full **3,744 packets ×64×9 =2,156,544 lanes**, with invalid/complete packet exits. Then two packet-authority compiler passes each inspect **3,744 ordinals**, dispatched as **59 groups / 3,776 lanes**. One compiler builds masks using up to 27 neighboring spatial tiles per B8 packet; the other compacts the lists. |
| **2. Face preparation** | Two **468-workgroup**, 256-thread leaf passes clear retired support and publish accepted-cell velocity support. Publication writes **18,432 vec4 records**; invalid leaves exit. Then **912 groups / 58,368 rows** inspect support and either zero unsupported faces or RK2-trace supported ones. Reset extension produces **5,940 traced rows**; see exact support calculation below. |
| **3. Conservative transport** | Clear six receipt planes over **18,432 cells**, exactly **110,592 32-bit stores / 442,368 bytes**. Then backward departure/gamma/beta trace, forward deficit scatter, and conservative gather of density, gamma and three momentum components. Selected packets have up to 64 cells each and an eight-donor cached stencil. If M packets are selected, each main pass dispatches **64M lanes**, M≤288 for the initial accepted topology. Dirty velocity, scalar masks and generation changes determine M; all 288 can be selected. Coarse packing has no B8 cells to pack at reset. |
| **4. Tracer advection** | Markers are off: **zero marker dispatches / points**. Enabling the overlay creates a separate marker workload, not fluid particles needed for the solver. |
| **5. Gamma diffusion** | One iteration: clear gamma receipts over **18,432 cells**, scatter over **58,368 accepted rows**, finalize over **18,432 cells**. Total **95,232 item visits**, with scalar/row eligibility guards and paired fixed-point transfers. Additional iterations add intermediate snapshot commits, but are not enabled here. |
| **6. Surface sharpening** | Native clears of one receipt plane and one sharpening-delta plane over **all 242,244 capacity cells each: 1,937,952 bytes**. Compile sharpening packets where applicable, compute dose and trace/scatter local return over the selected packet list, then finalize over **18,432 accepted cells**. The configured trace has at most **7 half-cell substeps**, bounded by D=2.1 and early exits; each density-gradient evaluation can require multiple interpolations. Exact trace counts are data dependent. |
| **7. Density capacity repair** | **8 rounds ×3 accepted-cell passes**: initialize receipts, scatter excess, finalize. Exactly **442,368 cell visits** for this reset-sized topology, even if no excess needs redistribution. Scatter's costly neighbor work is conditional on excess; that does not remove the initialize/finalize dispatches. |
| **8. Scalar publication** | Begin/seal final-scalar masks and publish frame scalar completion. Mask publication dispatches **468 groups ×64 lanes**, each group loops over **8 B8 packets**: **239,616 packet-lane fact evaluations**, with **18,432 valid accepted cells** at reset and invalid capacity slots elsewhere. |
| **9. Body forces** | Gravity/scene acceleration on **58,368 accepted rows**, **912 groups**; physical support/solid predicates determine the actual velocity update. No rigid-body coupling or inflow geometry in this scene. |
| **10. Pressure topology** | Begin membership/cache/repair receipts. Bootstrap cell classification can visit **18,432 cells**; the ordinary dirty-cell pass also dispatches the accepted cell list and tests flip masks. Repair compact dirty membership leaves. **Row publication visits all 816,276 capacity rows**, dispatched as **12,755 groups / 816,320 lanes**, and republishes **25,509 membership words**. Accepted rows inspect incident scalar-change flags; other rows are rejected. Fine coefficient work uses compact pressure cells. The frozen membership copy visits **7,571 cell-bit words**, padded to **7,616 lanes**. Finalize execution-image and topology receipts. The named coarse/hierarchy seams encode no numerical aggregate/hierarchy work in current production. |
| **11. Pressure RHS / initialization** | Begin solve; compact-pressure-cell initialization of RHS, Jacobi direction, true residual and pipelined image, with scalar reductions; publish dispatch gate and copy it. If P pressure cells exist, each cell kernel uses **ceil(P/64) groups**. The reset has 32 cells above rho=.5, but actual P here is determined after transport/diffusion/sharpening. |
| **12. Pressure solve** | Jacobi-preconditioned pipelined CG, using compact pressure cells and their incidence operator. Budget **128×3=384** main dispatches. Fifteen intermediate eight-iteration checkpoints each encode **8** dispatches; the final residual tail encodes **3**. Total **507 compute dispatch commands**, plus **32 buffer copies** for gated indirect arguments. Early convergence can zero subsequent cell dispatches, but scalar reductions/checkpoint commands remain encoded and guard themselves. This schedule cost is substantial for a solve with only tens of cells. |
| **13. Velocity projection** | Open activity epoch; scan compiled interior tiles and seam/sparse-air packets with pressure/dirty masks, plus accepted dynamic-row pass; enforce inflow on the accepted row list (no inflow here). Collocate and diagnose over **18,432 accepted cells** and their incidence, reduce divergence and publish face completion. Physical projection count is mask dependent; dispatch includes immutable face-address domains and accepted-row/cell domains, not solely P liquid cells. |
| **14. Activity measurement / frontier** | Scalar mask publication over **468 leaf groups**; topology/history scalar passes over **468 slots**, padded to 512 lanes. Activity measurement dispatches **468 groups**, rejects clean/invalid leaves, and scans the cells and incidence of dirty accepted bricks (up to **18,432 cells / 110,592 incidences** at reset). Computes mass, moments, deformation, velocity, surface/thin-feature, curvature and swept support; ages history and seals census. Frontier allocation considers **36×26=936 accepted-leaf/neighbor pairs**, padded to 960 lanes; demands are masked. Directory finalization visits **1,024 hash slots**, and page synthesis dispatches **432 page groups**. |
| **15. Resolution planning** | Frontier classification dispatches **468 groups** and explicitly examines **26 neighbors per slot =12,168 neighbor directions**; its entry does not reject inactive leaves before these lookups. Policy-tile classification is bypassed for scale-one policy tiles, though scheduling/metadata work remains. Plan, activate and retire; **3 grading rounds** for the B8 ladder; validate and schedule up to 64 ordinary preparations (urgent work has a separate lane). Build shadow leaf/structure/row lists and copy their GPU-authored indirect arguments. Scalar leaf scans cover 468 slots; full leaf-group scans also use capacity. |
| **16. Candidate transfer** | Delta-indirect cell/face transfer, shadow-face validation, effects preflight, boundary-image construction/validation, transport-image compilation, authorization, pressure-effect publication, seal, accepted-state publication, retired-image replay. Numerical transfer is **delta-sized**, and can be zero. Singleton transaction work still exists, and frontier connection/acceptance/execution-image passes each dispatch **432 page groups** and check page state. Exact delta sizes are not inferable from the authored sphere alone. |
| **17. Brick retirement** | This stage publishes post-commit activity masks; the retirement decision already happened in planning. **8×64=512 lanes** cover **468 leaf slots**, then one mask finalizer. |
| **18. Presentation publication** | Allocate/sort page directory; build/seal frame plan and compact publication packet; execute/commit selected page payloads; verify, publish surface coarsening proofs, reject faults; retire/compact pages and commit frame control. **Nine separate capacity-sized leaf-group passes** in the frame-plan/publication helper alone (**9×468=4,212 groups**), plus singleton passes, allocator/retirement scans and indirect payload work. Reset has **36 P8 pages / 18,432 scalar sample slots**; changed tile/page masks determine how many need exact resampling. Column-height solve is off. Renderer mesh/ray rendering is outside this advance stage. |

In addition, each of the 18 stage entries performs a four-byte failure-receipt
copy, and frame-plan publication has one more. These enforce pass boundaries
even with timing disabled. Hardware capture adds its substage pass boundaries;
the displayed measurement is the captured schedule. CPU planning, encoding,
submission, queue waiting, and renderer work must not be added to this exclusive
GPU stage partition as though they were extra fluid stages.

### First extension and face-work cross-check

The CPU census walks the exact reset composite graph. All leaves are unit-width,
so physical opposite-side adjacency is six-neighbor adjacency; no mixed-seam
approximation is involved. Starting from the 32 rho>.5 seeds, valid cells after
depth 0 through 8 are:

```
32, 80, 160, 280, 448, 672, 960, 1,320, 1,760
```

Exactly **5,940** accepted rows touch those final valid cells. At initial zero
velocity, RK2 chooses one substep: initial velocity sample, midpoint velocity
sample, terminal face sample. Each uses eight corners: **142,560 logical
corner queries**. These are source-level calls, not distinct addresses or
measured hardware memory transactions. The first face stage still performs
the support predicates for all **58,368 rows** and their **110,592 terms**.

Shader anchors: `sparse-cm12-velocity-extension.wgsl.ts` (`seal...Schedule`,
`initializeVelocityExtensionPackets`, `advanceVelocityExtensionPackets`);
`webgpu-sparse-cm12-resident.wgsl.ts` (`prepareTransportFaceRow`,
`traceFaceDeparture`, `sampleFaceVelocitySupport`).

## Existing browser measurement

The paused tab was at **0.8000 s**. The panel explicitly reports a **12-sample
hardware-timestamp mean**. These are readings of that existing panel, not new
Dawn measurements and not reset-frame timings:

| Stage | Mean ms |
|---|---:|
| Velocity extension | 1.05 |
| Face preparation | 1.57 |
| Conservative transport | 1.11 |
| Tracers | 0.00 |
| Gamma diffusion | 0.13 |
| Sharpening | 0.66 |
| Capacity repair | 0.26 |
| Scalar publication | 0.20 |
| Body forces | 0.07 |
| Pressure topology | 0.72 |
| Pressure RHS | 0.33 |
| Pressure solve | 4.85 |
| Projection | 0.33 |
| Activity/frontier | 1.90 |
| Resolution planning | 0.98 |
| Candidate transfer | 0.52 |
| Retirement mask | 0.07 |
| Presentation | 0.46 |
| Displayed total | **15.20** |

Rounded row values sum to 15.21 ms. **10.35 ms** of the displayed total is outside
the pressure-solve stage. The panel also reports 40 resident bricks/pages,
17/128 executed/encoded pressure iterations, and a matched pressure receipt with
16,672 accepted cells, 52,400 accepted rows, 20 pressure cells and 104 pressure
rows. However, its topology attribution explicitly says **unavailable**, and
the pressure receipt generation (24) differs from its end-frame topology label
(14). These asynchronously displayed counters must not be presented as one
fully synchronized frame census or multiplied into the 12-frame mean.

## What explains the excessive work

1. **The support footprint is large before any time integration.** Sixty-four
   liquid-bearing cells become 18,432 accepted cells. Some support is necessary;
   full B8 bricks plus two face-normal brick layers are the current policy,
   not a theoretical lower bound.
2. **Growth headroom compounds that footprint.** Twelve times the already
   air-expanded active set reserves 432 pages. Using four wet bricks would
   request 48 pages under the same multiplier, but that comparison is only a
   sizing hypothesis: shrinking it requires swept-frontier/live-insertion tests.
3. **Reserved capacity leaks into repeated work.** Full row membership,
   scalar-mask loops, frontier classification, presentation metadata passes,
   sharpening clears, and topology-change VEX all pay some capacity-shaped cost.
   A guard after dispatch is cheaper than full arithmetic, but is not free.
4. **Accepted air remains a broad scan domain.** Face support, row preparation,
   diffusion, receipt clearing, 24 capacity passes, collocation and activity
   census process accepted air even when the useful wet subset is tiny.
5. **Pressure arithmetic is small but its fixed command graph is large.** The
   507-dispatch ceiling schedule persists when convergence happens early. The
   displayed 4.85 ms cannot be interpreted as 4.85 ms of useful arithmetic on
   20 pressure cells, nor can iteration-count scaling predict its speedup.

The first targets to measure are capacity-only row publication/frontier work,
generation-change VEX versus cached VEX, pressure command-tail overhead, and
air-support/headroom sizing. The browser already identifies activity/frontier,
face preparation and extension as substantial non-pressure costs. Exact
per-kernel timings, selected packet counts, and delta volumes at 0.8 s still
require a synchronized GPU capture; this report does not invent them.
