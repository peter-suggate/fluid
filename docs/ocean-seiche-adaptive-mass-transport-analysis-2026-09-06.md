# Ocean-seiche + adaptive-mass: transport-stage performance analysis

Source snapshot: `eaea9b9b`, inspected 2026-09-06. This is a source audit with existing GPU receipts, not a new GPU benchmark. No simulation code was changed and no browser or Dawn workload was started. Existing unrelated workspace changes were left alone.

The main cost is repeated work per represented surface/air cell: resolving adaptive owners, sampling overlapping stencils, traversing composite incidence, and publishing conservative receipts. Deep-water compression works extremely well, but does not eliminate that band. Execution is only partly proportional to useful adaptive work: some operations use compact packets, others scan accepted cells, leaf capacity, or spatial support tiles.

“Memory-bound” would be too precise a hardware diagnosis without counters. The code establishes dependent address loads, redundant queries, scattered atomics, and uneven lane work. It does not establish DRAM saturation, cache miss rates, register occupancy, or which of these dominates current elapsed time.

## Scene and adaptive work domains

[`createOceanSeicheScene`](../lib/core/scenes.ts) authors an 8 × 2.4 × 2 m tank on a 320 × 96 × 80 lattice at h = 0.025 m. Water fills 72 cells vertically; an extra slab at the negative-x wall launches the wave. Physical surface tension is zero, but numerical surface sharpening is separately controlled and can still run.

The [2026-09-05 initial-atlas reconstruction](sparse-cm12-ocean-transport-cost-investigation-2026-09-05.md) records the ordinary B8 scene with its surface coarsening bias:

| Physical cell width | Leaves | Represented cells | Initially wet cells |
|---|---:|---:|---:|
| h | 70 | 35,840 | 10,240 |
| 2h | 1,530 | 97,920 | 25,600 |
| 4h | 400 | 3,200 | 3,200 |
| 8h | 400 | 400 | 400 |
| 16h | 140 | 140 | 140 |
| 32h | 20 | 20 | 20 |
| Total | 2,560 | 137,520 | 39,600 |

There are 416,335 initial gradient rows. The 160 deepest cells represent 66.3% of liquid volume, but **97.3% of represented cells remain at h or 2h**, and 97,920 cells are initially dry. These are initial construction counts from the earlier reconstruction, not a census of a current evolving frame.

Keep four domains distinct:

* **Physical leaves and accepted rungs:** a leaf has spatial span and a chosen number R of cells per axis. Width is `B * span / R`; B8 does not mean every leaf contains 512 accepted cells.
* **Execution packets:** each workgroup has 64 lanes arranged as 4³ cells. An R1 leaf uses one lane; R2 uses eight; R4 uses 64; R8 uses eight full packets. Compact packet scheduling removes absent packets, but does not fill the unused lanes inside R1/R2 packets.
* **Accepted versus dirty:** accepted means the cell/row exists in the published topology. It does not mean its numerical state changed. Transport selects dirty packets; face preparation does not use the same dirty selection.
* **Capacity:** storage includes alternate rungs and frontier reservation. The prior ordinary construction reserved 399,664 cells, including 512 future B8 pages, while initially representing 137,520 cells. Capacity-sized clears and dispatches can therefore exceed accepted work substantially.

## Existing timing evidence and its limits

The following values come from one [2026-09-02 control receipt](../artifacts/ocean-seiche-min8-capacity-control-20260902.json): B8/P8, **full-domain minimum-cell-size 8**, dt = 0.016 s, 8 warmup and 24 measured frames, 110 ms capture gaps, hardware timestamps with 65.536 µs quantum. This much coarser override is not the ordinary scene above, and predates current VEX scheduling and face-support changes.

| Stage | Median ms | Informative substage medians, ms |
|---|---:|---|
| Velocity extension | 2.8180 | Eight sweeps 2.2938 |
| Face preparation | 3.8011 | Row traces 3.0147; support publication 0.8520 |
| Mass + gamma + momentum | 4.3909 | Trace 2.6214; deficit scatter 1.8350 |
| Surface sharpening | 4.2598 | Transform 2.7525; capacity repair 1.3107 |

Medians of components do not add to the median of their sum. These receipts establish that expensive trace and repair work existed even in a coarse scene; they do not measure the current ordinary scene or prove a current speedup. Historical labels such as “dirty face rows” and “dense support” also do not accurately describe today's implementation.

## Velocity extension

Host sequence: [`webgpu-sparse-cm12-resident.ts`](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts), `stage("transport-velocity-extension")` around line 6258. Numerical kernels: [`sparse-cm12-velocity-extension.wgsl.ts`](../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts).

### What it computes

The projected/collocated liquid velocity must extend into nearby dry cells before a front arrives. Otherwise characteristic tracing encounters an artificial zero-velocity boundary and pins the front.

1. Seal frame/body authority and process moving-solid activity. This overhead belongs to the displayed stage even when interpreting its number as “extension.”
2. Check a cached packet schedule against topology generation and accepted slot; rebuild accepted packet addresses only when they change.
3. Initialize liquid cells (`rho > 0.5`) with source velocity and depth zero. Dry cells receive zero effective velocity and invalid depth. Initialize packet validity masks.
4. Run eight dependent sweeps. Already valid cells retain their velocity. Invalid cells average valid neighboring velocities using absolute composite gradient coefficients as weights. A donor is eligible only when its recorded depth is less than the current sweep, enforcing Jacobi dependency depth despite publication to one effective-velocity plane.
5. Publish packet masks and frame receipts. Effective velocity is published when initialized or first reached; there is no necessary full-cell copy at the end of every sweep.
6. Compile the separate conservative-transport packet lists from final-scalar and velocity activity. This compiler is also charged to “Velocity extension.”

### Work and memory costs

The current production shader uses `cacheAcceptedPackets: true`. Stable frames select compact execution below 75% packet occupancy. Topology-changing frames deliberately use full direct execution to invalidate retired masks. Thus **“always capacity-sized VEX” is now obsolete**. The prior ordinary census has 3,050 valid packets versus a 24,576-group direct domain including reservation; steady compact execution removes most empty groups, but all eight sweeps remain.

Full-validity packets take a uniform shortcut: load/decode packet and mask, copy the mask, return. Partly filled packets still perform shared-state setup, synchronization and validity ballots. Initialization reads a packet descriptor per lane; this repeats logical loads but the comment explicitly records the tradeoff against importing shared topology state into that pipeline. It is not evidence that a broadcast rewrite would be faster.

Interior extension uses six arithmetic neighbor addresses only when the cell is strictly inside its leaf and `!hasStaticSolidVoxels()`. Other cells traverse `incidence → row → row terms → neighbor → accepted depth → effective velocity`. The two-term shortcut reduces the inner term loop; composite seams can require several terms. These are dependent gathers whose addresses are unavailable until preceding topology reads finish. Do not assume every ocean interior gets the arithmetic fast path: the static-solid predicate is global.

The effective velocity is a vec4 per cell; contiguous packet x-runs are favorable. Cross-leaf and coarse/fine accesses scatter to separate ranges. Each unreached cell repeats neighbor eligibility checks in later sweeps until reached or the eighth sweep ends. An eight-hop adaptive band also has a varying physical thickness: eight coarse-cell steps cover more metres than eight fine-cell steps.

### Hidden work in the packet compiler

[`compileSparseCM12TransportPacketsFromFinalScalarMasks`](../lib/methods/adaptive-mass/sparse-cm12-transport-packet-authority.wgsl.ts), line 203, scans the direct packet domain each frame. One invocation constructs each valid packet's full geometric mask with a 64-iteration lane loop. If topology or velocity has not already marked it dirty, it walks every covered 4-fine-cell spatial tile and its 3³ neighboring tiles looking for nonexact, nonbulk scalar support.

The bound before early exit is `27 * product(ceil(scale * packet.counts / 4))` tile checks per packet. A single width-32 cell covers 8³ such tiles: up to 13,824 checks, despite being one accepted cell. This is a particularly important mismatch with fully adaptive bricks: **proving a coarse cell clean can still revisit its fine-space footprint**. Dirty detection short-circuits, so this is a work bound, not a measured average. The second compiler pass atomically appends selected packets. A topology-owned summary of support per adaptive leaf could avoid repeated overlapping tile queries while preserving the dependency closure.

## Face preparation

Host stage around line 6333 in the resident; numerical helpers at lines 1999–2130 and 3099 onward in [`webgpu-sparse-cm12-resident.wgsl.ts`](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts). Face address programs are in [`sparse-cm12-brick-tile-face-address-program.wgsl.ts`](../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-address-program.wgsl.ts).

### What it computes

Clear retired support and republish one velocity/span/flags record per accepted physical cell. Dispatch interior face tiles, boundary/seam packets, positive sparse-air packets and the accepted dynamic-row suffix. For each accepted open row:

1. Probe either side of the face, at ± one quarter of row distance, for extended-velocity support.
2. Return zero or solid velocity if neither side has support. This is a support test, not a dirty test.
3. Trace the face centre backward using midpoint RK2, with 1–16 substeps and solid-segment clipping.
4. Interpolate velocity at the departure, take the face-axis component, and apply solid/open-fraction blending.

Ordinary sampling uses a fixed unit lattice; an explicit refinement-policy region can select larger spans. This distinction matters when comparing ordinary ocean with full-domain min8. Selecting interpolation spacing from the owner at each successive point is explicitly avoided because crossing a 2:1 seam would change the basis discontinuously.

### Where the repeated accesses occur

Every trilinear sample performs eight `faceVelocitySupportAt` calls. Each calls `compactOwnerCellAt`, which resolves the world directory, reads leaf span/origin/rung/range, reconstructs cell coordinates, checks accepted activity, and finally reads a four-float support record.

[`cm12WorldOwnerAt`](../lib/methods/adaptive-mass/sparse-cm12-world-directory.ts), line 301, tries increasing dyadic spans. Each exact lookup probes a hash table and conditionally reads atomic state/hash/coordinates/span/leaf fields. An atomic **load** is not an atomic-add contention event, but these reads still form dependent memory accesses. Adjacent queries often rediscover the same leaf; multiple finest corners inside a coarse cell can load the same final value repeatedly.

For S RK2 substeps, a supported row makes **2 + 8(2S + 1) owner queries**: two side probes, two velocity samples per substep, and the final departure sample. S = 1 gives 26; each additional substep adds 16. Boundary clipping is additional. At 416,335 rows, 26 per row would mean 10,824,710 owner queries if all rows were supported—an upper-work illustration, not an observed trace count.

The new support allocation is correctly proportional to physical cell capacity rather than world volume, but it achieves that by moving address reconstruction into the hottest interpolation loop. Neighboring rows can retrace similar paths and fetch overlapping corners without explicit shared address reuse. Quiet submerged supported rows still trace.

A packet-local owner/coordinate cache or an unchanged-stencil address cache is the strongest source-supported target. It must retain the current sampling lattice, corner order, boundary handling and topology validity. A new dirty skip would need velocity, timestep, force/boundary, topology and characteristic-donor dependencies. A scalar-only dirty flag is insufficient.

Reducing launch counts alone has already failed: the [seam-family split experiment](sparse-cm12-face-preparation-seam-family-experiment-2026-09-02.md) halved seam groups while leaving useful traces unchanged and regressed stage median by 7%. Register pressure/latency hiding was a hypothesis, not measured proof.

## Mass + gamma + momentum transport

Host stage around line 6357; kernels `traceGammaAndBeta`, `scatterDensityDeficit`, `gatherConservativeDensity` at lines 3345–3475 in the resident WGSL. The volume-aware coefficient definitions are in [`cm12-numerics.ts`](../lib/core/cm12-numerics.ts), lines 47–100.

### Actual algorithm and relationship to face preparation

This is a conservative semi-Lagrangian **cell departure-stencil** algorithm. The UI description of transport “through the same oriented composite face fluxes” is misleading if read literally. These three kernels sample effective **cell velocity**, build eight-corner weights and balance donor columns; they do not read prepared destination face velocities in their transport loops. Prepared face velocities remain necessary staggered velocity state for subsequent force/projection work. The stages share geometry and velocity provenance, but face preparation does not supply a cached departure trace that mass transport reuses.

`rho` is an intensive liquid-density/occupancy quantity; physical mass scales with cell volume. `gamma` is persistent cumulative transport-operator state, not another independent liquid species. `beta` measures how much of each donor's volume-weighted column the backward receiver rows requested.

1. Clear six integer receipt planes over **all accepted cells**: beta, density deficit, gamma deficit and three momentum deficits.
2. **Trace:** for each selected cell, trace its centre backward using RK2 on a fixed source-cell interpolation lattice. Build and cache eight donor IDs and weights. Sample source gamma and condition it. Atomically add each receiver's volume-weighted gamma/weight contribution to donor beta. Inspect stencil support for the later persistent-bulk clearance certificate, and publish sharpening source masks.
3. **Deficit scatter:** a donor with `beta < 1` by more than one fixed-point quantum traces forward, builds an arrival stencil, and distributes its missing fraction to receivers. It adds density, gamma and three momentum components using five fixed-point atomics per nonzero destination. With no visible receiver, the contribution returns to the donor.
4. **Gather:** revisit the cached backward stencil. The coefficient is `advectedGamma * normalizedWeight / max(1, donorBeta)`. Gather density, gamma row weight, and density-weighted velocity; add forward-deficit receipts. Publish destination rho/gamma and velocity = momentum/rho above the dry cutoff. Near-dry gamma retains prior state under the implemented cutoff rule.

For unequal volumes beta receives `(Vreceiver / Vdonor) * rowCoefficient`; forward deficit transfers use `(Vdonor / Vreceiver) * deficit * weight`. This volume accounting is essential across adaptive seams. Removing it would make coarse/fine interfaces create or lose integrated quantities.

### Cost inside the loops

The packet trace/scatter path stages a 27-logical-brick directory into workgroup memory before numerical work. [`cm12TeiStageDirectory`](../lib/methods/adaptive-mass/sparse-cm12-transport-execution-image.wgsl.ts), line 93, resolves 27 owners and stores two arrays of 27 vec4 records, then synchronizes. The trace and scatter dispatches each do this; gather explicitly skips directory staging.

This already amortizes hash lookup across lanes and samples. Nevertheless, `cm12TeiOwnerAtFine` still reloads the owner's three global origin coordinates for a cache hit, then decodes local coordinates and bounds. Samples outside the 3³ logical neighborhood fall back to the world directory. The cache covers logical brick coordinates, not an equal physical reach for every coarse cell; a large-cell characteristic can exceed it quickly.

A backward trace with S substeps makes `1 + 16S` owner queries, plus eight for the final stencil, excluding boundary clipping. A deficient donor repeats comparable work forward. Trace also loops over the eight corners for gamma, beta and bulk certification; these revisit donor state and geometry even though donor IDs are already known.

The departure cache is 56 bytes per cell: six packed words for eight 24-bit IDs and eight float weights. It avoids a second backward characteristic during gather. Its layout is cell-major: at a fixed corner instruction, neighboring cell lanes access words 56 bytes apart. It has good per-cell locality but poorer across-lane contiguity than a packet-transposed layout. Gather first reads eight weights for normalization, then rereads them; ID decoding expresses three word loads for each corner in a four-ID group. Compiler elimination/cache reuse may reduce physical traffic, so these are source-level repetition opportunities, not claimed DRAM transaction counts.

Beta adds up to eight atomics per receiver; deficit scatter adds up to 40 per deficient donor. Coarse owners can be repeated among corners and shared across neighboring lanes, increasing same-address serialization. Scalar fields occupy separate planes and donor velocities use four-float records: regular packet ranges are friendly, while donor gathers cross adaptive ranges. Combining repeated-target fixed-point contributions is worth testing only after preserving the existing rounding and overflow behavior.

The hybrid coarse path can trade staging/locality for lane utilization. Its selector enables an accepted-cell scan only when dirty coarse packets exceed `max(4, ceil(acceptedCells / 16))`; the packed kernels use direct owner resolution rather than the shared packet directory. Thus “pack every coarse cell” is not an unconditional win. Selected ordinary packets also use their whole valid-cell mask, not a mask containing only the individual dirty cells.

## Surface sharpening

Host stage around line 6416; resident WGSL lines 3818–4165 onward.

### What it computes

Numerical advection spreads intermediate density into air. Sharpening removes part of that air-side density and returns its integrated mass along the frozen density gradient toward the liquid. It is not merely a render filter or an SDF redistance, despite the UI phase identifier.

1. Clear sharpening receipts and dose storage over template cell capacity. Assemble/copy sharpening packet arguments.
2. **Prepare dose:** for selected current sources with `0 < rho <= 0.5`, traverse composite incidences. For each row, count positive/negative terms, then traverse terms again to collect opposite-side density, area and distance. Compute directional Godunov differences and the configured nonpositive density correction. Freeze the dose in scratch.
3. **Trace/scatter:** for a negative dose, convert `-delta * cellVolume` into fixed-point mass. Follow the density gradient in half-local-cell steps. Stop at liquid density, maximum travel, invalid ownership, small gradient, or configured iteration limit. Default distance is 2.1 source-cell widths and default limit seven iterations (hard shader bound 40). Scatter the removed mass to the final eight-corner stencil; floor shares and return the indivisible remainder to the source.
4. **Finalize:** over accepted cells, publish conditioned density + dose + incoming mass/volume, and conditioned gamma. Snap near-capacity quantization endpoints.
5. **Capacity repair:** eight rounds, each accepted-cell initialize → scatter → finalize. Above-capacity mass is split among open incident neighbors with paired integer debit/credit. Full neighbors may receive excess and relay it next round; this is not a one-pass move only into empty capacity.
6. Publish final-scalar packet masks for changed/nonexact/bulk/membership facts, which drive later scheduling and the next transport step.

Capacity repair and mask publication remain encoded even if sharpening is toggled off. Turning off the dose/trace therefore does not remove the whole displayed stage.

### The expensive nested sampling

`sampleSharpeningField` actually uses **central differences**, despite the older nearby comment saying “analytically.” It obtains local half-width, samples density at the centre, then at ±x, ±y and ±z. Each of those seven values is an eight-corner finest-lattice interpolant resolved onto accepted owners.

That is **56 density corner-owner queries plus one width-owner query per field evaluation**. A continuing trace iteration makes two further owner queries for current and candidate positions: up to 59. Seven fully continuing iterations can therefore perform 413 owner queries, plus eight final receiver queries, per nonzero-dose source. They are logical queries, mostly benefiting from TEI caching when in range; they are not 421 independent hash misses.

The density threshold and distance stopping conditions are checked **after** the full field evaluation. Even a terminal iteration can compute all six gradient probes unnecessarily at source level. The seven interpolants overlap; coarse ownership often maps multiple corners to the same cell. Each `conditionedDensity` fetch and owner-origin reconstruction is repeated without an explicit per-trace value/address cache. This combines high per-source work with divergent iteration counts across lanes.

The scatter kernel stages its 27-entry directory before per-lane current-source/dose tests. The fine/fallback sharpening list is seeded from transport packets, so a packet can still pay staging cost after its current useful source mask becomes empty. `prepareSharpeningField` itself does not stage the directory; it uses incidence only. This distinction matters when changing its execution domain.

Capacity repair scans all accepted cells 24 times, even if most are below capacity. Each overfull cell traverses incidence/terms twice: once to count recipients, once to scatter equal integer shares. This repeats row acceptance, openness and neighbor-active loads. Final-scalar publication additionally dispatches leaf capacity. The coarse surface may have very few true sharpening sources while these broad passes still run.

The [capacity early-exit experiment](sparse-cm12-ocean-min8-sharpening-capacity-early-exit-experiment.md) found a bit-inert density field after one round in its sampled min8 case. But retaining dispatches and adding per-lane continuation loads worsened repair median from 1.3107 to 1.3763 ms. Shortening eight rounds globally is neither validated nor safe: the current ceiling exists to relay impact excess toward free capacity. A useful optimization must remove enough scheduling/domain work to exceed its bookkeeping cost.

## Implications for fully adaptive bricks

The architecture needs both spatial adaptivity and adaptive execution. It currently has strong spatial compression and several good execution mechanisms: stable cell/packet addresses, topology-cached VEX packets, arithmetic interior neighbors, packet-shared transport directories, volume-aware conservative transfers and frozen departure stencils.

The remaining mismatches are specific:

| Mismatch | Consequence | Direction to investigate |
|---|---|---|
| Fine surface/air band dominates represented cells | Coarsening deep liquid yields diminishing transport savings | Measure and bound necessary support reach without discarding front donors |
| Physical cell storage, repeated point-to-owner resolution | Sparse memory saves space but every trace pays address work | Cache leaf origin/stride and owner stencils with topology validity |
| Scalar dirty detection walks covered fine-space tiles | One very coarse cell can require thousands of support checks | Adaptive leaf/packet dependency summaries |
| Accepted packet still has 64 lanes | R1/R2 masks waste lanes and staging | Occupancy-aware coarse packing with locality measured |
| Face rows use support, not change certificates | Still water continues to trace | Valid characteristic dependency receipts |
| Capacity/accepted-wide maintenance around sparse sources | Fixed cost survives low interface activity | Compact affected domains with conservative neighbor closure |
| Characteristic and sharpening loops revisit overlapping samples | Metadata and scalar loads multiply | Reuse addresses/values while retaining exact interpolation semantics |

The next profiling run should use ordinary ocean and full-domain min8 as separately named configurations. Record accepted cells/rows, supported face rows, VEX direct/compact/rebuild counts, packet occupancy, dirty-support tile checks, TEI hits/fallbacks, RK2 substep counts, sharpening source/iteration counts and overcapacity cells per round. Separate packet compilation from extension sweeps and sharpening dose from trace/scatter. Instrumentation with many atomics can perturb timings, so collect work census separately from timing A/B runs.

Priority is face owner reuse and sharpening sample reuse, followed by packet-support compilation and transport locality. First verify current cached VEX numerically and measure it; promoting compaction is no longer an outstanding implementation task. Any subsequent substantial simulation change should run the repository's canonical `npm run test:dawn:sparse-cm12` gate serially with browser GPU work stopped, retaining all existing lane limits. No numerical regression run was required for this documentation-only audit.
