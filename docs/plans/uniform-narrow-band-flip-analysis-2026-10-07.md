# Uniform 4h pressure with narrow-band FLIP: feasibility and design analysis

7 October 2026. Source audit of the working tree based on `5638a7729114a818eafef781a2255405db146135`, including pre-existing uncommitted particle work. [Evidence hashes](../papers/narrow-band-flip-2026/repository-evidence.json) identify the files read. This is a research/design result, not a solver implementation or a new performance measurement. The [paper archive](../papers/narrow-band-flip-2026/README.md) contains source PDFs, complete text, page images, native raster images, provenance and reproduction instructions.

## Decision

**The proposed architecture is sound enough to prototype, and is substantially closer to the existing Uniform method than an ordinary narrow-band conversion of the APIC solver.** Keep the global pressure solve at H = 4h; use particles to carry surface detail and momentum; rebuild a compact h MAC grid around those particles for transfers and local pressure corrections. Preserve the current coarse-to-band flux contract initially.

The intended benefit is to pay the global solve once per large step while limiting expensive fine work to the surface. This can preserve a large **global** timestep even when particle trajectories or local surface dynamics need smaller steps. It does not require every surface operation to take the global timestep.

Three qualifications determine whether the simplified method succeeds:

1. Fine-band pressure connectivity remains necessary. We can remove persistent fine transport ownership, h volume remapping, hanging surface fields and much of their maintenance; we cannot remove the fine pressure grid needed by local FLIP projection.
2. Existing sharpening and repair can be replaced only after their mass and surface-consistency responsibilities have another owner. Changing to particles does not establish those invariants by itself.
3. The coarse solve can supply conservative flux to the band without reproducing a fully coupled fine-grid pressure solution. That is a reasonable initial approximation already present in Uniform; assess its error rather than claiming equivalence.

The strongest first experiment is **one H solve and one connected narrow-band h solve per existing Uniform step**, with particle trajectory substepping as necessary. Add local pressure subcycling only if this first version fails a timestep/fidelity test. Introduce EXNB-FLIP activity selection after the fixed-band coupling works.

## Evidence from the repository

### The proposed pressure decomposition is already present

[`UniformPressureBand`](../../lib/methods/uniform/uniform-pressure-band.ts), especially lines 132-151, describes the current contract:

- the global projection runs on all-4h ownership;
- every h simulation tile containing liquid pressure rows participates in the fine solve;
- its interior boundary receives prescribed normal velocity from the coarse projection, with coarse flux equal to the sum of fine fluxes;
- the actual fine free surface uses ghost-fluid Dirichlet pressure;
- band multigrid uses h, 2h and 4h aggregates;
- solid faces carry matching open-face weights.

[`uniform-mixed-frame.ts`](../../lib/methods/uniform/uniform-mixed-frame.ts), lines 769-813, performs the root projection, transfers its result to simulation ownership, and calls `band.encodeSolve`. Consequently, the proposal can reuse the numerical pressure split. Porting it to a new particle-band storage layout is still real implementation work: the current class depends on mixed ownership, field binding and topology machinery.

The current h band is not necessarily a thin shell: it includes liquid rows in all selected h simulation tiles, including refined bulk. A particle-defined surface band can make this set smaller. Conversely, a connected liquid surface can span the entire scene. “Local” must mean local to the band or a connected component, not independent uncoupled solves on every allocation brick.

### What the particle timings establish

The [7 October redesigned comparison](../verification/apic-parallel-redesign-2026-10-07.md) reports the following for minidam64, measured over 0.9 simulated seconds:

| Quantity | Redesigned APIC | Uniform Full |
| --- | ---: | ---: |
| Solver wall time | 42.750 s | 0.261 s |
| Physical substeps | 892 | 27 |
| Average simulated duration per physical step | 1.009 ms | 33.333 ms |
| Reconstructed volume drift at t = 1 s | +17.154% | -0.0208% |

This Uniform arm is **Full**, with all 4096 tiles fine, not an all-4h simulation. Its global pressure solve is nevertheless at 4h followed by the band solve. This supports retaining the pressure decomposition rather than using a coarse-only result as the comparator.

The defaults are not matched fidelity: APIC uses a fresh residual ceiling of 0.001 s^-1; Uniform uses an absolute ceiling of 5 s^-1 plus relative reduction. The APIC surface error is also substantial. The 164x runtime ratio is therefore a measured default-path comparison, not a proof of equivalent-accuracy speedup.

In the redesigned early GPU profile, pressure is 486.1 / 923.0 ms = 52.7%; P2G is 171.6 ms, surface reconstruction 115.5 ms, motion/G2P 92.9 ms. Even deleting *all* non-pressure work would cap an ordinary particle-count optimization at about 1.90x for this profile, assuming unchanged step count and pressure cost. The proposed H-plus-local-pressure architecture attacks that remaining cost and the global solve frequency, so this Amdahl bound does **not** bound the proposed new method.

The current particle timestep is explicitly chosen by [`mac-shared/schedule.ts`](../../lib/methods/mac-shared/schedule.ts), lines 33-42, from particle/grid speed, gravity, CFL, and optional viscosity/capillarity limits. [`particle/shader.ts`](../../lib/methods/particle/shader.ts), lines 178-185, includes affine support velocities in APIC's speed bound. A narrow-band implementation should have its own justified scheduling policy; inheriting this entire full-particle advance loop would preserve much of the present cost.

### Is fine-topology maintenance the main bottleneck?

**Not established as a single stage. The broader collection of work caused by persistent h simulation is a credible target.**

In the [Phase 2 Figure 9 Dynamic measurements](uniform-compiled-topology-2026-10-06/boundary-phase2.json), late-splash census/layout/remap costs 1.456 ms of 24.675 ms GPU time, about 5.9%. Eliminating that stage alone gives an optimistic 1.063x GPU speedup, before adding particle work. These measurements do not isolate every topology-dependent instruction embedded in other kernels.

The later [Phase 3 measurement](uniform-compiled-topology-2026-10-06/spatial-phase3.json) has 24.846 ms late-splash GPU time, including 3.706 ms surface work and 2.979 ms extension. Earlier same-investigation captures also identify substantial sharpening. These are different checkpoints; their timings must not be summed into a fictional unified profile.

The current [`frame sequence`](../../lib/methods/uniform/uniform-mixed-frame.ts), lines 734-761, independently advects/redistances h phi, transports volume, cleans orphan volume, aligns surface volume, gathers geometry and sharpens. Removing this representation can remove more than the named layout stage. The default [`detail storage`](../../lib/methods/uniform/uniform-detail-fields.ts) is domain placement, so a general atlas-directory lookup is not the present production bottleneck.

## What the papers contribute

These summaries distinguish the published algorithms from the proposed hybrid. Section and page references refer to the archived PDF page order.

| Reference | Relevant result and limit | Use here |
| --- | --- | --- |
| **Ferstl et al., NB-FLIP (2016)**, §§3-4, PDF pp. 3-7 | Particles occupy an interior surface band; a grid level set identifies the particle-free interior. Uses R = 3h and a smaller velocity-combination region r = 2h with trilinear transfers; some examples use R = 4h. Naive density-weighted transition can inject energy. Pressure remains unchanged. The teaser reduces about 23.9M particles to 1.08M, but total time only 24.78 to 11.15 s/step. Reports useful tracking at CFL 5+, without a general error guarantee. Does not exactly conserve mass. | Baseline band, interior representation and safe transfer transition. Published cell widths refer to the simulation grid, not an independently chosen coarse H grid. [Source](https://www.cs.cit.tum.de/fileadmin/w00cfj/cg/Research/Publications/2016/NBFlip/nbflip.pdf). |
| **Sato et al., EXNB-FLIP (2018)**, §§3-5, PDF pp. 5-7 | Selects particle-active surface regions using a transported/cooled activity field based on velocity variation and geometry. Uses APIC, position correction, reconstructed particle surfaces and blended level sets. Its activity metric is insensitive to uniform translation but can miss relevant motions. Transition overhead can erase gains when the whole surface is active. | Second-stage reduction of surface particles and calmer resting surfaces. Not a promise of no repair or automatically faster frames. [Source](https://pub.ista.ac.at/group_wojtan/projects/2018_Sato_XNBFLIP/XNBFLIP.pdf). |
| **Lentine, Zheng & Fedkiw (2010)**, §§3.2-3.5 and §4, PDF pp. 3-7 | Coarse projection uses area-weighted face restriction; its velocity increments drive fine boundary faces; local Neumann projections restore fine incompressibility. Crucially, water requires a connected fine free-surface solve to remove artifacts caused by coarse air/water classification. This is a much closer pressure precedent than NB-FLIP alone. | Retain Uniform's connected fine surface solve; do not replace it with disconnected brick projections. [Source](https://physbam.stanford.edu/papers/stanford2010-02.pdf). |
| **Ando & Batty (2020)**, §7.2 and §9.2, PDF pp. 12, 15-16 | Demonstrates EXNB-FLIP in an adaptive Eulerian solver, restricting particles to the finest level. Reports remaining calm-surface noise, nonconservative advection and damping from coarse resolution. Spatial adaptivity does not supply temporal adaptivity. | Evidence that adaptive pressure and selective particles combine, while surface and timing tradeoffs remain. [Source](https://cs.uwaterloo.ca/~c2batty/papers/Ando2020/Ando2020.pdf). |
| **Narita & Kanai (2026)**, §§3-5 | Adapts the depth of a fine optical layer over coarse tall cells, uses EXNB-FLIP, and measures roughly 2.5-3x pressure acceleration and 1.5-2x overall against its fixed-layer reference. Runs with CFL 2. | Recent evidence that reducing pressure-layer work complements narrow-band particles. Different grid/operator; no evidence for our large global step. [Source](https://diglib.eg.org/server/api/core/bitstreams/36c619dc-cc90-4617-ba6c-817dc5587116/content). |
| **Nielsen & Bridson, Bifrost (2016 talk)** | Adaptive particles, level sets, interpolation and pressure all use the adaptive representation. Reverting to a uniform narrow band can surrender scaling gains. This is a two-page implementation overview, not a complete derivation. | A warning to account for temporary-grid halos and reconstruction as carefully as particle count. [Source](https://history.siggraph.org/wp-content/uploads/2022/09/2016-Talks-Nielsen_Spatially-Adaptive-FLIP-Fluid-Simulations-in-Bifrost.pdf). |
| **Lentine et al. (2012)**, §§3-5 | Large-step free-surface advection fails when extrapolated air velocities incorrectly move the interface. Uses conservative color transport to address the resulting errors. | Uniform's large-step transport is valuable independently of pressure. Keep that capability when changing the surface representation. [Source](https://physbam.stanford.edu/papers/stanford2012-02.pdf). |
| **Braun et al., ST-FLIP (2026)**, §§3-5, already archived | Space-time sampling and slab-integrated P2G target high-CFL aliasing; examples reach target CFL 30. Local trajectory integration remains substepped. It uses particle weight accumulation for phase/projection coefficients and has noise and damping tradeoffs. | Optional later high-CFL transfer experiment, not necessary to adopt the initial H-plus-band split. Missing interior particles mean its phase estimator needs a new grid/particle combination; direct transplantation is unvalidated. [Source](https://ge.in.tum.de/download/ST-FLIP.pdf). |

Also indexed: the existing APIC and Adaptive Phase-Field-FLIP archives. The latter offers a different surface/projection representation, which would enlarge this first experiment's scope. [Koike et al. (2020)](https://diglib.eg.org/items/a25b4a24-7b37-4612-88c8-3ddea8a81200) is a relevant temporal extension: its abstract explicitly identifies regional pressure, interpolation and volume-preservation requirements. Its PDF download returned an authorization error; only the abstract was consulted, and no downloaded full text is claimed.

## Pressure coupling: what can be preserved exactly

Let H = 4h. Let `B` contain the fine liquid band and a support collar. Let `Gamma` be its artificial boundary with the coarse liquid interior. At the actual free surface impose the appropriate atmospheric pressure (or surface-tension jump). At `Gamma`, prescribe the accepted H-grid normal flux.

For a coarse face F subdivided into fine faces f, the essential invariant is

```
Q_F = A_F * alpha_F * U_F
    = sum(f in F) A_f * alpha_f * u_f .                 (1)
```

Here `alpha` is the consistent open/liquid face measure used by the chosen operator. For moving solids, include the same solid-displacement contribution on both representations. A plain average is valid only for equal fully open subfaces.

First restrict the combined pre-projection velocity to H, project globally, and prolong the **change** in face flux into B while preserving (1). Solve the fine pressure equation on B, with prescribed normal boundary velocity on Gamma and actual fine free-surface conditions. Local correction must satisfy

```
D_h (u_B_star - dt/rho * G_h p_B) = s_B
u_B · n = g_H                         on Gamma.       (2)
```

`s_B` is zero for pure incompressibility; if the existing method supplies a volume-correction target, preserve and name it explicitly. Homogeneous Neumann correction at Gamma prevents the local pressure update from undoing the global boundary flux. A divergence-compatible prolongation and internally shared face values are prerequisites, not optional interpolation details.

For a closed all-Neumann connected component, equation (2) is solvable only when its integrated divergence source matches imposed boundary flux. Fix the pressure gauge and check compatibility component by component. A free-surface Dirichlet boundary usually removes the constant-pressure nullspace, but cannot justify discarding flux-accounting errors. Separate components inside one coarse cell must not be spuriously joined.

**Result:** with these conditions and accepted residuals, the assembled field can be divergence-controlled and flux-conservative across the coarse/band interface. It need not have the pressure, vorticity or wave response of a monolithic all-h projection. Divergence-free is not an equivalence proof.

### How the band affects the bulk

Do not run a detached H simulation that ignores particle momentum. Before each global projection, the band must contribute its restricted velocity/momentum and surface occupancy to the H predictor. It then participates in global pressure communication at that step. Fine corrections after that solve hold the interface flux fixed; their unresolved influence on bulk pressure is deferred or approximated.

If impacts, narrow jets or submerged fine features require stronger coupling, measure the discrepancy and add a coarse residual correction or outer coarse/band iteration. A monolithic reference can establish the error. A Schur-complement formulation makes the issue explicit: eliminating fine unknowns gives `S_H = A_HH - A_HB A_BB^-1 A_BH`; an ordinary rediscretized H operator is generally not this exact interface response. The existing [Liu et al. analysis](../papers/liu-2016-schur-complement-fluids-notes.md) describes that algebra, but is not a ready-made adaptive liquid discretization.

This does **not** mean an outer iteration is mandatory from day one. The existing Uniform approximation is the first baseline. Add stronger coupling only when the tests show a material failure.

### What “local solves” should mean

Use spatial bricks for allocation and parallel kernels. Adjacent active bricks share faces and solve one connected band system, with its own multigrid aggregates. Independent per-brick Neumann solves freeze extra internal fluxes and can leave seams or suppress waves. The 2010 paper's additional free-surface solve directly motivates this distinction.

Geometrically disconnected fluid components can solve independently. A thin shell has O(area * thickness) unknowns but may have long tangential wavelengths and poor conditioning; its convergence is not determined by thickness alone. Retain band coarse levels or another mechanism for those long modes.

## Large timesteps: the three clocks

The user's intended mechanism is to keep the expensive globally coupled work at H. That is a valid computational strategy. Implicit incompressible pressure does not itself impose the particle-travel CFL bound, however; H projection reduces the cost and frequency of global work, while transport and coupling determine whether a large step remains accurate.

| Clock | Work | Initial policy |
| --- | --- | --- |
| Global step `DeltaT` | H transport/predictor, restricted band feedback, H pressure | Retain the existing requested Uniform step and acceptance rules. |
| Particle trajectory step `delta_t_adv` | RK tracing, collision sweeps, support checks | Subdivide locally; no pressure solve for every trajectory subdivision. |
| Local physics step `delta_t_B` | P2G, band forces, fine pressure, G2P | Initially once per global step. Subcycle only if needed. |

For the same speed U, `CFL_h = U*DeltaT/h = 4*CFL_H`. Spatial coarsening alone therefore buys a factor of four in cell-travel units, not the observed factor of about 33 between the current APIC and Uniform step counts. That larger difference also includes their different transfer/transport dynamics, velocity bounds and trajectories.

With particle trajectory subdivision only, a frozen/interpolated projected grid improves path integration and collision handling, but does not recompute intermediate pressure or solve high-CFL P2G aliasing. Test this approximation at matched simulated times. In particular, current collision segment checks are not evidence that pressure and free-surface evolution remain accurate over arbitrarily large steps.

### If local pressure subcycling is needed

Use `s` surface substeps within one DeltaT. For each shared coarse face, accumulate the integrated fine flux:

```
I_F = sum(k=1..s) delta_t_k * sum(f in F) A_f*alpha_f*u_f(k)
I_F must match the H transport's integrated face flux.             (3)
```

A temporally constant coarse boundary can enforce a consistent budget but may miss changing pressure response. A time-interpolated boundary needs an actual predictor; future accepted values are not available automatically. Momentum and liquid volume crossing Gamma require matching exchanges. If fine subcycling changes a face's net integrated flux, synchronize the coarse state with that discrepancy and reproject as needed; a volume-only fix can leave divergent velocity.

Do not apply the entire coarse pressure increment or gravity increment `s` times to each particle. Define the combined pre/post grid pair for each transfer, and distribute coarse forcing over the macro interval exactly once. FLIP's grid increment must include both coarse and local corrections exactly once, on matching positions/support. Newly seeded particles receive the current velocity field, not a fictitious historical FLIP increment.

Changing solid contact, a disconnected component appearing, a fast impact or rapid band expansion may require an early global synchronization. If that becomes frequent, the assumed global/local timescale separation has failed in that scene. Explicit capillarity and viscosity can impose their own h-scale limits independently of the H pressure solve.

## Proposed state and step

### Persistent state

- Compact H MAC velocity, pressure, liquid volume/occupancy, coarse surface/interior sign and solid geometry. The pressure multigrid hierarchy is at H and coarser.
- Band particles with position, velocity, sampling weight, identity and lifecycle data; affine rows only in APIC mode. Activity/age is optional initially.
- Compact particle spatial bins, an active-brick directory, accepted publication state and volume/flux receipts.
- Optional detailed render surface cache, kept separate from simulation authority.

### Temporary or regenerable fine state

- h MAC velocity before and after projection, transfer weights, band phi/occupancy, pressure rows, residuals, multigrid aggregates and consistent solid-face measures.
- A support collar for interpolation, trajectory sampling and pressure boundary placement.
- Surface reconstruction at the requested rendering resolution.

Recycle allocations across frames. “Temporary” means the values can be rebuilt from accepted H state plus particles; it does not require allocating/freeing GPU buffers each step. A stationary band should reuse its spatial slots and connectivity. Storing a previous projected fine velocity can be a performance cache, but its invalidation and interpolation on new support must be defined.

### One synchronized step

1. Choose DeltaT; predict a swept active band and its support collar from the accepted surface, velocities, solid motion and expected deformation.
2. Advect particles and the H bulk predictor in a consistent advection/projection ordering. Use trajectory substeps where needed. Reconstruct fine-band occupancy and retain an unambiguous interior mask.
3. P2G into the fine band; combine with the H predictor only where particle support is valid. Restrict the resulting band predictor into H before the global pressure solve. Apply forces once.
4. Run the existing H pressure solve and acceptance gate.
5. Transfer the H correction conservatively, then solve the connected h band with the existing Neumann/interior and Dirichlet/free-surface contract.
6. Update particle velocity from the consistent total grid change (FLIP) or projected affine fit (APIC). Restrict accepted band feedback and reconcile its surface with the authoritative H volume.
7. Reseed/cull only in the permitted transition region; preserve visual surface and moment constraints. Publish after pressure, flux, volume and support checks pass.

This sequence is a proposed contract, not a verbatim combination of the paper algorithms. Pin the exact time levels before implementation. In particular, a second independent H update under a fine update must not double-advect material or momentum.

## Band width and what h topology actually disappears

Distinguish four widths: particle occupancy band, particle-to-grid combination band, pressure band, and trajectory/renderer support. They can differ. NB-FLIP's 3h/2h example uses trilinear support; the present APIC quadratic stencil has 1.5h radius per coordinate, so copying those widths unchanged has no support proof.

For a fixed world-space allocation valid over an entire macrostep, a conservative support envelope is roughly

```
allocated band >= required physics band + transfer footprint
                  + maximum relative travel + surface prediction error.
```

For a band rebuilt or advected with the particles at intermediate times, absolute translation need not thicken the particle band by `U*DeltaT`; the relevant issue is relative surface/particle error and newly required support. Do not unnecessarily destroy savings by conflating a swept allocation envelope with persistent particle occupancy. Conversely, a fixed sparse grid must cover all stages of the trajectory, not just endpoints.

The principal deletions are persistent h liquid-volume transport, fine/coarse ownership remapping of those fields, hanging-vertex phi reconciliation, associated sharpening and fine surface-volume correction. Fine pressure adjacency, particle binning, solid contact, transfer support, and surface reconstruction remain. This is a simpler set of responsibilities, but it is not topology-free.

Thin sheets and droplets are nearly all band. A flat deep pool is favorable; a highly fragmented splash can approach full-particle work. For a smooth closed sphere of radius a and inward band R, the particle fraction is `1-(1-R/a)^3` for R<a. At a=16h and R=3h it is about 46%; at a=64h it is about 13%. These are geometric examples, not predictions for our scenes. Grid padding and surface-area growth add cost.

## Can sharpening and repair be removed?

**The existing h-grid sharpening pipeline can plausibly be removed. “No correction anywhere” is not supported.** Particle-carried interfaces do not require the same anti-diffusion operation as transported Eulerian volume, but they require sampling management and a surface/volume contract.

There are two viable authority choices:

| Choice | Consequence |
| --- | --- |
| H liquid volume is the mass authority throughout; band particles are detail/velocity samples | Easiest first prototype. Reseeding does not add material mass. Reconstructed surfaces must agree with the H volumes, and particles cannot be counted again as additional liquid. Detached features need explicit ownership. |
| H bulk and disjoint particle-band volumes partition material | More direct particle material accounting, but crossing the partition needs conservative mass/momentum exchange, remapping and lifecycle rules. This adds back a form of interface maintenance. |

Recommend the first for the initial experiment, with H volume updated by the accepted combined fluxes. It permits the existing coarse conservative transport to survive. It does not justify calling the initial method repair-free: coarse transport may still need its present bounding/correction, and fine particle geometry can disagree with it.

To eliminate coarse sharpening as well requires a separate bounded transport scheme whose update preserves `0 <= V_K <= openVolume_K` while conserving total volume, including moving-solid displacement and source/sink flux. Conservation alone does not imply boundedness or a sharp interface. Merely disabling the existing sharpening, cleanup and volume-correction flags retains the defects those stages currently address.

A local reconstructed-surface adjustment to match authoritative volume is a possible replacement, but it must preserve neighboring continuity and connected components; independently shifting each H cell can introduce seams. A global volume offset can conceal local mass migration. Initially measure both material volume and reconstructed volume and make every correction visible in diagnostics.

Detached droplets must either remain resolved particle liquid with conservative exchange on reentry, or be explicitly classified as secondary visual spray. Lifetime deletion from the EXNB-FLIP rendering model cannot silently delete primary liquid mass. Reusing the current APIC surface reconstruction unchanged is particularly risky given its measured +17% volume error.

## Performance model and acceptance experiment

Let N be the number of cells in an equivalent full-h grid, f the actual fraction in the padded fine pressure support, q the particles per sampled h cell, and s the number of band physics steps per global step. A useful accounting model is

```
T_macro = T_H_transport + T_H_pressure(N/64)
        + s * [T_bins(q*f_particle*N) + T_P2G + T_band_pressure(f*N)
               + T_G2P + T_surface + T_local_management]
        + T_sync + T_render_publication.
```

Particle fraction and padded pressure fraction must be measured separately. `N/64` is a cell-count reduction for isotropic H=4h, not a 64x runtime promise. Pressure iterations, cut cells, dispatch overhead, bandwidth and global reductions matter. For a sanity check, if f=0.1 and s=4, band pressure already visits the equivalent of 0.4N cells per macrostep before iteration factors; at f=0.5 it is 2N. Large surface substep counts can consume the intended gain even with a cheap root.

Compare these arms, without changing existing Uniform quality gates:

| Arm | Question answered |
| --- | --- |
| Current Uniform Full and Dynamic | Same-scene quality and throughput references. |
| Existing all-H endpoint | Bulk cost floor and missing surface detail. Not the fidelity reference. |
| H + full-surface NB-FLIP, one band solve/DeltaT | Does replacing fine geometric transport buy enough to pay for particles? |
| Same method with progressively smaller DeltaT | Is the chosen large step accurate, beyond being finite/converged? |
| Same method with local pressure subcycling | Does local temporal refinement recover fidelity without repeated global work? |
| Activity-selected EXNB-FLIP | Does lower particle/pressure support compensate for selection and transition cost? |

Start with frozen-state pressure/transfer checks, then complete trajectories:

1. **Projection invariants:** face-flux restriction/prolongation; final composite divergence; correct free-surface pressure; Neumann compatibility and pressure gauges; disconnected liquids within one H cell; agreement with an all-h reference at identical geometry.
2. **Transfers and lifecycle:** constant/affine field reproduction, translation and rotation, no energy growth at the band transition, no double coarse impulse, no mass/momentum jump on reseeding or support migration.
3. **Rest/contact:** fractional waterline, sloping shore, thin barrier, narrow channel, moving body, and long resting-pool evolution. Measure spurious velocity and represented volume.
4. **Dynamics:** minidam64 through the full existing interval, Figure 9 splash, falling drop/crown, a thin sheet, and a deep-pool impact sending a wave beyond the active patch. The last case specifically tests coarse/band pressure feedback.
5. **Time accuracy:** compare DeltaT, DeltaT/2 and DeltaT/4 at equal physical times; report actual root solves, local solves and trajectory steps. Measure wave speed/amplitude, splash timing, energy, material/reconstructed volume, local flux error and surface distance. Pressure residual alone is insufficient.
6. **Cost:** fenced solver wall time, p50/p90, complete GPU stage accounting, dispatches, allocated/peak memory, particle counts, padded band size, rebuild frequency, and renderer cost. Record cold compilation separately. Use serial interleaved runs with pinned source hashes.

First success means a measured gain over the existing Uniform surface pipeline at an explicitly accepted fidelity, while retaining its global step in the target scenes. A gain over the much slower current full-particle implementation alone is insufficient. Record any required synchronization, particle redistribution or volume correction rather than hiding it under a repair-free label.

Do not run Dawn concurrently with a browser or another Dawn process. A future code change must pass `npm run check:types`, `npm run test:unit`, and `npm run test:dawn` with unchanged maintained Uniform thresholds. This research-only change runs no GPU benchmarks and makes no clean-repository assertion.

## Recommended implementation boundary

Create a separate experimental method sharing Uniform's H pressure, coarse geometry, acceptance and publication infrastructure. Begin with static solids, an all-surface particle band and one band pressure solve per global step. Reuse the existing sorted-bin/scan/gather particle primitives, but replace their whole-domain dispatch and full-particle scheduling. Port `UniformPressureBand` to compact particle-derived pressure bricks while preserving its connected-domain boundary contract.

Keep the initial coarse conservative volume transport and expose its correction cost. Once the coupled method passes volume, pressure, timestep and surface tests, replace remaining coarse sharpening only with a demonstrated bounded alternative. Then test local pressure subcycling and activity-selected EXNB-FLIP separately. This isolates the central claim: **existing inexpensive global 4h pressure plus a particle-carried, locally projected surface can replace expensive persistent h geometric transport without forcing the entire domain onto the particle timestep.**
