# Realtime particle method research and implementation plan

Research date: 7 October 2026. Target: the current Apple M1 Max, 32 GPU cores, 32 GB unified memory, WebGPU/Metal, with 60 presented frames and one simulated second per wall-clock second. Repository inspected at `5638a772`. This document began as the research and benchmark plan. The subsequent requested APIC implementation is described below; neither this prototype nor the literature establishes a measured particle-method winner.

## Implemented APIC preview

The method catalog now installs **APIC particles**, method id `particle-apic`, through the same `SimulationMethod` contract as the other fluid methods. Open [Minimal dam break 32³ with APIC](http://localhost:3000/scene?scene=minimal-power-dam-break-32&method=particle-apic), or enter EDIT mode and choose **Fluid solver → APIC**. Uniform Geometric remains the default.

Implementation lives in [`lib/methods/particle`](../../lib/methods/particle/method.ts). The descriptor owns controls, diagnostics, the pipeline graph, harness registration and factory. Its transport owns GPU particle positions, material volumes, velocities and three affine rows. It seeds up to eight particles per wet cell, builds integer linked bins, gathers quadratic B-spline APIC momentum onto staggered faces, and reconstructs velocities and affine rows from the projected grid. RK2 particle motion includes subdivided point-versus-voxel collision checks. Escaping particles at an open top are counted in a separate material ledger.

The common MAC projection/publication engine is in [`lib/methods/mac-shared`](../../lib/methods/mac-shared/solver.ts), behind an optional transport interface. The existing Uniform MAC baseline uses its original Eulerian transport through that engine; APIC does not call another method's factory. Shared initial-liquid sampling and tetrahedral surface fill helpers moved to `lib/core`, with compatibility exports for existing Uniform callers. The method registry is the only application integration point; the UI and renderer require no APIC branches.

The preview includes gravity, grid viscosity, a capillary pressure boundary, fixed voxel solids, open/closed tank tops, scene wall mode, incompressibility projection and the shared liquid renderer. Pressure uses Galerkin multigrid-preconditioned CG with a fresh fine-grid residual check and rejects a failed solve before publishing. Particle state stays on the GPU between steps; asynchronous receipts report diagnostics without fencing the interactive advance.

The transport bar's **dt** control now changes APIC's requested step live (1–50 ms), without resetting particles or the running clock. Its former hidden 1/120 s method cap has been removed. GPU velocity, gravity, viscosity and capillary limits may subdivide the request; diagnostics show the last accepted substep. Pressure tolerance, iteration safety limit and CFL are method-owned pipeline controls. APIC defaults to CFL 2 and a 32-iteration MG-PCG safety limit, with GPU convergence guards. Host advances use a grid-work budget: up to 64 conservative step estimates at 32³ and eight at 64³, with command buffers split every four reserved substeps to bound browser work. Existing APIC pages need a reload to create the new solver. Uniform MAC retains its original diagonal-PCG configuration and indirect dispatch path.

This is **not Uniform Geometric feature parity**. Inflows and rigid bodies are rejected with explicit messages. Live detail changes, continuing solid/water edits without restarting, cut-cell geometry, adaptive grids, particle reseeding, PIC/FLIP controls and ST-FLIP are not implemented. Fixed solids use occupied cells, not geometric cut cells. Surface reconstruction uses a compact weighted particle centroid and a fixed radius; it can smooth thin features and change represented volume even when particle material is conserved. The initial particle material is a subcell sample of the authored shape. Surface-volume drift and material-balance error are therefore separate diagnostics, and neither is corrected to hide error. Allocation rejects particle counts above the device binding limit or one million particles.

### Verification and initial performance

[`tests/particle-apic-dawn.test.ts`](../../tests/particle-apic-dawn.test.ts) checks affine transfer reproduction, hydrostatic rest, ballistic free fall, a dam break through 0.5 simulated seconds, empty scenes, measured open-top outflow, and shared renderer consumption. The unit tests check plugin identity, seeding, capacity failure and unsupported sources. The browser check selected Uniform MAC and then APIC through the standard menu, stepped and played APIC, and displayed a moving reconstructed surface without console warnings or errors. [UI capture](../verification/apic-ui-2026-10-07.jpg).

The initial Chrome/Metal run on this M1 Max used the authored 32³ mini dam, default APIC controls and 12.1 MiB of solver allocation. The live hardware stage trace showed approximately **20–28 ms per advance**, including roughly 13.5–18.3 ms for pressure and 4.8–6.9 ms for grid-to-particle transfer, motion and surface work. These are short, instrumented UI observations, not a controlled p95 benchmark, and one advance is not necessarily one presented frame or a fixed amount of simulated time. The high paused presentation FPS does not measure simulation throughput. **The 60 FPS / realtime simulation target has not been achieved.** Those observations predate the pressure upgrade. Multigrid and larger advances are now enabled; pressure acceptance retains the original residual tolerance. The updated performance report records timestep counts and fidelity differences, with a browser-contention caveat on the latest wall timings.

Run the focused check with `npm run test:dawn -- particle-apic`. The full repository gates remain `npm run check:types`, `npm run test:unit`, and `npm run test:dawn`; stop browser WebGPU use before Dawn. The broader matched-scene performance/fidelity campaign below remains to be run.

The dedicated first performance pass and reproducible commands are recorded in [APIC performance verification](../verification/apic-performance-2026-10-07.md). This short solver-only sweep establishes a bottleneck and timestep baseline; it does not replace the longer paced-browser and matched-fidelity protocol below.

Implement a separate **particle-grid liquid method with APIC as its initial transfer scheme**, retain PIC/FLIP as controlled comparison modes, and evaluate **ST-FLIP from SIGGRAPH 2026** as a subsequent temporal-sampling extension. This is the best engineering fit for Uniform Geometric parity: grid projection, solids, fluid properties, diagnostics and surface publication can follow existing contracts, while particles replace transport. The recommendation is an inference from the papers and repository architecture, not a published comparison on this hardware.

Keep **IPBF from SIGGRAPH Asia 2025** as the strongest recent alternative to investigate if the particle-grid prototype misses its budget. DFSPH is a useful mature SPH reference. Neither SPH approach avoids the cost of neighbor searches, boundary treatment or surface reconstruction, and both require substantially different integration work here.

The [paper collection](../papers/realtime-particles-2026/README.md) contains original PDFs, searchable text, complete page images and embedded raster images, with source URLs and checksums.

## Candidate methods and evidence

| Method | What matters for this project | Evidence and decision |
| --- | --- | --- |
| APIC, 2015 | Carries a local affine velocity per particle; reduces PIC dissipation and FLIP noise, preserving angular momentum across the transfer under the method's assumptions | Start here. Conservation of the transfer alone does not guarantee conservation of the complete simulation. [Paper](https://media.disneyanimation.com/uploads/production/publication_asset/104/asset/apic-aselle-final.pdf) |
| PIC/FLIP | PIC provides a dissipative control; FLIP provides a familiar low-dissipation control with possible particle noise | Implement in the same particle/grid infrastructure so transport comparisons do not change pressure, rendering or boundaries. [APIC comparison](https://doi.org/10.1145/2766996) |
| ST-FLIP, 2026 | Jitters particles in time and uses temporal transfer weights and a phase-field projection; designed to reduce large-timestep aliasing | Highest-priority recent extension. The authors report 2–8× speedups, but on large CPU simulations. Their Table 1 has **1.86 seconds per step**, versus 2.3 for FLIP at the same CFL. Test locally before expecting any speedup. [Project](https://ge.in.tum.de/2026/07/16/siggraph26-spatiotemporal-flip-for-liquid-simulations/) · [Paper](https://ge.in.tum.de/download/ST-FLIP.pdf) |
| Adaptive Phase-Field-FLIP, 2025 | Adaptive grids and particles, phase-field treatment of water and air, and adaptive pressure multigrid | Source of design ideas for later adaptivity; not a realtime implementation to port wholesale. Large examples take minutes per output frame. Explicit air dynamics exceed current parity scope. [Paper](https://animation.rwth-aachen.de/media/papers/94/2025-Siggraph-Adaptive_Phase_Field_FLIP.pdf) |
| IPBF, 2025 | Implicit position-based SPH energy minimization improves robustness and compression control at large steps | Strong research alternative, not a demonstrated 60 FPS solution here. Table 1 reports **50 and 70 ms/frame on RTX 4090** for two cases at approximately matched density errors; the large example reports **159 ms/frame**. Stability does not guarantee low density error or undistorted motion at any timestep. [Paper](https://graphics.cs.utah.edu/research/projects/ipbf/ipbf.pdf) |
| DFSPH, 2015 and subsequent work | Corrects both density and velocity divergence | Mature SPH comparison with reference code in SPlisHSPlasH. Neighbor iteration costs and boundary handling need local measurement. IPBF's difficult large-step failures for DFSPH do not establish failure at DFSPH's appropriate timestep and tolerance. [Paper and code](https://animation.rwth-aachen.de/publication/054/) |
| IPIC, 2024 | Extends APIC using impulse transport to preserve circulation and vortical detail | A later fidelity experiment once ordinary APIC passes parity. It introduces additional state and numerical choices; more visible energy is not automatically more accurate. [Project and paper](https://studios.disneyresearch.com/2024/04/25/the-impulse-particle-in-cell-method/) |
| Particle Flow Maps, 2024; Vortex Particle Flow Maps, 2025 | Long-range advection and preservation of complex vortices; VPFM includes moving-solid boundary treatments | Useful fidelity references. Their evidence does not establish a ready replacement for this free-surface editor at 60 FPS. Extra derivative/map state increases implementation risk. [PFM](https://zjw49246.github.io/projects/pfm/) · [VPFM](https://arxiv.org/abs/2505.21946) |
| Tube Maps, 2026 | Fast SPH solid-boundary density calculation | Optional SPH boundary component, not a complete liquid solver. Its orders-of-magnitude claim concerns boundary evaluation. Smooth-surface and curvature assumptions are a concern for editable voxel corners. [Project](https://gatc.cs.columbia.edu/projects/tubemaps.html) |

ST-FLIP still reconstructs and smooths a surface for output. Its tests exclude rendering and use a 32-core Threadripper with 256 GB RAM. It does not remove splitting-related energy/angular-momentum loss, and capillary timestep limits reduce its advantage for small-scale liquids. Calm-surface noise is a stated limitation. These qualifications are particularly relevant to our pond, bowl and thin-liquid scenes. See paper Sections 4–5 and Algorithm 1.

The existing [GVDB FLIP paper and analysis](../papers/wu-2018-gvdb-flip-assets/README.md) remain relevant for GPU particle binning and gather-based transfers. They are historical implementation references, not evidence of the latest state of the art.

The public [MSBG repository](https://github.com/tum-pbs/MSBG) supplies CPU sparse-grid infrastructure and a surface-reconstruction demo; do not treat it as a downloadable WebGPU ST-FLIP solver. [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) supplies mature SPH algorithms and material/boundary references. The Tube Maps project currently labels code as forthcoming. The consulted ST-FLIP and IPBF project pages provide papers and videos; a complete implementation was not verified there.

## Uniform Geometric parity contract

Use the maintained `uniform-volume` implementation as the baseline. The top-level README describes older architectures in places; the executable method, scene and solver contracts are the authority. Retired Sparse CM12, adaptive and Losasso lanes are outside the clean-repository gate.

| Current behavior and source | Particle requirement and acceptance evidence |
| --- | --- |
| [Method descriptor](../../lib/methods/uniform/uniform-volume-method.ts): shared scenes, parameters, pipeline graph, harness and renderer | Register an independent method and harness; serialize its settings and preserve scene switching, reset, pause and stepping. Keep Uniform's defaults intact. |
| Conservative liquid volume plus vertex level set | Track particle material volume separately from reconstructed surface volume. Report source, escaped-liquid and resampling ledgers; test geometry as well as mass. A constant particle count alone is not a volume-fidelity result. |
| Full, Requested and Dynamic detail; `h`/`4h` layouts and live adoption | Support the same user intent with conservative particle resampling and coupled grid refinement, or explicitly remain a partial prototype. Changing only particles per cell cannot recover missing pressure-grid resolution. |
| [Live detail input](../../lib/core/solver-detail.ts), authored regions and runtime controls | Adopt detail changes at completed frame boundaries without restarting time; report actual resolution, allocated memory, rejected requests and support work. |
| Voxel/terrain solids, moving solids and live edits | Prevent leakage and handle displaced water without silent deletion; preserve cut geometry and relative wall velocity. Test thin barriers, corners, insertion and removal. |
| [Mixed rigid coupling](../../lib/methods/uniform/uniform-mixed-bodies.ts) and GPU rigid system | Reproduce current two-way impulse exchange, body motion, live roster adoption, picking and pose publication. Current coupling is not a claim of an exact monolithic pressure/body solve. Avoid applying boundary and body reaction impulses twice. |
| Inflow and live liquid-ball injection | Seed particles with the authored volume and velocity and preserve the running simulation. Current `editFluid` accepts **add + ball only**; arbitrary removal and other shapes are not required for parity. |
| Gravity, viscosity, capillarity, wall mode and open/closed top | Respect scene units and properties; expose actual timestep restrictions and failures. Do not substitute arbitrary smoothing for the authored viscosity. |
| [Shared solver and presentation contract](../../lib/core/method-contract.ts), dense surface ABI and renderer | Publish the correct current-frame field and transforms to the existing water renderer, including paused edits and initialization. Keep comparable lighting, surface extraction and render resolution. |
| GPU stage timings, physical diagnostics, health receipts and completion fences | Expose particle count, bin occupancy, substeps, pressure acceptance, invalid particles, surface work and memory. Never publish an unconverged or incomplete frame. |

Diagnostic views should carry the same useful information where it exists, with particle-specific views for particle state. Do not invent geometric volume-page or tile data for a solver that does not own those structures. A feature inventory must distinguish comparable diagnostic meaning from identical internal data structures.

There is also an independent [Uniform MAC baseline](../../lib/methods/uniform/mac/README.md). Its MAC projection, residual tests and publication conventions are valuable references. It explicitly rejects inflows and dynamic bodies, uses binary solids and full-grid storage, and uses diagonal PCG. Wrapping it with particles would therefore be a useful prototype, **not** Uniform Geometric parity or a proven performant production pressure solver.

## Proposed solver design

Create `lib/methods/particle/` and method id `particle-apic` as proposed names. Keep transfer choice (`pic`, `flip`, `apic`) separate from projection choice and temporal sampling. An APIC-based implementation of the ST extension must be labeled as such rather than implying it is the paper's exact default configuration.

One initial substep:

1. Consume pending source and solid/body edits. Create or redistribute particles with an explicit material-volume ledger.
2. Bin particles on the GPU using count, scan and scatter. Maintain stable particle identifiers for seeded sampling; detect capacity overflow and refuse the frame rather than dropping particles.
3. Gather mass and momentum onto a staggered MAC grid. Save the pre-force grid velocity for FLIP; use a consistent MAC APIC transfer and moment matrix treatment near truncated boundaries.
4. Reconstruct fluid support and the pressure interface; build solid face apertures and velocities. Apply gravity, viscosity and surface tension under their physical timestep limits.
5. Solve and verify the incompressibility projection with a freshly evaluated residual. Pressure boundary semantics must match the selected scheme; Geometric's separating-boundary LCP and a conventional signed ghost-fluid solve are not interchangeable APIs.
6. Extend valid grid velocity into the interpolation band. Transfer back to particles with the selected scheme; advect using RK2 initially and collision-safe trajectories against moving solids. Account for reaction impulses once.
7. Reseed or merge only when required, conserving material volume and linear momentum and measuring changes in angular momentum and energy. Rebuild the render field from the final particle positions and publish a completed frame.

Initially use a fixed uniform pressure grid to isolate transfer and reconstruction correctness. Test 4, 8 and 16 initial particles per liquid cell, including partial-cell seeding, then choose from measured quality/cost results. Empty air cells should not carry material particles. Density and boundary reconstruction must not rely on a fixed eight-particles-per-cell count once masses vary.

Prefer a bin-based gather transfer as the first portable WGSL implementation. Core WGSL atomics are integer atomics; do not assume CUDA floating-point atomic accumulation or warp intrinsics are available. Tile-local reductions and multi-pass gathers are candidates to profile later. Account for bin indices, scans, sorting/reordering copies and empty-grid clearing in timings. [WGSL atomic types](https://www.w3.org/TR/WGSL/#atomic-types)

Budget particle memory before allocating. An illustrative padded APIC record with position/mass, velocity/id and three affine rows is 80 bytes; one million records are 80 MB decimal before double buffering, sorting scratch, grids, surfaces or renderer allocations. Prefer structure-of-arrays where measurements support it. Query device binding and allocation limits and report full resident and peak transient allocations.

For the production pressure path, adapt suitable multigrid infrastructure behind an explicit operator contract or implement particle-owned multigrid. Avoid importing Uniform's monolithic solver and running hidden geometric transport. The initial diagonal-PCG reference may validate small cases but is not a performance commitment.

Surface reconstruction is part of the method. Begin with a deterministic particle-to-field reconstruction; measure disconnected-drop survival and sheet thickness before introducing anisotropic kernels or smoothing. Publish a compatible surface field with an explicit scalar convention, not a particle-density texture mislabeled as signed distance. Measure reconstructed volume before and after any correction. Global volume adjustment cannot compensate for a disappeared sheet or misplaced liquid.

Only after fixed-grid acceptance add sparse work lists, coupled `h`/`4h` grid ownership and particle split/merge. Keep fine support around solid contact and thin liquid; crossing a coarse/fine boundary must not create a momentum or surface impulse. Share scene/editor contracts rather than assuming Uniform's ownership and compiled stencils work unchanged for particles.

For ST-FLIP, first add an ordinary phase-field projection arm with temporal jitter off, then the temporal arm. This separates changes in interface treatment from time sampling. Implement output-time synchronization, seeded offsets and calm-region handling from the paper. Retain collision and capillary bounds. Sweep timestep and CFL rather than adopting the paper's high CFL values as safe defaults. A frame already solved in one step has less opportunity for the reported substep-reduction speedup.

## Comparison protocol

The target is **16.67 ms for a complete presented frame at normal simulation speed**. Proposed initial allocation: simulation 10 ms, reconstruction/publication 2 ms, rendering 3 ms, and 1.67 ms headroom. These are planning allocations, not measured stage costs; record CPU/GPU overlap rather than adding asynchronous clocks indiscriminately.

Existing evidence establishes the scale of the challenge. The [6 October frozen-snapshot comparison](../../artifacts/uniform-performance-2026-10-06/methodology.txt) reports the following app-Dynamic simulation throughput at 1/60 s per step:

| Scene | Mean wall time per simulation step |
| --- | --- |
| Minimal dam 64³, two repeats | 12.46–12.76 ms |
| Minimal dam 128³, two repeats | 37.82–38.01 ms |
| Figure 9 dam | 25.15 ms |
| Pool impact | 18.14 ms |

These are historical snapshot results from [summary.json](../../artifacts/uniform-performance-2026-10-06/summary.json), excluding rendering, compilation and field readback. They are not current-head browser FPS or a new particle comparison. Re-establish a frozen baseline when implementation begins.

Run three distinct comparisons:

- **Matched discretization:** identical geometry, finest pressure-cell width, fluid properties, physical time and output cadence. Compare full-grid APIC/PIC/FLIP to Uniform Full first. Report actual particle sampling and grid topology; equal grid width is not proof of equal accuracy.
- **Matched fidelity:** tune each method to predeclared physical and surface error limits, then compare total cost. Include Uniform Requested and Dynamic and the particle adaptive configuration when available.
- **Matched frame budget:** find the highest fidelity each method sustains within 16.67 ms on this machine. Report unsupported features, failed gates and missed frames, rather than allowing quality to disappear from the score.

Keep runtime settings, browser/OS/Dawn versions, GPU adapter, physical scene JSON, seed, code revision and uncommitted source hashes with every run. Lock camera, lighting, viewport pixel dimensions and renderer settings; disable automatic render-resolution adjustment. Do not render side-by-side panes during timing: run sequentially and compare synchronized captures afterwards.

Use 64³, 96³ and 128³ pressure-grid targets where the physical scene admits them, preserving dimensions for noncubic domains. Use finer/smaller-step offline runs for convergence references. Start capacity sweeps at approximately 32k, 128k, 512k and 1M particles, recording the actual counts generated by physical sampling. These are workload sweeps, not interchangeable quality presets.

For each representative run, simulate 10 physical seconds (600 output frames at 60 Hz); use longer rest and source runs up to 60 seconds. Compile/warm pipelines separately and reset to a reproducible initial state. Capture early impacts, turbulent motion and settled intervals separately so a cheap settled tail cannot hide an expensive splash. Use five timing repeats and three seeds for stochastic fidelity evaluation; report uncertainty and outliers. Run correctness diagnostics separately from minimally instrumented throughput.

Report p50/p95/p99 completed-frame wall latency, hardware GPU stage times, CPU encode time, missed-vsync rate, simulation-time/wall-time ratio, initialization latency, peak memory, cells/particles, substeps and pressure iterations. Publish both paced browser runs and offline simulation-throughput runs; a queue of unfinished frames is not 60 FPS.

Proposed interactive acceptance is p95 completed-frame latency at or below 16.67 ms, at least 0.99 simulated seconds per wall second, and at most 1% missed 60 Hz presentation deadlines over the measured interval, with all physical gates passing. Record the p99 and longest stall even when these aggregate gates pass. Measure edit/initialization spikes separately and include them in the interaction report.

The GPU lease is mandatory. Dawn runs serially, one process per file, and never concurrently with a browser GPU benchmark or another Dawn job. No timing ceiling, Uniform residual criterion or fidelity gate may be loosened to create a passing comparison.

## Scene matrix and fidelity measurements

| Scene or fixture | Principal question | Measurements |
| --- | --- | --- |
| `garden-pond`, `stationary-bowl`, coarse-solid-rest fixtures | Can quiet water stay quiet, including cut walls? | RMS/max speed, surface-height noise, hydrostatic pressure error, shoreline motion, wall leakage |
| `geometric-uniform-translation`, `coarse-surface-translation`, free-fall variants | Does transport preserve shape without help from gravity or damping? | Center of mass and velocity versus analytic motion, surface distance, per-component volume, kinetic energy |
| `minimal-power-dam-break-32`, `minimal-power-dam-break-64`, serialized 128³ variant | What is the basic speed/accuracy curve? | Wavefront travel, water-height gauges, volume, pressure/divergence, stage scaling |
| `mass-conserving-figure-9-dam-break`, `twin-dam-collision` | Do violent impacts retain coherent splashes? | Splash height and timing, symmetry, thin-feature persistence, energy and momentum, surface topology |
| `coarse-first-pool-impact` and its resolution variants | Does a crown survive at affordable resolution? | Crown radius/height, sheet breakup time, droplets, displaced volume |
| `thin-droplet-ladder`, `thin-sheet-ladder`, `thin-wall-films` | Are small visible features preserved? | Survival by physical size, thickness error, disconnected component count, material-versus-surface discrepancy |
| `ocean-seiche`, `coarse-surface-standing-wave` and live variant | Are phase and dissipation right? | Period, phase error, amplitude envelope, mean water level and noise after refinement |
| `hose-tank`, `garden-hose` | Can sources run without drift or allocation stalls? | Analytic integrated inflow versus material and surface volume, jet velocity, particle growth and peak memory |
| `rigid-hydrostatic`, `rigid-float`, `rigid-sink`, `dam-break-boxes` | Is two-way coupling comparable? | Buoyancy/trajectory, rotation, paired impulse balance, wet contact, leakage, body penetration |
| Existing mixed live-solid-edit and prescribed-motion fixtures | Does editing preserve a running state? | Clock continuity, displaced-liquid balance, transient energy, stale publications, edit latency |
| Existing capillary/viscous numerical fixtures plus proposed drop oscillation and shear decay | Are material parameters physical? | Laplace pressure, oscillation frequency/damping, viscous decay versus analytic solutions, required substeps |
| Proposed isolated rotating-liquid/vortex fixture | Is APIC/IPIC detail physical? | Angular momentum, circulation and kinetic-energy decay; forces and boundary torques accounted for |

Use executable scene generators rather than descriptive card copy; some older cards disagree with their current thin-liquid dimensions. Reuse scenes even where their historical IDs contain retired method names, while explicitly selecting the maintained solver baseline.

Measure incompressibility in physical units: volume-weighted RMS and maximum divergence, plus `dt × divergence`; pressure residual alone is operator-dependent. Report material-volume error against initial volume plus source minus measured outflow, reconstructed-surface volume error separately, and penetration volume. For SPH also report bulk density compression and its distribution, distinguishing free-surface neighbor deficiency from actual compression.

Use analytic motion and hydrostatics where possible and resolution/timestep convergence elsewhere. Uniform is a competitor, not ground truth. For chaotic late-time splashes compare gauges, spectra, feature distributions and repeated seeds rather than requiring particle-by-particle or pixel-identical agreement. Render identical materials and also inspect neutral surface/particle views so attractive shading cannot hide defects.

## Implementation milestones and decision gates

1. **Freeze the benchmark and contracts.** Serialize the scene matrix, baseline profiles and error measures. Record current test assertions and designate physical invariants versus Uniform-internal invariants. Calibrate additional tolerances against analytic/converged references before optimizing either arm.
2. **Build fixed-grid APIC/PIC/FLIP.** Implement GPU particle storage, seeding, binning, transfers, projection, advection, surface publication and receipts. Verify constant/affine transfer reproduction, transfer momentum, rest, free fall, translation and failed-solve rejection. This milestone is explicitly a partial prototype.
3. **Establish the performance envelope.** Run the small dam, rest, translation, thin-sheet and pool cases. Profile binning, transfer, pressure and reconstruction separately. If no configuration reaches the agreed fidelity at the budget, identify the limiting stage before adding temporal or adaptive complexity.
4. **Complete physical and editor parity.** Add inflows, injection, cut solids, live edits, rigid coupling, material properties, runtime controls and diagnostic integration. Extend the shared harness with method-owned observables rather than weakening existing assertions.
5. **Complete live detail parity.** Add conservative particle split/merge and coupled coarse/fine grid changes. Test repeated refinement cycles, contact support and motion across resolution boundaries. A dense-only implementation cannot pass this milestone.
6. **Evaluate the recent research extensions.** Compare ordinary APIC/FLIP, phase-field projection without jitter, and ST temporal sampling at matched fidelity and output times. Add IPIC only if vortex benchmarks expose meaningful transport loss. Prototype IPBF separately if evidence favors the SPH cost structure; Tube Maps is conditional on boundary geometry suitability.
7. **Run the full comparison and promotion gate.** Produce raw JSON, plots, synchronized scene captures, parity results and a per-scene quality/time frontier. Promote only the configurations whose fidelity and full-frame latency both pass; retain failure cases in the report.

Every solver implementation milestone must satisfy the repository's clean gate: `npm run check:types`, `npm run test:unit`, and `npm run test:dawn`. Targeted Dawn filters are diagnostic aids, not substitutes for the final serial suite. Implementation verification is recorded with the performance pass linked above.

The decision after the first prototype is concrete: proceed if particle transport preserves the thin liquid and motion that matter while leaving enough frame time for parity features; optimize the measured limiting stage if the quality passes but the frame budget fails; change method family if projection/transfer/reconstruction cannot meet the measured frontier. No paper establishes that any one method wins across every scene.
