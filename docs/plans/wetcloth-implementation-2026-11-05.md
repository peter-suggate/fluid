# Water and cloth implementation and paper scene reproduction

Date: 2026-11-05; evidence updated 2026-11-06. Status: implementation in progress; paper reproduction is not yet certified.

## Current implementation evidence

All 36 water-only released inputs load with checksummed original topology and source scripts. Effective material defaults are still being audited against the released constructor and force code; successful import is not proof of physical parameter fidelity. Three separately identified variants preserve the original inputs while recording paper-panel corrections or an explicitly exploratory packing change. Spot's canonical author mesh has been recovered with provenance; equality to the absent upstream copy cannot be established. The scene definitions remain validation fixtures until the physical and editing gates below pass.

Implemented subsystems include shell/DER mechanics, swept physical contacts with reciprocal rigid reactions, shared triangle/finite-radius yarn picking/rendering and through-cuts, actual production narrowband-FLIP drag/pressure coupling, conservative pore transport and local capture/release, source emission, and native resident-water termination. Framework features and publication ports separate their ownership. These are implementations, not a claim that all source trajectories match the paper.

Verified evidence so far:

The source audit corrected finite attachment springs (instead of hard pins), independently defaulted second fiber radii, deformation-specific viscosity coefficients, and four-component DER bending. Generic authored hard pins remain supported separately. Source bulk viscosity and free-surface tension switches are independent of Darcy viscosity and pore capillarity. Earlier full-source GPU first-step and 0.5-second dry receipts are **superseded for source-fidelity claims**; retain them only as historical implementation diagnostics. Corrected small GPU mechanics oracles pass; corrected full-source trajectories require rerunning after the audit closes.

- The finite-attachment CPU reference completes 500 original-timestep `simple_cloth` steps and the first full towel step. These intermediate results precede subsequent material-law corrections and are not recertified source trajectories. Contact regressions cover swept rod interiors, cut caps, friction and work-conjugate rod angular impulses. Intrinsic material-neighborhood exclusion resolves false contacts in an undeformed thick sheet. Elliptical segment geometry now has CPU picking, cut, fidelity, save/load and undo checks; GPU render parity and matching contact verification are in progress.
- The integrated production NB-FLIP solver completes two accepted submerged cloth steps in both tank and centered free-space fixtures, with matching initial free+bound water budget. Physical support reserves full detail-ring pages and a separate owner allowance alongside normal FLIP band capacity, seeds dry cloth without initial water, and follows its predicted footprint past the authored box through native page requests. The actual GPU rerun passes after resolving both slot and owner-budget eviction. No coarse substitute is accepted for the cloth MAC stencils.
- Actual GPU local-water tests pass paired budget exchange, phi-identical fully submerged absorption, release, and moving submerged fiber conservation. Native geometry/pressure/volume use free capacity after subtracting fibers and bound water. Manifold-edge remap support clears the earlier disconnected moving-sheet failure. Stronger moving-clamp reconstructed-volume checks exposed an initial rigid-body mirror omission; publishing poses/cuts before the initial census fixes the initial physical volume (GPU verified). Moving-clamp reconstructed volume still fails, both submerged and at the interface, including held-pose follow-up steps. Eight-point hard-cell sampling can entirely miss a thin plate between its samples; geometric coverage and moving-domain continuity require correction. Ledger-only passes do not certify physical conservation. The first source capillary trajectory also exposes a local receiver-feasibility failure; deterministic transaction diagnostics are being added.
- Wet cohesion uses source directional candidates and actual native free-liquid phi bridge samples, followed by the same implicit mechanical energy solve. CPU derivatives and equal/opposite linear and angular reactions pass for fixed/rigid endpoints. Shared edited-surface quadrature is wired into the host, preserving source sampled-solid flags (the towel clamps explicitly disable sampling). Full wet clumping and rigid-surface adhesion remain open. The constitutive audit corrects absorption timescale, drag/suction tensor axes, deposited grid fractions, nonlinear manifold mobility and the shell FEM metric; the newest actual grid constitutive test passes. The split pressure solve and nonlinear source-predictor differences still require trajectory/convergence assessment.
- Shared source camera roll now reaches the CPU and GPU camera basis, including polar views; actual GPU ray/picking checks and physical yarn depth/cut-gap parity pass. Source preview appearance follows pinned gray/blue liquid-content colors, independently of simulation and editing. Houdini dry/wet material records and camera parameters are audited separately; converting that full material graph remains pending. A complete production-renderer time-zero pore scene capture is saved as `artifacts/wetcloth/pore-mid-source-preview-t0.png`; it proves startup/publication, not wet trajectory reproduction. Empty static geometry now publishes a valid initial SVO generation through its normal finalizer (CPU and GPU regression).
- All 36 source imports have checksums and load successfully. Source-water emission's GPU tests pass volume, momentum and repeated-source checks. Resident termination removes actual live production FLIP particle records and native water, including an inverted terminator, with compensated budget accounting verified on the GPU.

- Coupled source runs on production NB-FLIP (2026-11-06, receipts in `artifacts/wetcloth/`):
  - `capillary-suction-16s.json` ran the full 16 s (16,000 steps, 34 ms/step). The wet front (S>0.01) rises roughly as t^0.6, reaching the strip top at 14.5 s; S>0.5 reaches 7.1 cm. Free+bound water drifts 2.9e-6 (relative), within f32 budget rounding.
  - The Figure 4 pair ran 0.02 s each (100 steps; ~1.3–2.2 s/step on a shared GPU). Over that time, with pressure on the solid, the submerged 250 kg/m³ sheet rises (+0.14 mm), the 1000 kg/m³ sheet holds still (+0.003 mm) and the 4000 kg/m³ sheet sinks (−0.33 mm). Without it, all three sink (−0.27/−0.34/−0.52 mm). See `buoyancy-figure4-comparison-0.02s.json`.
  - Full-size dam-break steps now take ~2.1 s (was ~12 s). Mechanics is 1.19 s of that (58 GPU sweeps, 0.5 Newton refinements); pore transport is 0.43 s and stencil preparation 0.17 s. A 4 s run (20,000 steps) is still ~12 h. The Figure 6 pair ran 0.01 s on the same mechanics (`dam-break-figure6-comparison-0.01s.txt`). Nonlinear drag deflects 1.5% more in x (min dx −0.357 vs −0.352 mm); relative speeds are still ≤0.11 m/s, too early for the paper's difference.
  - `splash_cloth_small` used to abort in Dawn `writeBuffer` at step 7. The cause is a webgpu 0.6.0 UploadData size check: it fails on a single upload over 4 MiB with an odd word count. Cloth coupling uploads now go in 1 MiB pieces (`uniform-chunked-upload.ts`). The run now reaches step 469 (0.094 s, 2.4 s/step). There the sagging cloth meets the solid sphere and stops with "contact constraints did not converge within 64 sweeps", before any water reaches it.
  - Exact (bitwise-verified) CPU speedups in pore transport, contact distance, stencil and cohesion preparation.
  - `pore_test_mid` ran its impact: 600 steps, 0.12 s, 6.7 s/step (`pore-test-mid-0.12s.json`). It exhausted the default 100-block page pool at commit 1. Its pool is now authored at 128 blocks: the measured peak was 105, plus margin (`pore-test-mid-pool-peak.json`). Water reaches the sheet at about 0.106 s. By 0.12 s it holds 8.1e-7 m³ bound water; S>0.5 spans y 0.173–0.239 m and z ±3.2 cm; the sheet bulges +16.5 mm in x. Free target plus bound water stays at 2.3652e-4 m³. Represented free volume runs 2–14% above target; that is NB-FLIP geometric drift, not ledger drift. A 4 s run would take about 37 h.
- The cloth's per-step water exchange now runs entirely on the GPU (2026-11-07): native support sample, Darcy capture, manifold transport, overflow release, the capacity table NB-FLIP reads, and the paired budget commit, in one submit and one readback (`uniform-cloth-gpu-pore-exchange.ts`; the CPU water exchange is its f64 test oracle). The budget commit splits each delta at the high word's ulp, because Metal's fast math folds the usual two-sum compensation to zero. On `capillary_suction` (30 steps, every step sampled), the steady step drops from 44.0 to 30.0 ms. The exchange takes 4.2 ms, down from 15.2 ms for sample, capture, transport and release. Its pipelines compile once, which makes the first step 1.9 s. Free and bound water match the paired ledger to 4e-12 relative. The phi measure plus bound water is +0.275% at step 1 and +0.354% at step 30 (the CPU exchange gave +0.331%).
- Phi measure, step-1 shift (handed to the fluid lane): the step-1 rise in measured phi volume does not depend on dt and scales as h². It appears without cloth: `capillary_suction` without cloth is +0.13% at dx 0.72 mm, at both dt 1 ms and 0.25 ms; `dam_break_nonlinear_drag_small` without cloth is +0.253% at h and +0.065% at h/2. With pore-free cloth, `capillary_suction` is +0.260% at both dt 0.25 ms and dt 2.5e-10 s. The budget is captured after the cloth capacity table is published, and volume control is off, so no mismatched target drives it.

### Source discretization

Each released source keeps its dx and dt unless a measured pure-fluid ledger shows it is under-resolved. A refined source gets an entry in `lib/cloth/scenes/discretization.ts`: local refinement only, the released Courant number, and no substeps.

- `capillary_suction`: refined to dx 0.72 mm and dt 0.25 ms (the tube and drop were 2h in radius).
- Dam-break drag family (`dam_break_linear_drag`, `dam_break_nonlinear_drag`, `dam_break_nonlinear_drag_small`): keep the released h. The rounded slab is 10 cells thick. `dam_break_nonlinear_drag_small` without cloth, at dt 0.2 ms, is +0.253% at step 1 with +0.064% creep over the next 29 steps. At h/2 the figures are +0.065% and +0.029%, but the step goes from 24.7 to 70.1 ms. That is not the capillary-style loss, so the family is not refined.
- Candidates still to check with a ledger before any refinement: `tighten_towel` (2.7-cell slab, boards about 0.35 cells) and the ring jets (0.5h).
- Fluid-only scenes (`simple_fluid`, `simple_surf_tension`, `double_surf_tension`, `simple_fluid_mesh`): not yet audited for resolution.

- Open issue (pre-existing at HEAD 37cbeadc; matters for the towel's moving clamps): after an impulsive submerged rounded-clamp move (11 m/s for 1 ms, then held), phi and physical volume oscillate against a paired ledger of 0.2547586 m³ (2038.07 cells). Volume control engages from step 3. The interface variant holds exactly. The tolerance is 0.01 cell and was not changed. **Known failing GPU gate:** `tests/cloth-nbflip-gpu.test.ts`, "moving rounded clamp ... when submerged", until the fix lands. It is GPU-only (skipped without `WEBGPU_NODE_MODULE`), so it is not a unit-suite failure. Diagnosis (2026-11-06): the drift is unchanged with the cloth coupling plugin never attached, so it is native NB-FLIP. In the hold steps, bulk liquid cells about 5 cells below the surface lose occupancy (0.56–0.86) only beside the container walls and floor. Phi transport at static walls appears to read air-valued phi outside the open domain. The fix must take liquid-side extrapolation there, never air, with no clamps.

  | Hold step | HEAD phi volume (m³) | 11 mm fixture (m³) | vs ledger |
  |---|---|---|---|
  | 3 | 0.253925 | 0.254029 | -0.29% |
  | 4 | 0.251844 | 0.251785 | -1.2% |
  | 5 | 0.254008 | 0.254100 | -0.26% |
  | 6 | 0.252991 | 0.253535 | -0.48% |

- Open issue (native NB-FLIP; passed to the NB/perf lanes): a sharp, 4h-aligned liquid box with no fine tiles measures 6–9% low on the 4h band, walled or in free space. Seed 3456 cells (12×24×12), after step 1: 3256.2 or 3154.3, depending on position mod 8h. A 32×24×32 block in a 0.64 m window measures 23994.6 of 24576 (-2.37%), bit-identical walled and free. The same box with fine tiles measures -0.5%. Measure errors outside the 1% volume-control dead band drive divergence (free-fall cloth: +3% velocity). Free-space initial liquid now keeps a 16h far-air margin from the seed window (`assertFreeSpaceSeedMargin`). At 4h and 8h, low faces differed from walled by about ±3.3%; at 16h, free space equals walled.

Open reproduction gates include complete wet trajectories and paper comparisons; the full moving towel sequence; capillary/contact behavior and coupled timestep convergence; remaining arbitrary yarn edit shapes; and production cloth shading. Exact rounded-clamp aperture, extraction and local-cut/undo GPU tests pass, as do fluid-renderer plate intersections; primary-renderer picking parity also passes. No scene is marked reproduced merely because its input loads or one step passes. The source terminator profile temporarily uses the existing h/4h NB-FLIP hierarchy with BULK16 disabled, because native BULK16 retirement is not implemented. It does not replace NB-FLIP or force the full domain to h.

Validation artifacts are written under `artifacts/wetcloth/`, with original input metadata and `reproductionValidated:false` until the per-case gates pass. Run GPU checks and captures only through `tools/gpu/fluid-gpu-run.sh`; shared-device correctness and exclusive timing runs must honor that runner's locks.

Latest required checks (2026-11-06): `npm run check:types` passes. `npm run test:unit` reports 2,039 tests: 1,733 pass, 305 skip, one failure in concurrent reservoir work's shader-text assertion (`reservoir-particle-cost.test.ts`, expected `if(deep< -1.0&&count>cap)`). Log: `/tmp/fluid-wetcloth-research/unit-integrated-interval-clock.log`. This receipt precedes the final integer-time coupling and new solid-integration health changes. GPU-only skipped tests require separate actual-device evidence. Shared elliptical render/depth/cut and rolled-view parity now passes all nine checks (six actual GPU cases), complementing the CPU geometry/edit and mechanics oracles. Source imports and implemented capabilities remain separate from unverified scene reproduction.

Implement water interacting with triangle cloth, yarn and fuzzy towels using the actual existing `uniform-narrow-band-flip` method. Keep Box3D for rigid bodies. Use framework feature composition and typed publication ports. Simulated deformation, water exchange and contact must be physical; only prescribed supports and liquid sources may follow the original scene scripts. The paper is Fei, Batty, Grinspun and Zheng, *A Multi-Scale Model for Simulating Liquid-Fabric Interactions*, SIGGRAPH 2018.

The user's water-only restriction excludes honey, olive oil and acetaldehyde. Do not introduce a high-viscosity solver for those examples. The source inventory contains 39 XML files; 36 water cases are in implementation scope. Figure 10 panels h/i and `pore_test_honey`, `pore_test_oil`, `pore_test_acetal` remain reference-only. Never call replacing those liquids with water a faithful reproduction.

## Evidence and fidelity

- [Paper and demonstrations](https://www.cs.columbia.edu/cg/wetcloth/).
- [Paper PDF](https://www.cs.columbia.edu/cg/wetcloth/figs/wetcloth_reduced.pdf), especially figures 3, 4, 6, 10–16.
- [Supplement](https://www.cs.columbia.edu/cg/wetcloth/supplemental.pdf).
- [Released implementation and assets](https://github.com/raymondyfei/libwetcloth/tree/b43993d775a0f9e3a9e32611f26936f2528fac37), pinned commit `b43993d775a0f9e3a9e32611f26936f2528fac37`.
- [Machine-readable source inventory](wetcloth-scene-inventory-2026-11-05.json): original paths, checksums, integration settings, structural counts, material inputs, Houdini companions and asset issues. These are inventory records, not completed simulations.

Three separately identified configurations are necessary: unchanged upstream inputs, corrected paper-panel inputs where published values conflict with the release, and explicitly labelled interactive approximations. Retain provenance for every correction. Paper geometry, material distribution, physical dimensions, source timing, supports and duration are authoritative. NB-FLIP is deliberately different from the paper's full-domain APIC algorithm: reproduce physical behavior and scene appearance, not bit-identical trajectories. The original executable is an offline comparison oracle, never the application's fluid solver.

Source audit findings that must be resolved before a reproduction is certified:

1. `pore_test_low.xml` specifies 20 micrometres whereas Figure 10b labels 25. Preserve both inputs under distinct IDs.
2. `pore_test_zero.xml` sets surface tension to zero; Figure 10j describes a 90-degree contact angle. The corrected panel must suppress pore suction without automatically suppressing free-surface tension.
3. Several files specify `fabricThreadCount`, but that name is absent from the pinned XML parser. Resolve effective material properties from equations, initialized upstream state and history; do not silently ignore the field.
4. `ring_test.xml` enables nonuniform initialization without explicit vertex fractions; the parser's vertex default is 1. `ring_test_nonuniform.xml` contains 40,000 explicit `vf` values. Preserve the latter and investigate the former rather than trusting filenames.
5. `unit_tests/simple_fluid_mesh.xml` references `assets/meshes/spot_triangulated.obj`, absent from the release. Recover the exact model from its original source and verify scale/topology. A substitute must stay labelled and cannot close this case.
6. Render camera/material settings in Houdini are distinct from XML preview cameras. Extract them where available; otherwise record measured camera reconstruction and its uncertainty. Main-paper element counts also need comparison with the released resolutions.

## Papers to review alongside the reproduction

Research checked 2026-11-05. These inform implementation choices; none changes the requirement to use the production narrowband FLIP solver.

| Paper | Why it matters here |
| --- | --- |
| [Fei et al., A Multi-Scale Model for Simulating Liquid-Fabric Interactions (2018)](https://www.cs.columbia.edu/cg/wetcloth/) | Primary physical model, source scenes and reproduction target: pore-scale constitutive laws, surface transport, shell/rod mechanics and bulk liquid coupling. |
| [Kim and Lee, Unified GPU-based simulation of porous flow, absorption, and diffusion in cloth–liquid coupling (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13578283/) | Recent integrated GPU wet-cloth implementation with saturation exchange, emission and diffusion. Its SPH/DFSPH formulation is useful comparative reading, not a substitute for NB-FLIP. |
| [Chen et al., Offset Geometric Contact (SIGGRAPH 2025)](https://graphics.cs.utah.edu/research/projects/ogc/) | A contact model and bounded-displacement strategy for scalable codimensional contact; relevant to thick folds and dense towel pile. Its guarantees require implementing the full algorithm, not borrowing its name for ordinary collision projection. |
| [Lu et al., High-performance CPU Cloth Simulation Using Domain-decomposed Projective Dynamics (SIGGRAPH 2025)](https://sig25ddmpd.github.io/) | Domain decomposition and local/global parallel scheduling, useful when fluid and rendering already occupy the GPU. |
| [Lan et al., Efficient GPU Cloth Simulation with Non-Distance Barriers and Subspace Reuse (SIGGRAPH Asia 2024)](https://wanghmin.github.io/research-areas/cloth/) | Solver and contact acceleration context for larger cloth meshes. Review the linked author paper before adopting its constitutive or contact model. |
| [Using The Polynomial Particle-In-Cell Method For Liquid-Fabric Interaction (2023 preprint)](https://arxiv.org/abs/2308.01060) | Transfer-method sensitivity in wet cloth. Relevant when evaluating differences from the 2018 APIC baseline; not authorization to replace FLIP. |

## Architecture and ownership

Add a `lib/cloth/` domain with authored contracts, runtime state, materials, geometry, dynamics, contact and fluid-exchange features. Definitions use `lib/framework/composition.ts` and typed ports; the host owns allocation and execution. Do not add a competing water method or represent the towel as thousands of independently integrated Box3D bodies.

| Owner | Responsibility | Integration seam |
| --- | --- | --- |
| Cloth document plugin | Versioned material/topology/support/source references, validation, persistence | `SceneDescription`, scene store and worker serialization |
| Cloth geometry plugin | Deformed triangles and finite-radius rods; shared picking/editing/render authority | `visibleToolSurface`, scenery/solid publication and renderer |
| Cloth mechanics plugin | Shell membrane/bending and rod stretch/bend/twist | GPU buffers and ordered solver phases |
| Cloth contact plugin | Self-contact, cloth/rod contact, terrain/rigid contact, friction and attachments | Shared geometry; compact reactions to Box3D |
| Porous coupling plugin | Pressure, anisotropic drag, suction, saturation, wicking, capture/release | NB-FLIP grid forces/projection/volume/source phases |
| Existing NB-FLIP | Water surface, bulk grid momentum, particles and pressure hierarchy | `uniform-mixed-frame`, `uniform-mixed-surface-volume`, narrow-band transfer |
| Existing Box3D | Rigid body transforms, supports/grippers, rigid contacts | Existing interaction host fixed clock and impulse exchange |
| Paper-scene plugin | Reproducible imported assets, controls, cameras and diagnostics | Existing scene shelf and method profile system |

Expose phase contracts for support preparation, pre-force prediction, pressure assembly, post-projection momentum gathering, manifold transport, state publication and rigid reactions. A rigid-only post-pressure callback is insufficient. Every publication carries generation, tick, topology revision and accepted simulation time. Disallow stale or duplicate exchanges. Zero cloth scenes allocate no cloth buffers and retain the original solver path.

Initial material models must match the paper's shell and discrete elastic rod energies. VBD is an integration/optimization candidate, not permission to replace the material with an arbitrary spring net. First verify derivatives and converged trajectories against a small CPU reference. Modern OGC/DAT contact may be used after separate physical-contact verification; it must not be described as the paper's AMLS-MPM collision implementation. If a changed contact model fails reproduction, investigate that discrepancy before tuning materials to conceal it.

## Authored and runtime data

Store immutable rest positions, triangle indices, ordered rod vertices, material IDs, rest areas/lengths, fiber directions, support membership, source scripts and source hashes. Store physical dimensions in SI. Runtime state includes positions/velocities, rod material frames and twist, deformation/thickness, dry mass, absorbed water volume, saturation, capacity, contact state and solver history. Preserve explicit per-vertex material fields.

The importer must convert CGS quantities by dimensional meaning: cm to m (0.01), g to kg (0.001), g/cm³ to kg/m³ (1000), dyn/cm² to Pa (0.1), dynamic viscosity in g/(cm s) to Pa s (0.1), dyn/cm to N/m (0.001). Source angles need explicit degree/radian semantics. Do not apply one scale to every numeric field. `yarnDiameter` is a microstructure input, not necessarily the geometric rod diameter. Validate the source's thread-count convention instead of guessing from its label.

Import all faces and rod connectivity directly. Resolve omitted values using a versioned, audited defaults table. Parse source/intersection/termination fields and initial impulses faithfully. Preserve cubic easing, rotation origins, group composition and endpoint semantics from `Script.cpp`; linear endpoint interpolation is insufficient. A zero-duration source is an event, not an inflow evaluated every step.

Separate document save/load from runtime checkpoints. Checkpoints include pore water, velocities, rod frames, time, source event consumption and solver history. Undo of a paused edit restores geometry and wet state. Live cuts are transactions at a tick boundary: split local topology and transfer material/rest/water quantities conservatively; detach affected pins and fuzz roots consistently. Account explicitly for water/solid deliberately deleted by an editor operation.

## Shared geometry and physical collision

Use one authoritative deformed topology for visible surfaces, nearest-visible hits and collision. Triangle picking returns stable element ID and barycentric position; rod picking returns segment ID and material coordinate. Both sides of cloth must be selectable. Depth comparison includes terrain, scenery, cloth and rods. Keep original materials, UVs and all geometry outside a cut.

Triangle and rod representations require regressions for nearest-hit occlusion, local edits, unchanged detail outside the edit, collision opening, save/load and undo before being admitted to scene shelves. Unknown kinds fail explicitly. Do not voxelize the whole towel after a local cut. Do not leave a hidden collision membrane spanning the opening. Visible pile fibers must be addressable, not render-only decoration.

Use swept broad-phase bounds, finite material thickness/radius, vertex-face and edge-edge contacts, adjacent-element exclusions, friction and continuous or provably bounded motion. Validate folded thin layers, high-speed pulls and rod/shell junctions. Porous fluid passage is not cloth self-penetration: water is allowed through the physical porous model rather than repelled as if every fabric face were impermeable.

## Cloth and water physics

1. **Dry mechanics.** Shell stretch/shear/bend, damping and fixed/moving supports; rods additionally carry twist and material frames. Fuzzy cloth shares attachment constraints with its real shell. Compression and pore capacity must respond to deformation so squeezing can release stored water.
2. **Material response.** Derive positive anisotropic permeability/drag and suction tensors in material coordinates, rotate them into world coordinates, and update them under deformation. Keep physical pore radius, geometric thickness and contact separation distinct. Check dilute, dry, saturated and near-impermeable limits without singular branches.
3. **Grid coupling.** Scatter structural volume/momentum/material tensors onto the active MAC grid. Integrate stiff nonlinear relative drag implicitly, with paired equal/opposite impulses. Include the porous solid contribution in pressure assembly and gather its reaction from the same discrete operators. Do not add empirical buoyancy on top of pressure buoyancy. Build matrix and small-system reference tests before trusting hierarchy modifications.
4. **Wicking.** Solve conservative nonnegative transport on triangle/rod manifolds using the capillary/gravity model; conserve volume across material changes and junctions. Redistribution changes saturation without changing total stored water. Use physical tensor orientation, not isotropic texture diffusion.
5. **Capture and release.** Exchange volume and momentum between free water and the material reservoir. Limit capture by available local liquid and pore capacity. Distribute competing requests conservatively. Emit only actually removed reservoir water with the correct local velocity/momentum. Sub-resolution release accumulates in an explicit reservoir until represented; it must not disappear or appear twice.
6. **Wet contact and appearance.** Add capillary cohesion between nearby wet structures where required, separately from collision/friction. Saturation drives appearance from the simulated state. Wetting never comes from a painted animation mask.

### NB-FLIP conservation and sparse support

The existing fluid derives volume from reconstructed geometry and corrects toward a scalar budget. Particle count is not total liquid mass. Maintain the accounting identity

`free water + pore water + unresolved emitted water + measured outflow = initial water + sources + explicit edit delta`.

Capture decreases the free-water target by exactly the increase in pore storage; release reverses it. Update local physical occupancy/transport and momentum consistently, as well as the target. Updating only the global target would shrink water remotely; removing only particles would let reconstruction/refilling restore it. Distinguish water inside pores from free water so neither mass, occupied volume nor inertia is counted twice. Derive the mixture continuity source for changing pore volume; do not create artificial air holes around submerged fabric.

Submerged cloth must couple to grid-owned liquid outside the narrow particle band. Add mandatory support around swept cloth/rod influence, including pressure/force/transport halos. Wake settled regions for motion, capture, pore transport and emitted drops even if the free surface is quiet. Retain coarse/fine flux agreement and explicit failure on insufficient allocation. Do not keep the whole domain hot to hide a missing support rule. FLIP's force snapshot must include the cloth impulse once, so the particle update does not omit or double it.

Use the actual existing water viscosity path within its verified stable timestep. Audit the distinction between pore capillarity and optional bulk surface tension in upstream flags. The application already has these water force terms; the water-only scope does not require introducing honey viscosity.

### Stepping and Box3D

At an admitted shared interval: evaluate support trajectories; prepare swept support; predict dry mechanics; assemble paired drag/pressure exchange; solve the existing fluid projection with coupled terms; gather impulses; solve contact/elastic correction and manifold transport; commit volume transfer; publish accepted cloth/fluid state; return compact rigid impulses to Box3D at the established fence.

That ordering is a starting split whose convergence must be demonstrated. Light cloth and stiff supports can require inner coupling iterations or smaller shared intervals. Repeat from saved provisional state rather than applying impulses twice. On failed residual/contact tolerances, reduce and retry the interval; never silently drop interactions. A dry reference solve and wet solve must advance identical physical time. Use source `dt` (often 0.0002 s) for the first reference comparisons; real-time playback is not a correctness requirement.

Prescribed source supports are kinematic and do external work, which the diagnostic energy balance records. Dynamic Box3D attachments receive equal/opposite cloth reactions. Preserve one transform owner per rigid body. Do not treat an unvalidated one-frame-delayed reaction as equivalent to synchronous coupling.

## Complete scene coverage

The inventory is the exhaustive release checklist. The following groups are all required except the three explicitly excluded liquids. Every row needs an import checksum, runnable scene, matching method profile, initial-state checks, timed diagnostic capture, visual comparison and acceptance record.

| Group and released filenames (without `.xml`) | Required reproduction or verification |
| --- | --- |
| `ring_tests/ring_test`, `ring_test_nonuniform` | Figure 3 and heterogeneous spreading comparison; wet area/front and retained volume versus time. Resolve initialization discrepancy first. |
| `buoyancy_tests/buoyancy_test_with`, `buoyancy_test_without` | Figure 4: pressure-reaction on/off comparison with original material groups; compare vertical trajectories and wet mass. |
| `drag_tests/dam_break_linear_drag`, `dam_break_nonlinear_drag`, `dam_break_nonlinear_drag_small` | Figure 6: preserve source geometry and drag ablation; compare cloth deformation and entrained water, including resolution variant. |
| `parameter_tests/pore_test_mid`, `pore_test_low`, `pore_test_high`, `pore_test_ft_low`, `pore_test_ft_high`, `pore_test_fd_low`, `pore_test_fd_high`, `pore_test_zero` | Eight water variants for Figure 10 a–g/j, plus corrected paper-panel variants where values disagree. Compare at 4 s and across the trajectory, not one selected frame. |
| `general_examples/impermeable_small`, `impermeable_medium` | Pinned near-impermeable water retention; quantify leakage and stretch under water load. |
| `general_examples/splash_cloth_small`, `splash_cloth_medium`, `splash_cloth_large` | Figures 1/11 at released resolutions; impact, penetration, retained water, sag and drainage. |
| `general_examples/splash_yarn_small`, `splash_yarn_medium`, `yarn_small` | Figure 12 and released variant; preserve actual weave gaps/strands and characterize penetration/jet formation. Inspect actual source behavior of `yarn_small`; do not infer it from its name. |
| `general_examples/fuzzy_cloth` | Figure 13: shell and physical pile strands; compare capture and stronger drag with plain cloth. |
| `general_examples/tighten_towel` | Figure 14: complete 12 s lift/twist/hold/untwist sequence; verify discharge and persistent drips with physical compression and contact. |
| `general_examples/splash_plastic_medium` | Additional porous-plastic water example; preserve material differences instead of merely renaming the cloth preset. |
| `unit_tests/capillary_suction` | Vertical uptake, gravity balance and storage conservation. |
| `unit_tests/simple_cloth`, `simple_plastic` | Small scripted shell/material regression fixtures. |
| `unit_tests/simple_yarn`, `simple_yarn_loop`, `splash_single_yarn` | Rod connectivity, closed-loop semantics, scripted motion and liquid/rod exchange. |
| `unit_tests/simple_fluid`, `simple_fluid_mesh`, `simple_surf_tension`, `double_surf_tension` | NB-FLIP water/control cases; exact external mesh remains an asset dependency. |
| `parameter_tests/pore_test_acetal`, `pore_test_oil`, `pore_test_honey` | Reference-only, excluded by water-only scope; never required for acceptance. |

Recreate the main-paper comparison layouts, not just individual scenes: a synchronized buoyancy pair, linear/nonlinear drag pair, eight water parameter panels, and the three general splash sequences plus towel. Figure 1 reuses the appropriate demos; Figures 15/16 require measured volume curves and our own performance breakdown, not copied timings. Figures explaining models/algorithms are documentation rather than extra simulations. Link the original real-world experiment as a qualitative reference; do not claim it is simulation ground truth.

The source towel has 47,072 authored particle records, 48,954 faces, 22,272 two-point hair records and 32 motion scripts. Import their shared vertex connectivity exactly. The full source sequence includes later untwisting; stopping after the first wring does not complete it. The large cloth splash has 159,201 authored structural vertices and 316,808 triangles. These source counts are not fluid particle counts or automatically the counts in the paper's table.

## Implementation sequence and acceptance gates

| Stage | Deliverable | Must pass before proceeding |
| --- | --- | --- |
| 0 | Pin sources, update water-only inventory, audit defaults/units/scripts, obtain reference captures | Every water case mapped; discrepancies and missing files explicit; no fabricated defaults |
| 1 | Versioned cloth/rod document schema and framework feature contracts | Invalid kinds/materials rejected; persistence round trips; old scenes unchanged |
| 2 | Shared triangle/rod geometry, picking, local editing, rendering publication | AGENTS.md geometry regressions; no invisible duplicate collision surface |
| 3 | Dry shell/rod mechanics and physical contact with CPU reference | Force/energy consistency, mesh/time convergence, folds/pulls without crossings, pinned work accounted |
| 4 | GPU porous grid transfer, support wakeup, pressure/drag coupling | Equal/opposite impulses, uniform-flow invariance, submerged cloth without particles, pressure ablations |
| 5 | Pore storage, capillary/gravity transport and conservative exchange | Localized capture, no volume restoration, nonnegative saturation, conservative release and cut handling |
| 6 | Box3D attachments/reactions and shared-clock integration | No duplicate integration; stable light-cloth test; stale feedback rejected; reset/checkpoint replay |
| 7 | Source importer and all water scene assets, small fixtures first | Exact counts/units/topology/source events; explicit asset-gap handling |
| 8 | Ring, suction, buoyancy, drag and eight water parameter panels | Quantitative curves and ablation ordering agree with reference; disagreements investigated |
| 9 | Mesh/yarn/fuzzy splash and full towel sequence | Complete physical-time runs; physical pile/contact; water discharge and appearance comparable |
| 10 | Full-resolution and interactive profiles, editor UX and diagnostic controls | Reference profile remains unchanged; performance and approximation error reported separately |
| 11 | End-to-end reproduction dossier and regression suite | All 36 in-scope source cases and required corrected panels accepted; no unresolved fidelity substitutions |

Parallel ownership after the plan: mechanics agent owns new cloth dynamics/material modules; scene agent owns importer/assets/catalog; fluid agent owns porous grid coupling. Main integration owns the document/render/editor/runtime seams and cross-system acceptance. Agree shared types before edits, and assign each existing file one owner. Re-read before patching because unrelated NB-FLIP work is active in this checkout. Never reset that work.

## Validation protocol

Record commit, source hashes, adapter, precision, grid spacing, timestep, solver tolerances, transfer blend, contact settings, seeds and output times. Reference captures use original physical dimensions and event timing. Compare the same simulation time; do not time-warp the result or select only flattering frames. Preserve failed runs and diagnostics.

Numerical tests: dry energy derivatives; rigid translation/rotation invariance; pressure/drag momentum conservation; nonnegative pore storage; conservative exchange and manifold diffusion; saturation/capacity changes under deformation; coarse/fine seam transport; passive contact energy; supported and unsupported droplet accounting. Momentum tests remove gravity and kinematic work or include their recorded impulses. Water ledger tests distinguish exact transfer bookkeeping from existing FLIP geometric volume drift.

Proposed initial engineering thresholds, to be finalized from reference convergence rather than represented as paper guarantees: cumulative exchange bookkeeping error below 1e-5 of initial water; full-scene total-water drift below 1%; no accepted geometric crossings; wet area, retained water, cloth sag and major event timing within 10% of the converged reference where observable. Use a spatial tolerance tied to cell/mesh resolution for near-zero metrics. Tighten tolerances after refinement studies; never loosen them just to mark a failed scene complete. Visual agreement also requires matching folds, outflow patterns and wet boundaries, not only scalar metrics.

For the towel, plot free/pore/total water, emitted flux, contact counts, maximum stretch, coupling residual and rigid-support work for all 12 s. For the ring, compare wet-front contours and anisotropy. For the parameter panels, verify causal ordering across variants. For buoyancy and drag, reproduce both the enabled physics and its deliberately disabled counterpart. The near-impermeable case must retain water without replacing the cloth with a hidden rigid bowl.

Run `npm run check:types` and `npm run test:unit` after each integrated milestone. Add meaningful CPU, real-WebGPU and actual Box3D tests; source-string assertions alone do not verify physics. Use the existing Dawn/headless GPU harness for numerical captures and the application for visual/editing/transport checks. Attribute unrelated baseline failures separately. Record GPU timings and peak memory only after correctness; no unsupported real-time claims.

Completion means functioning scene presets in the application, editable geometry, save/load/undo, measured water/cloth/contact behavior, original-source attribution, and a per-case reproduction record. Imported assets, passing unit tests or attractive screenshots alone do not meet this definition.
