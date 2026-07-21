# Power-Diagram Unified Octree Implementation Plan

Status: implementation handoff  
Primary target: WebGPU octree liquid simulation at 16³ and 32³ maximum leaf sizes  
Representation constraint: one unified sparse octree; no persistent dense simulation fields  
Pressure-solver direction (revised 2026-07-20): the authoritative power path
now uses the paper's Section 4.3 hybrid PCG preconditioner: an SPD first-order
L1 V-cycle with adjoint transfers, bracketed by paired `k=8` second-order
boundary/transition smoothing. Compact-row Chebyshev and compatibility
projection remain explicit fail-closed rollback paths (see "Critique outcomes
and direction changes").

## Implementation checkpoint and restart handoff (2026-07-20)

This is the authoritative handoff after the first production-integration
push. The local WebGPU/Metal runtime entered a machine-wide bad state: shaders
and submissions validated, but even trivial compute readbacks returned zero.
No further Dawn or browser result from that state is admissible evidence. The
next session starts after a computer restart and must re-establish a minimal
compute sentinel before running the liquid acceptance gates.

### Late implementation/evidence checkpoint (2026-07-20)

This checkpoint supersedes the implementation-status portions of the older
late-audit and continuation lists below; those lists remain as historical
rationale. The paper and this plan's acceptance gates remain authoritative.

Product-path cutover: the normal balanced `water-box-dam-break` preset now
requests the 384-column (`24 x 18 x 16`) cubic safety grid, compact face
transport, factor-4 global fine phi, authoritative power projection, and
`leafSolver=auto`. Once power authority is admitted, `auto` selects the
Section 4.3 hybrid PCG path; terrain, imported/seeded geometry, anisotropic
spacing, or any failed power publication retains the axis/Chebyshev rollback.
Browser WebGPU startup remains manual, so changing the product preset does not
submit GPU work on page load. This cutover is configuration and CPU-test
evidence only: the staged Dawn and one-step browser gates below are still
required before claiming a physical dam-break acceptance pass.

#### Current safe bring-up checkpoint

- The isolated 384-column `solver-resources` Dawn gate passes the adapter,
  known-value compute sentinel, all pipeline/resource construction, and delayed
  uncaptured-validation check. The former 12-storage-buffer Section 5 pipeline
  is gone; every compute entry point stays within the portable limit of ten.
- The following combined sparse-t=0 warmup timed out after 120 seconds. Its
  Dawn child remains in macOS's unreaped exiting state at 0% CPU, and the
  exclusive `/tmp/fluid-webgpu-exclusive.lock` deliberately remains. Do not
  run another Dawn or browser-GPU process until the OS reaps that PID; a reboot
  is the expected recovery if it does not clear.
- To remove and localize that blocker, interface discovery, external seed
  insertion, and the exact one-brick topology dilation are now parallel. The
  21-cell default fine redistance retains all 42 half-cell causal buckets and
  all 595 ordered dispatches, but batches them into 86 compute passes instead
  of roughly 595 pass transitions.
- Sparse t=0 startup is now five dependency-ordered fenced submissions: cold
  topology, power/operator authority, surface/global-fine redistance, Section
  5 face band, and sparse render world. A future timeout reports the exact
  phase, and render attachment becomes ready only after the final fence.
- The final Section 5 regular-face to power-face publication now uses the
  paper's regular-cell trilinear or catalog containing-tetra barycentric
  interpolation at the physical face centroid, followed by projection onto the
  generalized face normal. The former inverse-distance approximation is gone.
- Consolidated non-GPU evidence after these changes: TypeScript passes, the
  complete water WGSL validator passes, 84 focused tests pass, five optional
  Dawn tests skip, and `git diff --check` passes. Runtime, tall-cell parity,
  and browser-image acceptance remain outstanding.

- A known-value Dawn compute/readback sentinel now passes, so the current
  runtime trace is admissible evidence rather than a repeat of the poisoned
  Metal state.
- Recurring topology candidate/worklist publication is GPU-transactional:
  invalid, stale, overflowing, and failed-empty candidates retain the previous
  bounded generation, while valid-empty is represented explicitly.
- The bounded analytic bootstrap now seeds genuine size-16 and size-32 owner
  leaves; focused GPU census evidence no longer shows the former size-8
  fallback cap.
- Authoritative power projection constructs and selects the Section 4.3 hybrid
  PCG path. It captures the Cartesian/GFM L1 rows before the second-order
  operator, applies the explicit free-surface anchor, uses adjoint multilevel
  transfers in the SPD first-order V-cycle, and brackets that cycle with the
  paired `k=8` second-order boundary/transition smoother. The aggregate PCG
  and compact Chebyshev machinery remain rollback/diagnostic paths.
- Surface allocation is capacity-planned from the interface surface-area and
  band bound, with explicit overflow/fail-closed publication rather than a
  box-volume allocation.
- Global-fine rendering claims authority only after an actual surface
  crossing. Otherwise extraction transactionally falls back to the adaptive
  leaf/page source, or retains the last valid mesh when neither source can
  publish; validation failures are not hidden.
- The clean staged trace currently reaches resource construction. The first
  sparse-t=0 submission timed out before it could publish runtime evidence; the
  now-parallel topology, batched redistance, and five-phase warmup have not yet
  been rerun because the old Dawn PID is still OS-unreaped. This remains a
  diagnostic checkpoint, **not** a physical dam-break acceptance pass; visible
  advancing water and the Dawn, tall-cell parity, browser, convergence,
  memory, and endurance gates are still required.

### Completed production work

- The descriptor → catalog topology → generalized power faces → compact
  operator → authoritative Section 4.3 hybrid PCG → same-face projection chain
  is connected to `WebGPUOctreeProjection`. Aggregate PCG and compact
  Chebyshev remain bounded rollback paths; physical convergence evidence is
  still required.
- Catalog version 3 contains 6,475 canonical configurations, a packed direct
  map for all raw descriptor/orientation combinations, compact local
  tetrahedron ranges, and reproducible generation. GPU descriptor resolution,
  bounded site hashing, deterministic scans, generalized face emission,
  reciprocal CSR incidence, open/world boundaries, compact row assembly, and
  volume-normalized divergence are implemented.
- Section 5 velocity reconstruction is present as an area-weighted
  least-squares cell-centre stage followed by indexed trilinear/tetrahedral
  point interpolation. The fine level set does not allocate velocity or
  pressure; every trajectory segment samples the reconstructed octree field.
- The fine interface is one global indexed sparse lattice at factor 4 or 8,
  not leaf-local storage. It has GPU page generations, hash/worklist indexing,
  interface-plus-exactly-one-ring initial activation, transport, exact
  distance-ordered Cartesian fast marching with causal page activation, volume
  correction, sparse summaries, fine-to-coarse correction, and atomic
  publish/rollback. The first exact factor-4 implementation used a serial GPU
  heap and failed the production runtime gate: the `60 x 45 x 40` first
  submission had not returned after three minutes. It has been replaced in
  source by half-cell parallel causal buckets, live-page indirect dispatch,
  and a compact between-bucket page-request allocator. Static WGSL validation
  passes, but the production one-step timing/telemetry rerun is still required.
  The same global-coordinate marcher now accepts factor-8 B4 generations:
  two bricks per finest octree cell axis are addressed, activated, and clipped
  against the doubled sample lattice without weakening the physical band or
  any capacity/publication gate. A valid fine
  sample owns the narrow band; a miss falls back to the compact coarse-phi
  directory. Samples deliberately skipped outside the transport core are
  copied from the source generation before the transactional commit, rather
  than read from shared redistance scratch.
- The renderer consumes the published global fine source first and the compact
  coarse source second. Coarse cubes are suppressed only when all eight fine
  corner samples validate, and an unpublished next generation retains the
  previous mesh.
- Analytic dam-break and tank-fill initialization no longer allocates or
  uploads a box-sized phi field. It binds a one-texel `r32float` format
  placeholder, evaluates the authored analytic SDF in GPU kernels, and keeps
  dense initialization only as a compatibility path for imported/seeded
  geometry, terrain, and rigid bodies.
- Analytic `t=0` no longer scans the finest domain. A constant-time host planner
  derives conservative bounds from authored scene scalars without enumerating
  tiles; one GPU pipeline writes deterministic x-major topology-tile indices,
  active/retired and candidate indirect dispatch words, and tile states. The
  range includes every negative analytic-liquid tile, interface support, and
  one topology grading ring. Missing tiles are analytically proven
  non-negative until compact coarse phi publishes.
- Fine-summary hierarchy allocation now follows exact sparse ancestor bounds.
  The factor-4 balanced checkpoint dropped from 32 MiB to 8 MiB and factor 8
  from 256 MiB to 32 MiB. Face-transfer radix/scatter dispatch is authored from
  the live face count rather than running every pass over maximum capacity.
- Power stages were separated into portable shader modules to stay within the
  observed ten-storage-stage/binding limit. Legacy texture extrapolation is
  disabled when canonical face transport is authoritative.
- The WGSL unreachable-code warnings observed in Chrome were removed from
  projection, overlay, pressure-to-body coupling, and surface-page free-list
  shaders without changing their retry/fallback semantics.
- Simulation scheduling and authority selection remain GPU-resident. Explicit
  QA/telemetry readbacks exist, but they are observational and must never steer
  topology, publication, pressure, or rendering decisions.

Static checkpoint results:

- `npx tsc --noEmit`: pass;
- production WGSL validation: pass;
- power-catalog reproducibility check: pass;
- focused integration suite: 89 pass, 5 optional GPU tests skipped;
- production build: pass;
- full unit suite: 1,071 pass, 13 fail, 71 skip. The 13 failures are existing
  SVO/glass/UI source-contract tests outside this liquid cutover; the focused
  octree/power/fine/renderer suite is green.

### Paper-technique visualization and `t=0` inspection contract (2026-07-20)

The Render & debug panel now exposes the algorithm's live compact GPU data on
the existing movable 3D slice. The original Structure, Cell scale, Surface
band, sparse-face, phi, divergence, pressure, and projection views remain the
primary octree/fine-band audits. Five power-specific views extend that base:

- **Power cells** distinguishes regular Cartesian configurations from
  catalog-resolved transition configurations and marks the pressure site at
  each leaf centre.
- **Power faces** draws the generalized face plane through its published
  centroid, its stored normal, and the dual link between incident pressure
  sites; boundary faces and invalid publication are explicit.
- **Delaunay tetrahedra** reconstructs the exact Section 5/6.2 catalog-local
  tetrahedra for non-uniform rows using the live row transform and selector
  buffers. Regular rows intentionally remain clear.
- **Transitions** identifies the boundary and resolution-transition rows that
  require the paper's special interpolation and localized second-order work.
- **Operator** colors rows by their strongest live `area * inverseDistance *
  openFraction` incident coefficient and fails red when the reciprocal face
  graph is unpublished.

These are presentation-only fragment passes. They use owner materialization,
compact leaves, topology metrics, tetrahedron tables, generalized faces,
centroids/normals, and CSR incidence directly. They perform no topology
readback, create no dense diagnostic mesh, and never participate in authority
or scheduling. The configuration is URL-shareable through `grid`,
`gridSlice`, and `gridMode`.

The async solver is now attachable only after the complete initial sparse
authority command stream has passed its queue fence: cold owner/topology
publication, compact rows and power graph, coarse/fine phi and surface pages,
the Section 5 regular-face/transition-tetra interpolation band, then sparse
residency. Renderer attachment uses this explicit
`initialSparseAuthorityReady` invariant, not `encodedSteps > 0`, so the paused
scene at `t=0` binds the same compact face/page fallback and technique buffers
that the first step will consume. A zero initial velocity is therefore a
published value, not a missing-source fallback. Play, Step, recording, and
timestep mutation remain locked during WebGPU startup/rebuild, and the
controller independently refuses to accumulate or advance GPU target time
until that ready state is published.

### Full-volume diagnostics and failure localization (2026-07-20)

Every paper-technique view can now switch between an exact movable slice and a
camera-ray traversal of the complete simulation volume. The volume path
front-to-back composites compact leaves, power faces, catalog tetrahedra, and
transaction state directly; it does not construct a stack of slices or a CPU
debug mesh. A UI opacity control scales the compositing contribution without
changing simulation data.

The second diagnostic tranche adds seven focused views:

- **Octree lifecycle** expands the live active/retired topology-tile lists into
  a presentation-only GPU membership buffer and distinguishes unchanged,
  active, retired, and invalid tiles across the full domain.
- **Fine-band lifecycle** hashes camera samples through the current sparse fine
  generation and distinguishes interface, fast-march frontier/trial, known
  support, newly activated/unresolved support, coarse fallback, stale data, and
  transaction failure.
- **Operator diagonal** and **operator RHS** expose the assembled compact row
  values, including non-positive or non-finite failures and signed forcing.
- **Operator reciprocity** validates endpoint/sign agreement and reverse CSR
  incidence for each generalized power face.
- **Operator open fraction** shows the area-weighted incident-face aperture and
  fails closed on malformed or out-of-range face data.
- **Tetra validity** audits catalog selectors, degeneracy/handedness, and the
  tetrahedral volume reconstructed from each local topology entry.

Modes 12–23 share the same `grid`, `gridSlice`, and `gridMode` URL contract;
`grid=volume` selects the ray traversal and `gridSlice` becomes the diagnostic
opacity. All membership expansion, hashing, traversal, and validation remain
GPU-resident and observational.

### Important divergences from the paper and original plan

1. **GPU paged storage replaces CPU virtual-memory SPGrid.** The logical model
   remains the paper's background octree plus one uniform fine narrow band,
   but allocation, generation validation, and publication use WebGPU buffers,
   hashes, free lists, and indirect worklists. There are no CPU page-table
   decisions or simulation readbacks.
2. **Pressure migration is staged.** The power operator feeds both compact-row
   Chebyshev and experimental matrix-free PCG. The PCG preconditioner currently
   adds independent geometric-aggregate diagonal corrections to fine Jacobi;
   it is not a first-order multigrid V-cycle and has no paired second-order
   boundary/transition smoothing. Section 4.3 therefore remains outstanding,
   and Chebyshev remains the explicit A/B rollback.
3. **The fine grid stores phi only.** Velocity stays on octree power faces and
   is reconstructed/interpolated on demand, following Section 5. Pressure is
   never allocated on the fine grid.
4. **Fresh SPGrid construction is approximated by A/B page generations.** A
   next generation is initially built from interface blocks plus exactly one
   topology-safety brick ring. Factor-4 and factor-8 B4 redistance then perform exact causal
   fast-marching activation to complete the configured physical band before
   validation and atomic publication; failed generations retain the last good
   source and old-only pages have no target-generation authority. The former
   single-invocation binary heap was too slow for production scale and has been
   replaced by a bounded parallel half-cell bucket schedule. The replacement
   is statically validated but still awaits the production one-step timing and
   numerical telemetry gate. Factor-8 source, mapping, and static shader tests
   pass; its full physical Dawn/endurance evidence is still outstanding.
   Transport-core samples are updated while skipped support
   samples must be preserved explicitly across the swap.
5. **Dry-side face-band interpolation is an incomplete Cartesian bridge.**
   Fine bricks bound the work frontier, while rows now resolve through the
   live owner topology and published compact coarse phi. Uniform size-one
   owners use regular axis-neighbor faces; non-uniform owners fail `BAD_ROW`
   until the paper's local Delaunay-tetrahedron connectivity is wired. The
   final regular-face-to-power-face transfer is also outstanding. Synthetic
   guard phi/connectivity has been removed and must not be reintroduced.
6. **Wall extension is specialized.** The current rectangular Neumann ghost
   clamp is an engineering boundary extension, not Section 5's explicit domain
   geometry channel. General terrain/body boundaries remain outside this
   surface-fidelity claim.
7. **Analytic cold start is specialized.** Dam/tank bounds are derived from the
   authored SDF and emitted on GPU. Imported shapes, terrain, and bodies remain
   on the dense compatibility bootstrap until a bounded sparse voxelizer can
   provide equivalent sign and solid coverage.
8. **Factor 4 and factor 8 are explicit quality/memory modes.** Factor 8 is not
   silently narrowed to fit a device. It must pass its stated capacity gate or
   fail closed.
9. **Tall-cell regular is a qualitative reference, not a golden image.** Its
   dam-break motion, mass trend, and visible surface are indicative; the power
   implementation is not expected to reproduce its artifacts exactly.

### Key implementation learnings

- Bootstrap, recurring topology, and rendering publication are three distinct
  authority transitions. Making `t=0` sparse does not imply that later frames
  remain sparse, and compiling a fine renderer does not prove that it receives
  a published fine generation.
- An empty GPU worklist is ambiguous. It may mean valid-empty liquid, a producer
  that has not run, overflow, or a failed generation. Publication therefore
  needs an explicit generation/validity/error contract; the host must not infer
  authority from a count or read it back to choose the next simulation path.
- Missing fine pages are expected outside the narrow band. They must fall back
  to validated compact coarse phi, never implicit zero and never an automatic
  dry classification before the coarse generation proves that sign.
- Narrowband surface residency and global pressure topology have different
  extents. The paper's separate coarse octree phi/frontier is what preserves
  deep-liquid sign and pressure rows; widening every fine or topology worklist
  through the whole wet volume defeats the intended scaling.
- Capacity bytes and active bytes must be reported separately. The factor-8
  checkpoint exposed large payload fragmentation even when the active page
  count was valid; a single total obscures whether memory is useful residency,
  page-table overhead, or safety headroom.
- WebGPU portability constraints shape architecture. Splitting stages fixed the
  ten-storage-binding limit, and storage/indirect aliasing must be avoided even
  when a backend happens to accept it. Indirect arguments should be copied to
  dedicated buffers before dispatch when their producer buffer is storage-bound.
- Shader compilation, Naga validation, static source contracts, and even a
  submitted command buffer are not execution proof. The poisoned Metal state
  accepted all of those while returning zeros. Known-value GPU sentinels and
  physical invariants are required at the start of every runtime investigation.
- Device loss can poison WebGPU beyond a browser restart on the current machine.
  After a repeated zero-compute/device-loss sequence, stop issuing GPU work,
  preserve the static checkpoint, and restart the computer before collecting
  more evidence.
- Browser startup is now fail-safe. `?gpu=off` is the UI-only inspection mode
  and returns before `navigator.gpu.requestAdapter()`. `?gpu=manual` waits for
  the explicit **START WEBGPU** action, while `?gpu=on` restores automatic
  startup. With no override, the default `water-box-dam-break` octree scene is
  manual; other scenes retain automatic startup. Device recreation after a
  loss is disabled unless the diagnostic `?gpuRecovery=1` opt-in is present,
  so a deterministic Metal fault is not immediately submitted three times.
  Browser timestamp queries are likewise disabled by default and require the
  diagnostic `?gpuTimestamps=1` opt-in; correctness bring-up therefore matches
  the accepted Dawn path and reports GPU timing as unavailable rather than
  attaching timestamp writes to physics or presentation passes.

### Late audit findings: do not claim final cutover yet

1. **Recurring direct-paged topology still falls back to full-domain work.**
   The bounded analytic worklist is consumed at `t=0`, but
   `encodeSparseBrickWorld()` currently clears `topologyWorklistReady` for the
   direct-paged path. Subsequent rebuilds therefore take the volume-wide
   reset/refine/balance/frontier schedule. The next implementation must make
   candidate publication GPU-transactional and retain the last good bounded
   worklist on an invalid or unpublished generation.
2. **Analytic owner-page bootstrap does not yet prove max-leaf 16/32.** The
   analytic publisher seeds topology tiles and tile states, while owner-page
   activation consumes the brick worklist. With no cold owner page, lookup
   falls back to a canonical size capped at 8. A real GPU leaf census must
   demonstrate genuine deep size-16/32 leaves; otherwise add bounded coarse
   owner/page seeding or a hierarchical coarse-owner fallback.
3. **Narrowband candidates are not global liquid authority.** This is correct
   per the paper. Deep-liquid pressure rows are intended to survive the one-time
   cold-to-narrowband retirement through `appendFrontierRetired` and the
   persistent frontier, while compact coarse phi supplies far-field sign. That
   invariant is statically plausible but has not passed a real GPU transition
   test. Do not "fix" it by rebuilding every wet interior tile each frame
   unless runtime evidence disproves frontier preservation; that would restore
   volume scaling.
4. **Owner-page retirement and grading support need an endurance check.** The
   topology tile list includes a grading dilation while owner lifecycle follows
   brick residency. Pages allocated on demand in grading-only support must be
   shown to retire without leaks or stale ownership.
5. **No physical acceptance result exists after the latest fixes.** The earlier
   Chrome zero-row/device-loss observations occurred before the fail-closed
   bootstrap and publication fixes and on a subsequently poisoned runtime.
   They are useful history, not a pass or a current failure reproduction.
6. **The paper-directed pressure preconditioner is not implemented yet.** The
   connected power path can execute experimental aggregate PCG or Chebyshev,
   but the aggregate hierarchy is not Section 4.3's first-order V-cycle. Final
   production authority requires the first-order sparse-grid operator and
   adjacent-level ghost transfers, paired `k ~= 8` second-order Jacobi sweeps in
   a roughly three-voxel boundary/transition band, and the width-scaling
   convergence gate below.

### Post-restart continuation order

1. Run a minimal Dawn compute sentinel. If it does not write a known nonzero
   word, stop: the machine/runtime is still unsuitable for evidence.
2. Run the new analytic-worklist Dawn test and the constrained owner-page,
   power, fine-level-set, and global-fine renderer tests before opening Chrome.
3. Implement and test GPU transactional surface-candidate publication:
   validate publication/generation/overflow before mutation; preserve the
   previous worklist and states on rejection; distinguish valid-empty from
   failed-empty; avoid binding one buffer as writable storage and indirect in
   the same usage scope; keep analytic topology readiness after `t=0`.
4. Add a max-leaf-16/32 analytic cold census. Fix bounded owner-page/coarse
   owner seeding if deep leaves remain capped at 8.
5. Add a cold → valid narrowband → next-rebuild Dawn test. Assert that a
   sampled deep-negative row remains in the persistent pressure frontier while
   recurring topology work is narrowband-bounded.
6. Add moving-interface/grading-halo owner-page endurance coverage, including
   allocation, retirement, generation, overflow, and stale-owner checks.
7. Only after the equivalent Dawn warmup and one-step gates pass, load the
   balanced dam-break power/factor-4 URL in Chrome with `gpu=manual` and no
   diagnostics panel/readback flags. Press **START WEBGPU**, wait for the
   paused `t=0` publication, then press **STEP** once before running
   continuously. Inspect nonzero compact rows and
   CSR, face/operator controls, adapter candidates, fine active bricks and
   generation, coarse publication, overflow flags, mass, and the fine-source
   raster. Only then run longer motion.
8. Compare motion and surface quality qualitatively with tall-cell regular.
   Require visible advancing water, stable finite pressure/velocity, plausible
   mass trend, and correct fine-level-set rendering; do not require pixel or
   artifact parity.
9. Once topology and fine-surface plumbing are physically observable, replace
   the additive aggregate preconditioner with the Section 4.3 hybrid: paired
   second-order boundary/transition smoothing around an SPD first-order sparse
   V-cycle with adjacent-level ghost transfers. Keep aggregate PCG and
   Chebyshev as diagnostic A/B paths and measure residual reduction versus
   domain width before changing default authority.
10. After the short gate and solver-convergence gate pass, run max-leaf 16 and
   32, factor 4 and factor 8, followed by the 300-frame
   page-generation/endurance gate and memory/timing capture.

The work is not complete until those real Dawn and browser gates pass after
restart. Static mirrors and shader compilation are necessary but are not
substitutes for the physical dam-break acceptance scenario.

## Critique outcomes and direction changes (2026-07-20)

An external review of this plan against the paper, the code, and the measured
vast-ocean profiles produced two direction changes and a list of findings.
Where anything below contradicts older text in this document, this section
wins; scattered references to a frozen Chebyshev iteration should be read as
"the production pressure iteration during migration."

### Direction change 1: adopt the paper's pressure solve

The "keep Chebyshev unchanged" constraint is withdrawn. Two review findings
motivated this:

- A fixed-pass Chebyshev budget propagates pressure information roughly one
  stencil cell per pass. On a domain thousands of leaves wide, global modes
  (impacts, basin-scale seiche/swell) cannot equilibrate in any fixed budget;
  the hydrostatic split rescues calm water but not active global modes.
- Chebyshev, unlike CG, diverges on a wrong spectral-interval estimate, and
  the power operator changes the spectrum (edge-neighbor faces, theta-clamped
  ghost-fluid diagonals). Freezing the iteration while changing the operator
  was the plan's largest unguarded quality risk.

New direction: implement the paper's Section 4.3 hybrid PCG preconditioner as
the production solve for power authority, and optimize later. The wired
additive aggregate-PCG scaffold does not satisfy this direction. Completion
requires, in order, `k` damped-Jacobi iterations of the second-order power
operator in a roughly three-voxel boundary/transition band, a whole-domain
first-order SPD multigrid V-cycle, and the matching `k` post-iterations (the
paper reports `k ~= 8`). What is preserved from the original constraint:

- the deterministic compact power-face/row assembly contract (faces remain
  the single source for divergence, coefficients, and projection);
- the existing Chebyshev path as the documented A/B rollback during
  migration, after which it may be removed;
- fail-closed behavior: solver non-convergence or preconditioner overflow
  suppresses authority for the generation, exactly like topology overflow.

A solver-scalability gate is now required regardless of solver: measured
residual reduction versus domain width on calm and active ocean scenes, so
the width at which convergence degrades is known before a scene finds it.

### Direction change 2: interest-driven fine band

The fine narrow band will not remain a uniform shell around the entire
interface. Future direction: band residency is granted by interest —
high local velocity, proximity to rigid bodies or terrain contact, high
surface curvature/churn — while calm surface regions are served by the
corrected coarse octree phi (and column heights under the hydrostatic
split). Sparsity of the band follows activity, not interface area alone.

Related defect, not merely a checkpoint: the measured factor-8 band cost
(Section 18.12: ~710 MB fine level set on a 60×45×40 dam break, driven by a
12-brick-ring support width) scales with interface area times band thickness
and is incompatible with vast scenes. Shrinking the support width (velocity-
bounded backtrace ring, narrower redistance band with clamped far values) is
a required work item with a bytes-per-interface-area metric, ahead of any
factor-8 quality claims.

### Review findings to action

1. **Resolved:** the 7.7 MB `lib/generated/octree-power-catalog.{ts,bin}` pair
   is an explicitly ignored deterministic build artifact. Dev, build, unit,
   WebGPU, targeted power/runtime, and shader-validation npm entry points
   generate it before loading application modules. CI runs
   `npm run verify:octree-power-catalog`, which generates once, independently
   rebuilds and byte-compares once, then checks the embedded format version,
   generator-source hash, binary SHA-256, and decoder contract. A clean
   checkout therefore has no dependency on an unpublished local binary.
2. The production row arena caps incidence at 24 entries while the catalog
   manifest proves a maximum of 30. Bump the arena and memory plan before any
   authority attempt, or fail-closed overflow will silently roll back forever.
3. Remove or justify the `sceneHasTerrain` gate on authoritative mode in
   `lib/webgpu-octree.ts`; as wired, the body-free ocean gate scene may be
   structurally unable to exercise authority.
4. Run the Section 7 grading audit before further GPU work: the layered ocean
   (fine surface band over 16³/32³ bulk) is the adversarial case for the
   paper's stronger grading rule, and the 15% leaf-growth budget is genuinely
   at risk. Pre-decide the fallback if it fails.
5. Reconcile doc and code: the Section 10 flag list is mostly unimplemented
   (only `powerDiagramProjection` exists), while the row-arena wiring the
   checkpoint lists as "remaining" already exists behind the authority gate.
6. Sequence explicitly against
   `docs/HYDROSTATIC_SPLIT_TWO_TIER_VELOCITY_PLAN.md`: both plans claim
   velocity authority in the same files. They compose, but exactly one
   migration owns velocity forces/transport at a time, and the order must be
   named. The split plan delivers depth-independent cost; this plan delivers
   the transition correctness that makes deep coarsening safe.
7. Mirror mode gets an expiry: define the minimal authority slice (body-free,
   isotropic, dam break) and cut over early — the fail-closed machinery is
   what makes an early cutover safe. Dual-path operation is a standing tax on
   memory budget and parity maintenance.
8. Silent under-convergence replaces T-junction error as the dominant quality
   failure mode; convergence telemetry (residual ratio) is a first-class
   scene-gate metric for power authority, not a diagnostic.

## 1. Purpose

This plan adapts the most useful techniques from Aanjaneya et al., *Power
Diagrams and Sparse Paged Grids for High Resolution Adaptive Liquids* (2017),
to the existing WebGPU implementation. It adopts the paper's logical separation
between an adaptive dynamics octree and a uniformly sampled fine level-set
narrow band, while implementing both with bounded GPU-resident sparse storage
rather than CPU virtual-memory SPGrid or a dense full-domain fine grid.

The work has four intended outcomes:

1. Correct the geometry of pressure projection at octree T-junctions.
2. Encode every supported 2:1 transition as bounded local topology rather than
   storing an explicit unstructured mesh.
3. Track the interface on a single-resolution sparse fine-grid address space,
   while reconstructing its velocity from the adaptive octree and correcting a
   coarse octree level set for topology, pressure, and far-field sign queries.
4. Preserve the current sparse memory architecture and avoid reintroducing any
   box-sized velocity, pressure, occupancy, or signed-distance representation.

The new power-diagram coefficients are assembled into the current compact
`LeafHeader`/`LeafEntry` row format. Per the 2026-07-20 direction change, the
iteration that consumes them migrates to the paper's Section 4.3 hybrid PCG
preconditioner; existing aggregate PCG and Chebyshev iterations are diagnostic
scaffolds during that migration.

## 2. Source material

Primary paper:

- `/Users/petersuggate/Downloads/high_resolution_liquids-a.pdf`
- Section 4.1: power-diagram pressure discretization.
- Section 4.2: sparse uniform-grid pyramid and adjacent-level transfers.
- Section 4.3: matrix-free multigrid preconditioning. Per the 2026-07-20
  direction change, this is now in scope as the initial production solve for
  power authority.
- Section 5: multisegment level-set backtracing, transition-aware velocity
  interpolation, fast marching, extrapolation, and dynamic topology.
- Section 6: compact topology encoding and lookup-table reconstruction.

Existing project architecture and cutover status:

- `docs/UNIFIED_OCTREE_SIMULATION.md`
- `lib/webgpu-octree.ts`
- `lib/webgpu-octree-face-mirror.ts`
- `lib/webgpu-octree-face-transfer.ts`
- `lib/webgpu-octree-face-transport.ts`
- `lib/webgpu-octree-surface-adapter.ts`
- `lib/webgpu-octree-surface-pages.ts`
- `lib/octree-consumer-sampling.ts`
- `lib/octree-face-fragments.ts`
- `lib/octree-compact-allocation.ts`

## 3. Non-negotiable constraints

### 3.1 No dense authority

The implementation must not add or retain any persistent resource whose size
is proportional to the complete finest-cell domain, including:

- dense 3D velocity textures;
- dense pressure textures or dense row maps;
- dense signed-distance or occupancy textures after bootstrap;
- a dense finest-cell owner table;
- a full-domain face table;
- a full-domain page table;
- renderer-only dense publications of otherwise sparse fields.

Short-lived bootstrap data already present in the code may remain until its
separate removal milestone, but no new stage may depend on it after the first
sparse generation has been published.

Every new persistent allocation must scale with one of:

- live leaf capacity;
- live power-face capacity;
- live surface-page capacity;
- a fixed topology catalog independent of scene dimensions;
- bounded per-level worklists derived from live sparse keys.

### 3.2 Pressure solve (revised 2026-07-20)

Superseded: the original "keep Chebyshev unchanged" constraint is withdrawn
(see "Critique outcomes and direction changes"). The revised contract is:

- the compact row/face assembly ABI remains the single operator source:
  `LeafHeader.diagonal`, `LeafHeader.rhs`, `LeafEntry { row, coefficient }`,
  and the power-face records feed whichever iteration runs;
- the production solve migrates to the paper's Section 4.3 hybrid PCG
  preconditioner, sized by live sparse rows (no dense or domain-volume-scaled
  levels): paired second-order Jacobi smoothing in the boundary/transition band
  around an SPD first-order sparse-grid V-cycle with adjacent-level ghost
  propagation and accumulation;
- the wired additive geometric-aggregate diagonal preconditioner is explicitly
  experimental and must not be used as evidence that the preceding item is
  complete;
- `iterateChebyshev` is retained unmodified as the A/B rollback during
  migration and may be removed after the solver gates pass;
- solver non-convergence, preconditioner overflow, or non-finite reductions
  fail closed for the generation, like every other authority failure;
- the solver-scalability gate (residual reduction versus domain width) is
  mandatory before authority on large scenes.

### 3.3 Maintain one velocity authority

Normal velocity continues to live once per canonical physical face. A power
face is allowed to be non-axis-aligned, but it must still have exactly one
normal-velocity degree of freedom shared by:

- advection;
- body forces;
- divergence assembly;
- pressure projection;
- surface advection sampling;
- rendering and diagnostics;
- topology transfer.

Never create permanent Cartesian `u`, `v`, and `w` duplicates of the power-face
field.

### 3.4 Preserve fail-closed behavior

All bounded catalogs, face stores, incidence slabs, row entries, page pools,
and worklists must expose overflow. Overflow must suppress authoritative
publication for that generation and select the existing supported fallback, or
fail the scene gate explicitly if no sparse fallback remains. Truncating a
stencil, silently dropping an edge-neighbor face, or drawing a partial surface
is prohibited.

### 3.5 Optimize for large scenes

Small-scene fixed overhead may grow. The acceptance target is the large,
deep-water workload where 16³ and 32³ leaves remove the most bulk state. Any
fixed topology catalog is acceptable only if its size is independent of domain
dimensions and its cache behavior is measured.

## 4. Current implementation baseline

The implementing agent should understand the current ABI before editing it.

### 4.1 Current pressure rows

`lib/webgpu-octree.ts` stores:

```wgsl
struct LeafHeader {
  cell: u32,
  entryStart: u32,
  entryCount: u32,
  size: u32,
  diagonal: f32,
  rhs: f32,
  pad0: u32,
  pad1: u32,
  gradient: vec4f,
}

struct LeafEntry {
  row: u32,
  coefficient: f32,
}
```

The row assembler currently samples the six Cartesian sides of a leaf at
finest-cell granularity. It merges repeated neighbor rows and computes a
coefficient approximately equal to `openArea / pressureDistance`. The compact
Chebyshev loop only gathers those cached entries.

### 4.2 Current canonical face record

`lib/webgpu-octree-face-mirror.ts`, `lib/webgpu-octree-face-transfer.ts`, and
`lib/webgpu-octree-face-transport.ts` share this effective record:

```wgsl
struct FaceRecord {
  negativeRow: u32,
  positiveRow: u32,
  packedOrigin: u32,
  axisSpan: u32,
  normalVelocity: f32,
  area: f32,
}
```

The current face is an axis-aligned square fragment. `axisSpan` contains an
axis and an integer span. Each row owns a fixed slab of at most 24 incident
faces: four fragments on each of six sides.

This representation is conservative, deterministic, and suitable for an
ordinary octree MAC discretization. It cannot represent the additional
non-axis-aligned faces that a 3D power diagram creates between some original
octree edge neighbors.

### 4.3 Current velocity reconstruction

Compact face transport and the surface adapter reconstruct Cartesian velocity
components using inverse-distance weighting of nearby axis-aligned face
velocities. The search is bounded by the two incident rows' fixed incidence
slabs.

This works as a migration sampler, but it does not enforce a least-squares
normal fit and has no tetrahedral transition interpolant.

### 4.4 Current surface transport

The authoritative surface uses leaf-attached 2³ or 4³ signed-distance pages.
`SurfaceLeaf.motion` stores one reconstructed velocity at the leaf center.
Every page sample currently backtraces using that constant leaf motion and then
performs one signed-distance lookup. Redistance uses a fixed number of local
sweeps.

### 4.5 Immediate page-resolution inconsistency

Before beginning power-diagram work, resolve the current 2³/4³ ABI mismatch.
The page planner defaults to 2³, but the following consumers currently contain
4³-only availability tests or indexing:

- `lib/webgpu-octree-face-mirror.ts`: `surfaceParams.shape.z == 4u`;
- `lib/webgpu-octree-surface-adapter.ts`: `pageParams.shape.z == 4u`;
- `lib/webgpu-grid-overlay.ts`: `octreeSurfaceParams.shape.z == 4u` and fixed
  4³ center sampling;
- `lib/octree-consumer-sampling.ts`: `array<f32,64>`, fixed `4.0`, fixed
  `vec3u(3u)`, and fixed 4³ indexing.

At page resolution 2, a consumer can therefore reject the authoritative page
arena and fall back to a retired or 1³ placeholder dense texture without a
WebGPU validation error. Treat this as a correctness blocker, not a cosmetic
cleanup.

## 5. Target architecture

The final dataflow for the supported body-free path should be:

```text
sparse owner pages + live leaf frontier
                 |
                 v
       per-leaf topology descriptor
                 |
                 v
   fixed canonical power-topology catalog
                 |
                 v
 deterministic compact power faces + incidence
          |                         |
          |                         +--> compact row assembly
          |                               |
          |                               v
          |                      existing Chebyshev solve
          |                               |
          +-------------------------------+
                          |
                          v
                 projected face normals
                          |
            +-------------+-------------+
            |                           |
            v                           v
 transition-aware velocity      sparse surface pages
 reconstruction                 transport/redistance
            |                           |
            +-------------+-------------+
                          |
                          v
             direct adaptive rendering/consumers
```

The topology catalog describes geometry; it is not a spatial simulation field.
It is shared by pressure assembly, face transport, velocity reconstruction,
surface transport, redistance, and diagnostics.

## 6. Mathematical contract

### 6.1 Weighted sites

For an octree leaf with center `c` and side length `s` in finest-cell units,
associate a weighted site with radius:

```text
r = s / sqrt(3)
```

in 3D. A point `x` belongs to the site with minimum power distance:

```text
power(x; c, r) = |x - c|² - r².
```

The bisector between sites `i` and `j` is the plane:

```text
2 (c_j - c_i) dot x
  = |c_j|² - r_j² - |c_i|² + r_i².
```

Uniform same-size neighborhoods must reduce exactly to the ordinary Cartesian
grid.

### 6.2 Face orientation

Every internal power face has a deterministic orientation from its negative
incident row to its positive incident row. Store one scalar:

```text
u_f = velocity dot n_f
```

where `n_f` is the unit normal in that orientation.

For each face, the catalog or face record must provide:

- face area `A_f`;
- face centroid `x_f`;
- oriented unit normal `n_f`;
- dual distance `d_f` between pressure sites, or its inverse;
- the two incident leaf/site identifiers;
- a stable local face signature;
- optional open fraction for solid support;
- optional free-surface distance adjustment.

### 6.3 Integrated divergence and gradient

For row `i`, use the integrated flux:

```text
rhs_i = sum_f s(i,f) A_f u_f - boundaryPressure_i
```

where `s(i,f)` is `+1` when the face normal points outward from row `i` and
`-1` otherwise, following one documented convention consistently.

For an internal liquid/liquid face, use one shared coefficient:

```text
w_f = openFraction_f A_f / d_f.
```

Add `w_f` to both incident diagonals and add the same neighbor coefficient to
both rows. The exact sign representation must match the current Chebyshev row
equation. Do not independently recompute the two sides.

Projection updates the same face:

```text
u_f <- u_f - pressureScale (p_positive - p_negative) / d_f.
```

Use the project's existing pressure scaling and units. Confirm the formula on
a uniform grid before changing any timestep or density factor.

### 6.4 Free surface

For liquid/air adjacency, keep the existing ghost-fluid zero-crossing idea but
apply it along the power dual edge rather than a Cartesian axis. Sample phi at
the two pressure sites, compute bounded `theta`, and use:

```text
d_boundary = theta |c_air - c_liquid|.
```

The pressure boundary remains `p = 0`, or the current hydrostatic boundary
value when that mode is supported. The face area and normal still come from
the power geometry.

### 6.5 Cell volume

The integrated pressure equation does not require division by cell volume, but
post-projection divergence diagnostics, volume-normalized source terms, and
some future couplings do. Store or reconstruct the power-cell volume per row.

The sum of power-cell volumes over any closed test patch must equal the patch
volume within the stated floating-point tolerance.

### 6.6 Physical cell isotropy

The paper's construction assumes cubic cells. Before enabling power authority,
verify that the finest physical spacings in `params.cellRelax.xyz` are equal
within a small relative tolerance.

Initial policy:

- enable power geometry when `max(h) / min(h) <= 1 + 1e-5`;
- otherwise fail closed to the current axis-aligned operator;
- do not silently build the diagram in index space and claim physical
  orthogonality for anisotropic cells.

A metric-aware anisotropic extension is separate future work.

## 7. Topology and grading contract

### 7.1 Paper grading is stronger than ordinary 2:1 balance

The paper encodes a cell under one of two cases:

1. all relevant neighbors are the same size or one level finer; or
2. all relevant neighbors are the same size or one level coarser.

Ordinary pairwise 2:1 balance may still allow one leaf to have a finer neighbor
on one side and a coarser neighbor on another. Do not assume the current tree
already satisfies the paper's stronger local restriction.

### 7.2 Mandatory grading audit

Add a read-only topology audit that reports:

- live leaf count;
- leaves with same/finer neighborhoods;
- leaves with same/coarser neighborhoods;
- leaves with mixed finer and coarser neighbors;
- maximum face-neighbor level difference;
- maximum edge-neighbor level difference;
- counts by leaf size;
- counts by scene and time step.

Run it on:

- balanced dam break with maximum leaf 16;
- balanced dam break with maximum leaf 32;
- ocean seiche at `320x96x80`, maximum leaf 32;
- at least one topology-changing frame after motion begins.

### 7.3 Decision gate for mixed neighborhoods

Preferred approach: enforce the paper-compatible local grading invariant by
refining a coarser neighbor whenever a leaf simultaneously observes a finer
neighbor. Apply refinement until stable, then re-run ordinary 2:1 balancing.

Accept stronger grading only if:

- leaf count rises by no more than 15% on the ocean gate;
- surface-page count rises by no more than 10%;
- no new capacity overflow occurs;
- topology construction time rises by no more than 20%.

If those limits are exceeded, stop and extend the topology generator to mixed
2:1 configurations. Before doing so, prove with exhaustive local enumeration
that face/edge neighbors still determine every clipped power cell; otherwise
expand the descriptor radius deliberately. Do not guess.

Document the chosen route and delete the unused experimental route.

## 8. Proposed runtime data ABI

The exact packing may change after catalog generation, but the semantic fields
must remain explicit.

### 8.1 Per-row power metadata

Add a compact buffer parallel to live pressure rows:

```wgsl
struct PowerRowMetric {
  topologyCode: u32,
  transformAndFlags: u32,
  volume: f32,
  reserved: u32,
}
```

`topologyCode` identifies a canonical catalog entry. `transformAndFlags`
encodes rotation/reflection, grading case, boundary flags, and authority state.
Do not reuse `LeafHeader.pad0/pad1`; Chebyshev owns those words during the hot
loop.

### 8.2 Generalized face record

Replace the axis-only semantic contract with a power-face contract. A suggested
32-byte record is:

```wgsl
struct PowerFaceRecord {
  negativeRow: u32,
  positiveRow: u32,
  geometryCode: u32,
  flags: u32,
  normalVelocity: f32,
  area: f32,
  inverseDistance: f32,
  openFraction: f32,
}
```

The catalog plus the anchor row reconstructs centroid and normal.
`geometryCode` should contain a catalog-local face slot and the canonical
transform. Boundary faces must have a documented anchor rule when one incident
row is invalid.

Do not store a full centroid and normal on every face unless profiling proves
catalog reconstruction is slower enough to justify the extra bandwidth.

### 8.3 Stable transfer key

The current `(packedOrigin, axisSpan)` key is insufficient for diagonal power
faces. During topology transfer, construct a temporary stable key from:

- ordered negative and positive leaf site keys;
- face kind or catalog-local signature;
- boundary identity when one side is invalid.

A site key consists of packed leaf origin plus size exponent. If a 64-bit key
is insufficient, use a fixed 128-bit radix-sort record in temporary transfer
storage. Do not keep that key in the persistent face record unless measurement
shows it is cheaper than reconstructing it during rebuild.

### 8.4 Incidence and row-entry bounds

The current bound of 24 is not automatically valid after adding edge-neighbor
power faces. Derive exact constants by exhaustive enumeration:

- maximum physical faces incident to a row;
- maximum distinct liquid neighbor rows;
- maximum boundary faces;
- maximum faces gathered by a two-row transport query;
- maximum local Delaunay tetrahedra needed by interpolation/redistance.

Use the derived constants everywhere. Until enumeration exists, a provisional
48-entry incidence slab may be used only in a diagnostic branch. It must not
become production authority without a proof and overflow test.

### 8.5 Catalog buffers

Generate compact immutable buffers containing:

- descriptor-to-catalog lookup;
- per-topology face range;
- per-face neighbor selector;
- normalized area;
- normalized centroid;
- normalized normal or a compact normal selector;
- normalized dual distance;
- normalized cell volume;
- optional tetrahedron range and byte-sized relative vertex selectors;
- optional redistance update stencil.

Canonicalize configurations under cube rotations and safe reflections. Store
the inverse transform needed to reconstruct world orientation.

Initial catalog budget:

- target: at most 8 MiB total GPU storage;
- warning gate: more than 16 MiB;
- stop gate: more than 32 MiB without a measured end-to-end benefit.

Do not copy the paper's uncompressed approximately 128 MiB Delaunay table into
the browser build.

## 9. Work packages

Each work package should land independently with tests and a fail-closed
feature flag. Do not combine the complete migration into one unreviewable
change.

### WP0 — Repair and freeze the sparse surface ABI

Objective: make the existing unified representation trustworthy before using
it as the geometry source for new operators.

Tasks:

1. Replace every 4³-only availability test with validation that accepts exactly
   page resolution 2 or 4.
2. Replace fixed 64-sample function parameters with buffer-backed dynamic page
   sampling, or provide explicit specialized 2³ and 4³ functions selected at
   pipeline creation.
3. Remove fixed `4.0`, `3u`, and `x + 4(y + 4z)` indexing from shared consumer
   code.
4. Make renderer, face mirror, projection, surface adapter, grid overlay, SVO
   staging, diagnostics, and tests consume the same `pageResolution` and
   `samplesPerPage` fields.
5. Verify that no resolution mismatch falls back to the retired dense texture.
6. Add a single helper that validates page parameters and use it at every host
   binding boundary.
7. Add opt-in water diagnostics that can read back active pages, overflow,
   active cubes, and emitted vertices without affecting normal frames.
8. Classify live coarse interface leaves that have no resident detail page.
   The active-page dispatch alone is insufficient: a valid sparse hierarchy
   can represent phi on a coarse leaf only through its affine fallback plane.
   Extract one scale-aware coarse cube from that leaf, while resident pages
   continue through the fine page dispatch. Do not require `leaf.size == 1` for
   all visible surface geometry.

Required tests:

- identical analytic plane sampling through 2³ and 4³ pages;
- real Dawn classify/polygonize test at page resolution 2;
- the same Dawn test at page resolution 4;
- page-resolution-2 coarse live/core leaf with no resident page emits a finite
  triangle from its affine phi fallback;
- production balanced/max-leaf-16 lifecycle, not only a hand-built size-1 leaf;
- dense bootstrap destroyed before a second successful surface update;
- zero page overflow and nonzero triangle count;
- renderer screenshot containing visible water.

Exit criterion: the default 2³ configuration renders water and every adaptive
consumer remains page-native after bootstrap.

### WP1 — Build the CPU power-geometry oracle

Suggested new files:

- `lib/octree-power-geometry.ts`
- `lib/octree-power-topology.ts`
- `tests/octree-power-geometry.test.ts`
- `tests/octree-power-topology.test.ts`

Tasks:

1. Represent a local leaf/site by relative integer origin, dyadic size, center,
   and squared weight.
2. Construct bisector halfspaces using the power-distance equation.
3. Clip a convex polyhedron for the anchor site using deterministic Float64
   arithmetic and stable plane ordering.
4. Extract face polygons with ordered vertices.
5. Compute face area, centroid, outward normal, and incident site.
6. Compute cell volume and centroid using oriented tetrahedral decomposition.
7. Merge coplanar numerical fragments using scale-relative tolerances.
8. Match shared faces generated from both incident cells and reject asymmetric
   geometry.
9. Include domain boundary clipping as separate tagged planes.
10. Produce an ordinary uniform cube exactly, within a strict tolerance.

Robustness rules:

- sort all sites and planes by stable integer keys;
- make epsilon proportional to anchor size;
- reject negative areas, inverted volumes, NaNs, and non-manifold edges;
- never silently repair a face by dropping vertices;
- emit a human-readable local configuration dump on failure.

Oracle tests:

- uniform cell: six faces, axis normals, unit normalized volume;
- coarse/fine face transition;
- edge-neighbor face creation in 3D;
- all rotations and reflections of representative cases;
- shared-face area/centroid agreement;
- opposite normals for both incident cells;
- local patch volumes sum to the enclosing patch volume;
- no cell is empty or unbounded in an interior patch;
- deterministic byte-identical output across repeated runs.

### WP2 — Enumerate reachable topology and generate the catalog

Suggested files:

- `tools/generate-octree-power-catalog.ts`
- `lib/generated/octree-power-catalog.ts`
- `lib/octree-power-catalog.ts`
- `tests/octree-power-catalog.test.ts`

Tasks:

1. Enumerate only neighborhoods reachable from complete dyadic leaf tilings.
2. Apply the selected grading rule from Section 7.
3. Encode face and edge neighbor state into a compact descriptor.
4. Canonicalize under cube symmetries.
5. Run the CPU oracle once per canonical configuration.
6. Assign deterministic catalog indices.
7. Quantize only after comparing reconstruction error against Float32 storage.
8. Generate runtime arrays and descriptor lookup tables.
9. Emit a manifest containing version, generator hash, configuration count,
   maximum face incidence, maximum neighbor rows, maximum tetrahedra, byte
   count, and worst geometry error.
10. Make generation reproducible and checked by CI; ordinary builds must not
    regenerate the catalog.

Catalog validation:

- every reachable descriptor resolves;
- unreachable descriptors fail closed;
- every face has a reciprocal incident face;
- normalized area and distance are positive;
- volume is positive;
- uniform entries reproduce the existing operator;
- transformed entries match direct oracle geometry;
- catalog byte count obeys the budget.

### WP3 — Generate topology descriptors on the GPU

Suggested files:

- `lib/webgpu-octree-power-topology.ts`
- `tests/webgpu-octree-power-topology.test.ts`

Tasks:

1. Allocate `PowerRowMetric` proportional to pressure row capacity.
2. For each live row, query the bounded face/edge neighborhood from sparse
   owner pages and the frontier hash.
3. Encode the descriptor and canonical transform.
4. Resolve the catalog index using an immutable GPU lookup buffer.
5. Store reconstructed normalized volume.
6. Record invalid descriptor, mixed-grading, anisotropic-cell, lookup miss, and
   capacity errors in a compact control block.
7. Publish a deterministic indirect dispatch for later power-face construction.
8. Avoid a CPU topology readback in the normal path.

Tests:

- CPU/GPU descriptor parity for every catalog entry;
- production topology parity on dam and ocean scenes;
- deterministic output over repeated rebuilds;
- malformed owner page fails closed;
- mixed topology follows the selected grading policy;
- no descriptor lookup probes beyond its declared bound.

### WP4 — Construct deterministic compact power faces

Primary files:

- `lib/webgpu-octree-face-mirror.ts`
- `lib/webgpu-octree-face-transfer.ts`
- `lib/webgpu-octree-face-transport.ts`
- `lib/octree-face-fragments.ts`
- `lib/octree-compact-allocation.ts`

Prefer renaming the implementation to `powerFaces` only after the migration is
complete; avoid a disruptive rename during the dual-path phase.

Tasks:

1. Extend the CPU face oracle with general power faces.
2. Count faces per live row using catalog face ranges.
3. Establish canonical ownership so each physical face is emitted exactly once.
4. Include the bounded edge-neighbor faces introduced by the 3D power diagram.
5. Prefix-scan counts and emit stable face IDs.
6. Append equal/opposite row incidence.
7. Reconstruct physical area, centroid, normal, distance, and volume by scaling
   and transforming catalog data.
8. Initialize face velocity by least-squares reconstruction from the previous
   generation or, on first generation, from the bootstrap Cartesian velocity.
9. Replace `(packedOrigin, axisSpan)` transfer keys with stable site-pair/local-
   face keys.
10. Update memory planning and all diagnostics for the derived incidence bound.

Determinism requirements:

- stable row order;
- stable catalog face order;
- stable ownership rule;
- stable prefix scan;
- stable transfer sort;
- no atomic append order in the public face sequence.

Tests:

- unique physical faces;
- reciprocal incidence;
- equal/opposite signed flux;
- no catalog face omitted;
- exact uniform-grid compatibility;
- bounded maximum incidence proven by enumeration and observed at runtime;
- exact transfer for unchanged topology;
- conservative restriction/prolongation under refinement changes;
- finite normals, areas, centroids, distances, and velocities;
- no overflow on 16³/32³ target scenes.

### WP5 — Assemble power coefficients into the existing pressure rows

Primary file: `lib/webgpu-octree.ts`.

The authoritative order should become:

```text
emit live leaves
-> build topology descriptors
-> build power faces/incidence
-> assemble LeafHeader/LeafEntry rows from those faces
-> run existing Chebyshev passes
-> project the same faces
```

Tasks:

1. Add a row-assembly kernel that walks each row's bounded power-face incidence.
2. Compute RHS from signed `area * normalVelocity` using deterministic face ID
   ordering.
3. Compute `w_f = openFraction * area * inverseDistance` once per face.
4. Add the identical `w_f` to the diagonal of both incident liquid rows.
5. Merge duplicate neighbor rows deterministically if catalog decomposition can
   produce more than one face between a row pair.
6. Treat liquid/air faces using the power dual-edge phi crossing.
7. Store the result in the existing `LeafHeader` and `LeafEntry` buffers.
8. Leave `iterateChebyshev` byte-for-byte unchanged if possible.
9. Project normal velocity using the same face's `inverseDistance`.
10. Recompute post-projection divergence from the same incidence and area.
11. Divide diagnostic divergence by catalog power-cell volume where a
    volume-normalized value is reported.
12. Keep the ordinary axis operator behind an A/B feature flag until all gates
    pass.

Do not keep two independently assembled authoritative operators. During A/B,
the old operator may exist only for comparison and must be excluded from final
supported-path allocations.

Pressure tests:

- matrix symmetry: `A_ij == A_ji` within Float32 tolerance;
- positive energy: `x^T A x >= -epsilon` for randomized vectors;
- null-space behavior for a closed all-liquid domain;
- uniform-grid row equality with the existing implementation;
- manufactured linear pressure gradient;
- manufactured quadratic pressure convergence study;
- hydrostatic equilibrium across a coarse/fine transition;
- projection reduces volume-normalized divergence;
- projection does not create non-finite or excessive face velocity;
- face-derived RHS exactly matches row RHS;
- fixed Chebyshev pass count remains unchanged.

### WP6 — Generalize velocity reconstruction

Primary files:

- `lib/webgpu-octree-face-transport.ts`
- `lib/webgpu-octree-surface-adapter.ts`
- `lib/octree-consumer-sampling.ts`

Stage A: least-squares cell velocity.

For each cell center, reconstruct a vector `v` from incident face normals by
minimizing:

```text
sum_f weight_f (n_f dot v - u_f)².
```

Build the symmetric 3x3 normal matrix and right-hand side from bounded
incidence, then solve with a guarded analytic inverse or Cholesky factorization.
Use face area as the base weight and apply a condition-number guard. In a
uniform Cartesian neighborhood this must reduce to ordinary component
averaging.

Stage B: point interpolation.

- use trilinear interpolation in catalog entries marked uniform;
- use catalog Delaunay tetrahedra near transitions;
- interpolate reconstructed cell-center vectors barycentrically;
- project the result onto a target face normal only when a scalar face value is
  required.

Tasks:

1. Replace `faceAxis`/per-axis filtering assumptions.
2. Add catalog reconstruction for face center and normal.
3. Implement least-squares cell velocity.
4. Implement tetrahedron lookup and containment with deterministic boundary
   tie-breaking.
5. Fall back to the nearest valid reconstructed cell velocity only when the
   catalog explicitly lacks a containing tetrahedron, and count that fallback.
6. Keep all searches within the proven 2:1 local bound.
7. Share this sampler among face transport, surface transport, spray, renderer,
   and diagnostics.

Tests:

- exact reconstruction of constant velocity;
- exact reconstruction of linear velocity within expected discretization
  tolerance;
- uniform-path parity;
- continuity across representative T-junctions;
- bounded condition number/fallback count;
- no axis assumptions in general-face paths;
- CFL reduction uses physical displacement and smallest local support scale.

### WP7 — Multisegment sparse surface backtracing

Primary file: `lib/webgpu-octree-surface-pages.ts`.

Borrow both the paper's trajectory integration and its logically independent
fine level-set grid. The grid is a sparse, uniform-resolution narrow band, not
a dense texture and not a second velocity/pressure simulation.

For a page sample at `x_0`, choose a local segment count:

```text
m = clamp(ceil(|u| dt / h_page), 1, maximumSegments)
```

and trace backward with `dt / m`, resampling adaptive velocity after each
segment. Sample phi once at the final departure point and redistance after the
complete page transport.

Tasks:

1. Bind the canonical power-face source and shared velocity sampler directly to
   the surface transport stage.
2. Stop using one constant `SurfaceLeaf.motion` for every sample in a leaf.
3. Retain `SurfaceLeaf.motion` only as a conservative halo-sizing estimate or
   remove it if a face-derived bound replaces it.
4. Compute `m` from the configured fine-to-octree resolution ratio (4 or 8),
   then take `m` equal substeps over the octree timestep as the paper does.
   An additional CFL-derived increase may be used when required for safety,
   subject to the configured maximum.
5. Use midpoint or second-order Runge-Kutta backtracing if it does not double
   topology lookups excessively; otherwise use the paper's piecewise forward-
   Euler backtrace first and benchmark.
6. Ensure the active halo contains the complete swept trajectory plus
   interpolation/redistance support.
7. Count samples whose departure leaves the resident signed band.
8. Fail the page-authority gate if that count is nonzero rather than sampling a
   dense fallback.

Tests:

- translating plane;
- rotating slotted sphere/circle analogue;
- deformation field comparable to the paper's Figure 7 setup;
- page-boundary crossing;
- coarse/fine transition crossing;
- factor-4 and factor-8 fine-grid parity at equivalent physical resolution;
- volume drift versus the current one-segment method;
- visual retention of a thin sheet and crest.

### WP8 — Transition-aware redistance

Primary file: `lib/webgpu-octree-surface-pages.ts` plus catalog data.

Treat the two level-set discretizations separately:

- the fine SPGrid is uniformly sampled, so its fast-marching/redistance stencil
  is Cartesian even when the background octree changes resolution;
- the coarse octree level set requires local Delaunay tetrahedra near
  T-junctions, exactly as described in the paper.

Do not port a serial CPU priority queue. Use GPU-suitable sparse schedules for
both fields.

Tasks:

1. Fine grid: identify interface seeds in the transported uniform fine grid.
2. Fine grid: implement a bounded bucketed distance schedule or ordered sweeps
   over active fine-brick worklists.
3. Fine grid: carry distance information across physical page boundaries using
   cached six-neighbor page IDs or bounded hash fallback.
4. Fine grid: preserve the sign from the transported field and initialize newly
   activated fine pages from interface seeds, not only a leaf-affine fallback.
5. Coarse octree: copy/correct phi from valid fine samples after fine-grid
   advection and redistance.
6. Coarse octree: use Cartesian fast marching inside uniform octree regions and
   catalog Delaunay tetrahedra only near octree T-junctions.
7. Use sparse double-buffered phi/scratch channels; do not allocate a dense seed
   volume for either field.
8. Report unresolved samples, maximum Eikonal residual, and sweep/bucket count
   independently for fine and coarse fields.

Tests:

- signed distance to a plane and sphere;
- Eikonal residual `||grad phi| - 1|`;
- transported core updates while every skipped support sample is preserved
  across commit;
- newly activated support is initialized from transported interface seeds and
  reaches the configured physical band;
- continuity across fine-page boundaries;
- coarse-octree continuity across leaf-resolution transitions;
- reference fast-marching/ordered-sweeping agreement on translating and
  deforming surfaces;
- no sign inversion;
- deterministic result for a fixed schedule;
- bounded work independent of full-domain volume.

### WP9 — Dynamic topology transfer

When leaf topology changes, borrow the paper's trace-back principle while
retaining conservative normal flux.

Tasks:

1. For every new power face, trace its centroid backward through the current
   velocity field.
2. Interpolate the old full velocity at the departure point.
3. Project onto the new face normal.
4. For exact site-pair matches, preserve the existing exact transfer.
5. For one-to-many refinement, preserve area-weighted parent flux and add only
   a zero-net interpolated detail component.
6. For many-to-one coarsening, use area-weighted restriction.
7. For genuinely new connectivity, use trace-back interpolation.
8. Measure total boundary and internal flux before and after transfer.

Tests:

- unchanged topology is bitwise stable;
- refine/coarsen round trip preserves aggregate flux;
- rotating velocity field does not pop at a transition;
- new edge-neighbor power faces receive finite values;
- transfer never reads the retired dense velocity textures.

### WP10 — Solids, terrain, and boundary completion

Do this only after the body-free path is stable.

Tasks:

1. Compute solid aperture on general power-face polygons.
2. Compute moving-solid normal velocity at the power-face centroid.
3. Use `openFraction * area` consistently in divergence and row coefficients.
4. Apply the post-projection solid normal constraint on general normals.
5. Compute pressure reaction using the same oriented face geometry.
6. Port terrain boundaries to the same aperture contract.
7. Validate inflow/outlet semantics on general faces.
8. Remove the corresponding dense compatibility publications only after each
   consumer has passed its sparse gate.

Tests:

- static wall no-penetration;
- moving rigid displacement;
- equal/opposite pressure impulse;
- partial aperture symmetry;
- terrain contact without leakage;
- combined terrain/rigid scene;
- no dense field allocation on the newly supported path.

### WP11 — Consumer completion

Port every remaining consumer to the generalized face and dynamic surface-page
ABI:

- raster water extraction;
- optical SVO water queries;
- grid overlays;
- spray emission and motion;
- raw voxel inspection;
- pressure/divergence/projection diagnostics;
- picking;
- performance captures.

No consumer may infer a Cartesian axis from a general power face. Uniform
catalog entries may use an optimized axis path selected by an explicit flag.

## 10. Host and shader organization

Avoid copying topology math into many WGSL strings. Establish shared generated
WGSL fragments or one source module for:

- page-resolution validation and phi sampling;
- catalog decoding;
- transform decoding;
- face centroid/normal reconstruction;
- incidence iteration;
- least-squares velocity reconstruction;
- tetrahedron lookup and barycentric interpolation.

Every host-side buffer plan must have a TypeScript oracle with the same byte
layout. Add ABI round-trip tests that decode representative GPU records on the
CPU.

Suggested feature flags during migration:

- `powerDiagramProjection`: `off | mirror | authoritative`;
- `powerVelocityInterpolation`: `off | transition | all`;
- `surfaceTrajectorySegments`: integer bounded range;
- `powerRedistance`: `off | mirror | authoritative`.

Do not expose these as permanent user-facing complexity. Once the gates pass,
make the best path the default and retain only a diagnostic rollback switch.

## 11. Validation matrix

### 11.1 Pure CPU tests

- geometry clipping and catalog generation;
- symmetry transforms;
- face reciprocity;
- volume partition;
- incidence and neighbor bounds;
- stable transfer keys;
- memory planners;
- analytic interpolation and redistance oracles.

### 11.2 Shader compilation

Compile every new module through Dawn, including both page resolutions and all
feature-flag specializations. Treat warnings about unreachable code as cleanup
work, but distinguish them from validation errors and runtime failures.

### 11.3 Real Dawn execution

Required GPU tests must execute, not merely compile:

- topology descriptor generation;
- power-face count/scan/emit;
- row assembly;
- one complete fixed-budget Chebyshev projection;
- face transport;
- 2³ and 4³ page transport;
- water classify/polygonize and finite triangle readback;
- post-projection divergence reduction.

Avoid relying on `device.destroy()` as test correctness if the native Dawn
binding crashes during teardown. Isolate teardown instability from GPU work,
await submitted work, unmap buffers, and report native crashes separately.
Production validation still requires a clean browser run.

### 11.4 Browser integration

For balanced max-leaf-16 and max-leaf-32 scenes, collect:

- browser console errors and WebGPU uncaptured errors;
- shader compilation messages;
- surface active-page count;
- surface overflow code;
- active water cube count;
- water vertex count;
- face count and maximum incidence;
- pressure row/entry count;
- non-finite counters;
- screenshot before motion;
- screenshot after visible motion.

The favicon 404 and browser-extension connection messages are not simulation
failures. Do not use their absence as a GPU correctness gate.

### 11.5 Physical scene gates

Dam break:

- exactly the requested encoded steps;
- no face, row, entry, page, or worklist overflow;
- finite pressure and velocity;
- bounded CFL;
- post-projection divergence lower than pre-projection divergence;
- visible raster water;
- no topology-transfer flux spike.

Ocean seiche at `320x96x80`, maximum leaf 32:

- successful construction and at least three steps;
- nonzero 16³ and 32³ leaf counts;
- no overflow or validation error;
- visible water surface;
- stable free surface across coarse bulk;
- measured allocation and traffic results.

Manufactured projection gates:

- uniform parity;
- coarse/fine linear pressure;
- hydrostatic patch;
- randomized SPD energy;
- convergence under refinement.

## 12. Memory and bandwidth accounting

For every work package, update `allocatedBytes` and the compact allocation
planner. Report at least:

- per-row topology bytes;
- face record bytes;
- incidence bytes;
- row-entry bytes;
- fixed catalog bytes;
- transfer scratch bytes;
- surface-page bytes;
- redistance scratch bytes;
- total steady-state bytes;
- peak bootstrap bytes.

Traffic instrumentation should count or model:

- owner/hash probes during topology construction;
- catalog bytes read per face and per row;
- face/incidence reads during row assembly;
- `LeafEntry` reads per Chebyshev pass;
- velocity sampler face reads;
- surface sampler face and phi reads per trajectory segment;
- redistance reads/writes;
- renderer reads.

Pressure-iteration traffic claims must be measured against the completed
Section 4.3 hybrid preconditioner, not the current additive aggregate-PCG
scaffold and not assumed from the paper. Other likely gains come from:

- replacing repeated finest-face owner scans during row assembly with compact
  face/incidence streaming;
- sharing one geometry catalog among stages;
- retaining sparse surface and velocity authority;
- avoiding dense compatibility materialization.

Likely costs include:

- additional edge-neighbor faces;
- a larger incidence bound;
- per-row topology metadata;
- catalog lookups;
- extra surface velocity samples during multisegment backtracing.

Acceptance limits:

- no new domain-volume-scaled persistent allocation;
- total steady-state compact memory no more than 15% above the current unified
  path unless projection accuracy materially improves and the user approves;
- retain at least 65% memory reduction versus the documented dense ocean
  baseline;
- fixed catalog no more than 32 MiB and preferably no more than 8 MiB;
- no more than 20% construction-time regression after stronger grading;
- surface work proportional to active pages, never total domain cells;
- publish measured end-to-end bandwidth before claiming a speedup.

## 13. Failure and rollback rules

Disable power authority for a generation if any of the following occurs:

- anisotropic physical cell spacing outside the supported tolerance;
- unknown or invalid topology descriptor;
- catalog version mismatch;
- face or incidence overflow;
- asymmetric shared-face metrics;
- nonpositive face area, distance, or cell volume;
- non-finite coefficient or velocity;
- row matrix asymmetry above tolerance;
- surface departure outside resident sparse support;
- page or redistance overflow.

During development, rollback selects the current compact axis-face operator,
not a dense simulation path. Unsupported terrain/rigid modes may retain their
existing compatibility behavior until WP10, but the final goal remains removal
of those dense publications.

## 14. Recommended commit sequence

1. `fix: unify 2x and 4x sparse surface page sampling`
2. `test: add production adaptive water counters and Dawn render gate`
3. `feat: add deterministic CPU power-cell geometry oracle`
4. `feat: generate canonical octree power topology catalog`
5. `feat: encode GPU power topology descriptors`
6. `feat: publish deterministic compact power faces`
7. `feat: transfer velocity across power-face topology changes`
8. `feat: assemble Chebyshev rows from canonical power faces`
9. `test: add SPD and manufactured T-junction projection gates`
10. `feat: reconstruct adaptive velocity from power-face normals`
11. `feat: add sparse multisegment surface backtracing`
12. `feat: add transition-aware sparse redistance`
13. `feat: port solid apertures and terrain to power faces`
14. `perf: measure unified power-octree memory and bandwidth`
15. `cleanup: remove superseded axis-only and dense compatibility paths`

Each commit must leave the default branch buildable and must not depend on
uncommitted generated catalog data.

## 15. Definition of done

The project is complete only when all of the following are true:

- the production factor-4 and factor-8 fine narrow-band modes are supported by
  every consumer; factor-2 may remain only as a diagnostic compatibility mode;
- balanced 16³ and 32³ scenes visibly render water;
- every reachable supported topology has a catalog entry;
- power faces include required 3D edge-neighbor connectivity;
- divergence, row assembly, and projection use exactly the same face geometry;
- the assembled matrix is symmetric positive semidefinite within tolerance;
- the existing Chebyshev solver runs unchanged on the new coefficients;
- production PCG applies paired second-order boundary/transition smoothing
  around an SPD first-order sparse V-cycle with adjacent-level ghost transfers;
- velocity reconstruction is correct for non-axis-aligned faces;
- surface advection resamples velocity along multisegment trajectories;
- fine-grid redistance constructs the configured signed band from transported
  interface seeds with a meaningful Eikonal residual gate; skipped support
  samples are preserved across the transport/commit transaction;
- coarse redistance and velocity interpolation use catalog local Delaunay
  tetrahedra at octree T-junctions;
- dry-side face extrapolation is transferred regular-face -> power-face without
  synthetic guard connectivity entering authoritative topology;
- topology changes preserve aggregate face flux;
- no supported body-free stage allocates or reads a persistent dense field;
- dam and ocean physical gates pass without overflow or non-finite values;
- real Dawn execution and browser rendering both pass;
- memory and bandwidth numbers are measured and documented;
- obsolete axis-only compatibility code is removed after the rollback window.

## 16. Explicitly deferred work

The following are not part of this implementation:

- (removed 2026-07-20: the Section 4.3 hybrid PCG preconditioner is now in
  scope; the aggregate-PCG scaffold does not complete it; see "Critique
  outcomes and direction changes" and Section 3.2)
- a dense or full-domain fine level-set grid;
- OS virtual-memory SPGrid;
- particle or mesh surface authority;
- unrestricted non-2:1 octrees;
- anisotropic power diagrams;
- multiphase pressure;
- viscosity and surface-tension redesign.

These deferrals must not be used to weaken the unified sparse representation or
the power-face consistency contract described above.

## 17. GPU SPGrid early-warning assessment and topology-update checkpoint

This section records the topology-performance risk identified after the main
plan was written. It is normative for the implementation handoff: do not begin
a wholesale SPGrid rewrite merely because the aggregate `OCTREE` timer is
large. First isolate the work inside that timer, then adopt the useful GPU
SPGrid mechanisms without giving up adaptive leaf resolution.

### 17.1 Decision

Do **not** replace the adaptive octree with a globally uniform SPGrid at this
stage.

Instead, treat the desired topology substrate as:

> an adaptive leaf graph whose storage, allocation, work scheduling, and
> neighborhood lookup use GPU-SPGrid-style hashed brick pages.

This is a refinement of the dynamics-octree plan, not permission to create a
second pressure/velocity representation. The separate global sparse fine
level-set grid required by Section 18 is the one intentional exception: it is
interface-only phi storage, carries no velocity or pressure, and corrects the
coarse octree phi without replacing the octree's simulation authority. Do not
introduce any additional leaf-local fine grid, dense fine field, or second
dynamics authority.

The repository already implements several important pieces of the GPU SPGrid
idea:

- physical 8 cubed owner pages rather than a mandatory box-sized owner array;
- an open-addressed hash from logical brick coordinates to physical pages;
- compact active and retired brick/tile worklists;
- indirect dispatch sized from those worklists;
- canonical coarse owners for missing pages;
- compact page payload arenas with explicit capacity and overflow behavior;
- block-granular kernels instead of recursive CPU tree traversal.

Therefore a data-structure rename or a second hashed page pool is not a useful
optimization. The open question is whether the *adaptive topology algorithms*
operating on those pages are efficient.

### 17.2 What the cited GPU SPGrid paper actually provides

The relevant design in Gao et al., *GPU Optimization of Material Point
Methods* (2018), is not CPU SPGrid virtual-memory paging transplanted directly
to the GPU. Its GPU structure is effectively a tiled sparse grid:

1. Divide the grid into fixed 4 x 4 x 4 blocks.
2. Hash occupied logical block offsets into consecutive physical block IDs.
3. Build explicit mappings for neighboring blocks because GPU code cannot rely
   on CPU page faults or a CPU-style TLB.
4. Sort particles by the compact block ID and then by the dense cell ID inside
   the block.
5. Execute transfers at block granularity and stage neighboring blocks in
   shared memory.
6. Rebuild the active block mapping after particle motion changes occupancy.

The paper's strongest performance claims are for MPM particle-grid transfers,
particularly its CUDA warp reduction for scatter conflicts and its histogram
particle sort. Those gains do not transfer directly to this Eulerian liquid
topology rebuild:

- this simulation does not use MPM particles to define every active block;
- WebGPU does not expose all CUDA warp intrinsics used by the paper on every
  target device;
- the expensive adaptive decisions include leaf sizing and 2:1 grading, which
  a uniform GSPGrid does not perform;
- the paper assumes topology construction is small relative to the subsequent
  MPM work, while our warning is precisely that topology is not small;
- the paper's uniform block grid does not by itself provide 16 cubed or
  32 cubed simulation cells.

What *does* transfer is the principle that sparse addressing and neighborhood
discovery should happen once per brick or leaf publication, not repeatedly per
sample in every downstream kernel.

### 17.3 Current likely costs that SPGrid lookup alone will not remove

The current aggregate topology range includes all of the following:

- owner-page lifecycle work;
- resident/retired worklist copies and indirect-argument staging;
- optional dirty-tile detection and compaction;
- resetting every cell in each selected topology tile;
- one refinement dispatch per supported leaf size;
- exact `leafNeedsRefinement` scans over every fine sample inside a candidate
  leaf;
- cooperative 16 cubed and 32 cubed leaf scans;
- `ceil(log2(maximumLeafSize))` 2:1 balance rounds;
- face-area scans when testing whether coarse leaves neighbor overly fine
  leaves;
- persistent frontier filtering, replacement-leaf appending, and hash repair.

For a 32 cubed candidate, the exact sizing predicate can inspect 32,768 fine
samples. Balance can then inspect six 32 x 32 faces, and the fixed grading loop
can revisit the candidate several times. A page hash makes the owner and phi
loads sparse, but it does not change those sample counts. This is the leading
hypothesis until sub-stage timestamps prove otherwise.

Also note that the page-native configuration intentionally disables the dense
phi snapshot. Consequently the current change-driven dirty-tile path is not
eligible there. Page-native topology can therefore rebuild all resident tiles
even when most retained topology is unchanged. Restoring a dense snapshot is
not an acceptable fix; the replacement must be page-local and sparse.

### 17.4 Mandatory profiling split before structural changes

Split `gpuLayerConstruction_ms` into at least these timestamp ranges:

1. `topologyResidency_ms`
   - owner-page activation/retirement;
   - worklist and indirect-argument preparation.
2. `topologyDirtySelection_ms`
   - page-local change reduction;
   - dirty tile/leaf compaction and halo expansion.
3. `topologyResetRefine_ms`
   - reset;
   - all leaf-sizing evaluations;
   - splits and owner publication.
4. `topologyBalance_ms`
   - every 2:1 grading round;
   - any convergence or queue maintenance.
5. `topologyFrontier_ms`
   - old-frontier filtering;
   - replacement append;
   - origin-to-row hash publication.

Also publish counters, not only time:

- resident, retired, clean, and dirty page counts;
- candidate leaves tested per size class;
- fine phi samples read by leaf sizing per size class;
- leaves split by sizing per size class;
- leaves tested and split by balance per round;
- balance rounds that performed zero splits;
- owner words written;
- frontier rows retained, invalidated, and appended;
- page-hash probes, maximum probe length, and failed insertions;
- neighbor-page mapping hits and hash fallbacks;
- bytes read/written by each topology sub-stage, estimated from counters where
  hardware counters are unavailable.

Measure cold initialization separately from steady-state frames. Report median,
p95, and worst frame over at least 300 simulated frames. A single calm frame is
not representative of dam-break topology churn.

### 17.5 Optimization order

Apply the following experiments in order. Each experiment must be separately
switchable until its differential and performance gates pass.

#### Experiment A: sparse hierarchical sizing summaries

Build a small summary for each resident phi page during or immediately after
surface publication. At minimum record:

- minimum phi;
- maximum phi;
- minimum absolute phi;
- minimum and maximum solid fraction when solids are present;
- optional bounded velocity-span and curvature summaries when detail-based
  refinement is enabled;
- generation number and validity bits.

Reduce these page summaries into a sparse hierarchy aligned with the supported
leaf sizes. A 16 cubed or 32 cubed leaf-sizing decision should combine a bounded
number of child summaries and inspect fine samples only for ambiguous cases.
The hierarchy must contain records only for resident pages and their ancestors;
it must not allocate a dense mip pyramid.

Required correctness contract:

- summary rejection may prove that a leaf is uniformly far from every feature;
- summary acceptance may prove that refinement is required;
- an inconclusive summary falls back to the exact current predicate;
- the optimized decision must never coarsen a leaf that the exact predicate
  would refine;
- generation mismatch is inconclusive, never clean;
- all reductions are deterministic for identical page payloads.

Primary success metric: reduce fine phi reads in 16 cubed and 32 cubed sizing by
at least 8x on representative large scenes without changing the emitted owner
map.

#### Experiment B: page-native dirty detection

Store the previous sizing summary beside each resident page summary. Mark a
page dirty when a refinement-relevant interval crosses a decision boundary or
when its generation/residency changes. Expand the dirty set by the proven
sizing and grading halo, then rebuild only leaves intersecting those pages.

This replaces the disabled dense phi snapshot with sparse metadata. It must not
compare every logical domain cell and must not allocate a domain-volume-sized
bitset. Stable page generations should produce an empty rebuild worklist.

Pages near the interface need conservative threshold handling so sub-cell phi
motion cannot leave a stale wet/dry owner. Use interval tests against the same
refinement predicate, not a generic maximum-delta threshold.

#### Experiment C: persistent topology mutation instead of tile reset

Do not reset an entire dirty tile to maximum-size owners before rediscovering
the same tree. Begin from the existing compact leaves:

- re-evaluate only leaves touched by the dirty-page halo;
- split leaves whose sizing summary requires finer resolution;
- mark potential sibling groups for coarsening;
- coarsen a sibling group only when all eight children agree, no child is
  protected by interface/detail criteria, and grading remains valid;
- publish mutations into a next-generation leaf buffer or through a
  deterministic plan/scan/emit sequence;
- swap generations only after overflow and consistency checks pass.

Avoid in-place concurrent split/coarsen races. Planning and emission may use
additional compact buffers proportional to live leaves, but never proportional
to the full domain.

#### Experiment D: queue-based local 2:1 grading

Replace fixed global balance rounds with a compact work queue seeded by:

- newly split leaves;
- newly coarsened leaves;
- leaves on activated or retired page boundaries;
- immediate face neighbors of all of the above.

For each queued leaf, use the canonical face/incidence topology to identify
neighbors. If a violation is found, enqueue the required coarser neighbor split
and its affected face neighbors. Deduplicate queue entries with a generation
stamp stored per live leaf or in a compact hash keyed by leaf origin.

The queue must terminate because splits monotonically decrease leaf size and
the minimum leaf size is bounded. Add a fail-closed iteration/work limit; on
overflow, reject the new generation rather than silently publishing an
unbalanced tree.

This is where cached SPGrid-style explicit neighbor-page IDs help: resolve the
six brick neighbors once during residency publication, then use direct physical
page IDs in topology kernels. Hash lookup remains the fallback for generation
changes and cross-page adaptive leaf neighbors.

#### Experiment E: optional flat hashed-leaf directory A/B

Only after Experiments A-D are measured should the team test whether the owner
page payload itself is still a bottleneck. The A/B representation is:

- a compact array of live leaf records;
- an open-addressed hash from packed leaf origin/size to row ID;
- explicit bounded face-neighbor incidence generated at publication time;
- resident phi pages for fine surface data;
- no per-fine-cell owner word inside coarse leaves.

This is closer to an adaptive GPU SPGrid than a classical pointer octree. It
can remove repeated owner-word writes and per-cell owner lookup, but downstream
consumers must use the leaf and face catalogs directly. Missing fine pages
still resolve through a canonical coarse leaf, not a dense fallback.

Do not make this A/B authoritative unless it beats owner pages in end-to-end
frame time and memory bandwidth. Owner pages can be faster for arbitrary
point-sampling even though they write more metadata; the comparison must include
advection, pressure, surface extraction, and rendering, not topology alone.

### 17.6 WebGPU-specific constraints

The implementation must not copy CUDA-specific assumptions blindly:

- do not require 64-bit WGSL integer offsets; pack validated domain brick keys
  into 32 bits when the configured dimensions permit it;
- do not require subgroup ballot/shuffle for correctness;
- subgroup acceleration may be an optional pipeline selected only after
  feature detection and parity testing;
- use workgroup memory for bounded page-neighborhood staging where it reduces
  repeated global reads;
- respect minimum WebGPU storage-binding limits and the existing bind-group
  budget;
- keep Dawn validation and browser Metal compilation in every experiment's
  gate;
- do not introduce GPU-to-CPU topology readback or submission barriers.

The cited implementation also reserves substantial auxiliary hash/sort memory.
Our page hash is currently capacity-bounded by resident physical pages and is
far smaller than a domain-sized table. Preserve that property.

### 17.7 Migration gate and kill criteria

Call the current system "topology-bound" only if steady-state topology update
is at least 20% of total GPU simulation time or is one of the two largest
stages in p95 frames on the target 16 cubed/32 cubed large scenes.

Proceed from page-level optimizations to the flat hashed-leaf A/B only if:

- Experiments A-D have passed correctness gates;
- topology remains at least 15% of total simulation time;
- profiling shows owner publication/lookup, rather than sizing or balancing,
  is the remaining majority of topology time;
- projected additional compact metadata remains below the memory acceptance
  limits in Section 12.

Kill the flat-directory experiment if any of the following holds:

- end-to-end GPU time improves by less than 10% on both target large scenes;
- total compact memory increases by more than 15%;
- point-sampling consumers regain a dense compatibility cache;
- hash probe tails or overflow make p95 worse;
- the design requires a second fine simulation grid;
- 2:1 balance, flux conservation, or deterministic publication regresses.

### 17.8 Required benchmark matrix

Run all topology experiments on:

- static tank fill, maximum leaf 16;
- static tank fill, maximum leaf 32;
- dam break during initial collapse, maximum leaf 16;
- dam break during initial collapse, maximum leaf 32;
- broad calm ocean with a thin active surface sheet;
- high-curvature/splashing surface with maximum resident-page churn;
- one rigid-body case, even if it remains a compatibility gate;
- the smallest supported scene as a regression observation, not an optimization
  target.

For every case compare:

1. current owner-page topology;
2. sparse sizing summaries;
3. summaries plus page-native dirty selection;
4. persistent mutation plus queue grading;
5. optional flat hashed-leaf directory.

Record correctness hashes of the leaf set and owner sampling, balance audits,
active rows, topology milliseconds, total simulation milliseconds, resident
bytes, auxiliary bytes, estimated bandwidth, and all overflow counters.

### 17.9 Expected outcome

The likely winning result is not "SPGrid instead of octree." It is:

> SPGrid-style sparse bricks and explicit block neighborhoods for storage and
> scheduling, plus compact adaptive leaves and power faces for discretization.

That architecture retains 16 cubed/32 cubed coarse degrees of freedom, removes
dense representations, and attacks the observed update cost without expanding
the active fine grid through the liquid volume. If profiling disproves the
coarse-scan and balance hypotheses, the sub-stage counters in this section will
identify the next bottleneck before a costly representation rewrite begins.

## 18. Two-resolution sparse level-set architecture

This section supersedes any earlier wording that rejects a separate fine
level-set *discretization*. The rejected design is a dense or full-domain fine
grid. The accepted design is the one in Section 5 of Aanjaneya et al.:

- pressure and velocity live only on the adaptive octree;
- a coarse level set lives on the octree and supplies global sign/topology
  information;
- a uniform fine level set lives only in a sparse narrow band around phi = 0;
- the fine level set is normally 4x or 8x finer than the finest effective
  octree spacing;
- the fine grid never receives pressure or velocity degrees of freedom.

This is still a unified sparse simulation. "Unified" means one authoritative
field for each physical quantity, one sparse residency system, and no dense
compatibility copies. It does not require quantities with fundamentally
different sampling needs to share identical cell keys.

### 18.1 Why the separation is desirable

Bulk incompressibility needs velocity and pressure throughout the liquid, but
surface appearance and volume preservation need geometric resolution only near
the interface. Refining octree velocity merely to sharpen phi would multiply:

- face velocity degrees of freedom;
- pressure rows and matrix entries;
- Chebyshev bandwidth;
- topology construction and grading work;
- rigid coupling and projection work.

A fine narrow-band level set increases surface work approximately with
interface area and band width instead of liquid volume. It lets a 16 cubed or
32 cubed deep-water leaf coexist with sub-finest-cell surface detail while the
actual interface neighborhood remains refined appropriately on the octree.

Direction (2026-07-20): interface-area scaling is still too much for vast
scenes. The band becomes interest-driven — resident only where local velocity,
rigid-body/terrain proximity, or surface churn justifies fine detail, with
calm surface regions served by the corrected coarse octree phi. See "Critique
outcomes and direction changes", direction change 2, which also downgrades the
Section 18.12 factor-8 support-width measurement from checkpoint to defect.

### 18.2 Current implementation and the gap

`WebGPUOctreeSurfacePages` already has a useful prototype of the paper's
storage budget:

- resident/core/halo/desired/activated flags;
- two transported phi buffers;
- a seed/redistance buffer;
- compact active-page worklists;
- hashed lookup and bounded air-side aliases;
- page activation and retirement;
- multisegment backtracing;
- sparse redistance and volume correction.

However, it is currently *leaf attached*:

- page identity is an octree leaf row;
- only finest octree leaves become candidates;
- each page contains 2 cubed or 4 cubed samples inside one finest leaf;
- page topology is rebuilt by scanning or classifying octree rows;
- a split/coarsen or frontier-row change can change fine-page identity even
  when the physical interface brick is unchanged;
- the default factor is 2, below the paper's normal factor of 4 or 8;
- velocity is currently constant `SurfaceLeaf.motion` within a leaf and is only
  resampled when the trace reaches another leaf;
- redistance is a small fixed PDE iteration rather than a complete narrow-band
  distance construction;
- the page-native affine fallback is not yet a fully specified independently
  advected and fine-corrected coarse octree level set.

The target replaces leaf-row page identity with global fine-grid brick
identity. Existing leaf-attached 2 cubed/4 cubed support remains useful as a
compatibility and migration oracle, but it is not the final surface ABI.

### 18.3 Fine-grid coordinates and brick keys

Let the finest effective octree cell width be `h` and the fine ratio be
`m in {4, 8}`. The fine level-set spacing is:

```text
h_fine = h / m
```

Define integer fine sample coordinates in a domain-global coordinate system:

```text
q_fine = floor((x - domainOrigin) / h_fine)
```

Divide this logical lattice into fixed `B cubed` bricks, where `B` is selected
by an explicit 4-versus-8 benchmark. Use:

```text
brickCoord = floor(q_fine / B)
localCoord = q_fine mod B
brickKey   = pack(brickCoord)
```

Requirements:

- `brickKey` is independent of octree leaf rows and generations;
- a physical page can be reused while the octree splits or coarsens below it;
- missing bricks carry no implicit zero phi; queries fall back to the coarse
  octree sign/distance representation;
- all key packing is range checked when scene dimensions are configured;
- 32-bit keys are preferred where sufficient; 64-bit integers are not a WebGPU
  correctness requirement;
- hash capacity scales with maximum resident fine bricks, never logical domain
  volume;
- cache physical IDs for the six same-resolution brick neighbors after each
  topology publication.

The fine grid is called "single" because every active brick has the same
`h_fine`, not because it is allocated as one dense texture.

#### 18.3.1 Initial sparse authority

The first generation follows the same indexed-band model as every later
generation. For analytic `dam-break` and `tank-fill` scenes without terrain,
rigid bodies, or explicit imported/seeded liquid shapes:

1. evaluate the authored analytic SDF in the cold compact-octree classifier;
2. publish compact leaf rows, power faces, and coarse-octree phi;
3. derive `SurfaceLeaf` interface candidates from those validated compact rows;
4. seed the global fine `brickKey` set from the interface candidates, dilate it
   by the complete advection/redistance/publication support width, and publish
   the first indexed fine generation;
5. build owner-brick residency from that compact candidate worklist.

This path may bind a one-texel `r32float` placeholder where WebGPU layouts still
require a sampled 3D texture, but it must not allocate, upload, sample, or
publish a box-sized initial phi field. A valid empty or partially inserted
coarse/fine directory is not authority: row count, generation, insertion
count, capacity, and finite/error flags must all validate before publication.

Explicit initial brick seeds, imported meshes/volumes, terrain, and rigid-body
cut cells remain on the bounded compatibility bootstrap until a sparse
voxelizer can emit signed coarse rows and interface brick keys directly. That
compatibility is not permission to make the dense field persistent or to use
finest-domain indexing after the first sparse generation. The follow-up is a
bounded sparse seed/import voxelizer, not a dense-to-dense reconstruction.

This is the GPU equivalent of Section 5's fresh-SPGrid construction: storage is
created from the interface and its required band, while the coarse octree
supplies sign outside the indexed fine pages.

### 18.4 Fine SPGrid channel layout

Provide the logical equivalent of the paper's four channels:

1. `flags`
   - domain geometry;
   - valid/interface/known/trial state;
   - optional sign and generation bits.
2. `phi`
   - authoritative transported signed-distance value.
3. `workA`
   - next phi/distance value, fast-march bucket state, or sweep ping buffer.
4. `workB`
   - seed/original sign, alternate distance buffer, or second fast-march
     scratch channel.

The flags may be bit packed rather than a full 32-bit scalar per fine sample.
Scratch channels may alias lifetimes when Dawn/WebGPU usage constraints allow
it. The externally visible ABI must nevertheless expose explicit offsets and
generation information so consumers cannot confuse current phi with scratch.

For one finest octree cell, an uncompressed four-32-bit-channel payload costs:

```text
factor 4: 4^3 * 16 bytes = 1,024 bytes
factor 8: 8^3 * 16 bytes = 8,192 bytes
```

This is acceptable only because allocation is restricted to the narrow band.
Report actual active-brick payload, hash, worklist, and fragmentation bytes.
Do not extrapolate savings from a single smooth plane; include folded and
splashing interfaces.

### 18.5 Dynamic topology with two page generations

Follow the paper's fresh-SPGrid logic using GPU page generations rather than
CPU virtual-memory allocation:

1. Advect current fine phi on the current resident brick set.
2. Detect zero-crossing fine cells and compact their brick keys.
3. Add every interface brick and its required one-ring/swept-band neighbors to
   a next-generation desired-key set.
4. Sort/unique keys or insert them deterministically into the next page hash.
5. Reuse physical pages for keys present in both generations when this is safe;
   otherwise copy only interface seeds into next-generation pages.
6. Initialize newly activated bricks from neighboring interface seeds and the
   coarse octree sign, then complete fine-grid redistance.
7. Validate band completeness, hash load, finite phi, and capacity.
8. Atomically publish/swap the next page table and worklist.
9. Return old-only physical pages to the free list after no encoded consumer can
   reference the previous generation.

This reproduces the simplicity of constructing a new SPGrid without allocating
a second maximum-capacity payload arena if page reuse is carefully sequenced.
If safe reuse complicates the first implementation, use two bounded page-table
and metadata generations while sharing a capacity-checked payload pool.

Do not scan the full logical fine lattice. Interface discovery starts from
resident pages; new coverage comes only from bounded dilation and swept
trajectory support.

### 18.6 Fine level-set advection

Retain the octree timestep `dt`. The fine ratio does not require four or eight
complete simulation substeps. For every fine sample:

1. set `m` to the fine ratio, or a greater bounded CFL-safe count;
2. trace backward `m` pieces of duration `dt / m`;
3. reconstruct and resample velocity at every segment endpoint;
4. sample old fine phi once at the final departure point;
5. run redistance once after the complete advection.

Velocity reconstruction must use the power-face field:

- reconstruct full velocity vectors at octree cell centers by least squares
  from incident face-normal values;
- use trilinear interpolation in uniform cubic regions;
- use barycentric interpolation on catalog Delaunay tetrahedra near octree
  transitions;
- query this interpolant at every fine trajectory segment;
- never allocate velocity on the fine SPGrid.

For the factor-4 vertical slice, the compact simulation topology stores power
faces only for wet rows; there are deliberately no dry power-face records onto
which the paper's final regular-face-to-power-face interpolation can be
published. The current dry-side fallback consumes a temporary regular-face
band for positive-air trajectory samples while retaining power-cell
interpolation for wet samples. This is a bounded engineering approximation,
not an equivalent implementation of Section 5: the implemented increment
resolves real size-one owners and coarse phi, but deliberately fails at
non-uniform owners rather than inventing transition connectivity. It still
needs local catalog-Delaunay tetrahedron adjacency at T-junctions, and the
result is not transferred back to newly created power faces. The temporary
band is not persistent power-face authority and must write only trajectory
result/status buffers. Synthetic guard phi or edges are prohibited from
authoritative face rows and topology acceptance. Newly connected wet power
faces still require WP9's trace-back/restriction/prolongation transfer.

This is a major fidelity requirement. Merely increasing page resolution while
retaining one constant leaf velocity will sharpen sampling but will not recover
the paper's reduced advection dissipation.

The active narrow band must cover:

```text
maximum backtrace displacement
+ velocity interpolation support
+ redistance stencil/bucket support
+ one topology-publication safety ring
```

Any departure outside the resident band invalidates that generation. Do not
silently clamp to a page edge or sample a dense fallback.

### 18.7 Fine redistance

Because the fine lattice is uniform, fine redistance must not use octree
transition tetrahedra. Implement either:

- bucketed parallel fast marching with deterministic integer bucket order; or
- ordered directional fast sweeping over the compact active-brick set.

In both cases:

- seed exact/subcell distances from transported sign changes;
- exchange boundary values through cached neighbor-page IDs;
- preserve the transported sign;
- continue until the configured physical band is valid, not for an arbitrary
  four iterations;
- record unresolved cells and maximum Eikonal residual;
- keep work proportional to resident fine cells.

The fixed-schedule Jacobi Eikonal scaffold has been removed. Its first
replacement, an exact serial binary-heap FMM, established the required causal
semantics but failed the production runtime gate (>3 minutes without returning
the first `60 x 45 x 40` factor-4 submission). The connected replacement uses
distance buckets of width `h/2`, below the `h/sqrt(3)` minimum 3-D upwind
increment, so a value accepted in one bucket can depend only on prescribed
interface seeds or earlier buckets. Sample passes dispatch indirectly over the
live page count. Accepted boundary fronts emit page requests in parallel; a
bounded compact allocator deduplicates those requests, completely initializes
and links dense target-generation pages, then publishes their hash entries
before the next bucket. Capacity, hash, generation, request, residual, and
unresolved failures remain publication-fatal. Static shader validation passes;
production timing and numerical telemetry still gate Section 5 fidelity. The
same fine-coordinate implementation now supports factor-8 B4 pages (including
the eight pages mapped to each finest octree cell); factor-8 physical Dawn and
endurance evidence remains outstanding.

Use the paper's local Delaunay tetrahedra only for coarse-octree redistance and
octree velocity interpolation near T-junctions.

### 18.8 Coarse octree level set

The coarse level set is required even though fine phi is authoritative near the
surface. It supplies:

- inside/outside information beyond the fine band;
- wet-leaf classification and pressure-row construction;
- refinement/coarsening criteria;
- a safe signed fallback for a missing fine brick;
- initialization signs for newly activated fine bricks.

Store at least one signed phi value per live octree cell. A bounded gradient or
min/max interval may also be stored for interpolation and conservative topology
classification. This allocation scales with compact live leaves, not finest
domain cells.

Per timestep:

1. advect coarse octree phi using the shared reconstructed velocity;
2. redistance it on octree connectivity;
3. wherever valid fine samples overlap a leaf, restrict/correct the coarse phi
   and its conservative interval from the fine field;
4. use the corrected coarse field for the next topology update and pressure
   free-surface fractions.

Restriction must preserve the zero crossing. A plain average can erase a thin
sheet. At minimum publish center phi plus fine-band min/max; use a fitted local
plane or closest signed sample when a leaf contains a sign change.

### 18.9 Relationship to the slow octree-update warning

The fine SPGrid should help, not worsen, topology construction:

- fine interface bricks directly identify which octree regions require finest
  refinement;
- each fine brick can publish phi min/max/min-absolute summaries;
- coarse leaf sizing can combine sparse summaries instead of scanning 16 cubed
  or 32 cubed finest samples;
- stable fine-brick keys avoid rebuilding surface storage merely because leaf
  row indices changed;
- the coarse octree can remain unchanged while subcell surface geometry moves
  within the already refined interface band.

Topology still needs page-native dirty detection and queue-based 2:1 grading
from Section 17. Do not rebuild the octree at the factor-4/factor-8 fine
resolution.

Implementation checkpoint (2026-07-20): the production topology shader now
consumes both GPU-published sparse levels without CPU readback. A power-of-two
octree leaf maps to one dyadic fine-summary key (one brick per finest cell at
factor 4, two at factor 8). Any observed fine-summary sign crossing forces a
split even when the sparse node is incomplete; only a complete sample/brick
count may replace the legacy sizing scan. Missing, unpublished, errored, or
stale data is inconclusive.

The next-step coarse sign authority is the compact corrected coarse-phi hash.
Its publication generation must match the expected global-fine generation,
and its state, dimensions, cell width, capacity, finite flags, and phi interval
must all validate before use. `minimumPhi < 0` classifies a crossing leaf as
wet even when its centre phi is positive, so a restricted thin sheet cannot be
coarsened away. After coarse publication, exact corrected leaf intervals are
GPU-merged into the same fine-summary hash (word 7 marks coarse authority), so
the Metal refinement kernel performs only one hash lookup. Refinement uses that
interval without rescanning only when the directory contains the exact
candidate leaf; a partition mismatch executes the retained exact rollback
predicate. A valid missing point key is positive
air only because publication remains invalid unless every requested compact
row inserted successfully.

Both consumers remain within the portable ten-storage-buffer limit. Fine
summaries alias the otherwise-unused pressure input only in refinement bind
groups; the coarse directory aliases binding 15 in global-fine topology and
pressure groups, while sparse extrapolation retains separate binding-15 bulk
worklist groups. This completes migration step 8's consumer wiring, but does
not authorize removal of the dense/leaf rollback fields or completion of the
page-native dirty detection and local grading experiments.

### 18.10 Migration sequence

1. Keep current 2 cubed/4 cubed leaf pages as the differential oracle.
2. Introduce global fine coordinates, packed brick keys, and a standalone page
   hash without changing phi transport.
3. Copy current page samples into globally keyed factor-4 bricks and prove
   sampling parity.
4. Make renderer and surface extraction consume the global fine-grid source.
5. Move advection, redistance, and volume correction to global fine bricks.
6. Add next-generation interface-plus-ring topology publication.
7. Add the compact coarse octree phi field and fine-to-coarse correction.
8. Make octree sizing consume fine-brick summaries.
9. Factor 4 is now the balanced dam-break product default behind manual
   browser GPU startup; retain the staged Dawn and browser physical gates as
   release evidence rather than silently reverting the preset.
10. Add factor 8 as a high-quality mode with an explicit memory/capacity gate.
11. Remove leaf-row ownership from the production surface ABI.
12. Remove bootstrap dense phi publication after every consumer binds either
    fine sparse phi or compact coarse octree phi.

### 18.11 Acceptance gates

- No persistent dense phi texture beyond a one-texel format placeholder.
- Fine-grid memory scales with resident narrow-band bricks.
- Coarse phi memory scales with live octree leaves.
- Factor-4 translating plane and rotating/deforming sphere retain sign and
  converge redistance across page boundaries.
- Factor-8 improves measured surface error or visual detail enough to justify
  its memory/bandwidth cost.
- No fine-grid velocity or pressure allocation exists.
- Fine advection samples reconstructed velocity at every segment.
- Fine-to-coarse correction preserves all detected zero crossings.
- Newly activated pages receive valid signed-distance values before
  publication.
- Page topology can grow, shrink, and translate for 300 frames without leaks,
  stale generations, overflow, or CPU readback.
- Octree topology time does not regress; fine-brick summaries reduce coarse
  sizing sample reads materially on 16 cubed/32 cubed target scenes.
- Real Dawn tests and browser screenshots show visible, moving water at both
  factor 4 and factor 8.

### 18.12 Measured factor-8 capacity checkpoint

The production `60 x 45 x 40` dam-break checkpoint uses a factor-8 fine
lattice, `4 cubed` bricks, and the complete Section 18.6 support width:

```text
8 fine cells maximum backtrace
+ 1 fine cell interpolation support
+ 32 fine cells redistance support
+ 1 whole-brick publication safety ring
= 12 brick rings
```

With a configured capacity of 337,500 bricks, generation 2 required 249,808
bricks and generation 3 required 266,555 bricks. Generation-3 headroom was
70,945 bricks (21.0% of configured capacity), after required residency grew by
16,747 bricks (6.7%) from the bootstrap generation. Both generations published
without topology flags or rollback and retained both signs. This is a two-step
capacity checkpoint, not evidence for the 300-frame endurance gate.

Capacity telemetry must distinguish:

- configured brick capacity;
- exact required bricks when discovery completes, or an explicitly labelled
  strict lower bound when discovery reaches capacity;
- active payload bytes;
- capacity payload bytes and fragmentation bytes;
- hash, metadata, and worklist bytes;
- exclusive fine-level-set, power-diagram, and total allocation bytes.

At generation 3 the active payload was 272,952,320 bytes, capacity payload was
345,600,000 bytes, and payload fragmentation was 72,647,680 bytes. The complete
fine-level-set allocation was 709,879,732 bytes and total solver allocation was
862,709,112 bytes on the measured adapter. Factor 8 therefore remains an
explicit high-memory quality mode; its physical R12 band must not be narrowed
silently to meet a smaller device budget. A device that cannot allocate the
reported capacity must reject the mode or publish a fail-closed capacity error.
