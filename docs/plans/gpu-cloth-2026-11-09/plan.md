# GPU-native cloth: VBD + offset contact, then voxel solids, then fluid

Date: 2026-11-09. Status: approved 2026-11-09; implementation in progress (lanes C1 and C2, progress in `progress-lane-c1.md` and `progress-lane-c2.md` beside this file).

**Decisions (user, 2026-11-09):**
- **No substepping, ever** (user, 2026-11-09, overriding the earlier substep approval). Cloth takes exactly one implicit step per 1/60 s frame, the same dt as the fluid. Contact re-detection rounds inside that step's iteration loop are allowed: they refresh the contact set during one solve and do not advance time.
- Two-way rigid coupling uses option (b): bodies that touch cloth become AVBD 6-DOF blocks in the cloth's GPU solve.
- The libwetcloth reproduction is retired as a goal. Yarn, rods and towels are parked.
- **The old cloth code is deleted, not parked.** That covers mechanics, contact, rods, porous transport, cohesion, the wet coupling in NB-FLIP, wetcloth scenes, assets, tools, tests and plan docs. Keep the tree neat: only what the new triangle cloth uses survives, rewritten where that is simpler.
- CPU work is allowed at setup and edit time: mesh build, partitioning and initial colouring. Per-step work is GPU only, with no readbacks in the step path.

Paper texts (pdftotext) for the lanes: `~/code/me/fluid-scratch/gpu-cloth-papers/` (vbd, avbd, ogc, jgs2, pfb, webgpu).

## Why replace the current cloth

The current cloth reproduces libwetcloth (Fei et al. 2018). It uses an implicit-Euler incremental potential and solves it with GPU Newton-PCG (`lib/cloth/gpu-newton.ts`, 2,012 lines). The design blocks real-time speed for structural reasons, not tuning ones:

- **The solver is global.** Each Newton iteration assembles a block-CSR Hessian and runs PCG with Eisenstat-Walker forcing, then an 8×5 Armijo line search. On full-size dam-break, mechanics alone is 1.19 s per step (58 GPU sweeps).
- **Contact is CPU-in-the-loop.** The Float64 contact set detects on the CPU. Augmented-Lagrangian multipliers and tangent planes update on the CPU between GPU solves, so each outer iteration reads positions back. `splash_cloth_small` stops with "contact constraints did not converge within 64 sweeps" as soon as cloth meets a sphere.
- **Readback-driven control flow.** There is a status readback every two Newton iterations, plus position readbacks. This contradicts the all-GPU rule.
- **Scope coupling.** DER yarn, pores, cohesion and NB-FLIP coupling are all wired into one mechanics step. None of it can be made fast in isolation.

## Research summary (state of the art, late 2026)

| Method | What it is | Real-time? | Fit for us |
|---|---|---|---|
| **VBD**: Chen, Liu, Yang, Yuksel, SIGGRAPH 2024 | Implicit-Euler variational energy, minimised by per-vertex 3×3 Newton blocks, Gauss-Seidel over graph colours, Chebyshev acceleration. Unconditionally stable, purely local. | Yes; used in NVIDIA Newton/Isaac Lab (`SolverVBD`) | **Core solver.** No global matrix, no precompute, so cuts and topology edits are cheap. |
| **AVBD**: Giles, Diaz, Yuksel, SIGGRAPH 2025 | Augmented Lagrangian on VBD: hard constraints with progressive stiffness and warm-started duals. Fixes VBD's poor convergence at high stiffness ratios. | Yes: 110K rigid blocks in 3.5 ms on a 4090; 10K-vertex cloth in 16 ms | **Pins, attachments, strain limits.** Later, rigid bodies in the same solve. |
| **OGC**: Chen, Hsu, Liu, Macklin, Yang, Yuksel, SIGGRAPH 2025 | Offset-geometry contact: block-wise offset of each face, so normal forces are orthogonal. Per-vertex conservative displacement bounds instead of CCD. Only local operations, no global sync. | Yes: 4090, 15.6K-vertex cloth on body 1.2 ms/step at dt 1/200 with 20 iterations; 1M vertices (50 layers) 6.3 ms/step at dt 1/1200 | **Self-contact model.** Penetration-free by construction. |
| JGS2: Lan et al., SIGGRAPH 2025 | Jacobi/GS with a precomputed perturbation subspace (cubature) to cancel overshoot; near-Newton convergence. | 2M-element cloth with IPC: 469 ms/step | **Rejected.** Per-mesh precompute breaks on cuts, and too slow with contact. |
| Barrier-free AL: Zheng, Luo, Li, SIGGRAPH 2026 | Second-order AL with CCD-detected active set; IPC robustness without barrier ill-conditioning or time-of-impact locking. | No: 0.016–1.7 s/step | **Reference for robustness**, not the solver. Its OGC comparison is our risk list (below). |
| GIPC / StiffGIPC / C-IPC | GPU IPC | No: seconds per frame | Rejected for real time. |
| XPBD / mass-spring (incl. WebGPU paper 2507.11794) | Constraint projection, iteration-dependent stiffness | Yes | Rejected: does not converge to the physical solution. The fidelity complaint would come straight back. |

**Independent evidence on OGC's weak points** (Zheng et al. 2026, Appendix B, profiling Newton's official OGC):

1. **Motion locks at high velocity.** Conservative bounds truncate displacement, so in contact-dense, fast regions motion locks unless the step is split. Their twisting test needed 80 substeps × 80 iterations.
2. **Float32 crashes in stiff cases** (high bending, 50 layers) without an energy line search.
3. **Contact force/Hessian accumulation and edge-edge detection dominate** (about 4.9 s of 6.5 s in the stacked test). Warp also issued 13,206 `memset`s per run. That is a launch-overhead pattern an M1 Max one-pass design must avoid.

## The design: "VBD-OGC, patch-resident"

### State and energies

- **Mesh.** A triangle shell, stored as a structure of arrays. Per vertex: `x`, `x_t`, `v`, inertia target `y`, mass, OGC bound `b_v`, detection position `x_det`, AVBD duals. Per face: `Dm⁻¹` (2×2), area, material. Per hinge: 4 vertices, rest angle, stiffness. Vertex→element adjacency is CSR. Each vertex keeps its local index in each incident element.
- **Membrane.** Orthotropic StVK, so warp and weft can differ for woven fabric. The per-vertex 3×3 Hessian is projected to PSD, and compression is softened so the cloth buckles instead of resisting.
- **Bending.** Discrete-shell dihedral energy with a Gauss-Newton Hessian `k ∇θ ∇θᵀ`, which is PSD and cheap. The rest angle comes from the authored shape.
- **Damping.** Rayleigh damping in the VBD form, `k_d H (x − x_t)/dt`.
- **Hard constraints (AVBD).** Pins, moving attachments, and an optional per-edge strain limit `|e| ≤ (1+ε) L`. The strain limit lets near-inextensible fabric use soft stretch energy without the high stiffness ratio that stalls plain VBD.
- **Precision.** Float32 in a cloth-local frame. Solve for per-step increments `Δx` from `x_t`, which the current GPU-Newton already does well. Keep that idea.

### Step (all on the GPU, no readback)

```
per frame (dt = 1/60 s, one step, no substeps):
  y = x_t + dt v + dt² g                       (inertia target)
  x = VBD adaptive init(y), truncated to b_v    (penetration-free start)
  for k in 1..K:                                (K authored per scene; the only budget)
    for colour c:                               (≈ 6–8 colours for a cloth mesh)
      per vertex i in c: gather f_i, H_i from incident faces, hinges, contacts
                         (and AVBD terms)
                         Δ = H_i⁻¹ f_i, local backtracking on the vertex's own energy
                         x_i += Δ, truncate |x_i − x_det,i| ≤ b_v,i
    Chebyshev extrapolation; AVBD dual update
    count vertices that hit their bound  → if ≥ γ_e N, re-detect (indirect dispatch)
  v = (x − x_t) / dt
```

- **The vertex's own line search** fixes OGC failure 2. The local energy is a sum over a dozen elements, so backtracking is cheap.
- **Re-detection is GPU-decided.** An atomic counter writes the indirect args for detection: zero workgroups normally, N when bounds are exhausted. The CPU never sees it.
- **One 1/60 s step, no substeps.** The published OGC scenes use dt 1/100–1/1200, so a full 1/60 step is beyond what the papers demonstrate. Three things make it work:
  - **Velocity-adaptive envelope.** Per vertex, `r_q,v = r + s·|v|·dt`, capped by an authored limit with a status word. The query uses the larger envelope of each pair, so no pair is missed. The bound is `d_min` clipped at the vertex's own envelope, so free-flying fast cloth gets a bound as large as its motion, and only cloth near other cloth is truncated.
  - **Re-detection rounds inside the step.** They run between iterations through the exceed-count and indirect dispatch, with an authored cap per step.
  - **Convergence at large dt.** AVBD strain limits and stiff terms, Chebyshev acceleration, warm-started duals and stiffnesses, and the vertex line search.
- **BDF2 by default.** Implicit Euler at 1/60 visibly damps swinging cloth. BDF2 is the same variational VBD solve with a different inertia target and weights.
- **Truncation is Planar-DAT, not OGC's isotropic sphere.** Chen, Hsu, Ayman and Macklin, *Divide and Truncate* (arXiv 2604.15513, April 2026), is the OGC authors' successor. Each contact pair gets a division plane, so only motion toward the other primitive is truncated and tangential motion is free. It keeps OGC's penetration-free guarantee and removes OGC's damping and deadlock, for about 10% more per-iteration cost. Its stated limitation applies directly here: motion beyond the query radius falls back to isotropic truncation and damps.
- **Velocity-adaptive envelope (our extension; unpublished).** The swept, predictive envelope comes from Lewin, *Cloth Self Collision with Predictive Contacts* (GDC 2018). It uses one detection per 30 Hz frame, with the envelope swept along each primitive's displacement and half-space contacts, but it is not robust to fast motion on its own. Combined with DAT, the guarantee holds whenever pairs are searched at max(r_q,a, r_q,b): unpaired isotropic radii sum to γ·max, which is less than the distance. **No published paper demonstrates robust self-contact at a single 1/60 s step.** DAT, OGC and Wu 2020 all use steps of 1/100–1/2520 s, so the 1/60 gates are where this is proven or refuted.
- **Fallback if Planar-DAT still locks at 1/60.** Use the active-set idea from Zheng, Luo and Li (2026): add every pair found along the swept motion as AVBD inequality constraints at once, with augmented-Lagrangian multipliers, instead of only truncating. Penetration-freedom stays exact. Switching needs evidence from the gates.

### Self-contact (OGC)

- **Contact energy.** Offset radius `r` is the physical cloth thickness, about 1–2 mm. Use OGC's two-stage C² activation (quadratic, then log) over the block-wise offset geometry. Vertex-facet and edge-edge pairs go through OGC Algorithms 1 and 2, which never miss or duplicate a pair.
- **Friction.** IPC lagged friction, with the VBD paper's stabilisation.
- **Conservative bound.** `b_v = γ_p · min(d_min,v, d^E_min,v, d^T_min,v)`, with γ_p = 0.45. The atomic mins use the u32 bit pattern of non-negative f32, which is monotone, because WGSL has no float atomics.
- **Broad phase.** A sparse hash grid with cell size `max(r_q, mean edge length)` and a counting sort, with no BVH rebuild. Cloth triangles are near-uniform, so a grid beats an LBVH here. Keep an LBVH option for strongly graded meshes.
- **Fixed per-vertex contact capacity** (for example 16 vertex-facet + 16 edge-edge slots), authored per scene from the measured peak. Overflow increments a counter and fails the step explicitly: no runtime pool growth, no silent drop.
- **Gate check.** A GPU exact triangle-triangle intersection count after every frame of every gate run, as the OGC paper does. It must be 0.

### Execution on M1 Max: two tiers

**Tier A (baseline, lands first).** Global coloured VBD: one dispatch per colour per iteration. Use one SIMD-group per small vertex group: lanes take incident elements and contacts, then reduce with subgroup ops (Dawn `subgroups`; fall back to workgroup memory). Element terms are recomputed per incident vertex, as VBD does; there is no element pre-pass. Dispatch count is colours × iterations. That is fine for correctness, but it is the launch-bound pattern from the OGC profile.

**Tier B (the big swing).** Patch-resident solve, the cloth analogue of the brick-resident fluid step:

- Partition the mesh into patches of about 512–1024 vertices, with greedy BFS ordering so each patch's elements are contiguous.
- One workgroup owns a patch and keeps its positions in threadgroup memory. It runs all K iterations × C colours inside one dispatch, with `workgroupBarrier` between colours.
- Halo vertices from neighbouring patches stay frozen during the dispatch: Gauss-Seidel inside, block-Jacobi across.
- Alternate two shifted partitions between dispatches (two-colour multiplicative Schwarz), so seams don't stay put.
- Read-only rest data streams from device memory. A 64K-vertex cloth's whole working set (about 10 MB) fits in the 48 MB system-level cache.
- **Activity.** Sleeping patches (bounded residual and velocity) skip their workgroup, so cost scales with moving cloth, not authored cloth.
- **Gate.** At equal wall-time, Tier B's residual must beat Tier A's. If the Jacobi seams cost more convergence than the saved dispatches buy, attribute it before abandoning: overlap width, partition count, Chebyshev ω.

### Targets (M1 Max, 60 Hz frame, self-contact on)

| Cloth | Target per frame | Reference |
|---|---|---|
| 16K vertices | ≤ 4 ms | OGC "cloth on body", 15.6K vertices: 1.2 ms/step on a 4090 at dt 1/200 (about 4 ms per frame). That workload is launch-bound, so Tier B should match it on the M1 Max. |
| 64K vertices | ≤ 12 ms | |
| 256K vertices | interactive (≤ 50 ms) | |

These are targets to measure against, not predictions. Every gate also reports energy-residual decay and the intersection count.

### Editing contract (AGENTS.md)

- **Reuse the geometry and edit path.** Keep the existing cloth topology, picking, render and cut path (`topology.ts`, `edit.ts`, `render.ts`, `round-cut.ts`, `scene-edit.ts`), so hit testing and local cuts stay on the shared geometry path.
- **After a cut, recolour only the touched region on the GPU.** Use Jones-Plassmann greedy colouring, AVBD style, then rebuild the touched patches. VBD needs no global precompute, which is the main reason JGS2 and projective dynamics are out.
- **Regressions** for picking, a local cut mid-simulation, untouched fidelity outside the cut, and save/load/undo of the solver state, before any scene uses the new representation.

### What happens to the old code

Deleted (decision above). Lane C1 removes the old stack in its first commit:
- `lib/cloth` mechanics, contact, rod, porous and source modules;
- `lib/methods/uniform/uniform-cloth-*` and the cloth hooks in the NB-FLIP method;
- the wetcloth scene shelf and `public/assets/wetcloth`;
- the `tools/*wetcloth*` and `tools/probe-cloth-*` probes;
- `tests/cloth-*` and `tests/wetcloth-*`;
- the `docs/plans/wetcloth-*` documents.

The triangle-cloth document, picking, cut and render paths survive only where the new solver uses them. Yarn can come back later as Cosserat rods inside VBD, as OGC demonstrates.

## Phase 2: cloth against voxel solids

Solids are voxel solids (8³-cell bricks, 9³ SDF nodes, hashed, already on the GPU through `webgpu-solid-world-pages.ts`). Cloth reads that same field, so solid edits are seen with no extra work.

- **Vertex vs field.** The energy is the same OGC activation on `φ(x) − r`, using trilinear `φ` and `∇φ` from `sampleSolidNodes`, with Hessian `∇φ∇φᵀ`. The bound term is `b_v ≤ γ_p (φ(x_det) − r) / L_b`. Here `L_b` is a per-brick Lipschitz bound, computed at upload, because a trilinear SDF is not exactly 1-Lipschitz.
- **Solid features vs cloth faces.** A sharp convex solid edge can poke through a triangle while no cloth vertex enters the field. Fix this by sampling the solid's zero surface, near cloth only, into points and edges, and running them through the same OGC vertex-facet and edge-edge path in reverse. This is the analogue of the existing `solid-samples.ts`.
- **Thin features.** Features below one cell are not representable (stated in `procedural-solid.ts`). Scenes must resolve their thinnest solid, and an unresolved one fails at authoring.
- **Rigid voxel solids.** Sample in body frame; contact velocity comes from the body's twist.
  - Step 1: one-way, with the rigid body kinematic for cloth.
  - Step 2, two-way, needs a choice. Box3D steps on the CPU (`box3d/world.ts`).
    - (a) Accumulate the cloth→body wrench on the GPU and hand it to Box3D once per frame. It is lagged and needs a readback.
    - (b) **Recommended:** bodies that touch cloth become AVBD 6-DOF blocks in the same GPU solve, which AVBD was built for. Box3D keeps everything else.
- **Gates.** Drape on a voxel sphere, a box with sharp edges (corner poke-through), a cut in the solid under resting cloth (the cloth falls through the new hole), and cloth on a moving rigid plate. Intersection count 0, every frame.

## Phase 3: return to cloth-fluid (sketch only)

Coupling enters VBD as per-vertex external forces in the inertia target: NB-FLIP drag and pressure sampled at the cloth, with reactions splatted back. Porous transport returns only after phases 1–2 satisfy you. This phase is deliberately not designed yet.

## Milestones

| M | Deliverable | Gate (single short runs, per the GPU gate policy) |
|---|---|---|
| M0 | Float64 CPU oracle of one VBD vertex update: energies, gradients, PSD Hessians, OGC energy and bound | unit tests only |
| M1 | Tier A dry cloth: membrane, bending, damping, AVBD pins, ground | 64² and 256² hanging cloth; residual decay per iteration; ms/frame; matches the M0 oracle per vertex |
| M2 | OGC self-contact, hash broad phase, velocity-adaptive envelope, in-step GPU re-detection, vertex line search | twisting cloth (10K, from OGC Fig. 12); 10-layer drop; intersections = 0; ms/frame vs OGC Table 1 |
| M3 | Tier B patch-resident solve, sleeping patches | equal-time residual vs Tier A; 16K/64K targets |
| M4 | Edit contract: GPU recolour, cut mid-simulation, picking, save/load/undo | AGENTS.md regressions |
| M5 | Static voxel-solid contact, including solid-feature samples | sphere drape, sharp box, solid cut under cloth |
| M6 | Rigid voxel solids, one-way, then two-way (option a or b) | moving plate; momentum exchange check |

Two long-lived lanes:
- **Lane C1:** solver core, Tier B and edits (M0, M1, M3, M4).
- **Lane C2:** contact, then voxel solids (M2, M5, M6).

C2 starts once M1's per-vertex interfaces are frozen.

## Risks to watch

1. **OGC locking** under fast motion at a single 1/60 s step. This is the central risk. Mitigated by the velocity-adaptive envelope, in-step re-detection rounds, and bounds taken from `x_det` across iterations. If locking persists, the fallback is the swept active set as AVBD constraints. Gate it on truncated fraction, residual, rounds per step and intersection count.
2. **Float32 stiffness blow-ups.** Mitigated by the vertex line search and AVBD for stiff terms, not by stiffer penalties.
3. **Gauss-Seidel inconsistency.** Contacts between same-colour vertices are lagged within a colour, as VBD does. Watch the residual in dense stacks.
4. **Dawn limits:**
   - storage buffers per stage, so pack the structure-of-arrays into a few buffers;
   - no float atomics, so use u32-ordered mins;
   - subgroup availability, so keep a workgroup-memory fallback;
   - silent Metal aborts, so use the runner guard.

## Sources

- Chen, Liu, Yang, Yuksel. [Vertex Block Descent](https://graphics.cs.utah.edu/research/projects/vbd/). ACM TOG 43(4), SIGGRAPH 2024.
- Giles, Diaz, Yuksel. [Augmented Vertex Block Descent](https://graphics.cs.utah.edu/research/projects/avbd/). ACM TOG 44(4), SIGGRAPH 2025.
- Chen, Hsu, Liu, Macklin, Yang, Yuksel. [Offset Geometric Contact](https://graphics.cs.utah.edu/research/projects/ogc/). ACM TOG 44(4), SIGGRAPH 2025. Reference implementation in [newton-physics/newton](https://github.com/newton-physics/newton) (`SolverVBD`).
- Lan et al. [JGS2](https://arxiv.org/abs/2506.06494). ACM TOG 44(4), SIGGRAPH 2025.
- Zheng, Luo, Li. [Robust and Efficient Penetration-Free Elastodynamics without Barriers](https://arxiv.org/abs/2512.12151). ACM TOG, SIGGRAPH 2026 (includes the OGC profiling in Appendix B).
- Chen. [Toward Real-Time, Stable, and Physically Accurate Simulation: VBD and Beyond](https://cmis2026.usi.ch/media/com_papers/abstracts/5_chen_CMIS2026_231036.pdf). CMIS 2026 abstract.
- [Newton SolverVBD docs](https://newton-physics.github.io/newton/api/_generated/newton.solvers.SolverVBD.html); [Isaac Lab VBD tuning](https://isaac-sim.github.io/IsaacLab/main/source/how-to/solver_tuning_vbd.html).
- Wu, Wu, Yang, Wang. [A Safe and Fast Repulsion Method for GPU-based Cloth Self Collisions](https://wanghmin.github.io/publication/wu-2020-sfr/). ACM TOG 40(1), 2020 (the conservative-bound technique OGC builds on; dt 1/100 with many collision steps per time step).
- Chen, Hsu, Ayman, Macklin. [Divide and Truncate](https://arxiv.org/abs/2604.15513). arXiv, April 2026 (Planar-DAT truncation).
- Lewin. [Cloth Self Collision with Predictive Contacts](https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/gdc2018-chrislewin-clothselfcollisionwithpredictivecontacts.pdf). GDC 2018 (swept predictive envelope at one detection per frame).
- Sung et al. [Real-Time Cloth Simulation Using WebGPU](https://arxiv.org/abs/2507.11794). 2025 (mass-spring baseline; not adopted).
