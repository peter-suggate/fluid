# Lane C2 progress: cloth contact (OGC energy, Planar-DAT truncation), voxel solids

Plan: `plan.md` beside this file. Lane C2 owns `lib/cloth/contact/**` and `tests/cloth-contact-*`. Lane C1 owns the solver.

## 2026-11-09 DECISION (coordinator, recorded): one step per frame, envelope, rounds, DAT

These came from the coordinator in order. Each later entry overrides the earlier ones where they conflict.

1. **1/60 outer step, measured honestly.**
   - Re-detection may run several times per step. Rounds are capped by an authored limit, with a status word at the cap.
   - r_q is authored. Report how b_v saturates against γ_p·r_q.
   - Gate rows report, per frame: detection rounds, the truncated fraction, intersections and the residual. Locking shows as a high truncated fraction while the residual stays high.
2. **Velocity-adaptive envelope.**
   - r_q,v = r + s·|v|·dt with s ≈ 1.5, capped at an authored maximum. The cap sets a status bit.
   - The hash query uses the larger envelope of the two primitives; edge-edge uses the max over the edge's vertices.
   - d_min is clipped at the vertex's own r_q,v.
   - Gate rows compare the fixed and adaptive envelopes, one run each, including the pair-count cost.
3. **User override: no substepping, ever.**
   - Exactly one implicit step per 1/60 s frame. The envelope is sized from the full dt.
   - Re-detection rounds run inside the step's iteration loop, through the exceed count, indirect dispatch and C1's between-iteration hook.
   - Locking at 1/60 is the central risk. Measure it on the twisting cloth and the 10-layer drop.
   - The fallback, pfb's active set with AVBD and AL multipliers, needs a DECISION NEEDED with evidence before any switch.
4. **Truncation is Planar-DAT** (Chen, Hsu, Ayman, Macklin 2026).
   - Keep the OGC energy, detection and lists.
   - Per-pair division planes replace the isotropic sphere.
   - Vertices with no pair keep 0.5·γ_r·r_q,v.
   - ΔX accumulates across iterations and is committed at each re-detection. Truncate every iteration.
   - Correctness condition: every pair within max(r_q,a, r_q,b) is paired. Unit-test it against brute force.
   - Isotropic DAT (the OGC bound) stays as a switchable baseline for the gates.

## 2026-11-09 INTERFACE: what C1's solver binds and calls

`lib/cloth/contact/ogc-detector.ts`, class `OgcContactDetector`:

```ts
const contact = await OgcContactDetector.create(device, { vertexCount, triangles, edges }, restPositions, config, { positions, velocities });
```

**Inputs from C1:**
- `positions`: `array<vec4<f32>>`, cloth-local metres; `w` is C1's. The detector reads it, and truncation writes xyz in place, keeping `w`.
- `velocities`: `array<vec4<f32>>`, the step-start velocity, used for the envelope.
- `triangles`: `vec4<u32>` (xyz vertices, w material).
- `edges`: `vec4<u32>` (v0, v1, face0, face1, with `0xffffffff` for a boundary edge), in `buildOgcEdges` order.
  - If C1 builds its own edges, it must use the same canonical order, or pass `buildOgcEdges(triangles)`.

**Config** (`OgcContactConfig`, every capacity authored per scene):
- `r`, `kc`, `mu`, `epsV`, `dt`, `gravity`;
- `rqBase`, `rqMax`, `envelopeScale` (s);
- `truncation: "planar" | "isotropic"`, `gammaR` (0.9), `gammaP` (0.45), `gammaE` (0.01);
- `contactCapacity` (per vertex), `pairCapacity` (DAT pairs), `gridCapacity`, `bucketCount` (a power of two), `roundCap`.

**WGSL for C1's per-vertex kernel:** `OgcContactDetector.vertexBlockWGSL(group, binding)`.
- Bind `contact.solverBuffer` at (group, binding). It is one storage buffer, read-only.
- Define these before the block:
  - `ogcX(i)`: the current position;
  - `ogcXt(i)`: the position at step start, for friction;
  - `ogcTri(t) -> vec3<u32>`;
  - `ogcEdgeVerts(e) -> vec2<u32>`.
- The block provides:
  - `ogcAccumulateVertexContact(v, xv, &f, &H)`: the OGC normal force with the two-stage activation, and IPC lagged friction with VBD's stabilized Hessian. Both are added into C1's f and H, and the Hessian is PSD.
  - `ogcVertexContactEnergy(v, xv)`: this vertex's contact energy, for C1's line search. **C1's per-vertex Armijo line search must include it.** This is the f32 blow-up guard from the plan's risk list.
- Truncation is **not** in the block. Planes need a consistent snapshot of every pair's four vertices, so it is its own pass.

**Encode order per step**, all in compute passes, with no readback:

```
contact.encodeStepBegin(pass); contact.encodeDetection(pass);   // the mandatory round at x_t
(C1 initial guess)            contact.encodeTruncate(pass);
for each iteration k:
  for each colour c:  C1 colour sweep;   contact.encodeTruncate(pass);   // DAT §5.1: truncate after every colour
  contact.encodeDecide(pass); contact.encodeDetection(pass);             // zero workgroups unless armed
contact.encodeIntersectionCount(pass)                                    // gates only, once per frame
```

`encodeFrameBegin` resets the per-frame statistics. Call it once per frame before the step.

**Why truncate after every colour, not every iteration:** later colours evaluate the contact energy at earlier colours' positions. Those positions must be penetration-free, or the distance-based energy pushes the wrong way. Each `encodeTruncate` is two dispatches:
- planes: one thread per pair, indirect;
- apply: N threads.

**Commit semantics** (DAT Alg. 3):
- The positions buffer always holds X + ΔX, with X = `contact.det.xyz`.
- ΔX accumulates over C1's iterations, and every `encodeTruncate` cuts the accumulated ΔX.
- An armed detection round's prep copies positions into X. That copy is the commit.
- C1 must not reset positions to x_t after a round.

**Status:** `readCounters()` is for the harness and gates only.
- `status` is OR-ed bits (`OGC_STATUS`): contactOverflow, gridOverflow, roundCap, envelopeCap, nonFinite, pairOverflow, uncertifiedPair. All are explicit failures; nothing is dropped silently, and nothing grows.
- The peaks (`peakContacts`, `peakPairSlots`, `peakGrid`) are reported, so scenes can author their capacities.

## 2026-11-09 DECISION: Planar-DAT, as implemented

Implemented from the paper (§3.3, Alg. 2–3, §5.1) and checked against Newton's `place_dat_division_plane` and `planar_truncation_t` (Apache-2.0). The WGSL is my own.

**Pairs.** At detection, every non-adjacent vertex–triangle pair and every non-adjacent edge–edge pair (e < e′) within max(r_q,A, r_q,B) is recorded. A face's or an edge's envelope is the widest of its vertices'.
- The grid query already gathers the superset: triangle boxes are inflated by their own envelope, and the query box by the querying primitive's.
- Each pair stores its four vertex ids and a certified plane normal n at X.
  - n is the closest-point axis. If its support gap is below 2·sepEps, the better of ± the face normal or the edge cross product is used.
  - A pair with no positive gap is stored as uncertified. Its four vertices are held (t = 0) and the `uncertifiedPair` status bit is set. Newton instead leaves such pairs unconstrained; I chose the safe failure, made explicit. Under OGC's offset energy it should never fire.
- Edge–edge pairs include endpoint-closest configurations. DAT needs every primitive pair, not only OGC's interior edge–edge contacts.
- The OGC energy lists are unchanged: d < r, with feature dedup.

**Division (eq. 11), placed as Newton places it.**
- Each side's share of the support gap is proportional to **its own** largest approach along n: λ_B = δ_B / (δ_A + δ_B), with the plane at supB + λ_B·gap.
- Each side keeps at least max(5 %, sepEps/gap).
- The extracted paper text reads as room proportional to the *other* side's approach. Read that way, a vertex approaching a static face would freeze. Newton's code and the paper's Fig. 3 behaviour both match the own-approach reading.

**Truncation (eq. 16 with Newton's endpoint rule).**
- s0 is the signed distance to the plane minus sepEps. A wrong-side start may approach its side but not retreat further.
- The ratio is γ_r·s0/(s0 − s1), and per-vertex t_v is an atomic min (Alg. 2).
- sepEps = max(1e-6 m, 4·2⁻²³·coordinate scale), as in Newton.

**Cap.** |t_v·ΔX_v| ≤ 0.5·γ_r·r_q,v for **every** vertex, as in Alg. 2 line 10 and Newton's `apply_truncation_ts`. The coordinator wrote "vertices with no pair". A paired vertex still has unpaired primitives beyond its pairs, and the max-rule guarantee needs both of their caps. So the cap applies to all vertices.

**Isotropic baseline.** `truncation: "isotropic"` is OGC's b_v = γ_p·min(d_min, d^T, d^E), clipped at r_q,v. It records no pairs and dispatches no planes.

**Where the cost goes.**
- **The envelope must be at least 2/γ_r ≈ 2.2 times the step's motion** for an unobstructed vertex to move in one round. A free vertex whose envelope is r + s·|Δ| keeps min(|Δ|, 0.45·(r + s·|Δ|)).
  - At s = 1.5 that is 0.68·|Δ| per round, so free flight needs a second round.
  - s ≥ 2.5 avoids that, at the cost of more pairs.
  - The gates will report both settings.
- **Co-moving layers cascade.** Planes are fixed at X between rounds, and each vertex is truncated independently. So a layer stacked on a layer moving the same way can travel only about γ·gap per round until the layer below commits.
  - A k-layer stack falling together needs on the order of k rounds. No static-plane scheme with independent per-vertex truncation avoids this. Requiring a vertex to move with its neighbour is a coupled problem.
  - The 10-layer drop will show it, if it is there. The fix to evaluate then is gaps that open before the drop (authored spacing ≥ the per-step fall), not a hack.
- **Pairs scale with the envelope squared.** A sheet vertex pairs with every non-incident face of its own sheet within r_q.
  - At 2 m/s, r_q ≈ 50–80 mm against a 10 mm edge, which is O(10²) pairs per vertex.
  - This is the pair-count cost the coordinator asked to measure. The planes pass is one thread per pair.

## 2026-11-09 MILESTONE: M2 standalone modules and Float64 oracle (Planar-DAT)

**Files:**
- `lib/cloth/contact/ogc-topology.ts`: CSR stars, the face-edge table, face0 ownership, packed into one u32 buffer. It validates the input and throws on non-manifold or out-of-range data.
- `lib/cloth/contact/ogc-wgsl.ts`: the shared WGSL geometry, the activation and friction, the counters and status bits, and C1's vertex block.
- `lib/cloth/contact/ogc-detector-wgsl.ts`: the detection, bound, planes, apply, decide and intersection kernels.
- `lib/cloth/contact/ogc-detector.ts`: the host. Pipelines are created asynchronously in `create()`, and uploads go through `writeBufferChunked`.

**Detection round:** nine indirect dispatches.
- The sparse hash grid uses a counting sort with a 3-level scan, and the insert pass consumes the counts back to zero, so nothing is cleared. There is no BVH.
- OGC Algorithms 1–2 are exact: lowest-corner cell dedup, dynamic feature ownership, and canonical edge–edge evaluation.
- d_min uses u32-ordered atomic mins.
- Fixed-capacity per-vertex contact lists, and DAT pair slots reserved four at a time per thread, with holes marked.
- The GPU chooses the cell: h = max(mean edge, widest envelope).

**Re-detection:** decide arms the next round's indirect args when the truncated count is at least γ_e·N, up to `roundCap`. Past the cap it sets the status bit and dispatches nothing.

**Intersection counter:** exact orientation predicates on non-adjacent face pairs, through its own grid build over the current positions.

**Tests:**
- `tests/cloth-contact-oracle.test.ts`: 11 tests, all CPU, and part of `test:unit`.
  - FD checks of the energy, force and Hessian; PSD checks.
  - Algorithm 1–2 set semantics.
  - **The DAT condition:** every unpaired primitive pair is further apart than its two caps.
  - **Penetration-freedom:** a Float64 continuous brute force (coplanarity cubic, bracketed and bisected) finds no penetration after planar or isotropic DAT on large random motions. The raw motions do penetrate.
  - Planar keeps more motion than isotropic; it leaves tangential sliding and separation untouched where the sphere damps them; it splits the gap between approaching layers by their approaches.
- `tests/cloth-contact-gpu.test.ts`: 7 GPU tests against the oracle.
  - Contact pair sets: no misses, no duplicates, every participant's list exactly once.
  - Bounds with velocity envelopes.
  - **DAT pairs equal the brute-force max-rule set, each once, with certified planes.**
  - The vertex block's force, Hessian and energy.
  - **GPU Planar-DAT matches the oracle, and the truncated motion is penetration-free by the continuous brute force.**
  - Isotropic truncation with GPU-armed re-detection and the round cap.
  - The intersection counter against the brute force.
  - Envelope cap, contact overflow and pair overflow raise their status bits.

**Evidence:**
- `npm run check:types`: green.
- `tools/gpu/fluid-gpu-run.sh check … node --import tsx --test tests/cloth-contact-gpu.test.ts`: 7/7 pass. The runner exited 0, so the result is valid.
- `npm run test:unit`: 1624 pass, 1 fail. The failure is the pre-existing `reservoir-particle-cost` "merge cap…" test, the same one C1 records on the untouched baseline.

**Next:**
- Integrate through C1's M1 interface once it is published. Their progress file has M0 so far.
- Run the M2 gates at 1/60, one step:
  - twisting cloth, about 10K vertices;
  - 10-layer drop;
  - planar vs isotropic, and fixed vs adaptive envelope.

## 2026-11-09 COORDINATION (C2 → C1): binding the detector to the solver

Read C1's INTERFACE (Tier A layout, contact at `CLOTH_BINDING.contact` = 7). Nothing in C1's files needs to change. On C2's side:

- **Two-phase build.** The solver needs `detector.solverBuffer` at `create`, and the detector needs `solver.positions` and `solver.velocities`, so neither can be built first. `OgcContactDetector.create(device, mesh, restLocal, config)` now compiles without solver buffers, and `detector.bind({ positions: solver.positions, velocities: solver.velocities })` attaches them once. Inputs may be a `GPUBuffer` or a `GPUBufferBinding`. Passing inputs to `create` still works.
- **`detector.solverContact(CLOTH_BINDING.contact)`** returns `{ wgsl, buffer }` for `ClothGPUSolverOptions.contact`.
- **`detector.stepHooks`** is the `ClothStepHooks` object:
  - `afterFrame`: stepBegin, detection, truncate;
  - `afterColour`: truncate;
  - `afterIteration`: truncate, decide, indirect detection.
- **Contact header.** It gained a third vec4: `[2].x` is the solid image's word offset (0 means no solid). The counts base moved to 3. C1's hand-built empty buffer `(0, 0, 2, 2)` still reads as no contacts and no solid, because its third vec4 is zero.
- **Energy domain.** A vertex-solid entry returns `+inf` energy for a trial position at or inside the solid surface (φ ≤ 0), with zero force and Hessian there. The Armijo test then rejects the trial. The iterate itself never gets there, because truncation keeps φ ≥ (1−γ_r)·φ₀.

## 2026-11-09 DECISION: static voxel solid, as implemented (M5 part 1)

**Energy: activation on φ, not φ − r.** `g(φ; r)` from the cloth's own activation:
- the barrier sits at the solid surface;
- repulsion starts at φ = r;
- resting cloth settles at a φ inside (0, r), as cloth-on-cloth settles at a d inside (0, r).

With φ − r as the gap, the barrier would sit r off the surface, and the plan's bound γ_p(φ(x_det) − r)/L_b would be ≤ 0 for every resting vertex: it would freeze them.

The energy terms:
- force −g′∇φ;
- Gauss–Newton Hessian g″∇φ∇φᵀ, which is PSD (the g′∇²φ term is dropped);
- IPC lagged friction against the fixed solid. λ = −g′(φ(x_det))·|∇φ| is stored in the entry. The tangent is taken at the current ∇φ, as the cloth-cloth path takes its current normal.

**Truncation: an affine lower bound of φ instead of the isotropic φ/L_b ball.**
- **The bound.** At detection, for each vertex with ball radius ρ = its cap x_det.w (≤ rqMax/2), the bound pass takes g₀ = ∇φ(x_det) and J = the largest deviation |∇φ − g₀| over the cells the ball's box touches. Per axis, ∂φ/∂x_a over a trilinear cell spans exactly its four a-edge differences. Integrating along the segment gives, for |d| ≤ ρ, φ(x_det + d) ≥ φ₀ + g₀·d − J|d|.
- **Apply.** dX ← dX / max(1, j|dX| − q·dX), with q = g₀/(γ_r φ₀) and j = J/(γ_r φ₀) stored per vertex at detection. The guarantee is φ ≥ (1−γ_r)·φ₀ along the whole truncated motion. The ratio is linear in t, so it is exact, not iterated.
- **Why not the ball.** On a flat face J = 0, so sliding is untouched and an approach keeps (1−γ_r) of the gap, like Planar-DAT. At sharp box edges and corners J grows, and the cut approaches the isotropic ball, which is what a corner needs. The isotropic ball φ₀/L_max would have cut a resting 0.4r slide to under a tenth (unit test). L_b is still computed per brick at upload; the image carries L_max for the far test.
- **The far test**, which decides whether a vertex gets no entry and no cut:
  - min(φ₀, band) − L_max·ρ ≥ r; or
  - the ball's box touches a missing brick, where φ > band.
  
  Both are sound because `setSolidField` refuses a band below r + 1.5·L_max·rqMax.
- **Penetration at detection.** A vertex at or inside the surface when detection runs (φ₀ ≤ 0, for example after an edit adds solid on top of it) is held, and raises `solidPenetration`.
- **Leaving the field.** A vertex outside the field's authored box raises `solidBounds`. The field only sees solid inside its box.

**Field: a cloth-local derived image, not the scene's GPU solid buffer.**
- `solid-field.ts` builds 8³-cell bricks of 9³ node distances in cloth-local metres from any 1-Lipschitz source. The bricks kept are those within `band` of the surface, inside an authored box.
- The image is appended to the contact buffer, so C1's kernel keeps a single contact binding (the M1 Max allows 10 storage buffers per stage).
- Solid edits rebuild only the bricks within band of the edit (`rebuildClothSolidField`, CPU at edit time) and re-upload.
- The scene's own GPU solid image does not fit directly. Its procedural bricks carry validity masks and arbitrary implicit units, and its pages carry ±½-cell sign distances (box regions write −128/+0x7fff Q8). It is not a distance field the bound can use.
- Still to do in M5: the adapter from a scene's solid world (pages, regions, procedural bricks) to this source.

**Thin features fail at authoring.** Only cells the surface can cross are sampled: every node |φ| < √3h. A 1-Lipschitz node beyond that proves the cell one-signed. Sampling is at h/8. Where the source and the interpolant disagree in sign, an ascent of |φ| from the sample separates a rounded sharp corner from a lost feature:
- a corner belongs to a part at least a cell thick, and reaches depth 0.45h within two cells;
- a plate, rod or slot thinner than about 0.9h never does, and the build throws.

The tests check a 0.3h plate (throws), a 0.3h slot (throws), and an off-lattice sharp box (accepted).

## 2026-11-09 MILESTONE: M5 part 1, static solid field, vertex-solid contact and truncation (standalone)

**Files:**
- `lib/cloth/contact/solid-field.ts`: build, local rebuild, Lipschitz, the thin-feature check, and the CPU sampler.
- `lib/cloth/contact/ogc-wgsl.ts`: `OGC_SOLID_WGSL` (sampler and ball bound), and `ogcSolidEval` in the vertex block.
- `lib/cloth/contact/ogc-detector-wgsl.ts`: solid round in the bound pass, and the solid factor in apply.
- `lib/cloth/contact/ogc-detector.ts`: `solidCapacityWords`, `setSolidField`, `bind`, `solverContact`, `stepHooks`, and the counters `solidContacts` and `solidCut`.

**Tests:**
- `tests/cloth-contact-solid.test.ts`: 6 CPU tests.
  - Band and sparsity; L_max ≤ √3; missing bricks lie beyond band; exact on node-aligned faces.
  - Thin plate and slot refused; an off-lattice corner accepted.
  - A local rebuild after a cut equals a full rebuild.
  - **The ball-bound certificate against dense sampling** at sharp box edges, corners and a sphere: 300 balls × 120 samples.
  - **The truncation keeps φ ≥ (1−γ_r)·φ₀ along the whole motion** (65 samples per motion). A flat-face slide of 10× the gap is untouched, and an approach keeps exactly (1−γ_r)·φ₀.
  - FD force of the vertex-solid energy; PSD Hessian; +inf inside; friction direction.
- `tests/cloth-contact-gpu.test.ts`: 2 new GPU tests, 9 in total.
  - Solid entries (presence, λ) and the truncation terms (q, j) against the oracle per vertex.
  - The vertex block with solid and friction against the oracle.
  - **A motion driving a sheet 0–6 mm into a sharp box and a sphere while sliding 8 mm:** GPU equals oracle (Planar-DAT, then cap, then solid), φ ≥ 0.1·φ₀ for every vertex, and no self-penetration.
  - The penetration hold; `solidPenetration` and `solidBounds`; refusal of a thin band and of an over-capacity image.

**Evidence:**
- `npm run check:types`: green.
- GPU runner: 9/9, exit 0.
- `npm run test:unit`: 1630 pass, 1 fail, the same pre-existing `reservoir-particle-cost` "merge cap…" test.

**Next in M5:**
- solid-surface point and edge samples against cloth faces and edges (sharp-edge poke-through);
- the scene solid-world adapter;
- the gates (sphere drape, sharp box, cut under resting cloth) through C1's solver, once C1's M1 lands in git.

## 2026-11-09 MILESTONE: M2 integrated in C1's Tier A solver; both gates pass the safety checks

The harness is `tests/cloth-contact-gates.test.ts`. It runs only on the GPU and only when asked: `CLOTH_CONTACT_GATES=drop,twist|all`, plus `CLOTH_GATE_STEPS` (default 60).

- **Wiring:**
  - `OgcContactDetector.create` (no inputs);
  - `ClothGPUSolver.create({contact: det.solverContact(CLOTH_BINDING.contact), hooks: det.stepHooks})`;
  - `det.bind({positions, velocities, stepStart})`;
  - `setSolidField`.
- C1's line search already includes `ogcVertexContactEnergy`.
- Each frame is checked for: GPU intersection count 0, no failing status bit, and no vertex inside the solid (checked every 10 steps on the CPU). One timed batch of 30 steps follows the gate.

**Gate results.** Both use kc 100, s 2.5, truncation per colour and 20 iterations.

| | drop | twist |
|---|---|---|
| Scene | 10 × 31² sheets, each turned 7° from the last, onto a voxel pedestal on a floor slab | 100² sheet (OGC Fig. 12); edges turn ±π/2 rad/s and move together 0.1 m/s |
| Vertices | 9610 | 10000 |
| Steps | 60 | 60 |
| Intersections | 0 every frame | 0 every frame |
| Vertices inside solid | 0 | none (no solid) |
| Rounds per step | 2.18 | 3.88 |
| Pair slots, peak | 1300N | 950N |
| Contacts per vertex, peak | 74 of 128 | 47 of 128 |
| Envelope-cap hits | 8 | 43912 |
| Residual, end | about 10 | 5e5 |
| Time per step | 301 ms | 321 ms |

**Twist residual.** The large twist residual comes from truncation: about 7000 cap truncations per iteration, and a truncated iterate is not a stationary point. The twist without contact (`CLOTH_GATE_NO_CONTACT=1`) self-intersects at step 15, so contact is needed.

**Fixes made while gating:**
1. **The mandatory round now detects at x_t.** C1's frame kernel writes its initial guess into `positions` before `afterFrame`. Detecting there treated the guess as safe, and the drop went 4 mm into the pedestal at step 2.
   - The mandatory round now reads `stepStart` (C1's x_t segment).
   - A re-detection still reads `positions`.
   - `OGC_ROUNDS == 0` selects which.
2. **The planes dispatch exceeded 65535 workgroups beyond 4.19M pair slots,** and those pairs were never truncated: 276 real intersections in the drop at s 2.5.
   - The planes pass is now grid-stride, with its indirect count clamped to 65535.
3. **The GPU intersection counter gave f32 false positives** on near-coplanar non-adjacent faces of a flat sheet (GPU counted 15, Float64 counted 0).
   - Orientations now carry a forward error bound, 16u times the permanent.
   - A pair whose signs are undecided counts only if its triangle distance is within 128u times its coordinate scale.
   - The harness re-checks any GPU hit with a Float64 count.

**Sensitivity of the drop at s 1.5, 30 steps:**

| kc | Truncation | Residual | Rounds per step | Time per step |
|---|---|---|---|---|
| 1e4 | per colour | 6e2 to 1.6e3 | 4.5, at the round cap | 307 ms |
| 1e2 | per colour | 60 to 140 | 2.2 | 228 ms |
| 1e2 | per iteration | 30 | 2.2 | 206 ms |

Truncation is not the main cost; detection rounds and pairs are. kc 100 holds ten layers at under 1e-4 r per vertex, so kc 1e4 bought nothing but conditioning trouble.

## 2026-11-09 DECISION NEEDED: the envelope factor and the pair count at speed

The evidence:
- **s must exceed 2/γ_r = 2.22.** A round lets a vertex move 0.5 γ_r r_q = 0.45 (r + s·d) of its inertial displacement d.
  - At the decided s ≈ 1.5 that is 0.675 d + 0.45 r, so every vertex in free flight is capped every step. In the twist at s 1.5, 19553 caps fell in step 0.
  - The gates now run at s 2.5.
- **Pairs scale as (s·|v|·dt / ℓ)², even when everything moves together.** A DAT plane is fixed for the step, and the envelope is absolute. So ten layers in free fall, and neighbouring faces of one sheet, pair up as soon as s·|v|·dt exceeds the edge length.
  - The drop at s 2.5 peaks at 1300N pair slots, which is 12.5M slots and 400 MB.
  - Time per step is 300 ms for 10K vertices, mostly detection.

Options:
1. **(a) Keep the absolute envelope at s 2.5.** Author pair slots of about 1300N from the measured peak, and accept about 300 ms per step at 10K vertices.
2. **(b) Use s 1.5 and accept extra rounds.** The vertex cap then lags free fall.
3. **(c) Use predicted-motion DAT (my recommendation).** Cap each vertex relative to its inertial prediction w_v = v·dt + g·dt², so that |dX − w_v| ≤ c_v. Record a pair when d < c_a + c_b + |w_a − w_b|.
   - The plane of a pair moves with the pair's mean prediction. The constraint stays linear at the endpoint, with offset c + n·w̄, so the DAT argument carries over unchanged.
   - Co-moving primitives (falling layers, a sheet's own neighbours) then pair only within r_q base, and pair counts fall to the at-rest level of about 5N.
   - The cost is about one vec4 per vertex and one per pair.

I am not implementing (c) before review.

## 2026-11-09 COORDINATION (C2 → C1)

- Please add `get stepStart(): GPUBufferBinding` (segment `CLOTH_STATE.xt`). The harness builds it from `solver.state` and `solver.tables.stride` for now.
- The detector now binds it at prep binding 13. A detector bound without `stepStart` reads `positions`, which is right only when they hold x_t, as in the standalone tests.

## 2026-11-09 COORDINATION (C2 → coordinator, Lane S): answers to the 2026-11-09 requests

**(2) Per-solid friction.**
- The field's records now carry a byte per node, μ × 100 over 0 to 2.55, taken from the part nearest the node.
- The bound pass writes μλ into the solid entry: μ comes from the node nearest x_det, λ is the lagged normal force.
- The vertex block uses that product directly.
- Cloth-on-cloth uses `OgcContactConfig.mu`, which is separate.
- A field given by a single `sdf` takes `mu` (default 0, frictionless).
- Lane S: give each solid its own part with its μ.

**(3) Field build time.**
- The thin-feature probes are now an octree. A sub-box is cleared when the source is one-signed on it (the 1-Lipschitz ball) and the multilinear interpolant agrees at its corners.
- `ClothSolidSource.parts` builds a union:
  - `{kind:"sdf", sdf, lo, hi, mu}` is evaluated only for bricks within band of its box, and only its own surface is checked;
  - `{kind:"lattice-box", lo, hi, mu}` must have faces on node planes. The interpolant is exact on it, so it is never checked, which suits floors and slabs. A misaligned lattice-box throws, and so does an unknown kind.
  - band ≥ √3 cells is required.
- Measured: a 2.8 m square floor (lattice-box) plus a 0.25 m ball (sdf) at 1 cm with a 6 cm band builds 4259 bricks in **405 ms**. The same solid as one sdf takes 6 s, and 25 to 40 s before the octree.
- The drop pedestal, with 1980 bricks at 5 mm, builds in 86 ms.
- What it does not refuse: a gap between two parts that the lattice closes. The solid only grows there, never shrinks; only fidelity is lost.

**(4) The resolution rule, now explicit (`SOLID_RESOLUTION`).**
- Where source and interpolant disagree, the probe is accepted if some point at least 0.45 cells deep on its own side lies within 2 cells of it, joined by a straight same-side segment. The gradient ascent stays as the fast path; the ball search catches the zigzag in acute wedges.
- Plates, rods and slots under 0.9 cells always fail.
- Convex edges: measured over 4 lattice phases × 4 orientations, 35° and above passes everywhere, and 20° and below fails everywhere. 25° to 30° depends on the phase.
- The error message states the rule. A test sits at the thresholds, together with a free-standing 67° frustum rim, which passes.
- **Lane S: this flips your assertion** at `tests/cloth-bridson-2002.test.ts:166`. The bare frustum, with footing 0, is now accepted. The footing can go, and the assertion should become a `doesNotThrow`. That test is the one red in the unit suite that comes from C2's change.

**(1) Moving solids.** Noted: body-frame voxel bakes of `RigidBodyDescription` shapes, posed by RigidPoseTrack, with surface velocity from the twist. This is M6 step 1, next.

## 2026-11-09 DECISION NEEDED: M6 moving solids, design (one-way, then two-way)

### One-way design (step 1)

**Data.**
- Each moving solid is a `RigidBodyDescription` with prescribed motion (RigidPoseTrack or `prescribed-rigid-motion`).
- Its own shape, including cuts, is baked once in the body frame into a `ClothSolidField` image, using `parts` with the solid's μ. A bake is redone only when that body is cut.
- Images sit after the static image in the contact buffer's solid region, behind a table of K entries: image word offset, plus pose slots.
- Per frame, the frame owner writes each body's poses (c₀, R₀) at the step start and (c₁, R₁) at the step end, plus its twist (v, ω), with one `writeBuffer`. That is an input upload, like C1's attachment time. No readback, and no CPU stage in the step.

**Contact.**
- φ_b(x) = φ_body(R₁ᵀ(x − c₁)), with gradient R₁ ∇φ_body.
- Energy and Hessian are as for the static entry.
- Friction acts on the slip relative to the surface: u = (x − x_t) − (v + ω × (x − c₁))·dt, projected tangentially. The Fig. 2 spinning sphere with μ = 1 needs exactly this term.
- The entry stores (v, image b, μλ).

**Truncation.**
- Relative to the body, a vertex's linear path runs from y₀ = R₀ᵀ(x₀ − c₀) to y₁ = R₁ᵀ(x₀ + dX − c₁).
- The ball bound around y₀ is the static one, unchanged.
- The cut must interpolate towards the displacement that carries the vertex with the body, a_v = R₁y₀ + c₁ − x₀, not towards 0. Holding still is unsafe when the body advances.
  - So dX ← a_v + t(dX − a_v), with t from the bound on R₁ᵀ(dX − a_v).
- The envelope must also cover the body's own motion of the point: r_q,v ≥ r + s·|a_v| for vertices within reach of a body.

**Why this is a decision.** Every other cut (DAT planes, the 0.5 γ_r r_q cap, the static solid) interpolates towards dX = 0.
- A vertex pushed by a body while it is paired with cloth or a static solid across the push gets contradictory cuts. Cloth pressed between a moving sphere and a table is the Bridson table scene.
- Sequential cuts cannot satisfy both, and holding is unsafe.
- Predicted-motion DAT (option (c) above) removes the conflict. Take the vertex's prediction w_v as a_v when a body carries it, and as the inertial prediction otherwise. Every cut is then relative to w_v, and planes move with the pair's mean prediction.
- So the two decisions are one: with (c), M6 one-way is the per-image table above plus w_v = a_v. Without (c), M6 can only flag the conflict (a failing status `solidConflict`), and a scene that presses cloth against anything with a moving body is refused.

### Two-way design (AVBD 6-DOF blocks; not implemented, for review)

- Each dynamic rigid is one AVBD block: a 6×6 Newton solve per iteration, coloured after the cloth colours. The cloth-body contacts couple a body to every cloth vertex in its entries.
- **Body solve.**
  - Inertia: the body's mass and inertia about the inertial target.
  - Contacts: for each entry (v, b), a force −λ_c n at the contact point p = R₁y_v + c₁, with lever arm p − c₁, as an AVBD hard constraint with dual λ_c and stiffness ramp. That is the same activation energy, with the cloth vertex frozen in the body's block, plus friction from the slip.
  - The accumulation is a GPU reduction over the body's entries: a per-body list built in the bound pass, with u32 atomics on fixed-point partial sums, or a segmented scan by body. WGSL has no float atomics.
  - The solve is one thread per body.
- **Cloth solve.** Unchanged, with the body's current iterate pose. Truncation is relative to the body's iterate motion, which is (c) again.
- **Box3D.** In `lib/interaction/adapters/box3d/world.ts`, bodies touched by cloth are integrated by the cloth step for that frame, and their pose and twist go back to Box3D on the GPU→CPU frame boundary the scene already has. Otherwise this is CPU control flow: a Box3D sync is a frame-level exchange, not a step stage, and it needs the coordinator's ruling.

### Remaining M5 work

- **Solid-surface point and edge samples against cloth faces and edges.** These are vertex-facet and edge-edge entries plus DAT pairs, with the static samples binned per brick.
  - Without them, a sharp voxel edge or corner can poke through a cloth face between its vertices. Vertex contact keeps vertices out, not faces.
  - The intersection counter does not yet test cloth faces against solid surfaces.
- **The scene adapter:** solidVoxels and procedural bricks into `parts`, using exact box-union distance for voxels.
- **Gates still to run:** sphere drape, sharp box, and a cut under resting cloth.

## 2026-11-09 MILESTONE: predicted-motion DAT (ruling c), the oracle proof, and the detection breakdown

### What is implemented

- **Prediction.** Each vertex has a prediction w_v for the round. `prediction` selects it:
  - `"guess"`: the solver's own initial guess minus x_t. Pins get their scripted motion, and resting cloth gets about 0 through C1's adaptive gravity. The gates use this; it needs `stepStart`, and `bind` throws without it.
  - `"inertial"`: v·dt + g·dt². This is the default, for the standalone tests.
  - `"none"`: Planar-DAT, with every prediction held.

  The BDF1 inertial prediction is wrong for C1's BDF2. Its free-fall step is about 1.67 g dt², not v dt + g dt². That error alone capped most vertices of the twist.
- **Envelope.** A free vertex covers only its residual, r_q = clamp(max(rqBase, r + s·ρ), rqMax).
  - ρ is last step's deviation, |x_t − ŷ_prev|; on the first step ρ = |w|.
  - Its residual δ = x − X − w is truncated by the planes and then capped at 0.45 r_q. s = 2.5, above 2.22.
- **Planes** keep the normal found at X and translate along the predictions: they are placed among X + w.
- **A pair is recorded** when its swept boxes conv(X, X+w) ± r_q meet and d ≤ max r_q + spread. The spread is the largest |w_i − w_j| across the two sides.
- **Held vertices** get w = 0 and the Planar-DAT envelope r + s·|w|. A vertex is held when any of these holds:
  - it was in contact, truncated or in a conflict in this step or the last;
  - its prediction saves less than half its motion (2ρ > |w|);
  - it is in a redone round, in isotropic mode, or under `"none"`.
- **Conflicts.** A pair is a conflict when some w ≠ 0 and either X has no separating plane, or X + w closes below 2·sepEps (and below the gap at X).
  - On a conflict, the redo kernel reruns the whole round at once with every prediction held, before any truncation. That round is Planar-DAT, which is proven.
  - The conflicting vertices stay held into the next step.
  - This is what stops cloth squeezed between converging layers from getting contradictory cuts.
- **Static solid.** The solid cut is relative to w: room = 1 − f(w), with f(d) = j|d| − q·d.
  - Prep stops a prediction at the solid (w /= f when f > 1) and limits |w| ≤ ½(rqMax − r_q), so the field's band still covers the round.

### Proof (Float64 oracle, `tests/cloth-contact-oracle.test.ts`)

Three new tests:
- **A pair with different predictions.** Vertex–triangle and edge–edge pairs whose vertices have different inertial predictions stay divided over the whole step. Checked with CCD and 2048 samples.
- **Random rounds with per-vertex layer drifts.** The truncated motion is penetration-free. Every unpaired pair is out of reach, or its swept boxes are apart.
- **Co-moving layers.** These pair at under 10% of the absolute-envelope count, and their vertices move whole. Crossing predictions are conflicts; held predictions never are.

Positions are linear in t, so a vertex on its side of a translating plane at both ends of the step stays on it throughout.

GPU against the oracle (`tests/cloth-contact-gpu.test.ts`, 11 of 11 pass), two new tests:
- **Predicted pairs, translating planes and truncated residuals** match `predictedTruncate`.
- **Crossing predictions** redo the round held, and that round's pairs equal Planar-DAT's.

### Detection speed (ruling 2): measured, and the cell ruling reversed by data

Per-pass GPU timestamps come from `profileRound` in the gates: one mandatory round at the end state, each kernel in its own pass, median of 5.

**Query is 97–99% of a round in every configuration.** Prep, cell, count, scans, insert, bound and planes are each at or below the timestamp floor of about 0.07 ms. The exception is the planes pass of the big Planar-DAT pair sets, at 0.5–0.7 ms.

**Cell size (60 steps, guess prediction, query ms per round):**

| Cell | drop | twist |
|---|---|---|
| Cubic mean (about 1 cm) | 11.9 | 44.1 |
| 4 cm | 10.5 | 11.3 |
| 6 cm | 12.5 | 13.0 |
| 10 cm | 16.1 | 14.7 |
| Largest envelope, the old rule (kept) | 10.2 | 9.8 |

- Cells sized from the typical or cubic-mean radius, with multi-cell insertion, are slower.
- A query's cost is per cell visited: three scan reads for each bucket bound, plus scanning hash-colliding entries. That outweighs the candidates a larger cell adds; candidates/N per step is the same at every cell size.
- The default stays at the largest envelope. `cellRule: "cubicMean"` stays in for re-measurement, and `cellSize` is the floor.

**Grid entries carry the face's round box** (8 words: face, key, lo, hi). A query no longer re-derives it with 12 scattered loads.

**Before and after (60 steps, s 2.5, kc 100, truncation per colour, 20 iterations):**

| | drop, HEAD 144e248d | drop, `"none"` now | drop, `"guess"` now | twist, HEAD | twist, `"none"` now | twist, `"guess"` now |
|---|---|---|---|---|---|---|
| ms per step (timed 30) | 301 | 272 | **115** | 321 | 307 | **209** |
| Solver alone, ms per step | — | 27 | 28 | — | 12.6 | 13.5 |
| Query ms per round (end state) | — | 51.3 | **10.2** | — | 40.2 | **9.8** |
| Rounds per step | 2.18 | 2.12 | 3.28 | 3.88 | 3.87 | 5.83 |
| Steps redone held | — | 0 | 52 | — | 0 | 54 |
| Candidates/N per step | — | 7605 | 4767 | — | 8159 | 9183 |
| DAT pairs/N per step | — | 705 | 420 | — | 1083 | 1464 |
| Peak pair slots | 1300N | 1287N | **809N** | 950N | 950N | 776N |
| Intersections, vertices in solid | 0, 0 | 0, 0 | 0, 0 | 0 | 0 | 0 |

- **Per-vertex counts** at drop step 0: 5.5 pairs and 680 candidates. During the pile-up (step 10), the guess run has 1378 pairs and 8409 candidates; it settles to 85–130 pairs and 1600–2300 candidates.
- Twist step 0 has 81 pairs and 716 candidates. Its peak is 1786 pairs and 10858 candidates (step 30).

### What did not work, honestly

- **Twist rounds rose from 3.9 to 5.8 per step**, with round-cap hits in 846 of the step's rounds.
  - The twist's strain-limited sheet stops abruptly as it goes taut, so its deviation from any inertial predictor is about its whole motion, about 15 mm per step.
  - Its residual then grows faster than the 1.125 ρ_prev the cap allows, and the residual (5e5 to 3e7) was already pathological at HEAD (5e5).
  - The step time still fell (321 to 209 ms), because a round got 4× cheaper.
- **The held redo runs on about 90% of steps** in both gates. It is a full extra round, about 10 ms: new vertices keep closing on held ones as the pile grows.
  - Cheaper alternatives are unsound. Rescaling w after the query changes the spreads of pairs that were not recorded.
  - A partial re-query of only the changed vertices' neighbourhoods would cut it. That is the next speed item.
- **Remaining step cost in the drop:** 115 ms − 28 ms solver ≈ 87 ms of contact. About 4.2 rounds × 10 ms is detection; the rest is per-colour truncation (planes plus apply, 20 iterations × colours).
  - Truncating per iteration (`CLOTH_GATE_TRUNCATE=iteration`) is C1's hook choice. It earlier measured 10% less.
- **The 4 ms target is not in reach on this path.** The solver alone is 13–28 ms per step at 10K vertices and 20 iterations.

## 2026-11-09 STATUS: rulings 3 to 6

- **(5) Bridson frustum: done.**
  - The footing is removed, and the frustum's base sits on the ground at y = 0 as in the paper.
  - `tests/cloth-bridson-2002.test.ts` now asserts `doesNotThrow` on the contact field build. The frustum's 67° base corner is above the 35° edge that the 1 cm field holds.
  - `bridson2002-scenes.md` is updated, and 10 of 10 pass.
- **(4) Box3D: noted.** No per-frame pose readback. Two-way coupling waits for the review of one-way M6.
- **(3) M6 one-way moving solids: not implemented this round.** Predicted DAT is the precondition and has now landed. Steps 1–2 below then need only the per-body table and w_v = a_v (prep) as new mechanism; the redo resolves the squeeze. Steps:
  1. Body-frame images behind a pose table; per-frame pose and twist upload with no readback.
  2. In prep, a vertex within reach of a body gets w_v = a_v = R₁R₀ᵀ(x₀ − c₀) + c₁ − x₀. Its solid cut is relative to a_v, which the static path already does with room = 1 − f(w).
  3. Friction relative to the surface velocity.

  **Bridson table (cloth between a moving body and the floor):** the carried prediction a_v crosses the floor. The static-solid stop (w /= f) then shortens it, and pairs with cloth below conflict and redo held. A held vertex pressed by an advancing body has no feasible motion. Such a frame must fail explicitly (status `solidConflict`), not be cut.
- **(6) M5: not done this round.** The tests of solid-surface samples against cloth faces and edges, and the intersection counter for cloth against solid, are still to do. Detection speed came first under ruling (2).

## 2026-11-09 MILESTONE: the scene contact adapter (priority 1)

`lib/cloth/contact/scene-solids.ts` turns a scene's solid world into a `ClothSolidSource` on the scene's own voxel lattice. The field's nodes are the nodes the scene renders, hit-tests and edits on (new `ClothSolidSource.origin`). Every authored patch is accounted for, or the build throws:

- **Box fills** become `lattice-box` parts, exact under the trilinear field:
  - the container shell, at μ = `CLOTH_CONTAINER_SHELL_FRICTION` = 0, since no scene authors cloth friction for the vessel;
  - the ground slab, at the scene's ground μ.
- **Procedural bake patches** (`solidBoundary`) are covered by the analytic solids the author sampled them from, as `sdf` parts with each solid's μ. The build re-evaluates every stored boundary node against those parts and throws on any mismatch, so the contact surface is the visible surface.
- **Any other patch** throws `cannot follow solid patch i (op kind) yet`. That covers a clear, a paint, a round or sculpted kind, and a voxel block from an edit. Cloth contact does not follow solid edits yet. It must not silently disagree with the visible solid.
- **Unknown cases fail explicitly:**
  - an unknown Bridson id;
  - a box fill that no authored solid owns;
  - a spherical container;
  - a non-cubic lattice.
- **Field box:** the container lattice. Cloth that leaves it raises `OGC_ST_SOLID_BOUNDS`.

Per-scene authoring is a static table keyed by scene id prefix. Bridson's is `lib/cloth/scenes/bridson-2002/contact-solids.ts`:
- the ground is matched to its fill box;
- legs, frustum, the curtain ball and the spinning sphere are `sdf` parts at their t = 0 pose. Each is held until its pending time; M6 covers the motion after that.
- A solid that moves from t = 0 (the tablecloth sphere) has no part. The cloth system already halts before it reaches the cloth.

Measured on the CPU at setup, for the full container field:

| scene | band | source build (incl. node proof) | field build | bricks | image |
|---|---|---|---|---|---|
| curtain | 0.06 m | 54 ms | 1.2 s | 10,426 | 38.7 MB |
| curtain | 0.13 m | 43 ms | 1.6 s | 17,298 | 64.4 MB |
| rotating sphere | 0.06 m | 162 ms | 2.5 s | 6,910 | 25.6 MB |
| rotating sphere | 0.13 m | 155 ms | 2.8 s | 10,997 | 40.8 MB |
| tablecloth | 0.06 m | 170 ms | 3.4 s | 14,720 | 54.5 MB |
| tablecloth | 0.13 m | 168 ms | 4.0 s | 26,460 | 98.0 MB |

- About 60% of the bricks are the shell's walls and ceiling, which the Bridson cloth never reaches.
- The fix, if memory matters: an affine-brick record (one plane, a few words) for bricks whose nodes all come from one lattice-box face. That is a GPU format change, and it is parked: frame time is unaffected.

Tests: `tests/cloth-contact-scene-solids.test.ts`, 5 of 5 pass.
- part lists and μ for all three scenes;
- the field's φ, normal and μ at the ground and above each held solid;
- the contact field equals the visible procedural brick values node for node at a tablecloth leg (> 2000 nodes);
- explicit failures: clear, a foreign box, a round kind, a tampered boundary, an unknown id, and a box fill in a scene with no authoring;
- the cloth-frame translation.

## 2026-11-09 COORDINATION (C2 → C1): attach the scene's solids in `ClothSystem.build`

The interface is fixed:

```ts
import { clothSolidBand, clothSolidSourceForScene } from "./contact/scene-solids";
import { buildClothSolidField } from "./contact/solid-field";
import { OgcContactDetector } from "./contact/ogc-detector";

// Once per solid world: key it by solidVoxelPatchesKey(scene.solidVoxels) and the boundary. Not per cloth rebuild:
// it is 1.2-3.4 s of CPU at band 0.06 m (see the table above), so build it off the frame.
const band = clothSolidBand(config.r, config.rqMax);   // r + 1.5 sqrt(3) rqMax
const field = buildClothSolidField(clothSolidSourceForScene(this.scene, { band_m: band, frameOrigin_m: origin }));
// `origin` is the solver's origin_m: the source comes back in the cloth frame.

// ogcMesh = { vertexCount, triangles: Uint32Array of (v0, v1, v2, faceMaterial) per face, edges: topology.edges };
// restLocal = vec4 per vertex of restPositions_m - origin (as meshOf in tests/cloth-contact-gates.test.ts, which has origin 0).
const det = await OgcContactDetector.create(device, ogcMesh, restLocal, {
  ...config, prediction: "guess", solidCapacityWords: field.words.length,
});
const solver = await ClothGPUSolver.create(device, model, { ...,
  contact: det.solverContact(CLOTH_BINDING.contact), hooks: det.stepHooks });
det.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: solver.stepStart });
det.setSolidField(field);
// Per frame: det.encodeFrameBegin(pass) before the frame's solver.encodeStep(pass).
// The status word (det counters) is the explicit-failure channel: map OGC_STATUS bits to a halt, as for pendingSolids.
```

- **Config:** the gates' values are a start: r = 1.5e-3, kc 100, μ 0.2, εv 1e-2, rqBase 3e-3, rqMax 0.04, envelopeScale 2.5, contactCapacity 128, gridCapacity 32(N+T+E), bucketCount 2^18, roundCap 4.
- **pairCapacity** is 840 N in the gates. At the curtain's 38,100 vertices that is the dominant buffer, and that is what priority 3 attacks. Author it from a measured peak once priority 3 lands.
- **A defect in `scene-description.ts` (C1's file), please fix.**
  - `bridson2002SceneDescription` appends the procedural patches after the shell that `createEmptyScene` put there. It then sets `solidBoundary` with `patchCount`/`patchesKey` for the patches alone.
  - `proceduralSolidField(scene)` checks the prefix `solidVoxels.slice(0, patchCount)`, so it is `undefined` for all three Bridson scenes (measured).
  - As a result, the visible legs, frustum and balls are drawn and collided from the voxel blocks, not the smooth field.
  - Fix: prepend the patches (`[...patches, ...shell, ...fills]`), or set `patchCount: shell.length + patches.length` and `patchesKey: solidVoxelPatchesKey(scene.solidVoxels.slice(0, that))`, as `rebaseProceduralSolid` does.
  - The adapter accepts either layout: it finds the baked run at offset 0 or right after the shell.
- `ClothSystem`'s status text in `lib/core/scenes.ts` ("Not yet: cloth-solid contact (lane C2), so the cloth passes through the solids") can go once this is attached.

## 2026-11-09 MILESTONE (M5, part 1): surface samples against cloth faces, and the cloth–solid intersection counter

### What is implemented

- **Surface samples** (`clothSolidSamples` in `lib/cloth/contact/solid-field.ts`). These are static points on the field's zero set wherever vertex contact alone lets a face sag into the solid.
  - The criterion: a face of edge e over convex curvature κ dips about κe²/6, so samples go where κ > 3r/e². κ is measured as the normal's turn over two cells, from central-difference normals a quarter cell wide.
  - Concave creases get no samples (|∇φ| ≈ 0 there). Neither do flat faces or broad spheres.
  - Spacing is h/2: a sharp cell is seeded at its 8 half-cell centres, which are projected (Newton) and kept where the half-cell turn exceeds the threshold. Kept samples are at least h/8 apart.
  - CPU, once per field.
- **Detector.**
  - Samples sit in x_det at N + s (w = μ) and in the contacts buffer (header [2].y, count [2].z).
  - The query kernel runs N + E + S threads. `sampleFacet` does a point query against the face grid. A sample within r of a face (with OGC's feature acceptance) becomes an entry on that feature's vertices, carrying (s, t | feature, μλ, w = 1).
  - A sample within the envelope becomes a DAT pair with ids[0] = N + s. The sample is static: no truncation of its own, dn = 0, prediction 0.
  - Config `solidSampleCapacity` is authored. `setSolidField` throws if the field has more samples.
- **Vertex block.** A sample entry reads the sample from the contacts buffer. Friction uses μλ from the entry, and the sample's own motion is zero.
- **Counters.**
  - `sfPairs`: sample–face pairs per frame.
  - `solidIntersections`: cloth faces with φ < 0 at any point of a barycentric lattice of spacing ≤ cell/4, run in the intersection pass.

### Tests

| Test | What it checks | Result |
|---|---|---|
| `tests/cloth-contact-solid.test.ts` (CPU) | Samples lie on the zero set at the box's edges (none on face interiors) and on a 2.5 cm ball (κ 40/m > 30/m). None on a 10 cm ball or along a step's concave crease. r = 0 is refused. | 9/9 pass |
| `tests/cloth-contact-gpu.test.ts`, static solid | Sample–face pairs equal the Float64 brute force (sheet plus samples as free points). `sfPairs` equals the pair count. Force, Hessian and energy match the oracle with sample entries. Planar truncation matches with sample DAT pairs. After truncation, 0 face pairs and 0 faces in the solid. A vertex pushed 2 mm into the box edge puts 1–6 of its faces in the solid. A detector authored for 8 samples refuses the field. | 2/2 pass |
| Unit suite | Only the known `reservoir-particle-cost` "merge cap" failure. | 1662 pass, 1 fail |
| Types | | Clean |

### Gates (60 steps, then 30 timed)

| Gate | Samples | Faces in solid | Intersections | Other |
|---|---|---|---|---|
| drop (10 × 31², sharp pedestal) | 9,822 (capacity 16,384) | 0, every step and after the timed batch | 0 | 1,467 sample–face pairs at step 59; 122.9 ms/step (priority 3); peak pair slots 833.8N of 840N |
| **drape** (61² sheet on a 10 cm sphere on a slab) | 8,351 (slab edges; sphere κ 10/m below the threshold) | **34 at step 5** | 0 | **Fails** |

**Why the drape fails** (CLOTH_GATE_DEBUG):
- The impact drives the vertices to a minimum φ of 6e-5 m at step 5 (round cap, 4,628 capped).
- The faces then sag κe²/8 = 0.25 mm into the sphere.
- The sample criterion assumed that vertices rest r/2 out. During impact they do not, so any curvature lets faces in.

### The fix (M5 part 2, next)

Each cloth face and edge gets its own closest point to the field: the minimum of φ over the triangle and over the segment, found by local minimisation on the trilinear field (Macklin et al. 2020, *Local Optimization for Robust Signed Distance Field Collision*).
- Contacts:
  - an interior face minimum, or an interior edge minimum inside its normal cone, becomes an entry on the feature's vertices with frozen barycentric weights;
  - force −g′(φ) β_v ∇φ;
  - Hessian g″ β_v² ∇φ∇φᵀ.
- Truncation: a vertex's cap includes γφ_f/L over its faces and edges. Every point of a face moves at most max|δ_i|.
- Cost follows the cloth faces near a solid, not the solid's area.
- It covers creases and curvature alike. It also removes the samples' edge bound: a straight cloth edge crossing a crease passes within about h/4 of a sample, so it is held only where h/4 < r. That holds for the gates (h 5 mm, r 2 mm) but not for the Bridson scenes (h 1 cm, r 1.5 mm).
- Exact edge coverage by points would need spacing about r everywhere. Measured, that is 0.19, 0.20 and 0.35 M samples and 13–19 s of CPU for the curtain, rotating sphere and tablecloth.
- If the face and edge minima hold the drape and the drop alone, the samples go.

Sample counts at h/2 for the Bridson scenes (r 1.5 mm, e 2 cm):

| Scene | Samples | CPU |
|---|---|---|
| curtain | 17,170 | 2.0 s |
| rotating sphere | 14,856 | 1.9 s |
| tablecloth | 22,897 | 2.2 s |

## 2026-11-09 COORDINATION (C2 → C1): author `solidSampleCapacity`

`OgcContactDetector.create` now needs `solidSampleCapacity` whenever a solid field has sharp features. `setSolidField` throws otherwise, giving the field's count.

For the Bridson scenes, use 24,576, which covers the tablecloth's 22,897; per scene would be 17,170 / 14,856 / 22,897. The count is deterministic per field and r, so author it from these numbers.

M5 part 2 may remove the samples; I will say so here if it does.

## 2026-11-09 ANSWERS (C2 → coordinator, C1): items (a), (b), (c)

### (a) L_max and its f32 rounding: fixed at the source (02aea2c7)

- **The cause was not the rounding of L.** The shell measured L = 1.7320512 because the bricks measured their differences on the f32 node values. Lattice-box values such as 0.15 and 0.10 round independently, so a corner cell's three edge differences came out a few ulps over a cell. That is a storage rounding, not a gradient of the solid, and rounding L up alone still left it 3 ulps over the bound.
- **Fix:**
  - `brickLipschitz` now measures on the source's own double node values, before the image rounds them.
  - `finish()` stores `f32Up(L)`, so `field.maxLipschitz` is exactly image word 7.
  - `clothSolidBand` uses `SOLID_LIPSCHITZ_BOUND = f32Up(√3)`, the same expression the detector's band rule evaluates.
- **Measured:**
  - L_max is 1.732050896 (= the bound) for a bare shell and for all three Bridson scenes.
  - With band = `clothSolidBand(r, rqMax)` exactly, `setSolidField`'s rule passes.
- **Regression:** `tests/cloth-contact-scene-solids.test.ts` covers the bound and `f32Up`, and checks that a bare shell's L equals the bound, word 7 equals L, and the band rule passes at `clothSolidBand`.
- **Unit suite:** 1663 pass, 1 known failure (`reservoir-particle-cost` "merge cap…").

### (b) Turn-floor `clothSolidSamples`: landed (af83aa3b)

- The samples no longer depend on r or the edge length. The options are validated and the result is cached per field.
- An 8×8 sheet in an empty scene now builds in about 33 ms.
- Sample counts per field (with these, the COORDINATION 24,576 figure above is out of date): curtain 17,048, rotating sphere 21,376, tablecloth 14,472. C1's `createClothSceneContact` authors the capacity from the field's exact count, which stays right.

### (c) Rotating sphere at 22.5K vertices, K = 64: profile and per-iteration truncation

**Setup.** The harness is `tests/cloth-contact-scene-profile.test.ts` (`CLOTH_CONTACT_PROFILE=rotating`, `CLOTH_PROFILE_TRUNCATE=colour,iteration`):
- it builds the sheet as `ClothSystem.build` does;
- each tick is fenced alone;
- after every tick it runs the gate checks (status, intersections, faces in solid, vertices inside);
- it profiles each kernel in its own timestamped pass;
- it reads the step's last round back (`OgcContactDetector.readDiagnosis`, harness only) for the candidate diagnosis.

The sheet has 5 colours, so a step makes 385 truncations per colour and 65 per iteration.

| Schedule | Median tick (ticks 0–19) | Truncation per step | Rounds per step | Intersections | Faces in solid | Vertices inside |
|---|---|---|---|---|---|---|
| per colour (system hooks) | **622 ms** (max 741) | 385 × 0.85 ms = 328 ms | 2–5 × 38 ms | 0 | 34 at tick 17 | 0; min φ 0.37 mm |
| **per iteration** | **244 ms** (max 334); 217 ms over 14 ticks | 65 × 0.85 ms = 55 ms | 2–4 × 38–48 ms | 0 | 28 at tick 17 | 0; min φ 0.39 mm |

- **Safety holds per iteration exactly as it does per colour.** The truncation cuts the accumulated displacement against every pair's plane and the solid bound before any detection or commit. Colours inside an iteration only see untruncated neighbours in their Gauss–Seidel gradient. The faces in solid at tick 17 appear on both schedules, which is the M5 defect below.
- **Recommendation:** truncate per iteration. That is 2.5× on this scene. `stepHooks` is mine, so dropping `afterColour` is a one-line change. It changes every scene's solve, so I will make it once the coordinator agrees.

**Where the rest goes (per-iteration schedule, steady state, ticks 8–16):**
- **Detection, about 2 rounds × 38–48 ms.** Query takes 37.5–47.5 ms of each round. Everything else in the round costs 0.7 ms.
- **The query cost is the entries scanned, not candidates:**
  - r_q is at the 4 cm cap for the median vertex, so the cell is 4 cm against a 1.3 cm edge.
  - The grid holds 0.93–1.28 M entries in 7.5–8.4 K cells: a median of 100–200 faces per occupied cell, maximum 264.
  - A vertex query covers 27 cells and scans a mean of 3,550–6,430 entries (a median of 7,400 in the worst state).
  - Of those it tests 195–310 VF candidates. The GPU's all-roles count is 1,500–2,000 per vertex per round, mostly edge–edge.
  - DAT pairs are 160–250 per vertex per round, and they make `planes` 0.85–1.2 ms.
- **Why r_q is at the cap:**
  - The sheet moves 1.8–2.5 cm per step (median step length).
  - 92–100 % of the vertices are held at the mandatory round, because last step's contact or truncation holds them. The residual rule alone holds only 9 %: the median ρ is 0.25 mm, so the predictions are good.
  - A held vertex's envelope is r + s|w| = 6.2 cm, capped at 4 cm. Its 0.45 r_q budget (1.8 cm) is below its motion, so it is capped (40–50 K caps a step), truncated, and held again.
- **What I tried:** holding on last step's conflict only. The other holds are the residual's to make, and safety is unaffected because a conflicting round is redone held. It was **no better**: a median of 262 ms, with more re-detections and held redos (2–6 rounds), because the residual cap then truncates the unheld vertices inside the step. I reverted it.

**Next on cost (priority 3), in order of expected size:**
1. **Per-iteration truncation.** 2.5×, ready.
2. **Single-sided inflation.** Today the face boxes and the vertex query box each carry their own envelope, so a pair is found within r_v + r_t when max(r_v, r_t) is all that is needed. Inserting faces with max(own, query) and querying a point (as `sampleFacet` already does) cuts the scanned cells from 27 to 1–8. When envelopes are capped, as here, that is about 4–27× fewer entries per query.
3. **The held-at-cap loop.** A vertex whose step motion exceeds 0.45 rqMax can never leave the hold. Either rqMax is authored from the scene's speed (4 cm against 2.5 cm a step leaves s at 1.6), or a held vertex keeps its prediction for the part of its motion that the round's planes allow. This needs design before code.
4. **LBVH (Karras 2012) against the hash.** Measure after 2: with point queries the hash's cost is entries per cell, which an LBVH does not reduce for boxes this large.

### M5 on the real scene: faces in the solid at a 67° crease (the defect behind tick 17)

- **What it is:** the rotating-sphere scene has an sdf wedge on the floor. The sheet drapes over its top edge, a convex crease turning 67°, on a 1 cm cell.
- **Measured (Float64, stored field):** 25 faces reach 0.17–0.52 mm inside. Their vertices sit at φ 1.3–4.5 mm, and the longest edges are 1.8 cm.
- **The samples are there.** The nearest is 2.7 mm from the deepest point, and they are spaced h/2 = 5 mm.
- **The cause:** the trilinear field rounds the crease over about one cell (ρ ≈ h). Its zero set therefore bulges between samples by about κR_c²/2. For R_c = (h/2)/√2 that is h/16 ≈ 0.6 mm, and the barrier lets the face settle onto the samples themselves.
- **Fix (M5 part 2), now concrete:** lift each crease sample along its normal by κ_max R_c²/2, the bound on how far a convex patch of curvature at most κ_max rises above the plane through three samples with circumradius R_c. Smooth cells already get the equivalent (k1 + k2)h²/8 lift.
  - The lift is at most about h/16. That is inside the field's own O(h) rounding of the crease, and it makes the counter's exact test hold by construction.
  - The drape gate's µm misses come from the same rule applied with one irregular sample per cell. They need R_c taken from the actual sample spacing rather than h.

## 2026-11-09 COORDINATION (C2 → C1)

1. **The band margin can go.** After 02aea2c7, `clothContactBand(r)` can return `clothSolidBand(r, CLOTH_SCENE_CONTACT.rqMax_m)` without the `1.5e-4·√3·rqMax` term. A bare shell and all three Bridson fields now measure L_max = `SOLID_LIPSCHITZ_BOUND` exactly, and the detector compares the same expression. Please drop it in `lib/cloth/scene-contact.ts`, or tell me to.
2. **Per-iteration truncation is coming.** Once the coordinator agrees, `stepHooks.afterColour` goes away: 2.5× on the rotating sphere, and safety unchanged (above). Your dry-sheet oracle tolerance may need a look: truncation then bounds the iteration's move, not each colour's.
3. **The rotating sphere fails the cloth–solid face counter from tick 17 on.** This is the M5 crease defect above; it is mine to fix. Your system test checks vertices only, so it passes.

## 2026-11-09 COORDINATION (C2 → C1): truncation now runs once per iteration (coordinator ruling 1)

- **Change:** `OgcContactDetector.stepHooks` no longer has `afterColour`. Truncation now runs after the initial guess and after each iteration, so a step makes K + 1 truncations instead of K(colours + 1) + 1.
- **Measured on the rotating sphere** (22.5K vertices, K = 64): 622 → 244 ms a tick. There were no intersections and no vertices inside on either schedule.
- **Please recheck the dry 8×8 sheet oracle tolerance in `tests/cloth-system.test.ts`:** truncation now bounds an iteration's accumulated move, not each colour's.
- **The old schedule is still reachable** for comparison through `CLOTH_GATE_TRUNCATE=colour` (gates) and `CLOTH_PROFILE_TRUNCATE=colour` (profile).

## 2026-11-09 M5 part 2: no face inside the solid (coordinator ruling 2)

- **The lift is a measured coverage bound, not a constant.** `clothSolidSamples` (lib/cloth/contact/solid-field.ts):
  - **Facets:** each sample's neighbours, ordered around its normal, span a fan of triangles. Only facets of the samples' local convex hull count, meaning no neighbour lies outside the facet's plane: a face resting on the samples spans those.
  - **Bound:** a convex surface lies below each of its tangent planes, including each adjacent cell's own plane at a voxel edge or corner. So over a facet it rises at most the top of their lower envelope (`roofHeight`: corners, edge crossings, triple points). This is measured along the three samples' mean normal, and each sample is lifted by the largest bulge of its facets.
  - **Refinement:** a facet bulging past h/16 has its centre and edge midpoints sampled at half the spacing, and is measured again. If refining down to the ceiling's own scale cannot bring it under, the build throws a `RangeError` naming the place. It never clamps.
  - **Gaps:** a fan gap of half a turn or more is the edge of the sampled surface. So is a gap wider than a quarter turn whose middle holds no surface the seeding would sample: a concave crease, or flat surface just past the last row.
- **Seeding changes:**
  - Sharp cells are seeded at their 8 corners too, because a voxel corner is a node.
  - Corners and edges are kept first. The order is the number of cells' planes meeting at the point, so a nearby face sample never displaces the feature's sample.
  - A seed's turn is read a tenth of a cell out, so flat-face points next to a voxel edge are dropped.
  - Newton projection now converges to 1e-10 h, and tests φ before the gradient: at a voxel edge φ vanishes quadratically.
- **Measured sample builds (setup, CPU):**

  | Field | Samples | Time | Largest φ at a lifted sample |
  |---|---|---|---|
  | curtain | 10.1 K | 6.9 s | 0.027 h |
  | rotating sphere | 15.7 K | 8.8 s | 0.083 h |
  | tablecloth | 15.2 K | 8.2 s | 0.066 h |
  | drape sphere | 10.8 K | 5.1 s | 0.040 h |

  The largest lift is ≤ h/16 by construction; φ can read up to |∇φ| ≤ √3 times the lift. For comparison, the κR²/2 design took 33–38 s per field and hit the ceiling at exact voxel edges.
- **The exact face gate:** `faceInsideSolid` / `facesInsideSolid` in `tests/cloth-contact-oracle.ts`.
  - It clips each face into the cells it crosses. On each piece φ is a cubic in barycentrics, and its Bernstein coefficients (the symmetrised blossom of the cell's multi-affine interpolant) bound φ on that piece.
  - It subdivides until every coefficient is ≥ −1e-9 h (outside) or a corner is below −1e-9 h (inside), and throws if a piece is still undecided at depth 16.
  - It is unit-tested: a face cutting a voxel edge between vertices that are well outside is inside, one clearing it by 2e-5 m is not, and it agrees with a 48-step lattice on 300 random faces.
- **Gates (one GPU run each), all with zero faces inside:**

  | Scene | Faces inside (exact) | GPU lattice counter | φ over every face, exact lower bound |
  |---|---|---|---|
  | drop (sharp voxel pedestal) | 0 | 0 | ≥ 2.2e-7 m |
  | drape (sphere and floor) | 0 | 0 | ≥ 2.7e-6 m |
  | rotating sphere, 20 ticks | 0 | 0 | ≥ 3.1e-6 m |

  On the rotating sphere the minimum vertex φ is 4.5e-4 m and the median tick is 242 ms. The 25 faces that were inside at tick 17 are gone.
- **The GPU counter already counts faces.** `solidIntersections` tests every face at four lattice points per cell along its longest edge. The new `OgcContactDetector.countIntersections()` (harness only) submits the count and returns `{ intersections, solidIntersections }`.

## 2026-11-09 COORDINATION (C2 → C1): face-in-solid check for the system test

- **To see faces, not just vertices, in `tests/cloth-system.test.ts`:** call `await detector.countIntersections()` after a tick and assert `solidIntersections === 0`. For the exact Float64 check, use `facesInsideSolid(field, positions, faces, origin)` from `tests/cloth-contact-oracle.ts`, which returns `{ inside: number[], lower }`.
- **`clothSolidSamples` now takes 5–9 s on the Bridson fields** (once per field, cached). The rotating sphere has 15.7 K samples, so `solidSampleCapacity` in `createClothSceneContact` grows with it, as it already reads the count.

## 2026-11-09 Cost step 1 (ruling 3a): one-sided inflation

- **Change:**
  - A face's grid box now grows by the round's largest envelope. That value is the new counter word `roundEnvelope`, which the cell kernel sets.
  - The vertex and edge queries are their bare swept boxes.
  - `sampleFacet` was already a point query.
- **What it still guarantees:** every pair whose predicted paths come within the larger envelope is still found. A pair the search now skips stays more than R_max apart along its predicted paths, and the two caps (0.45 r_q each) cannot close that gap.
- **Oracle:** `bruteForcePredictedPairs` mirrors this: a bare swept box against a face box grown by max r_q, with an edge reached through its first face. The GPU predicted-pair sets still equal the oracle's.
- **The insert kernel no longer binds `work`.**
- **Rotating sphere** (one profile run, 20 ticks, K = 64), everything at the 4 cm cap:

  | Rotating sphere | Before | After |
  |---|---|---|
  | entries scanned per vertex query (mean) | 3,735 | 290 |
  | median entries scanned | 4,703 | 236 |
  | VF candidates per vertex (mean) | 200 | 116 |
  | all-roles candidates per vertex per round | 1,165 | 600 |
  | cells per query (median) | 27 | 1 |
  | round | 38.9 ms | 8.2 ms |
  | query kernel | 38.3 ms | 7.3 ms |
  | median tick | 242 ms | 161 ms |
  | grid entries | 0.93–1.28 M | 1.54 M |

  Rounds per tick are unchanged at 2.25. There were no intersections and no faces inside, by the GPU lattice or the exact check.
- **Gates (one run):** mixed envelopes pay for the wider face boxes.

  | Gate | Candidates/N per step, before → after | ms/step, before → after | Peak grid, before → after |
  |---|---|---|---|
  | drop | 4,379 → 7,690 | 113.7 → 118.2 | 0.64 M → 0.91 M |
  | drape | 1,425 → 3,278 | 28.1 → 27.4 | 0.23 M → 0.38 M |
  | twist | 15,316 (no comparable baseline) | 138.1 | — |

  All three pass.
- **Capacity:** the grid holds up to (⌈(D + M)/R⌉ + 3)³ entries per face. That is 64 when the face plus its motion fit in a cell. `tests/cloth-contact-gpu.test.ts`'s base grid went from 64 N to 128 N, because its velocity envelopes vary. Please check `gridEntriesPerPrimitive` against the new peak (C1, scene-contact).

## 2026-11-09 Cost steps 2–3 (rulings 3b, 3c)

### 3b: break the held-at-cap loop (tried, reverted)

- **Variant tried (the ruling's preference):**
  - A vertex is held only by a conflict. A conflict is a new hold bit, bit 2.
  - A vertex cut by a DAT plane or the cap keeps its prediction for the next step. Its residual is measured against its *truncated* motion, not the full inertial motion.
  - No rqMax was derived from the scene's speed.
- **Rotating sphere** (one profile run, 20 ticks):

  | Rotating sphere | 3a (hold on any cut) | 3b (conflict-only hold) |
  |---|---|---|
  | median tick | 161 ms | 174 ms |
  | rounds per tick | 2.25 | 3.55 |
  | conflicts per tick | not logged | 50–578 |
  | redo rounds | not logged | nearly every tick |
  | DAT pairs per vertex | 160–250 | 140–240 |

  No intersections and no faces inside, as before.
- **Diagnosis:** the hold is what keeps conflicts away.
  - A held vertex's envelope is r + s·plen. Its prediction is the previous truncated step, so neighbouring held vertices agree on direction and their pair planes are compatible.
  - Released from the hold, a vertex predicts its full inertial motion again. Pairs then disagree, a conflict follows, and the redo holds *every* vertex.
  - So the variant trades the cap loop for redo rounds, which cost more.
  - The cap loop itself is cheap: the envelope is at the cap because the cloth really is moving at about 0.45 r_q a step against the sphere.
- **What could work next (design needed, not started):** a redo that holds only the conflict's vertices and their 1-ring, repeated until no conflict remains (multi-redo). Each repeat is a full round of about 8 ms, so it only pays if most ticks converge in one repeat.
- **Where the about 161 ms tick goes now:**

  | Part | Time per tick |
  |---|---|
  | truncation (DAT planes, 0.85 ms × 65 iterations) | ≈ 55 ms |
  | detection rounds | ≈ 18 ms |
  | decides plus unarmed detections | 4.2 ms |
  | solver and the rest | ≈ 85 ms |

  Truncation cost scales with about 200 DAT pairs per vertex. That pair count, not the detector, is the next contact cost.

### 3c: LBVH against the hash (not implemented)

- After 3a, a vertex query scans 290 entries for 116 candidates, a 2.5× over-scan, in 1 cell.
- The round is 8.2 ms, about 5% of the tick.
- An LBVH would at best halve the scan, at the price of a per-step build (sort plus refit). It cannot recover more than about 4 ms of a 161 ms tick.
- Skipped. If detection becomes the bottleneck, the next step is the per-round cell size (R_max), not the structure.

## 2026-11-09 MILESTONE (M6, one-way): one moving solid, prescribed poses

**What it is.** One rigid solid whose motion is given (one-way coupling). Its field is baked once, in its body frame, with its surface samples and their hull edges. Each frame the owner uploads the poses at the step boundaries: n steps take n + 1 poses. Each step copies its start and end poses into the contact buffer on the GPU, so nothing is read back. Planar truncation only.

**How a step treats it.**
- **Detection.** Detection runs at the start pose in the mandatory round and at the end pose afterwards.
- **Carried prediction.** A vertex the solid can reach gets the carried displacement a_v = R1·R_detᵀ(x − c_det) + c1 − x. Its prediction is clamped to the band room about a_v, then bounded by the body field's ball bound.
- **Which vertices are carried.** A held vertex is carried (w = a_v). So is one whose prediction the bound would cut. The static solid's cut then applies.
- **Squeeze.** If the body bound still fails after the static solid's cut, the cloth is squeezed between the two solids. Status `bodySqueeze`.
- **Solver block.** The block uses φ_body(R1ᵀ(x − c1)) with gradient R1·∇φ. Friction slip is measured against the carried start position.
- **Body samples.** The body's samples are posed on the GPU. Their predicted motion is exact. They pair with cloth faces through a swept box.
- **Conflicts with the solid.** A sample's pair cannot be held apart, because the sample moves regardless. So the cloth vertices of a conflicting pair with a body sample are carried with the solid (flag bit 2).
- **Held rounds.** A held round can still conflict where carried cloth meets held cloth. Its conflicting cloth is then carried as well, and the round is redone.
  - `body.heldRounds` sets the number of held rounds. The default is 3.
  - A conflict after the last held round fails as `heldConflict`.
  - The pair that conflicted is kept in counter words 42–45 (`heldConflictPair`).
- **Missing poses.** A step past the uploaded poses fails as `bodyPose`. A frame encoded without `setBodyPoses` throws on the host.

**Hull edges (new, static and moving solids).** A cloth face can rest on the samples and still have one of its edges pass *between* two samples on a sharp edge.
- The truncation-only unit check found this: 7 faces inside the solid, with a face edge between two edge samples 2.4 mm apart.
- The overhang plate test found it again intermittently: 2 faces inside at step 22 in one run of two.
- **Fix:** `clothSolidSampleEdges` gives each pair of fan neighbours as a hull edge, and the query kernel pairs every hull edge with each cloth edge in its envelope (DAT, Planar mode). The sample–face and edge–edge pairs are what convex intersection needs. With them, truncation alone leaves no face in the solid. The unit test now asserts this exactly.
- **What it is not:** hull edges add DAT pairs only, not barrier entries. The faces' barrier against the samples is what holds the cloth off.

**Results (GPU, M1 Max).**

| Check | Result |
|---|---|
| Unit test vs the Float64 oracle | Body entries and terms, carried predictions, posed samples, the vertex block and truncation all match. After truncation φ_body ≥ 0.1 φ0, and no face is inside (exact and lattice). |
| Squeeze | A plate descending onto a sheet over a floor raises `bodySqueeze`. |
| Plate, in C1's solver (K = 20, BDF2) | Setup and measurements below. |
| Earlier plate without overhang | 0.32% RMS, 25.7 ms per step. |
| Rotating sphere (static), cost of hull edges | Round 9.1 → 12.2 ms (query 8.3 → 11.2 ms). Median tick 100.2 → 102.2 ms (profile at K = 64, 12 ticks). 0 faces inside in both runs. Face lower bound 3.1e-6 → 7.4e-6 m. |

The plate test:
- **Setup.**
  - The sheet is 26×26 and overhangs a 16×2×16 cm plate by 2 cm.
  - The plate rests for 15 steps, then moves for 45 steps at v = (0.3, 0.2, 0) m/s, turning at ω = 1 rad/s about y.
  - The plate has 404 samples and 876 hull edges, built in 0.17 s.
- **Clearances.** Minimum vertex φ is 1.96e-3 m (r = 2e-3). Over every face, φ ≥ 1.55e-5 m (exact).
- **Per-step checks.** Zero intersections and zero faces inside, on every step.
- **Friction.** The top of the sheet moves at 1.06% RMS of the plate's rigid velocity.
- **Redos.** One step was redone held, needing 2 held rounds.
- **Cost.** 33.7 ms per step, including the test's per-step readback.

**Limits.**
- One moving solid only, one way: its poses are prescribed.
- Planar truncation only.
- A squeeze between two solids fails explicitly; nothing pushes the cloth through.
- Hull edges carry no barrier. A cloth edge pressed onto a hull edge is held there by truncation (faces get to about 1e-5 m).
- Two-way coupling has not started; it is for review.

## 2026-11-09 COORDINATION (C2 → C1): wiring the moving solid, and what changed in the detector

1. **Failing status bits.** Add `OGC_STATUS.bodySqueeze | OGC_STATUS.heldConflict | OGC_STATUS.bodyPose` to `CLOTH_CONTACT_FAILING` in `scene-contact.ts`.
2. **The scene's rigid body.** Replace the pending-solids path with `config.body = { capacityWords, sampleCapacity, poseCapacity, heldRounds? }`.
   - `capacityWords` and `sampleCapacity` come from the body-frame field's `words.length` and `clothSolidSamples(field).length / 4`. The detector authors 8 hull edges per authored sample and throws past that.
   - `poseCapacity` is the most steps per frame + 1.
3. **Setup.** Call `setBodyField(field)` once, with the RigidBodyDescription shape baked in its body frame (body origin at 0, axes of the identity pose).
4. **Every frame.** Call `setBodyPoses(poses)` with the owed steps + 1 poses from the pose track (step boundaries k = 0..n) *before* `encodeFrameBegin`, which throws without them.
5. **Layout changes** (all read through the exports, so nothing should need editing):
   - `OGC_PARAM_WORDS` 64 → 68;
   - `OGC_COUNTER_WORDS` 42 → 46: `frameBodyContacts` 40, `frameBodyCut` 41, `heldConflictPair` 42–45;
   - the contact header's `countsBase` is 12;
   - with a body, `encodeDetection` encodes `heldRounds` redo/round pairs, plus a final `redoLast` check.
6. **Counters.** `solidIntersections` now includes faces inside the moving solid at the step's end pose. `OgcFrameCounters` gains `bodyContacts`, `bodyCut` and `heldConflictPair`.
7. **Static solids now get hull-edge pairs too.** No change is needed on your side. The cost is +3 ms a round on the rotating sphere, and the contact buffer grows 64 bytes per authored sample (1.4 MB at 21 K samples).

## 2026-11-09 M6 squeeze semantics: the bound keeps min((1 − γ_r) φ0, r), least carry, per-axis deviations

Coordinator's item (2): `bodySqueeze` must fire only when the gap between the solids really is below the cloth's 2r. Cloth pressed into a wider gap must resolve.

**Bound normalisation, static and moving.** The scale changes from s = 1/(γ_r φ0) to s = 1/max(γ_r φ0, φ0 − r).
- f(d) ≤ 1 now keeps φ ≥ min((1 − γ_r) φ0, r). A vertex may come down to the contact distance r, where the barrier holds it. Once nearer than r/(1 − γ_r), it may close γ_r of its clearance.
- The old scale stopped every vertex at 0.1 φ0 of its clearance, however far it was. So a 3r gap could not take a sheet resting at r below a plate coming down: carrying with the plate cut against the floor, which raised `bodySqueeze`.
- The squeeze check (the body's cut after the static cut > 1 + 1e-5) is now equivalent to a gap below about 2r along the path from hold to carry.

**Per-axis gradient deviations.** The bound changes from J|d| to D·|d|: φ(x0 + d) ≥ φ0 + g0·d − Σ_a D_a|d_a|, with D_a the largest |∂_a φ − g0_a| over the cells the ball's box touches.
- J was |D| (Cauchy–Schwarz). The per-axis form uses the same cell ranges, so it is strictly tighter and costs the same.
- Under a plate's edge, a slide along x sees D_x = 1, not |D| ≈ 1.16. That difference is what dragged the sheet in the gap test (below).
- **Layout.** Each set of terms is two vec4s, `OgcTerms {q: (g0 s, marker), D: D s}`. The marker is 1 for a bound, 3e38 inside, and 0 for none.
  - The static D sits at `solidDevOff`, which is 8N + T + E without a body and 16N + T + E with one.
  - The body's D sits at `bodyDevOff` = 20N + T + E. The body's terms stay in the body frame at the end pose; prep and apply rotate d by R1ᵀ.
  - The work buffer grows from (8 | 16)N + T + E to (12 | 24)N + T + E words. `OGC_PARAM_WORDS` goes from 68 to 72.
  - Apply now binds the contact buffer (binding 8) to read the end pose.

**Carry rule (prep, moving solid).**
- **Full carry, w = a_v**, in two cases:
  - held and in contact (φ0_body < r): the friction anchor;
  - any vertex the solid's bound reaches, in a round redone held. A conflict with the solid's samples carries the whole neighbourhood at once, instead of one ring per held round.
- **Least carry otherwise.** When f(w − a) > 1, w moves towards a_v only as far as the bound needs: w = mix(w, a, 1 − 1/f). A vertex the solid does not touch is not dragged.
- The old rule carried every held vertex the bound could reach. It pushed floor-held cloth under a descending plate into the floor (a false squeeze), and it would drag a sheet under a passing plate.

**Results (GPU, Metal).** `cloth-contact-body` (4 tests) and `cloth-contact-gpu` (11 tests): 15 of 15 pass.

| Test | Result |
|---|---|
| New: gap 3r | Setup and measurements below. |
| Plate in C1's solver (K = 20) | 0.99% RMS from rigid. 45 steps redone held, needing at most 2 held rounds. Min vertex φ 1.96e-3 m; faces φ ≥ 7.2e-5 m (exact). 31.6 ms per step with readback. |
| Squeeze | Still fails explicitly. The plate descends onto the sheet's own thickness. |
| Unit test against the oracle | The oracle (`solidBallBound` with D, `solidBoundDrop`, `solidTermScale`, `solidKeep`) is mirrored in Float64. |

The gap-3r test:
- **Setup.** A 21×21 sheet lies on a floor. A 30×2×30 cm plate comes down from 8r to 3r over 10 steps, then slides 10 cm at 0.2 m/s, so its trailing edge passes 2r over the sheet.
- **Measurements.**
  - No failing status and no intersections, on every step.
  - Min φ is 2.00e-3 m to the floor and 4.00e-3 m to the plate.
  - Sideways drift is 7e-6 m with 0 plate cuts. The least carry with J|d| had given 9.5 mm.

**Note for C1.** The static solid now also uses the r floor and the per-axis bound, so there is less truncation near solids. Nothing changes on your side.

## 2026-11-09 M6 fix: heldConflict with cloth resting on the moving solid

Repro: C1's `todo` test in `tests/cloth-system.test.ts`. A 24×24 sheet lands on a sliding, turning sphere (1 cm lattice) and
halted at tick 29. C1's riser probe (a translating sphere on the default lattice) halted at tick 9. The pair it recorded was
a cloth edge against a body hull edge. Read back at the halt, the two were about 5 µm apart, with a separating plane.

There were two causes, and both are fixed in `ogc-detector-wgsl.ts`:
1. **The conflict threshold equalled the gap the truncation keeps.** The planes pass keeps each side sepEps from its plane,
   so a pair pressed together rests 2 sepEps apart. The conflict test was `g1 < 2 sepEps && g1 < gap(X)`. For cloth carried
   rigidly with the solid, g1 equals gap(X) up to rounding, so every held round conflicted again. The test is now
   `g1 < sepEps` (oracle `predictedConflict` changed to match). To conflict, a prediction must close the gap by at least
   sepEps, which is the rounding scale sepEps is sized for. Between sepEps and 2 sepEps, the planes pass starts both sides on
   the wrong side of the plane and lets neither cross further.
2. **A pair with a moving-solid sample kept a world-fixed plane.** A co-rotating pair (cloth carried with a turning sphere)
   closes along a fixed axis by about turn × extent, roughly 0.5 mm a tick here. That registers as a conflict that carrying
   can never remove. The plane is now certified at X and turned to the end pose (`bodyFrame`: R1 R0ᵀ in the mandatory round,
   which detects at the start pose; no turn after it). The conflict test and the planes pass then meet the plane where the
   samples are fixed: in the solid's frame.

Result: `a sheet lands on the moving solid and rides it` passes. The nearest vertex is 0.1 mm off the sphere, the cap
1.0 mm above its top, the peak body contacts 752, and there is no failing status. **C1: please drop its `todo`.**
`cloth-contact-body` (4), `cloth-contact-gpu` (11) and `cloth-contact-solid` (10) pass, and so does check:types. Unit tests:
1669 pass; the one failure is the known `merge cap`.

The riser probe now runs all 36 ticks with no heldConflict, but from tick 12, 2–17 faces lie inside the rising sphere. This
is the same family as the tablecloth item below; see "lift".

Also in this commit:
- **Sample build:** the field is sampled without allocation, and bricks and point hashes use numeric keys. The output is
  bit-identical (sha1 of samples and edges unchanged on all three scenes).

  | Scene | Samples + hull edges, before | After |
  |---|---|---|
  | Rotating sphere | 8.1 s | 2.3 s |
  | Curtain | 5.1 s | 1.2 s |
  | Tablecloth | 7.3 s | 2.4 s |

  The remaining cost is seeding (0.8–1.2 s: Newton projections and κ probes) and lift measurement.
- **Query split into three pipelines** (vertices, edges, samples with hull edges) over the same indirect arguments. A
  workgroup with no thread of its role exits at once. The single kernel carried the largest role's registers into every
  thread. The isolated query, with envelopeScale 1 on the rotating sphere at K 8, went from 4.06 ms to 2.16 ms (vertex 0.26,
  edge 1.11, sample 0.79). The tick-level effect is not yet measured cleanly: another lane's K = 64 run shared the GPU.
- **Dispatch cost measured** (scratch microbenchmark, Dawn/Metal on the M1 Max): about 15 µs per indirect dispatch, even
  with zero workgroups, and about 6–8 µs per direct one-workgroup dispatch. Skipping validation changes nothing. One
  detection call is 36 indirect dispatches (with heldRounds 3) plus 4 direct ones, about 0.2–0.5 ms, and a step makes 9
  calls. That overhead is a floor under the "contact < 3 ms" target, before any work is done.

## 2026-11-09 Tablecloth faces inside a solid: the leg tops' bevel

**Which solid.** The faces are on the static leg-top 45° bevels, at scene (-0.187, 0.81, ±0.29). The 4 legs are symmetric
and the faces appear on two of them. The probe is the rigid-body variant at K = 64. Each face rests 11–25 µm above one
sample: the drape's load compresses the r = 1 mm barrier, so only DAT holds it at 2 sepEps. No hull edge pierces the
face, every sample lies below it, and the field pokes 0.15–0.21 mm through it.

**Why it was missed: two coverage defects in the sample lift.** The samples are lifted so that a face resting on them
stays outside the field. A static oracle, kept in scratch at `~/code/me/fluid-scratch/c2/cover.ts`, samples every
lifted hull facet against the field.

1. **Missing hull facets (fixed in this commit).** `measure()` only formed triangles between angularly consecutive fan
   neighbours. Where a consecutive triangle is not a facet of the local convex hull, because a neighbour stands above
   it, it was skipped "because other facets cover it". But the covering facet spans non-consecutive neighbours and was
   never formed. At the first face (tick 39, -z leg), all 15 field points under the face had no facet at all within 3 mm
   along the face normal. Every fan wedge short of half a turn is now a facet candidate. The face is then covered: 40
   hull edges pierce it.
   - Builds: rotating 5778 samples in 1.3 s, curtain 5005 in 1.1 s, tablecloth 15202 in 3.1 s (was 2.4 s).
   - GPU probe at K = 64, ticks 30–44: faces inside drop from the earlier 1 at ticks 36–42 to 1 at ticks 34–38 and 43–44.
     That remaining face is on the +z leg, with the field 0.10 mm through it.
2. **`roofHeight` underestimates (open).** The roof bound assumes the surface between three samples is convex. Over a
   lattice bevel the trilinear zero set is a saddle inside a cell, and it rises above every tangent plane at the samples.
   A sample lifted along its own normal also raises a facet across a sharp edge by less than its lift. On the tablecloth,
   6897 lifted facets enter the field, by up to 0.29 mm.
   - The +z face is covered by facet 6743-15009-15015. Its roof lift is 0.14 mm, and the field stands about 0.22 mm
     above it.
   - Tried: measuring the depth along the mean normal, sphere-traced from the unlifted triangle. Still 1084 facets enter,
     by up to 0.26 mm, because a lift along diverging normals is not a lift along the mean normal.
   - **Prototype:** saved at `~/code/me/fluid-scratch/c2/solid-field-coverage-refine.ts`; not committed. It checks each
     convex lifted facet directly at h/8 spacing. Where the facet enters the field, its samples rise by the depth divided
     by the samples' blended normal component along the facet. Each pass uses a max over facets, not a sum, and only
     facets around samples that rose are rechecked. A hole that goes past the ceiling is refined.
   - **Tablecloth with the prototype: 0 faces inside at ticks 30–44 on GPU** (K = 64, rigid-body). The build has 15434
     samples and 67454 edges, in 5.3 s.
   - **Blocker:** the rotating scene's pedestal box corner at (-0.49, -0.464, 0.49) needs 0.69–0.73 mm of lift. That is
     above the h/16 = 0.625 mm ceiling, and refinement cannot lower it (it raises the corner sample's roof). Without
     refinement, a sliver at (-0.8, -0.44, 0.5) needs 1.42 mm. So the build fails explicitly, as it should.
   - **Decision needed:** whether the ceiling may follow the field's own corner rounding. A lattice box corner rounds
     over more than a cell, and h/16 was argued from an edge's rounding, not a corner's. The alternative is a corner
     treatment that needs no lift above h/16. Until then, tick 222 (2 faces) is not re-run.

**Hull edges (C1's 8 cm sphere, item 3).** Each sample's edges are all of its neighbours within 1.6 × its nominal
spacing. The seeding keeps samples 0.5 spacing apart, so the actual spacing is about half the nominal one. The reach
then covers about 3 actual spacings, a degree of about 17–23. Measured on a 1 cm lattice, with the current build:

| Sphere | Samples | Mean spacing | Fan edges | Per sample | Mean length | Hull-facet edges | Per sample |
|---|---|---|---|---|---|---|---|
| R 4 cm | 1277 | 4.0 mm | 8091 | 6.3 | 5.6 mm | 4040 | 3.2 |
| R 5 cm | 1562 | 4.5 mm | 17868 | 11.4 | 9.6 mm | 4728 | 3.0 |

So the count is not intended: it comes from the nominal reach. The edges of the local hull facets (the triangulation the
lift is measured on) are the 1-ring set, at 3 per sample. A chord the hull leaves out lies under the lifted facets, so
it touches nothing outside the field.

**Done (next commit).** The hull edges are now the samples' Gabriel graph (a subgraph of their Delaunay graph; it joins
consecutive samples along a sharp edge's row, as the solid-samples test requires) plus the local hull facets' edges.

| Case | Before | After |
|---|---|---|
| Sphere R 4 cm | 6.3 per sample | 3.3 |
| Sphere R 5 cm | 11.4 per sample | 3.4 |
| Sphere R 10 cm | | 3.1 |
| Rotating scene | | 42974 for 15696 samples |
| Curtain | 34773 | 23857 |
| Tablecloth | 64636 | 46757 |

Builds take 2.7 s, 1.3 s and 2.3 s. A wide wedge is a facet only over neighbours strictly below it, so coplanar samples
yield one triangulation, not all of them.

Gates:
- CPU: `cloth-contact-solid` 10/10.
- GPU: `cloth-contact-gpu` and `cloth-contact-body` 15/15, and both cloth-system scene-contact tests (the held rotating
  sphere, and riding the moving solid) pass.
- Table probe at K = 64, ticks 30–44: 1 face at ticks 38–39, from the open roof item above.

(C) from the coordinator: the setup failure "a lifted facet … enters the field" came from the coverage prototype, which
was in the working tree for a short time. That prototype is not committed, and the committed build passes that test.

**(E) done.** `tests/cloth-contact-scene-solids.test.ts` (its shared `sceneOf`, used by all its scene tests) and
`tests/cloth-contact-scene-profile.test.ts` now pass `movingSolids: "held"` explicitly, and the 6 tests pass. **C1:** you
can flip `bridson2002SceneDescription`'s default to `"rigid-body"`.

**(D) done: the carried state is exposed.**
- New `OgcContactDetector.readCarriedState()` returns `OgcCarriedState`, which holds:
  - `reference`: per vertex, pred[N + v], which is the reference prediction xyz plus rho_v in w (-1 before any step);
  - `held`: the held flags (bit 0, read by the next step as "held in the last step").
- `writeCarriedState(state?)` restores that state exactly. With no argument it sets the state before any step. A
  vertex-count mismatch fails explicitly.
- Nothing else the detector holds survives a step.
- GPU regression in `cloth-contact-gpu`: a step replayed from the restored state is bit-identical to the original
  (predictions, carried state, held count and redos).
- **COORDINATION (C1, `system.ts`):** publish `contactFor(id).readCarriedState()` with the frame's positions. On undo
  or load, call `writeCarriedState(saved)`, or `writeCarriedState()` for a document with no saved state.

## 2026-11-09 (B) heldConflict at K = 8: curtain tick 121–122, diagnosed, not fixed

Probe: `~/code/me/fluid-scratch/c2/tests/table-faces-probe.test.ts`, with `SCENE=curtain K=8`, rigid-body. It now dumps
the held pair on a halt.

**Before the halt (curtain):**
- No face is inside the solid at any checked tick up to 120 (every 6th tick, then every tick from 118).
- The status is envelopeCap only. Body contacts are 300–400.

**At tick 121–122:**
- The ball starts moving at 2 s: it is at rest until `ballStart`, then rises at 1.12 m/s, 1.86 cm per step.
- Body contacts jump from 403 to 10049. Half of the 38100-vertex curtain is draped on the ball and is carried.
- The status is heldConflict | bodySqueeze | solidPenetration | envelopeCap | roundCap.
- The last pair is a cloth edge against a ball hull edge, 5.2 mm apart at the detection pose. It is 10 cm from any
  static solid.

**Mechanism: carry propagates one layer per held round.**
- A redone round moves only the solid's samples and the cloth carried with it. Everything else is held still.
- Cloth that is uncarried but lies within the ball's 1.86 cm sweep is then crossed by the ball, so it conflicts and is
  carried in the next round. The cloth next to it conflicts in the round after that.
- With heldRounds 3 the third round still finds new cloth in the sweep, which raises heldConflict.
- Scratch override (not committed) with heldRounds 16: the run halts one tick earlier, at 121. Body contacts reach
  20978, and the last pair is cloth against cloth (carried against held).
- So more rounds do not converge. Each round grows the carried set by the cloth the last carry swept into.

**Proposed fix (mine, not yet implemented):**
- In the first held round, carry at once every cloth vertex within the body's swept volume for the step: its distance
  to the body samples at X, within the samples' own displacement plus r.
- Carried cloth is exempt from cloth-cloth truncation, and its cloth-cloth conflicts carry the other side.
- That is one closure per step, not one layer per round.
- Whether rigidly carrying half the curtain for the step at the ball's start is acceptable motion is a ruling for the
  coordinator. It is what an infinite-mass solid starting at 1.12 m/s from rest does to the cloth resting on it.

The rotating sphere (tick 128) and the tablecloth (tick 226) are not re-probed yet. The 1154 faces inside the solid C1
saw come after the halt (zero up to it here), with the halt disabled.

**(A) truncation versus strain limit: not started.**

## 2026-11-09 Rulings (1) and (2): evidence

### (2) Curtain heldConflict: histogram of the conflicting pairs at tick 121

**Method.** A scratch-only GPU instrumentation (reverted, never committed) records every conflicting pair of the
halting step from inside `checkConflict`. For each pair it stores the ids, the gap d at X, the predicted gap g1, whether
the round was forced, and each cloth vertex's held and carried bits. The pairs are then classified on the CPU:
- **same sheet:** the rest distance is under 1.5 × the current distance + 1 cm;
- **separate layers:** anything else among cloth-cloth pairs;
- **cloth-body:** the pair includes a ball sample.

The probe is `~/code/me/fluid-scratch/c2/tests/table-faces-probe.test.ts` with `HIST=1`.

**Result:** 5179 conflicts in the step.

| Round | Pair class | Conflicts | Gap d at X | Closure d − g1 |
|---|---|---|---|---|
| Mandatory (free predictions) | cloth-body | 2323 | median 4–9 mm, range 0.9–17.9 mm | median ≈ d + 0.3–0.8 mm, max 18.4 mm |
| Mandatory | cloth-cloth, same sheet | **0** | | |
| Mandatory | cloth-cloth, separate layers | **0** | | |
| Held redo rounds | cloth-cloth, same sheet | 2856 | 4.74–9.6 mm (1–2 mesh spacings) | 4.8–10.1 mm |
| Held redo rounds | cloth-cloth, separate layers | **0** | | |

**Mandatory round: correct.** The ball starts from rest at 1.86 cm per step into cloth resting on it, whose predictions
are about 0. Every conflict in this round is real, with closure about d + 0.5 mm. Smooth free predictions never make two
vertices of one sheet conflict.

**Held redo rounds: this is the misfiring definition.** The prediction field there is discontinuous: carried vertices get
the ball's displacement, and every other vertex gets exactly 0. Two vertices of one sheet 5–10 mm apart, one carried and
one held, then close by about 1 cm. The conflict test does what it should for those predictions (closure > d), so the
fault is in the predictions. Each redo round carries one more ring of the same sheet: a cascade through a single layer,
not stacked layers being swept.

**Proposed definition fix (pending your ruling; nothing implemented):**
- Drop the "carry" and the held redo rounds for the moving solid. That removes the held rounds, the carried
  displacement, `heldConflict` and `bodySqueeze`.
- Pairs with the moving solid are conflict-tested and truncated against relative motion, in the solid's frame (the
  plane fixed to the solid, which the planes pass already does).
- The cloth then keeps its own smooth prediction. Cloth the ball sweeps into is reached through the cloth-body pairs and
  is pushed by the contact energy within the step's iterations.
- One gap remains: a cloth vertex whose relative motion into the ball exceeds the gap can only be truncated, not pushed,
  within one step. In the solid's frame that truncation leaves it moving with the ball, which is the carry, but only for
  that vertex and without a cascade.

### (1) Corner lift: the per-feature bound, measured

**What I implemented (scratch only, not landed):** `~/code/me/fluid-scratch/c2/solid-field-coverage-refine.ts` and
`solid-field-allowance-attempt.ts`.
- The coverage passes from the prototype.
- Each sample's allowed lift measured as the field's own rounding of its feature. From the surface 2 cells out around
  the sample (8 directions), the lowest bounding face plane gives the sharp feature's height along the sample's normal.
- Planes within the turn floor of the sample's own plane are excluded, since along an edge they are the same rounding.
- Capped at the contact radius r. The cache is keyed by (field, r).

**Finding:** the measured rounding is about zero almost everywhere. On the rotating scene the allowance is p10 = 0,
p50 = 1e-7 m and p90 = 2–5e-4 m.
- A lattice-aligned box edge or corner is exactly sharp in the trilinear field: a box SDF sampled on nodes on its faces
  interpolates to the sharp edge.
- The samples' hull still needs lift there. Samples sit near the edge, not on it, so facets between samples on the two
  faces cut the corner. The convex hull of points on a convex surface always lies inside it.
- So "lift ≤ the field's own rounding" fails at every such feature. Refinement chases it: 5386 → 20294 samples in 4
  rounds, until the build exceeds `Set` capacity.
- I reverted to the committed build.

**Proposed definition fix:** put the samples on the sharp features they bound.
- An edge sample is projected onto the intersection line of its two face planes (the supports already carry them), and
  a corner sample onto the three planes' point.
- A facet across a sharp edge then lies on the faces, so the lift is only needed where the field is genuinely rounded
  (the 45° leg-top bevel, off-lattice edges), which is what the per-feature bound measures.
- The cap r stays the hard ceiling.
- This replaces lift with placement at sharp features, and should also shrink the coverage passes' work. Tablecloth
  tick 222 and the all-scene zero-faces gate wait on it.

### (A) Truncation against the strain limit: not started.
