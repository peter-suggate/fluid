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
