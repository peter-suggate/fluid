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
