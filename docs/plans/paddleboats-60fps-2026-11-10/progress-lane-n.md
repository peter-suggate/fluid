# Lane N — go/no-go experiments (2026-11-10)

Two experiments decided whether two large bets are worth building. No solver files were changed, and nothing was dispatched on the GPU. Every number in Experiment 1 comes from compiling a pipeline, with no dispatch. Experiment 2 is a CPU Float64 2D prototype.

**Verdicts:**
- **Experiment 1 (hardware-filtered sampling for occupancy): NO-GO.** The paddleboats move stays at 384 threads in every hardware-tap variant.
- **Experiment 2 (high-order 4h cells): NO-GO.** At 6–12 coarse cells per wavelength, the phase and standing-wave decay of arm C are 4–10x worse than arm B. The higher-order criterion also stops flagging moving block corners.

Tools (new files only): `tools/experiments/lane-n/`. Artefacts: `~/code/me/fluid-scratch/lane-n/`.

---

## Experiment 1 — occupancy through hardware-filtered sampling

### Method (measured)

1. **Capture the WGSL.** `tools/capture-uniform-wgsl.mts` captures the WGSL with a mock device (paddleboats, beach, and figure-8 letters), with constants taken from each pipeline descriptor.
2. **Generate variants.** `variant.py` writes each variant:
   - `nbMoveSample` becomes one brick-record storage load plus 3 `textureSampleLevel` taps on `unitHalo`, one per staggered axis.
   - `nbMoveDepth` becomes one filtered `phi` tap.
   - `nowalk` drops the solid walk in `nbFinish`; `norigid` stubs `nbRigidContact`.
   - The surface advect and seed kernels get the same edits.
3. **Compile and read the cap.** `occ.sh` runs Dawn with `dump_shaders` to get the MSL. It strips `[[max_total_threads_per_threadgroup]]` and reads `maxTotalThreadsPerThreadgroup` with lane T's `mslprobe`.
4. **Run serially.** All probes ran one at a time under `tools/gpu/fluid-gpu-run.sh check`.

### The move (`advectParticlesDeferred`, module 080)

The paddleboats (body), beach (terrain) and letters (solid-free) builds give identical caps. Their module-080 descriptors share the same constants for these entries.

| variant | cap |
|---|---|
| baseline | 384 |
| (a) hardware velocity: 3 taps, 1 record | **384** |
| (b) (a) plus hardware depth: 1 tap | **384** |
| (c) (b) plus the solid walk and rigid contact stubbed | **384** |
| (b) plus only the walk stubbed, or only rigid stubbed | 384 / 384 |
| (b) plus the whole `nbFinish` and rigid stubbed | 384 |
| further: separation, surfaceShift, particleBlocked and gridSupported stubbed | 384 |
| further: sparseAge stubbed | 512 |
| further: merge, endpoint, pointKind and tileThin stubbed | 512 |
| further: moveHeat window lookups stubbed | 576 |
| lane T skeleton (all sections stubbed), software or hardware taps | 576 |
| skeleton, returning right after `first` | 1024 |
| skeleton, returning before the axis loop / after G2P | 896 / 896 |
| skeleton, returning after the G2P predicates | 832 |
| skeleton, with the trace cut after stage 0 | 768 |
| skeleton, returning after the trace loop | 640 |
| **real kernel with hardware taps, returning right after G2P** | **576** |
| **real kernel with hardware taps, returning right after the trace loop (no finish, no rigid)** | **384** |

**Skeleton with hardware taps plus one real section:**

| real section | cap |
|---|---|
| separation | 576 |
| surfaceShift | 576 |
| mergeRun | 576 |
| traceEndpoint | 576 |
| gridSupported | 512 |
| particleBlocked | 448 |
| sparseAge | 448 |
| rigidContact | 384 |
| finish | 384 |

**Where the cap is.** It is not the velocity or phi sampling. Replacing the 24 software taps with 3 hardware taps changes nothing anywhere.

Three sections hold the cap, and each is enough on its own:
- **The real 3-stage trace loop.** Even with hardware taps and the finish and rigid contact cut away, it is 384 by itself. Its cost is separation, surface shift, the blocked test and `gridSupported` per stage.
- **`nbFinish`**, which pins the cap to 384 on its own.
- **`nbRigidContact`**, which also pins it to 384 on its own.

The other measurements:
- **The prefix before the trace already caps at 576** with hardware taps: the merge plus the G2P predicates.
- **The skeleton reaches only 576.** Its loop alone costs 832 down to 640.

**Best cap with real behaviour: 384**, unchanged. No section split reaches 768:
- the prefix gives at most 576;
- the trace gives 384.

Lane T's earlier split already measured no step-time win.

The `hw-nomerge` probe failed to generate (its substitution did not match) and is not counted.

### Surface advect (module 028) and seed (modules 079/080)

Surface advect:

| entry | base | hw vel | hw vel + hw phi |
|---|---|---|---|
| advect | 576 | 576 | 576 |
| advectFine | 384 | 384 | 384 |
| advectTrace | 512 | 768 | 768 |
| advectFinish | 384 | 384 | 512 |
| advectCubic / advectCubicDeferred | 448 / 384 | 448 / 384 | 704\* / 704\* |
| advectWalls | 576 | 576 | 576 |
| advectDeferred | 384 | 384 | 384 |
| advectStagedTrace | 384 | 448 | 448 |
| advectStagedCell | 384 | 384 | 1024\* |
| advectStagedTail | 384 | 384 | 384 |
| advectOwners | 384 | 384 | 384 |

\* These caps are higher only because the probe swaps the cubic or trilinear departure phi for one linear tap. That removes behaviour (the cubic) and is not a valid gain.

**Paddleboats compiles the merged deferred entries** (`umMergedTiles = 1`), and they stay at 384.

**Probes per descriptor.** Every module-028 pipeline descriptor (57–76) was probed. The paddleboats (body) and beach (terrain) builds give identical caps descriptor for descriptor; the letters build has the same descriptor set.
- **Twinned deferred, descriptors 64 and 65:** 384 in every variant.
- **Solid-free deferred, descriptors 72, 73 and 75:** 384 at base. With the hardware velocity and hardware phi taps they reach 384, 448 and 512.
  - Descriptor 75 is the fast-surface variant, so its phi tap stands in for a trilinear lookup. Its 512 is real.
- **Nothing in the surface advect reaches 768 with its behaviour intact.** The one exception is `advectTrace` (fast surface), which is 768 with hardware velocity.

Seed:

| entry | base | hw vel + hw depth |
|---|---|---|
| seed / seedSources / seedRank | 512 / 512 / 576 | 896 / 896 / 896 |
| seedCells | 576 | 640 |
| seedInitial | 384 | 512 |
| airPlanSeeds | 512 | 512 |

The seed kernels run once per tile event and are not on the step floor.

### Apron atlas cost

These are measured from the code; the costs at the end are estimates.

- **The store exists but is unused.** `UniformFineBrickStore` has "sampled" channels: r32float 3D atlases, 10³ texels per 8³ brick. The one-texel apron is 488 extra texels per channel per brick, a 1.95x overhead. `apronWGSL` refreshes the apron. No product channel sets `sampled: true` today.
- **Filtering and format.** The adapter has `float32-filterable`, so r32float filters and f16 is not needed for filterability.
  - f16 would do for phi and the k2/k3 trace velocities.
  - G2P's FLIP delta needs f32.
  - Hardware weights round to about 1/256 of a texel (the particle-stage-gap doc).
  - The measured filter rate is 13.5 G samples/s for rgba32f and 28.4 G for rgba16f.
- **The h/4h velocity blend band** (`umVelocitySamplingWeightsAt`) has no hardware equivalent. It would have to be baked into the 4h atlas or dropped.
- **Estimated memory and refresh:** about 1,000 active bricks x 4 channels (u, v, w, phi) x 4 KB, which is about 16 MB. The refresh is about 2 M texels per step, roughly 0.05–0.1 ms or one fused epilogue.

**This is moot for occupancy. The atlas buys no thread cap on the move.**

### What is assumed

- The record lookup stands in for the real atlas indirection, using one dependent storage load per sample.
- The caps are compile-time register caps. Runtime occupancy was not measured, by rule.
- The shading caps assume the descriptor constants of step-300 paddleboats.

---

## Experiment 2 — high-order coarse (4h) cells

### Prototype (`wave2d.ts`, CPU Float64 2D)

**Discretisation:**
- MAC grid with nodal phi.
- Phi traced with RK3; u traced with RK2 semi-Lagrangian.
- Ghost-fluid pressure (theta ≥ 1e-3), PCG solved to 1e-11.
- Redistance with a narrow band of 8 cells.

**Sanity checks:**
- Still water stays exact.
- B converges at first order. A standing wave of 0.8 m (a = 0.5 h) has phase error −2.4% at h and −1.2% at h/2.

**Arms:**
- **A:** 4h, linear, semi-Lagrangian.
- **B:** h, linear, semi-Lagrangian.
- **C:** 4h, clamped Catmull-Rom cubic, MacCormack with the Selle limiter for phi and u.
- **D:** C plus a cubic-reference shape criterion.

The paddleboats scale is used throughout: h = 2.5 cm, depth 0.55 m, dt = 1/60 s, 5 s runs.

### Waves

Each cell is phase error % / amplitude decay per period %. λ is in coarse (4h) cells. Amplitude is 1 h unless noted.

| case | A | B | C |
|---|---|---|---|
| travel, λ = 4 | −50 / 13 | +1.4 / 10.7 | −50 / 5.7 |
| travel, λ = 6 | −31 / 12 | +0.7 / 9 | −32 / 6 |
| travel, λ = 8 | −21 / 14 | +0.3 / 7 | −12 / 4.3 |
| travel, λ = 8, 2 h | −19 / 18 | +2.9 / 12 | −3.7 / 11 |
| travel, λ = 12 | −9 / 14 | −0.1 / 3.3 | −5.8 / 1.6 |
| travel, λ = 12, 2 h | −8 / 18.5 | +2.3 / 6.8 | −1.4 / 3.5 |
| travel, λ = 16 (amplitude 1 / 2 / 4 h) | — | −0.4 / +1.3 / +5.7 phase, 1.4 / 4.1 / 10.2 decay | −4.4 / −2.5 / +0.7 phase, 1.1 / 1.7 / 6.4 decay |
| standing, λ = 4 | −50 | −6.4 | — |
| standing, λ = 6 | −32 | −3.8 | — |
| standing, λ = 8 | −22.5 / 12 | −2.7 / 4 | +2.6 / 11.7 |
| standing, λ = 12 | −13 / 16 | −1.7 / 2 | −13 / 15 |
| standing, λ = 16 | −8 / 15–19 | −1.2 to −2.5 / 1–4.6 | −8.5 to −10 / 8–17 |

Volume drift over 5 s is −1 to −3.5% for every 4h arm, and between +0.2% and −2.5% for B.

**Finding.** The 4h phase error comes from the coarse pressure and free-surface discretisation, not from transport. Arm A at 8, 16, 32 and 64 cells per λ gives −20%, −5%, −2% and −0.4%. Cubic transport plus MacCormack fixes the travelling decay for λ ≥ 8 cells. It does not fix phase error at 6–12 cells, nor the standing-wave decay (8–17% against 1–7% for B).

**The GO clause "within 2x of B for λ ≥ 6 cells" fails by 4–10x.** The paddleboats wake band, 0.3–1 m, is 3–10 coarse cells.

### Wake tiles over the 0.5 h tolerance

These come from `shape.ts` and `run-shape.ts`: a 48 x 48 tile survey of a synthetic directional wake. The two numbers are the fraction of tiles flagged by (i) the trilinear criterion and (ii) the cubic criterion.

| surface | (i) trilinear | (ii) cubic |
|---|---|---|
| slope 0.12, 0.3–1 m, smooth | 0.95% (22/2304) | 0% |
| slope 0.06 / 0.2 | 0% / 22% | 0% / 0.5% |
| slope 0.12 plus h-scale roughness of 0.1 / 0.2 / 0.3 h | 6.9 / 35 / 73% | 0.04 / 3.6 / 16% |
| spectrum 0.2–0.4 m / 0.6–1.6 m | 23% / 0% | 0% / 0.09% |

(ii) is under a third of (i), so that clause passes. But it passes against a tiny base: a smooth swell of slope 0.12 already releases 99% of its tiles under trilinear.

Paddleboats at step 300:
- It holds 3,526 h tiles.
- 2,491 of them are held by the join reason (particles, heat, collar); 1,740 of those hold particles.
- **Only 759 are held by the shape criterion.**

The cubic criterion could release at most those 759, before the join reason re-claims them.

### Features (the cases that must stay flagged)

These come from `run-features.ts`. The criterion is evaluated on B's h field at frames 0, 5, 10, 15 and 20. Feature tiles move with the fall. Cells are flagged / feature tiles.

| case | trilinear flags (frames 5–20) | cubic flags (frames 5–20) | volume error left coarse at frame 20: A / C (B) |
|---|---|---|---|
| a1 block, corners on 4h vertices | 12/12, 12/12, 6/6, 8/8 | **2/12, 2/12, 6/6, 4/8** | −52% / −16% (−2.3%) |
| a2 block, corners off the lattice | 10, 10, 11, 12 /12 | **8, 8, 4, 4 /12** | −46% / −7% (−2.5%) |
| a3 block at 45° | 15, 15, 16, 13 | **9, 9, 6, 8** | −52% / −13% (−1.5%) |
| b sheet 1 / 2 / 3 h thick | all | all (3 h: 7/8 at frame 20) | −100% / −100% / −72%, already lost at frame 0 |
| c column impact | 8, 13, 10, 4 | 4, 9, 7, 7 | −17% / −24% (−6.6%) |

At rest (frame 0), the trilinear criterion flags 0/7 corners of the lattice-aligned block (a known blind spot), and the cubic one flags 2/7.

**Once the block moves, the cubic criterion under-flags corners by 2–6x.** The trilinear criterion's corner and curvature terms are what catch them.

**This fails the generalisation clause.** Restoring those terms in D would re-flag the rough swell as well.

Sheets of 1–3 h cannot be represented at 4h at all.

### Arm C cost in the real solver (estimate)

- **Cubic phi** is a 4³ vertex stencil, radius 2. It already exists at 4h (`umCubicPhi`, the "Best" mode).
- **MacCormack** per field (phi, u) needs:
  - a backward re-trace and a second interpolation;
  - the limiter reading the departure stencil's min and max.
- That is one extra launch per field, or a second fused sweep. The halo is 2 coarse cells plus the trace reach (CFL x 4h), against today's single trace.

### What is assumed

- **The prototype is 2D.** The 3D criterion terms are carried over in their 2D form.
- **The wake is synthetic,** a 64-component directional spectrum. It is not a paddleboats capture.
- **The join and particle reasons are taken from the step-300 census.** They were not re-simulated.

---

## Smallest next step

None. Neither bet is worth building.

- **Occupancy.** The cap belongs to the move's control flow: finish, rigid contact, and the G2P predicates. Sampling does not hold it.
- **4h fidelity.** It is bounded by the coarse pressure and surface discretisation. In paddleboats, refinement is driven by particles and the join reason, not by the shape of the swell.
