# Halving the pour's late frame: an audacious plan

2026-10-20. Target: `nbflip-figure-7-pour` at 1/60 s, late frames f200–212, on the M1 Max. Peter asked for a plan to double performance, with physics changes allowed.

This plan was written read-only, from the code and the profiles. It made no GPU runs. Every ms figure comes from a profile named below or from a measurement recorded in another plan doc. Projections are labelled as projections.

## Summary

The late frame is 57.2 ms of pass time over 364 dispatches, plus about 8–11 ms of idle. Halving it means deleting about 29 ms of pass time.

The levers fall into three packages:
- **Package A** shrinks the work. Most of it is parameters, and four of them already exist.
- **Package B** deletes duplicated work across stages.
- **Package C** rewrites the remaining worst kernels.

At the central estimate, A and B alone reach about 33 ms (1.7×). Adding C reaches about 28.6 ms (2.0×). The honest range is 22–37 ms. The spread comes mostly from physics gates that have not been run.

| # | Lever | Package | Central saving (ms) | Range (ms) | Physics |
|---|---|---|---|---|---|
| 1 | The band in cells: particle band 3h, depth-graded density, padding 0 | A | −6.8 | −4 to −9 | Yes, new param plus `fineGridPadding` |
| 2 | Pressure in two cycles on one ownership | A+B | −5.9 (−1.7 of it already in flight) | −4 to −7 | Yes, `bandCycles` |
| 3 | A lazy level set: trilinear advect, advect band, one-candidate rotating redistance | A+C | −5.3 | −3 to −7 | Yes, three params |
| 4 | One nearest-surface search: extension from the redistance's bank B | B | −1.5 | −0.5 to −2.3 | Yes, new param |
| 5 | One tile summary at the head: census, local speed and work control | B | −1.5 | −0.9 to −2.0 | No (exact) |
| 6 | Lean particle transfer on brick windows | C | −1.7 | −1 to −2.5 | No (summation order) |
| 7 | Forces at the writers; sub-grid viscosity skipped | B | −0.8 | −0.4 to −1.2 | Yes, new param |
| 8 | Paired remap faces and publish | B | −0.6 | −0.4 to −0.8 | No (exact) |
| 9 | Structural tail: brick-resident cycles, persistent fine φ, head re-plan, lean advectFine | C | −4.6 | −2 to −7 | Mostly exact |
| | **Total (with the overlaps removed)** | | **−28.6** | **−20 to −35** | |

The headline swings are levers 1–3. They are big because each one deletes elements that the physics does not need, rather than making existing elements cheaper:
- **Lever 1** removes about 40% of the particles and about 20% of the fine tiles.
- **Lever 2** runs half the V-cycles.
- **Lever 3** advects and redistances only the vertices whose values survive.

**First step:** one late-frame pour profile and one letters A/B (two repeats an arm) for a single arm of existing parameters, with no code change:
- `phiFineAdvection` trilinear;
- `fineGridPadding` 0;
- `bandCycles` 3.

It projects about −4 ms. `bandCycles` 2 then gets an arm of its own.

## 1. Where the late frame goes

Profile `scratchpad/P/delta12.json`, which is HEAD 823ccc55 (the swept FLIP delta), frames 200–212:
- 10.9k–11.3k fine tiles;
- 46.6k coarse tiles;
- 1.33M–1.42M particles;
- 6.0k–6.4k pressure band tiles, at 4 cycles every frame;
- 364 dispatches.

Stages are summed from `summary.passMeans` by label (scratch script `plan2x/group.py`).

| Stage | ms | Largest passes (ms) |
|---|---|---|
| Surface | 17.84 | advect group 6.93 (advectFine 2.02, advectFinish 1.47, deferred 1.40, two cubics 1.13, trace 0.56); reconstruction 3.66 (coupleFine 2.33, splat 0.96); redistance group 5.22 (arenaDistance 2.16, arenaStore 1.33, arenaNearest 0.63, two gathers 0.62); phi resolve 1.11 (5 launches) |
| Pressure | 12.53 | solve 6.65 (123 dispatches: pRelax 2.26, coarse tail 1.40 in 60 dispatches, levels 1–2 1.53); all-4h mirror 1.88; authority 1.65 (12); pRows 1.67; arena projection 0.53 |
| Particles | 10.72 | advectParticles 3.60; transfer 3.98 (deposit 1.84, refine 0.76, properties 0.76, seams 0.56); sort 0.77; seed 0.75; recordDelta 0.71 |
| RFE | 9.42 | extension 3.07 (23 dispatches); remap 2.24; geometric fill 1.89; body forces 1.68; hanging taps 0.53 |
| Other | 5.24 | census 2.62 (classify 1.28); frame plan and local speed 1.93 (localSpeed 1.33) |
| Momentum | 1.49 | momentumFine 0.59, momentumTrace 0.43, momentumStep 0.36 |
| **Total** | **57.24** | |

The stage order the levers rely on is in `uniform-mixed-frame.ts`:

| Step | Lines |
|---|---|
| Head: relayout, remap, geometry, authority | 1433–1476 |
| Certificate | 1005 |
| Particle move | 1024 |
| Surface advect and resolve | 1027 |
| Reconstruct | 1035 |
| Air clear and resolve | 1044 |
| Retire, redistance and resolve | 1045 |
| Geometry | 1069 |
| Momentum, plan, authority, transfer | 1074 → 1200–1205 |
| Classify and forces | 1091 |
| Mirror transfer, split authority, band prepare and solve, 4h project, transfer back | 1105–1127 |
| recordDelta and the extension pair | 1142 |
| Tail plan | 1184 |

## 2. Five facts that set the shape of the plan

1. **The fine band is about one tile thick on each side of the surface.**
   - 5.4k of the 6.0–6.4k wet band tiles are interface tiles (`uniformMixedDynamicInterfaceTiles` 5427 at f212).
   - So "coarser pressure in the bulk, h only at the surface" reaches at most about 10% of band rows. It is not a lever.
   - The same thinness is why staged windows lost (sAdvect at 4.22 against 2.4 ms) and why the arena's per-brick boxes lost to per-tile lists (pressure-rfe §1.7–1.8).
   - The band's cost follows its thickness in cells, which only lever 1 changes.
2. **Seams are everywhere.**
   - 6709 of 10893 fine tiles are seam tiles (62%).
   - Seam variants cost more than the regular kernels in every stage that has both: extension seams 1.37 against regular 0.80, forces' seam launch about 0.76, transferSeams 0.56, hanging taps 0.53, phi resolve 1.11.
3. **The particles are about 9 a cell in all four depth layers** (particle-stage-gap §5).
   - Seeding fills 8 lattice sites a cell down to −4h (`uniform-narrow-band-flip.ts:1211–1213`).
   - The move retires below −4h, and crowding retires only above rank 16, below −1h (`:978–981`).
   - Layers 2–4 hold about 74% of the particles. Only the top layer defines the surface: couple clamps the advected value into the interval the nearest samples allow (`uniform-narrow-band-surface.wgsl.ts:42–94`). The deeper layers carry momentum.
4. **Pressure runs its cycle cap on every frame.**
   - The band receipt reads `[tiles, 4, residual]` with residuals 0.02–0.23 1/s against the 0.05 stop.
   - A cycle costs 1.50 ms: (6.65 − 0.67 fixed) / 4.
   - 0.35 ms of each cycle is the coarse tail: 15 dispatches a cycle (`uniform-pressure-arena.ts:1910`).
   - pTail itself is one 256-lane workgroup running 32 red-black sweeps (`:1518–1540`). It is latency, not work.
5. **Two stages compute the same "nearest surface" quantity, and three head passes load the same 125 vertices a tile.**
   - The redistance's arenaNearest writes each band cell's nearest crossing cell (bank B, `uniform-surface-brick-arena.wgsl.ts:16–22`).
   - The velocity extension then propagates (value, distance) pairs from the nearest liquid face by sweeps and a 4h pyramid (`uniform-mixed-extension.ts:48–75`). That is the same closest-surface question, on the same final φ: nothing writes φ between the redistance and the tail extension.
   - At the head, localSpeed (`uniform-mixed-frame-plan.ts:249–300`), census classify (`uniform-mixed-dynamic.ts:892`, `:934`) and geometry's work-control compare each load a fine tile's 125 φ vertices and its face extents.

## 3. The levers

Each lever gives:
- the passes it deletes or shrinks, with delta12 ms;
- the speed-of-light (SOL) figure for that work, at the measured ceilings (360 GB/s; gathers 15–34 G/s; workgroup memory 719 G/s; workgroup atomics 218 G/s; 2.3 µs a pass);
- the physics risk and how the letters gate (`nbflip-figure-8-letters`, two repeats an arm) catches it;
- its dependencies.

### Lever 1: the band in cells (package A, −6.8 ms)

**What changes.** These are three physics parameters, all under NB-FLIP method params with update "solver":

- **`particleBandDepth` (new): 4 → 3 cells.**
  - The seed floor (`uniform-narrow-band-flip.ts:1213`), the move's retirement (`:978`) and the census's fine-tile retirement depth move together.
  - Seed to −2.5h and retire at −3h, so the seed/retire hysteresis the particle doc asks for comes with it. It cuts the churn of about 7% of particles a frame.
- **`particleDensity` (new): "graded".**
  - 8 sites a cell above −1.5h, 4 below.
  - The crowding cap below −1.5h drops from rank 16 to rank 6 (`:979–981`).
- **`fineGridPadding` 0** (exists, `uniform-narrow-band-method.ts:29,68`).

**What it shrinks.**

| Change | Effect |
|---|---|
| Particles | Layer 4 goes (about −18%). Layers 2–3 fall from about 9.6 to about 5 a cell (about −27%). About −40% in all, so 1.4M → about 0.85M. |
| Fine tiles | Padding 0 measured −17% at f94 (brick-arena §6.5). Band 3h removes part of the liquid-side tile layer. Projected −20 to −25%. |

| Stage | Projected ms | Basis |
|---|---|---|
| Particles | −3.8 | About 9.5 ms of the 10.7 scales with the particle count (move, sort, deposit samples, refine, seams, properties, recordDelta, seed) |
| Surface splat | −0.35 | 0.96 ms, per particle |
| Surface | −1.4 | Fine-tile share. pad0 measured −0.7 ms surface for −17% tiles at f94; the band-3h tiles are liquid-side and costlier per tile |
| RFE / pressure / other / momentum | −0.6 / −0.3 / −0.2 / −0.1 | Fine-tile share, at the weak elasticity pad0 showed |
| **Total** | **−6.8 (range −4 to −9)** | |

**SOL.** Particles at the bench chain's 3 ns/p and 0.85M is about 2.5 ms. The design bound is 1.4 ns/p, about 1.2 ms.

**Physics risk: medium-high.**
- A 3-cell FLIP band is inside the range Ferstl et al. 2016 report as close to full FLIP. But the grid-advected interior starts nearer the surface.
- 4–5 samples a cell in the deep band make P2G noisier.

**How the gate catches it:**
- volume drift and occupancy L1 at f60/f120/f240;
- for padding 0, particles meeting hanging seams show as occupancy changes at the impact frames.

**The gate's particle-count test needs Peter's call.** This lever lowers the count by design. The proposal: replace the "count within noise" test with "count within noise of the expected ratio, and no growth trend".

**Dependencies:** none. Do it first, because it changes the base every later lever is measured on.

### Lever 2: pressure in two cycles on one ownership (packages A+B, −5.9 ms)

**What changes:**
- **2a. Retire the all-4h mirror (in flight, the pressure session; df23bacf is step 1).**
  - Deletes the transfer to pressure 0.56, the split authority (0.79 of 1.65), the root rhs 0.24, the root project 0.73, the transfer to simulation 0.31, root owners 0.05 and geometry's mirror (`uniform-mixed-frame.ts:1105–1127`).
  - The sizing doc nets −1.5 to −1.9 after the lean replacements (pressure-rfe §1.10). Count −1.7.
- **2b. `bandCycles` 2** (exists, `uniform-near-settled.ts:78`).
  - Each cycle is 1.50 ms, so −3.0.
  - The schedule already encodes min(cap, last wide count + 1) (`uniform-pressure-arena.ts:1903`), so frames that converge early stop early.
  - The conservative arm is 3 cycles (−1.5). Count −2.6 central.
- **2c. One tail cycle, not three** (`tailCycles` 3 → 1, `uniform-pressure-arena.ts:55`, `:1910`), with pTail's 32 sweeps raised to cover it.
  - This saves 2/3 of 0.35 ms a cycle, about −0.45 at two cycles, and 30 dispatches.
  - The 3e1db823 seam twin is the check: "deep natives need a stronger tail".
- **2d. Authority once and lean pRows.** −0.9 together.
  - After 2a, the simulation authority still runs twice: at the head after the remap (`:1475`) and after momentum (`:1204`). The head run feeds momentum's phase.
  - pRows (1.67) still spends about 0.2 on forced velocity and runs one lane a cell over every band tile (pressure-rfe §1.9).

**Projected:** pressure 12.53 → about 6.6 ms, with 123 → about 50 solve dispatches.

**SOL:** the solve is ≈0.26 ms at 2 cycles, ≈0.41 at 4 (pressure-rfe §1.4, §1.6). The projected 6.6 ms is still about 15× off. The prior that breaks is "per-tile in-place relaxation over a thin band". Lever 9's brick-resident cycles are the answer named for it.

**Physics risk: medium.**
- 4 cycles already end at 0.02–0.23 1/s, not 0.05.
- From a 4h start the old history was 30.6k → 1862 → 61 → 3.7 → 0.79. The warm start begins far lower, but how much lower is unmeasured.
- 2 cycles could leave 1–4 1/s in the worst cells: 2–7% local volume error a step, which particles then carry.

**How the gate catches it:**
- volume drift at f240 (unconverged divergence loses or gains volume);
- the band residual p50/p90 from the receipt;
- occupancy L1 at the impact frames 34–36 and 55–57.

**First measurement:** the band's residual history word (`UniformPressureArena.historyWord`) at f200–212 and on letters. It decides between 2 and 3 cycles.

**Dependencies:** 2a is in flight and independent. 2c follows 2b. 2d follows 2a.

### Lever 3: a lazy level set (packages A+C, −5.3 ms)

**What changes.** Four sub-levers, each behind a param.

**3a. `phiFineAdvection` trilinear** (exists, `uniform-narrow-band-controls.ts:20–22`).
- It deletes both advectCubic launches (1.13) and the cubic routing in advectFinish (about 0.1–0.2).
- The control's own hint records the cost: curved surfaces the particles do not carry shrink, and figure 9 held 5–9% fewer particles.
- In the band the particles pin the value. Couple clamps the advected value into the interval between the nearest sample's sphere and shell (`uniform-narrow-band-surface.wgsl.ts:44–60`), so the cubic's sharpness mostly survives only where θ < 1.
- Projected −1.2 ms.

**3b. `surfaceAdvectBand` k = 4 cells** (new, designed in brick-arena §6.3).
- Vertices with |φ_old| ≥ 4h keep φ_old. The redistance overwrites their magnitude, and only their sign is read.
- At f94 this removed 44% of traced vertices. After 3a, the remaining advect chain is trace 0.56, finish about 1.3 and deferred 1.40, plus advectFine's early-out.
- Projected −1.2 ms (k = 6, the conservative arm: about −0.6).

**3c. A one-candidate redistance** (new param, `surfaceDistance: "exact" | "nearest"`).
- arenaDistance evaluates the nearest crossing cell of each of a vertex's eight incident cells (`uniform-surface-brick-arena.wgsl.ts:20–21`): 1.47M pairs at f94, about 7 a vertex.
- The measured split of R is per-target setup 54%, edge sweeps 31% and Newton 13% (brick-arena §5.5). The cost is per pair.
- Evaluating only the incident cell whose bank B entry is nearest cuts the pairs about 7×, while each pair keeps its exact patch distance.
- Projected −1.0 ms of R's 2.16.

**3d. A rotating redistance** (new param, `surfaceRedistanceEvery` 1 | 2).
- A brick is redistanced and stored on alternate steps, by brick parity.
- A brick is forced every step when it gained a crossing, or when its tile's local-speed reach is saturated.
- The search (arenaCross, arenaNearest) still runs every step, because bulkDepth, the air update and the membership read it.
- It halves arenaDistance's remainder, arenaFar 0.17 and arenaStore 1.33. The cost is level per frame; this is not an alternating heavy frame.
- Projected −1.3 ms.

Two smaller pieces belong with these:
- **3e.** Fold the post-reconstruct and gated resolves into the arena's derived gather, which already re-forms hanging vertices at staging (`uniform-surface-brick-arena.wgsl.ts:10–12`): −0.5.
- **3f.** A lean class split of coupleFine, from the class product at f200 (24% θ ≤ 0, 28% bulk only, 49% finish; brick-arena §6.10): −0.6, range 0 to −1.0.

**Projected:** the surface stage takes −7.25 ms with lever 1's share, so 17.84 → about 10.6.

**SOL:**
- advect's interpolation floor through the texture unit is ≈0.35 ms (§6.2);
- R's exact ALU floor is ≈0.1–0.15;
- the surface's realistic floor is about 1.5 ms.

**Physics risk:**
- **3a: low to medium.** Particle count on letters.
- **3b: medium-high.** About 400 missed sign flips a step early in the pour. Liquid → air misses leave one-step phantom fragments. Watch occupancy L1 and the impact frames.
- **3c: medium.** Errors up to about 0.5h where the eight candidates disagree (thin features). reach 1 failed its A/B through the seeding depth: +6–8% particles. This lever reads the same way, so watch the particle count first.
- **3d: medium.** One step of |∇φ| drift on half the bricks. Seeding depth again, so particle count and drift.

**Dependencies:** 3b comes after 3a. 3d comes after 3c, which defines what a "stale" brick is.

### Lever 4: one nearest-surface search (package B, −1.5 ms)

**Duplicated work.**
- arenaNearest gives each cell of a reach-1 tile its exact nearest crossing cell (bank B) on the step's final φ.
- The tail extension then finds each air face's nearest liquid source again, with a seed, a sweep, two seam sweeps, a seam seed, publishes and a 4h nearest-source pyramid: 23 dispatches, 3.07 ms.

**What changes** (param `velocityExtension: "sweeps" | "closest"`):
- An air face in a reach-1 tile takes its value from the liquid face beside its cell's bank B crossing cell, in both channels of the pair, through one address resolution.
- Faces beyond reach 1 keep the 4h pyramid's far value (`uniform-mixed-extension.ts:48`).
- This deletes the h seed, sweeps and seam sweeps (seed 0.13, sweep 0.49, seam sweeps 0.61, seam seed 0.27, seed list 0.07) and most of publishSeams (0.49).

**Projected:** −1.5 ms, range −0.5 to −2.3. The pyramid (about 0.53) and retain (0.19) stay.

**SOL:** 576k air faces × 2 channels × 16 B ≈ 18 MB, about 0.05 ms.

**Physics risk: medium.**
- Closest-point extension and today's nearest-source sweeps solve the same constant-along-normals problem. They differ at concave corners and at seams.

**How the gate catches it:**
- occupancy and drift on letters;
- the band residual (extended air faces feed the next step's trace and G2P).

**Dependencies:**
- bank B must still cover every face the extension writes at h. Today reach is 3, so it does.
- With lever 3d, bank B is built every step, so it stays fresh.

### Lever 5: one tile summary at the head (package B, exact, −1.5 ms)

This is cross-stage duplication doc #3, widened.

**What changes:**
- Geometry's work-control compare, which already loads all 125 vertices a tile, emits a per-tile summary: crossing mask, inside/deep bits, value range and resolution error.
- The tail publish emits per-axis face extents.
- Census classify (1.28) and classifyCoarse (0.29) become lean passes over the summaries.
- localSpeed (1.33) takes the extents instead of reloading 125 vertices and every face a tile (`uniform-mixed-frame-plan.ts:252–277`).

**Projected:** −1.5 ms (range −0.9 to −2.0).

**SOL:** 11k tiles × 32 B, about 0.01 ms plus passes.

**Risk:**
- held regions skip the compare, so their summaries must stay valid;
- the face set must match `umFaceLow`'s skips;
- freeze steps write the head.

The gate is a bit-identical census verdict, checked with a scratch twin.

### Lever 6: lean particle transfer on brick windows (package C, −1.7 ms)

**What changes.** transferDeposit (1.84) runs one 192-lane workgroup per 4³ tile, with about 70% of its lanes idle and a halo ratio of 4 (particle-stage-gap §2b). It becomes the bench's DEPB:
- 8³ brick windows;
- subgroup pre-reduction;
- a halo ratio of about 1.6.

The fat-kernel rule holds. Seams, refine and properties stay separate lean passes, but read DEPB's compact per-brick product instead of re-walking records and cell runs:
- refine re-reads every record after the deposit (`uniform-particle-brick.wgsl.ts:215`, against `:146`);
- seams gather 4h owner faces from cell runs.

The refine-into-deposit fold is a measured null (+0.76) and is not retried.

Seed slab slack and a pure-stream sort join this lever:
- about 50% of seeds spill;
- the sort is about 3× off streaming.

**Projected** (on lever 1's 0.85M particles): about −1.2 transfer and −0.5 seed and sort.

**SOL:** at the bench's DEPB 0.72 ns/p plus FIN, about 0.8 ms at 0.85M.

**Risk:** summation order only. The gate is the letters A/B.

**Dependencies:** lever 1 first, because it sets the count.

### Lever 7: forces at the writers (package B, −0.8 ms)

**Why.** Body forces cost 1.68 ms in 4 launches. Knock-outs show about 0.6 ms of the fused seam launch remains with every term removed: the cost is the face walk, not the terms (pressure-rfe §2.5).

The pour's viscosity is water's. At h = 2.2 cm and dt = 1/60 s, ν·dt/h² ≈ 3·10⁻⁵, far below the scheme's own numerical diffusion. Yet viscosity costs 0.28 ms in the knock-outs.

**What changes:**
- **`subgridViscosity: "skip"` (new).** Viscosity is skipped when ν·dt/h² < 10⁻³.
- **Gravity moves to the writers.** It is added at the deposit flush and the momentum store wherever `uniform-mixed-forces.ts:360`'s occupancy test would add it.
- **Forces runs only over what is left.** That is inflow, behind an exact reach test, and capillarity, which is off in the pour.

**Projected:** −0.8 ms.

**Risk: low.** The occupancy rule must match exactly at the writers. The gate is the letters A/B.

### Lever 8: paired remap faces (package B, exact, −0.6 ms)

This is cross-stage duplication doc #4. The two remapFaces launches (1.67 ms) walk the same listed tiles and addresses. Pair them, and pair the two publishes (0.21).

### Lever 9: the structural tail (package C, −4.6 ms)

| Sub-lever | ms | Basis |
|---|---|---|
| Brick-resident pressure cycles: halo-1 shifted partitions, level 0 resident for two sweeps a load, only for bricks holding at least 4 band tiles; tile lists elsewhere | −1.0 | MG microbench: 0.29 ms a cycle at 603 bricks, 0.44 at twice that (pressure-rfe §1.6). The thin band is why per-brick boxes lost, hence the occupancy threshold |
| Persistent fine φ (brick-arena P1): arenaStore's rule chain, the two gathers and the remaining resolves go | −1.0 | P1 was sized at ≈1.0 ms and shelved alone. It is worth more once lever 3 has cut the store's share and lever 4 reads the same windows |
| Head re-plan: the head's geometry, authority and plan run after the remap, then run again after the surface. Fold the head's into the census frame's tail | −0.8 | geometryChanged 0.26, about 0.43 of authority, 0.18 of the frame plan seed. **Unverified:** momentum reads the head authority's phase |
| advectFine as a lean classify plus listed kernels (fat-kernel rule) | −0.8 | 2.02 ms at about 3 ns a vertex. Owner and burial walks set its occupancy (brick-arena §5.2) |
| The particle trace at RK2 with rgba16f k2 (param) | −0.4 | 9 → 6 fetches a particle; k2/k3 cost about 0.47 ns/p (particle-stage-gap §2a, §4) |
| Seam lean lists for the extension's remaining seams and the forces seam launch | −0.6 | Seam variants cost about 2× the regular ones (§2, fact 2) |

## 4. Projection, stage by stage

Applied in order, so each lever acts on the base the levers before it left. Central figures, in ms:

| Stage | Today | A + B | After C | Main contributors |
|---|---|---|---|---|
| Surface | 17.84 | 10.6 | 9.0 | L1 −1.75, L3 −5.5 |
| Pressure | 12.53 | 6.6 | 5.6 | L2 −5.65, L1 −0.3, L9 −1.0 |
| Particles | 10.72 | 5.2 | 4.8 | L1 −3.8, L6 −1.7, L9 −0.4 |
| RFE | 9.42 | 5.9 | 5.3 | L4 −1.5, L7 −0.8, L8 −0.6, L1 −0.6, L9 −0.6 |
| Other | 5.24 | 3.5 | 2.7 | L5 −1.5, L1 −0.2, L9 −0.8 (head) |
| Momentum | 1.49 | 1.4 | 1.2 | L1 −0.1, writers |
| **Total** | **57.2** | **33.2 (1.72×)** | **28.6 (2.0×)** | |

**Honest uncertainty.**
- Levers 1, 2b, 3b and 3c carry about 15 ms between them. All four are physics changes, and none has a letters result yet. Any one that fails its gate costs 2–5 ms of the total.
- The fine-tile elasticity is the largest single unknown. pad0 at f94 gave only −1.0 ms of frame for −17% tiles.
- The exact levers (5, 8, 2a) were each sized from knock-outs or twins. Their ranges are narrow.
- Range: about 22–37 ms.

**Idle.** Projected dispatches fall from 364 to about 230:
- pressure −73;
- extension −12;
- resolves −3;
- the tail and forces −10.

If idle scales with pass boundaries (brick-arena §1.4's prior), the 8 ms of idle becomes about 5. This must be measured, not assumed.

## 5. Considered and rejected

| Idea | Why not |
|---|---|
| Coarser pressure in the bulk, h only at the surface | 5.4k of about 6.2k band tiles are interface tiles: at most about 10% of rows, for a 2:1 seam inside the band |
| A particle-only surface (no grid advect) | Couple's answer depends on the advected value wherever it is deeper than the samples' interval, and on θ < 1 tiles (`uniform-narrow-band-surface.wgsl.ts:30–60`). The deep sign has no particle evidence. Lever 3b takes the part that is safe to take |
| The extension as one resident staged pass | The thin band: staged windows load mostly 4h texels nobody reads (sAdvect 4.22 against 2.4 ms). Lever 4 deletes the search instead |
| Tile context as a general lever | The coupleFine cell table was null; staging pays only where per-element chains dominate. It is already used where it won (move e5c890ad, recordDelta, deposit) |
| One MG over h and 4h bricks as a new build | The composite cycle already solves band h rows and native 4h rows in one cycle (3e1db823). What remains is lever 2a (the mirror) and 2c (the tail) |
| Refine folded into deposit; the advect certificate; the particle-free couple copy; a coarse band; redistance reach 1 | Measured nulls or failed A/Bs (cross-stage-duplication, brick-arena §5.5, §6.10) |
| A census every other frame | The swept margins must double, which grows fine tiles in every stage. It also alternates frame cost |

## 6. Order of work

Each step is one short switch-over, then a late-frame pour profile and a letters A/B (two repeats an arm), then a commit, per the rules: one Dawn run per arm, test at 1/60 s, land large chunks before GPU tests.

**Phase 0 (no code; one session):**
1. The combined arm `phiFineAdvection` trilinear + `fineGridPadding` 0 + `bandCycles` 3. If letters fails, bisect it.
2. The band residual history at f200–212 and on letters.
3. `bandCycles` 2 as its own arm.

**Phase 1 (small code, physics params):**
- lever 1's `particleBandDepth` and `particleDensity`, with the seed/retire hysteresis;
- then 3b, `surfaceAdvectBand`.

Re-profile after this phase: it moves every per-tile and per-particle baseline.

**Phase 2 (exact deletions):**
- 2a, the mirror (pressure session, in flight);
- lever 5, tile summaries;
- lever 8, paired remap;
- 2c, the tail;
- lever 7, forces;
- 2d, authority once and lean pRows.

**Phase 3 (rewrites behind params):**
- 3c then 3d, the redistance;
- lever 4, extension from bank B (after 3d settles bank B's freshness);
- lever 6, DEPB;
- 3e and 3f.

**Phase 4 (structural tail):** lever 9, starting with the brick-resident cycles and persistent fine φ.

Why this order:
- Shrinking first is cheap and sets the base everything else is measured on.
- Exact deletions need no physics sign-off.
- The rewrites depend on both.

## 7. The path toward 17 ms

At about 28.6 ms, these remain:

| Stage | ms | What is left |
|---|---|---|
| Surface | 9.0 | |
| Pressure | 5.6 | |
| RFE | 5.3 | |
| Particles | 4.8 | |
| Other | 2.7 | |
| Momentum | 1.2 | |

Reaching 17 needs the data-layer change the speed-of-light plan named: page-table-free kernels over a persistent fine store with per-tile class bits and seam records, so that no inner loop walks owners or seams. Rough sizes:

| Change | Sizing |
|---|---|
| Surface to about 4 ms | Advect at the gather floor plus the control (persistent masks, brick-arena §5.2); R at its ALU floor; no store or resolves |
| Pressure to about 2 ms | 0.3 ms a cycle, rows from staged channels |
| RFE to about 2.5 ms | Geometry and remap change-driven at brick refinement events (pressure-rfe §2.4) |
| Particles to about 3 ms | The bench chain at 0.85M |

That totals about 15–16 ms plus idle. Every one of these stays more than 2× off its SOL figure today. Per the reassess rule, each needs its own broken prior named before it is built. The thin band's effect on staging is the one already named.

## 8. Rules this plan keeps

- No indirect dispatch:
  - the rotation in 3d is a parity test inside direct grid-stride kernels;
  - the summaries in lever 5 and the bank B gather in lever 4 stride GPU-counted lists.
- Zero per-frame allocation:
  - levers 1 and 3 shrink lists, so no capacity grows;
  - DEPB's windows are workgroup memory.
- No compile during simulation: every physics lever is an update "solver" param, chosen at attach.
- No step limits or substeps. Lever 9's RK2 is one step with fewer stages, not a substep.
- One call site per heavy WGSL function:
  - lever 4 reuses the extension's slot addressing once;
  - lever 7 adds one fused multiply-add at each writer, not a sampler.
- Big swings only. The smallest lever here is 0.4 ms, and it rides in package C with others of its kind.

## 9. Measured: the first step (2026-10-20)

All three arms run at 2f6b6d5f, which includes the race-free level-0 relax. The pour columns are a late-frame profile over f200–212 (one run each). The letters columns are two repeats per arm.

| Arm | Pour GPU (ms) | Wall median (ms) | Pressure | Surface | Particles | Dispatches |
|---|---|---|---|---|---|---|
| Base | 52.37 | 63.4 | 11.41 | 15.58 | 9.60 | 364 |
| A: `phiFineAdvection` trilinear, `fineGridPadding` 0, `bandCycles` 3 | 50.41 (−2.0) | 60.5 | 10.01 | 14.00 | 10.59 | 335 |
| B: A with `bandCycles` 2 | 50.98 (−1.4) | 60.5 | 8.48 | 15.73 | 10.31 | 306 |

**Arm A gives −2.0 ms, half the −4 ms projected.**
- The cubic is gone (−0.95 ms in advectCubic, −0.5 in advectFinish), and one fewer cycle saves 1.4 ms.
- Padding 0 costs +1.0 ms in particles: transferSeams +0.28, deferred particle advect +0.27, recordDelta +0.16. More samples sit at seams.

**Arm B's pressure saving is eaten elsewhere.**
- The second cycle saves another 1.5 ms of pressure.
- The surface's deferred advect doubles, 1.38 → 2.90 ms. Over three earlier profiles it was steady at 1.37–1.40.
- The less-converged velocity sends more vertices off the fast trace. So pressure accuracy is coupled to surface-advect cost, and `bandCycles` 2 is not a lever until that coupling is understood.

**Letters gate:** both arms change the physics visibly. The 3→2 cycle step itself is within noise of arm A.

| | Base (a / b) | Arm A (a / b) | Arm B (a / b) |
|---|---|---|---|
| Volume drift f240 | −0.94 / −1.01% | −0.87 / −0.94% | −0.91 / −0.83% |
| Particles f30 | 39k / 39k | 5k / 5k | — |
| Particles f60 | 617k / 615k | 541k / 550k | 551k / 555k |
| Above-40 f60 | 20.4k / 20.4k | 18.8k / 18.8k | 18.9k / 18.9k |
| Above-40 f240 | 56.4k / 54.6k | 50.9k / 51.7k | 54.5k / 51.7k |
| Occupancy L1 vs base-a, f60 | 0.2% (b) | 0.7 / 0.7% | 0.8 / 0.8% |

- Late particle counts and volume drift stay within noise.
- The early band is much thinner: particles −12% at f60, and −87% at f30.
- Splash height (cells above y = 40) is −7.5% at f60 and about −7% at f240.
- Which of the three parameters causes the early difference is not attributed yet. Padding 0 is the likely one, since it removes fine tiles before the jet lands.
