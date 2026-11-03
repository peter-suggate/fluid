# Small scene, large step: paddleboats at 68 ms (2026-11-03)

**Status: for review. Nothing below is implemented.**

Paddleboats is a still pool: 12.8 × 4.8 m, 2.5 cm h, two moored boats and 16 paddles turning. Its clean step is **68.7 ms/advance**: the xctrace baseline, 150 advances, `--bodies`. Most of the water runs at 4h and 16h.

This plan says why each expensive stage costs more than 1 ms on the M1 Max, and how to bring each stage under 1 ms without fast paths or scene special cases.

## Evidence

Per-stage detail, with file:line references and per-kernel tables, is in the sibling files:
- [pressure.md](pressure.md)
- [census.md](census.md)
- [levelset.md](levelset.md)
- [momentum.md](momentum.md)

All measurements are paddleboats with the wheels driven and the hulls held (`--bodies`; `tools/moored-motor-bodies.ts`).

| Source | What it is |
|---|---|
| xctrace | Split-dispatch counter capture of 44 advances: per-kernel occupancy, ALU, LLC and read/write bandwidth. Table in [xctrace-kernels.txt](xctrace-kernels.txt); trace in `artifacts/xctrace-paddleboats/`. |
| Kernel timer | `tools/probe-reservoir-bulk.ts --kernels` per-dispatch timing, for attribution only. |
| Census reasons | Readback at step 300: why each h tile is h. |
| Experiments | Six A/B runs, each patch applied and reverted inside the GPU lock. Patches in [patches/](patches/). |

**Correction to the first answer.** "Global surface volume" (2.7 ms) is a diagnostic. It runs only when a reader asks: the probe every 60 steps, the app on pause. The live gauge costs 0.07 ms.

The kernel timer averages each kernel over the steps it ran on, so it overstated that stage. The xctrace numbers are per advance and do not have this problem.

## Diagnosis

### What the hardware says

Every expensive kernel runs at **8–17% occupancy, 14–30% ALU and 1–13 GB/s read**, on a part that does about 400 GB/s. Examples:

| Kernel | Time | Occupancy | Read |
|---|---|---|---|
| couple | 2.1 ms | 15% | 0.7 GB/s |
| buildRegions | 1.33 ms | 8% | 0.3 GB/s |
| advectParticlesDeferred | 4.5 ms | 10.5% | 13 GB/s |
| bodyMask | 0.4 ms | 0.6% | — |

pRelax is the only kernel near a proportionate rate (60 GB/s).

The arithmetic and traffic floors are far below the measured times:
- the particle move's traffic would take about 0.15 ms;
- a pressure V-cycle at the current band size would take about 0.2 ms.

The time is **items visited × serial dependent-load latency**, at occupancy too low to hide it.

### Where the waiting comes from, by measured size

1. **Loops over all bodies inside per-cell work.** Every cell near a body loops over all 18 bodies, often several times.
   - pRows (`umCellOpen`, `umCellInsideSolid`, `umBodyPoseNearTile`): 0.92 of its 1.50 ms, measured by removing the calls (pressure.md §3).
   - couple, coupleFine, buildRegions, bodyMask and `nbRigidContact` do the same.

   The cause is structural: every paddle moves every step, so the solid record is always stale, and every consumer falls back to evaluating live geometry per cell.
2. **Address chains re-walked per tap, per stage and per neighbour.** The chain is page word → block → tile key → topology → texel. It is re-walked per tap, per RK stage, per depth call (×8 at the surface) and per neighbour face, instead of once per tile or brick.
   - The particle move does about 100–150 dependent round trips per particle at a 384-thread register cap.
   - Separation's chains alone cost about 2.5 ms, measured by turning separation off (momentum.md §4).
   - Your hint holds where the chain is *serial* in a loop: hoisting it from pProjectNative's 16-subface loop saved 10%. Hoisting chains that already run in parallel (pRows interior faces) saved nothing. So the fix is to remove chains from inner loops, not to hoist every lookup.
3. **Sweeps over every slot instead of the items that can change.**
   - decide strides all 98,304 tile keys.
   - classify fully reclassifies all 4.8k h tiles; 52 cross the surface.
   - The surface advect traces 249k h vertices; about 10–15% are near the surface. Submerged liquid has no far-band exit at all.
   - Phi resolve runs four full seam passes; geometry fill re-fills every tile.
   - The surface pressure authority makes four 27-neighbour region scans.
   - Publish writes every face; forces visits every owner to add dt·g.
4. **Redundant recomputation.**
   - pProjectNative redoes pProject's fluxes.
   - Seam heads are rebuilt every cycle.
   - The arena re-stages the same brick neighbourhood 5–6 times.
   - classifyCoarse computes each vertex up to 64×.
   - The extension sweeps the same geometry twice, once per payload.
5. **Contention, launch shape and serial tails.**
   - couple's same-cache-line global atomics: **0.89 ms, measured** (census.md, experiment E).
   - Merge rounds are sized by capacity (22 dispatches); the hierarchy launches empty levels; pressure level-1 launches are 7× too wide.
   - The coarse tail is 57 tiny dispatches.
   - Serial tails: one lane per region in bodyMask, planeSums at 3% occupancy, classify's lane-0 tail.

### Why there are so many items

At step 300 there are 4,759 h tiles. Their census reasons:

| Reason | h tiles | Meaning |
|---|---|---|
| solid | 2,097 | A wet body-cut tile and all 26 of its neighbours |
| join | 2,320 | NB-FLIP particle heat plus padding, and body sweep |
| travel + closure | 290 | |
| crossing | **52** | The free surface |

Particles: 333k, held hot by the wake latch (one-pass plan §9). The pressure band holds 2,713 tiles.

Every stage's per-item fix runs out at about 1–5 ms because these counts are 4–20× what the surface itself needs.

## Stage budgets

All figures are ms per advance on paddleboats.

| Stage | Now | After the efficiency phases (1–4) | After phase 5, item counts |
|---|---|---|---|
| Pressure (rows, solve, projection, mixed project, surface authority) | ~14.5 | ~5 | ~0.9–1.0 (band of about 1k tiles) |
| Census (census, local speed, solid record, layout, frame plan, rigid couple) | ~8.9 | ~1.1 | <1 |
| Level set (surface advect, resolve, geometry fill, NB-FLIP reconstruction, arena, retirement) | ~13.3 | ~1.5–1.7 | <1, if decision 2 or the join cut lands |
| Momentum transport (particle move, separation, sort, transfer, seed, momentum, extension, remap, forces, halos) | ~19 | ~4–6 | ~0.8–1.2 (about 80k particles, about 1.2k h tiles) |

Measured so far: couple atomics −0.89 ms, pRows body loops −0.92 ms and the pProjectNative hoist −0.1 ms. The other savings are estimates from load counts and occupancy; each phase's gate measures them.

## Plan

The phases are ordered by shared mechanism, not by stage, because the same four mechanisms recur across all stages. Each item names its stage file for detail.

### Phase 1: bodies resolved once per step (~−6 ms)

1. **One body broad phase per step.** Extend markTiles (`uniform-mixed-bodies.ts:150`) to write a candidate-body bitmask per tile key, swept over the step. Every per-cell body query then loops only the tile's candidates, through `umBodyAtCandidates`, which already exists. The queries are in couple, coupleFine, pRows, buildRegions, `umBodyAmbient` and `nbRigidContact`. (census 1, pressure P0)
2. **Solid record split into static and body parts.** Static voxel and terrain openness is cached at build time and changes only on edit. Body cut fractions are refreshed once per step, one workgroup per dirty tile from a compact list. Every consumer takes the record path: pRows, pProject, the mixed project and the face velocities. (census 3, pressure P0) This also replaces the serial region rounds in buildRegions.
3. **couple and coupleFine over the body-cut list,** reducing in workgroup memory with one flush per workgroup (patch in [patches/census2.patch](patches/census2.patch)). Zero contributions add nothing. (census 2)
4. **bodyMask with one lane per (region, tile).** (census 4)

**Regressions** (required by the AGENTS.md contract):
- Record words equal a full rebuild after random body moves.
- Forces and torques equal the all-body loop.
- Save/load/undo still work.
- Edits stay local, with untouched geometry unchanged.

### Phase 2: visit only what can change (~−12 ms)

5. **Narrow-band transport certificate** (levelset P1). A tile with no crossing within two tiles last step is skipped entirely, for liquid and air alike, when:
   - the local speed bounds its displacement inside the stencil margin, and
   - no source, released wall or body sweep touches it.

   Phi outside the band carries sign and "≥ 5h", which is the arena contract already. This replaces the air-only exits. Expected: 4.9 → about 0.35 ms. It also shrinks every changed-tile list below.
6. **Changed-tile lists drive the work.** Producers append to the existing changed list, and the consumer dispatches indirectly over it (lists with nothing to do dispatch zero workgroups):

   | Consumer | Today | Detail |
   |---|---|---|
   | Phi resolve | 4 full seam passes | levelset P4; `resolveListed` exists |
   | Geometry fill | every tile | levelset P5 |
   | Extension publish | every face | momentum M3b |
   | Extension hierarchy | levels with no unknown faces | momentum M3a |
   | Surface pressure authority | four region scans | pressure P7: one deduplicated change list |
   | Layout and frame-plan dilates | all keys | census 9 |
7. **Live and candidate lists replace key sweeps.**

   | Kernels | List | Detail |
   |---|---|---|
   | decide, solidPromote, importance | band candidates | census 8 |
   | mixed project, rhs, toPressure | live 4h owners | pressure P6 |
   | coupleFine, surfaceClear, splat | particle tiles | levelset P2 |
   | retirement | its own list | levelset P6 |
8. **Certificates replace re-derivation.**
   - classify replays the last class when the tile's inputs are unchanged, a tile-level form of the existing cold replay (census 5).
   - localSpeed certifies dry, far tiles from their empty face set (census 7).

**Regressions:**
- Bit-exact against the full path for every certified, skipped or listed item. Example: far tiles bit-identical; a surface arriving at the certificate limit falls back and matches exactly.
- Seep and letters drift and step time at every commit.

### Phase 3: neighbourhoods staged once per job (~−8 ms)

9. **Particle brick window** (momentum M1). Extend `nbWinStage` so each slab job stages, in workgroup memory:
   - its window's tile sampling contexts;
   - its nodal depth;
   - its bin runs.

   `umSampleVelocity`, `particleDepthIn`, `nbRun` and `gridSupported` then resolve each tile from workgroup memory: same functions, same arithmetic. The separation pre-pass folds into the move. Expected: 5.6 → about 1.0 ms.
10. **Tile frame for grid kernels** (momentum M5, pressure P1). Each tile job loads its 27 neighbours' descriptors once: key, width, slot base, topology, band slot and brick. The kernels are momentum, remap, transfer, seed, affine fit, pRows and pProject. `momentumFine` already does this and runs at 2× the occupancy of its siblings. Apply it only where the chain is serial in a loop; that is where the hoist A/B paid.
11. **Fused arena per brick** (levelset P3). Stage the window, records, reach and capacities once, then run nearest → distance → publish in one kernel. Workgroup memory halves from 16.9 KB, which today caps occupancy at 17%.

**Regressions:** a staged-versus-device equality test per helper on the mixed h/4h/16h fixtures, including samples outside the window.

### Phase 4: redundant work, launch shape, serial tails (~−6 ms)

12. Gravity added in the momentum store; a forces pass runs only over faces with nonzero body or viscous terms (momentum M4).
13. One extension sweep carrying both payloads (momentum M2).
14. pProjectNative reads pProject's seam fluxes from a compact root list (pressure P2). Seam heads are computed once per solve, not once per cycle (pressure P3).
15. classifyCoarse is vertex-centric (census 6). localSpeed and seed share their vertex loads, and region atomics reduce per workgroup (census 7).
16. The coarse tail runs inside one workgroup: 57 dispatches → about 8 (pressure P4). The Krylov reduction folds into the apply kernels. Converged cycles launch zero workgroups (pressure P5).
17. Launches are sized from GPU counts, not capacity or the census reserve: merge rounds, level-1 grids and the hierarchy (pressure P5, momentum M6a).
18. Serial tails become parallel: planeSums gets a workgroup per part with a fixed-order combine; deposit atomics accumulate in registers per node (momentum M6b/c).
19. Halo and 4h-cache rebuilds run only on changed blocks once the samplers read staged contexts (momentum M7).

### Phase 5: item counts sized to need (gets every stage below 1 ms)

20. **Solid promotion stencil.** Today every wet body-cut tile pulls all 26 neighbours to h (`uniform-mixed-dynamic.ts:1615`). The comment there gives the reason: a coarse dry neighbour shares vertices and can interpolate liquid across a thin wall.

    First measure which of the 2,097 tiles carry a coupling term or a shared vertex the coupling reads. Then size the stencil to that, by rule, for every body. This is a correctness question, so it starts with a regression (thin wall, moving box), not with a change.
21. **Join tiles.** Split the 2,320 into particle heat, padding and body sweep with a census reason per source. Particle heat is the wake latch (one-pass plan §9): the census reads the particle surface's 0.2 h roughness as error at the 0.5 h tolerance.

    Earlier attempts traded the latch against the falling letters' volume, so this needs a physical fix to the particle surface's roughness, not a gate. APIC halved core roughness; its particle growth was explained there as retained wake.
22. **The pressure band follows.** With 52 crossing tiles plus halo, about 1k band tiles is the target (pressure P8). Fewer V-cycles come only from measured convergence (pressure P9), not from a cap.
23. **Remap stops running every step** once the census stops churning the layout (momentum §3).

## Decisions for you

1. **Staged windows with a device fallback** (items 9–11). A sample inside the job's window reads workgroup memory; one outside walks today's chain. It is one function with two address sources, used by every scene, and bit-identical. I read that as a cache, not a fast path. The stricter alternative sizes each window from the local speed certificate so that no sample can leave it; that makes larger windows and lower occupancy. Which do you want?
2. **Arena distance reach, cell-granular** (levelset §P8). Distance is solved only for vertices with |phi| < 6h, instead of a 3-tile reach; membership needs < 4h. This changes the distance contract and is the level-set stage's last 0.5 ms.
3. **Phase 5's scope.** Items 20–21 change what the census refines near bodies and how the wake cools. They need physics regressions before any timing. I would start them in parallel with phase 1, as investigation only, since every stage's last factor of 3–5 depends on them.
4. **Agents.** Following "fewer agents, bigger swings": two implementers.
   - One owns phases 1 and 2 (bodies and lists), which share the census and solid code.
   - One owns phases 3 and 4 (staging and launch shape).

   I take phase 5's investigation and judge each milestone.

## Gates

Each commit runs:
- `npm run check:types` and `npm run test:unit`;
- the item's equality regressions;
- one short timing run each on seep 300, letters 120 and paddleboats `--bodies` 150, reporting drift and step time on both scenes, per the GPU gate policy.

At the end of each phase, an xctrace counter capture of paddleboats confirms that occupancy and bandwidth moved, not only time. The capture is `tools/profile-mini-dam-xctrace.ts --scene=paddleboats --bodies --counters --split-dispatches --gpu-report-only`, reduced with [reduce_xctrace.py](reduce_xctrace.py) (`python3 reduce_xctrace.py artifacts/xctrace-paddleboats out.json`); the profiler's frame report rejects split labels on this scene and needs fixing first.
