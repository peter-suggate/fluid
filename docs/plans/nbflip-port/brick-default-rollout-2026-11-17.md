# Brick engine as the default for every scene (2026-11-17)

**User directive (2026-11-17):** "let's enable this new method for all scenes, making it the default". This is the
structural plan's M3–M6 (nbflip-structural-2026-11-15.md §4.2), started now.

## 1. Where it stands

`brickSceneSetup` was run over all 142 catalog scenes on main 20a233bb. The brick engine runs 1 of them: the
high-resolution dam.

| gap (brick-scene.ts) | scenes | owner, milestone |
|---|---|---|
| `scene.fluid.brickPools` not authored | 137 | E: a sizing rule from the scene (§3) |
| initial condition other than a plain dam-break (tank-fill, height fields, `initialLiquidVolumes`, brick seeds) | 111 | E (seeding the pool) with F (particles, level set) |
| container not whole pages: padding needs static solids | 83 | E, M3 |
| static voxel solids beyond the shell (`solidStamp`) | 37 | E, M3, with the AGENTS.md regressions |
| rigid bodies | 20 | P, M5 |
| sources: inflow and scheduled drops | 19 | F, M4 |
| open top: an air boundary above the lattice | 18 | E, M3 |
| initial liquid velocity | 5 | F |
| open world: the moving window | 3 | E, M6 |
| spherical container | 2 | E, M3 (a static solid) |

27 scenes need only pools and page padding: the power droplets and fills, the minimal and large dam breaks, the
sparse-cm12 ladder dams, the water-box dam and others. Static solids are the keystone, because padding is a solid.

## 2. The default switch, now

The `engine` parameter gains `auto`, which becomes the default.
- `auto` runs the brick engine when `brickEngineGaps(scene)` is empty, and the uniform engine otherwise.
- The choice is visible: the engine status names which engine ran and, for uniform, the first gap.
- An explicit `brick` still refuses a scene with gaps, naming every gap.
- An explicit `uniform` still runs the old engine.

So every scene that becomes enabled moves to the brick engine by default, and nothing else breaks. At M6 `auto` and
`uniform` go, together with lib/methods/uniform/.

## 3. Rules for enabling a scene

- **Enabling:** a scene joins `BRICK_ENGINE_SCENES` only after:
  - a smoke run on the brick engine, with no FAIL latch and no Metal abort;
  - the scene's own gates, where references exist (lane G's tools; seep G1–G8, letters L1–L5, paddleboats P1–P4).
- **Volume (the batch rule, coordinator 2026-11-17):**
  - Run 60 steps against the old engine. The brick's |peak| and |final| volume drift must be no worse than
    `bar = old worst + max(1%, old spread)`. The old worst and the old spread (max − min) come from the old engine's
    repeats.
  - For nondeterministic scenes, the brick's **mean over its repeats** is judged against the bar, and the brick's
    spread is reported beside it. The brick's spread never widens the bar: a noisier candidate does not get a looser
    gate.
  - **Splash scenes** (an impact inside the 60 steps, such as cm12-figure-7): the peak is judged as above. The final
    is judged at a settled step past the impact window, at 120 steps, from one old run and two brick repeats.
- **Pools:**
  - Sized per scene from measured peaks ("no runtime pool growth").
  - For scenes without an authored `brickPools`, E defines one deterministic sizing rule from the scene, such as the
    initial liquid's bricks and the lattice, with a stated margin.
  - A pool that would overflow latches FAIL.POOL and is never grown at run time.
  - **The splash term (E, 2026-11-17) is empirical.** It is fitted to five measured peaks, not derived. It applies to
    a scene whose liquid starts 16 cells or more above its rest depth: f ≥ 10·min(floorF, ⌈1.5 V/64⌉), meaning eight
    brick layers with the 1.25 margin, and particles ≥ 10 V.

    | scene | peak F | SPF after caps | margin |
    |---|---|---|---|
    | cm12-figure-7 | 2437 | 2890 | 1.19x |
    | cm12-figure-7-256 | 8948 | 10890 | 1.22x |
    | nbflip-stair-cube | 2600 | 3670 | 1.41x |
    | free fall | 289 | 405 (WF) | 1.40x |
    | tilted free fall | 238 | 405 (WF) | 1.70x |

    Stair is the weakest point: it is the only V-limited fit, and it sets the 1.5. FAIL.POOL remains the safety net.
    The term changes 10 catalog scenes, none of them enabled (the commit's audit). hero-garden-hose-x10 would be
    3.2 GB, so it must author its pools from a measured peak before it is enabled.
- **Unknown kinds fail explicitly.** A scene feature the brick engine doesn't handle stays a named gap.
- **Fixed scenes:** the dam's −10 ms must hold. Re-gate the dam at each milestone.

## 4. Lanes and order

| lane | first | then |
|---|---|---|
| E | the `auto` default; the pool sizing rule; general initial liquid (pool seeding) | M3: static solids (`solidStamp`, `phiSolid`, `pSolid`), page padding, the open top, the sphere; enable the 27 pools-and-padding scenes in batches; then seep |
| F | general initial liquid (particles and level set from any liquid volume); initial velocity | M4: sources (inflow, scheduled drops), volume control; letters |
| P | M5: bodies (`bodyStamp`, `couple`, the rigid integrator), motors, no-slip | paddleboats; then the 20 body scenes |
| E, last | M6: the moving window, open-world terrain, interactive balls and `editFluid`, delete the old engine | |
