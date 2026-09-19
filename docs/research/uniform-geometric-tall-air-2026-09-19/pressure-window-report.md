# The CM11a pressure lattice on the window

Uniform Geometric, tall-air fixture, 2026-09-19. Successor to
`solve-window-report.md`, which put every kernel on the liquid's box but left
the pressure hierarchy planned for the whole domain.

## What it is for

The solve window shrank the work but not the PLAN. A hierarchy is planned once,
for the domain, and a domain whose shortest axis will not divide the level count
falls out of lockstep coarsening into semi-coarsening. The tall-air fixture at
8x is exactly that cliff: 64x512x64 gets nine levels and a 4,228-pass plan where
the same liquid in 64x64x64 gets six and 2,008 — so the window arm still encoded
2,199 passes a step and still spent 26 ms of host time doing it.

A pressure instance is now planned for a CAPACITY, not the domain, and owns an
ORIGIN in simulation cells. Everything inside the hierarchy is window-local:
`simulation = id - 1 + origin`, dispatches cover the whole capacity directly
with a static plan, and the level records are not consulted at all.

## Headline: 8x

`pressure-window-8x.json` — four arms built up front and advanced in lockstep,
one frame each per round, 70 frames. Medians over frames 5-70.

| arm | GPU ms | CPU encode ms | wall ms | levels | passes encoded |
| --- | --- | --- | --- | --- | --- |
| 8x, solve window OFF | 30.80 | 26.03 | 57.17 | 9 | 2199 |
| 8x, solve window ON, lattice = whole domain | 17.30 | 25.56 | 43.12 | 9 | 2199 |
| 8x, solve window ON, lattice = liquid window | **12.78** | **11.86** | **24.96** | **6** | **948** |
| 1x reference (no window, small domain) | 11.86 | 11.66 | 23.91 | 6 | 948 |

The 8x cliff is gone: the tall domain now runs within 7.8% of GPU and 1.7% of
CPU encode of the same liquid in a domain an eighth the height, and its plan is
bit-for-bit the small scene's — six levels, 2,008 planned passes, 948 encoded.
Against the whole-domain lattice it is 1.35x on GPU and 2.16x on host encode;
against the window off, 2.41x and 2.20x.

The window itself is unchanged by this: both windowed arms ran at 7.0% of the
domain, 0 clipped steps, 1 dense step.

### 4x and 1x

`pressure-window-1x-4x.json`, same protocol.

| arm | GPU ms | CPU encode ms | wall ms | lattice | re-plans |
| --- | --- | --- | --- | --- | --- |
| 1x, window, domain lattice | 11.99 | 11.41 | 23.92 | 64x64x64 | – |
| 1x, window, window lattice | 11.93 | 11.23 | 23.62 | 64x64x64 @ 0,0,0 | 0 |
| 4x, window, domain lattice | 12.06 | 11.42 | 23.93 | 64x256x64 | – |
| 4x, window, window lattice | 12.26 | 11.49 | 24.07 | 64x64x64 @ 0,0,0 | 2 |

Neutral, as expected and as the brief predicted: at 1x and 4x the domain already
plans six levels and 2,008 passes, so the window lattice has nothing to win and
is asked only not to lose. At 1x it does not even allocate — the capacity IS the
domain, so it adopts the instance built at load and re-plans zero times. The 4x
+1.7% on GPU is inside this lane's noise (p25 11.86 against 11.80).

### The plan, on the CPU

`pressure-plan-census.mts` now walks the window plan beside the domain one, for
a capacity seated on the CPU-known starting box. No GPU needed.

| arm | domain plan | window capacity | window plan |
| --- | --- | --- | --- |
| 1x | L6 · 2008 passes · 1.34M workgroups | 64x32x64 | L5 · 1442 · 0.72M |
| 4x | L6 · 2008 · 5.05M | 64x32x64 | L5 · 1442 · 0.72M |
| 8x | L9 · 4228 · 10.01M | 64x32x64 | L5 · 1442 · 0.72M |

8x's window plan equals 1x's domain plan exactly at the 32-pad seating
(L6 · 2008 · 1,339,444 workgroups), which is the claim the brief asked to be
confirmed before any Dawn time was spent. The five-level row above is the
tighter settled seating, and it is what a calmer scene reaches.

Run: `node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/pressure-plan-census.mts`

## Halo semantics

A halo cell whose simulation coordinate leaves the domain keeps today's
behaviour exactly — it is the wall, floor or lid it always was, through
`mgOpenTopHalo` and the `mgInterior` test.

A halo cell INSIDE the domain is built as an ordinary interior cell:
`mgBuildFinestTopology` and `mgBuildFinestRhs` both branch on

    mgSimulationCell(p, d, simulation) = mgInterior(p, d)
                                       || (pressureWindowLattice() && valid(simulation))

This is stronger than the brief's "open, phi positive, Dirichlet p = 0". Far air
is what such a cell usually holds, and the LCP smoother already makes an air
cell Dirichlet p = 0 through `mgBakedLiquid`. But a window's halo can also land
on a solid, on terrain, or on a genuine free surface, and building it from the
real open fractions, face V and phi is correct in all four cases where "always
far air" is correct in one. It costs one extra branch in two setup kernels.

Audited and unchanged: `mgTrilinearPressure`'s out-of-grid rule (it clamps, and
a clamped coarse tap is a coarse-grid correction, not a boundary condition),
`mgInterior`'s other callers (all coarse-level, where the halo is the level's
own and not the window's), and the minimum/shifted chain.

Because a domain-capacity instance seated at origin zero is bit-identical under
the window rule — every one of its halos leaves the domain — the mode word stays
on whenever the lattice is enabled at all, and only the capacity and origin
move. There is no second code path.

## Readers of the pressure texture

`projectPressureValue` subtracts the origin and distinguishes the two ways a tap
can leave the lattice:

    let local = p + vec3i(1) - pressureWindowOrigin();
    if (valid(p) && (any(local < 0) || any(local >= pressureDims))) { return 0.0; }
    return textureLoad(pressureIn, clamp(local, 0, pressureDims - 1), 0).x;

Leaving on an IN-DOMAIN side returns 0 — far air, which is what the halo it just
walked off was built as. Leaving the DOMAIN keeps the old clamp, which is the
wall extrapolation the projection has always used.

Other readers: `projectGroup` is rebuilt whenever the instance changes;
`physicsFieldsForQA` now also returns `latticeOrigin` and `latticeDimensions`,
so the `{pressure, gamma}` field view can place what it reads. The coarsest
capture is per-instance and is re-armed on adoption.

## Re-planning

Instances are cached by capacity (LRU 3, evicted ones destroyed). Growth is
immediate. Shrinking waits for 30 consecutive steps — EXCEPT when the current
capacity is the whole domain, which is not a capacity to oscillate away from but
the start-up and fallback state; waiting there would make the plan the window
exists to replace the cost of every scene's first second and every recovery.

Creation is synchronous: the compiled module, bind-group layouts and pipelines
are shared from the instance built at load, so a re-plan is a plan walk plus
texture and buffer allocation. Measured: **75-81 ms on one frame**, twice in 70
frames at 8x (once to seat the window capacity at frame 3, once when the box
moved). That is a visible hitch and the largest remaining cost of the feature.
**Superseded by "Re-plan hitch" below**: the walk is now ~10 ms, and the growth
re-plan is prewarmed to zero.

There is no state to migrate, verified rather than assumed: `mgBuildFinestRhs`
stores p = 0 every solve; `fullCycleBackup` is written and consumed inside one
`fullCycle()`; `encode()` clears the diagnostics buffer first; the lagged
cycle-budget state lives on the solver, not the instance.

Memory caveat: the domain-capacity instance is created in the constructor
because it owns the compiled programs, and is kept as the violation and
start-up fallback. Window instances are therefore ADDITIONAL allocation, not a
saving. At 8x that is a 64x64x64 hierarchy beside the 64x512x64 one.

## Window pessimism: what the three layers actually cost

Peter's addendum: the window looks cautious, fix three layers.

**B. Host lag pad.** Was `LAG_STEPS * 2 * (ceil(travel*1.5)+1) + 4` with three
steps assumed against a measured lag of one. Now `2 * (travel_axis + 2) + 4`
with the per-axis travel from (C). Landed. The containment flag and the dense
fallback are untouched.

**C. Exact padding.** The scan now reduces travel per axis AND per direction
over wet cells, packed 3x10 bits into two free summary words, with gravity
`g dt^2/h` added to the downward travel and inflow to its direction. The
padding is

    padding_side = max( standing reach (k+s)*4 , max(travel_side, redirect) + stencil reach )

The two groups do NOT stack: a stencil is read after the motion, from where the
liquid ends up, while the two-level shell class is measured from where it is
now. The `+4` that used to sit on top of both is gone.

`redirect` is the term the brief did not have, and the measurements forced it.
Per-direction travel alone is not safe: where a stream meets a wall, the floor
or another stream, the pressure impulse turns its momentum into some other
direction WITHIN the step, so the velocity the scan measured says nothing about
the jet that leaves. What bounds that jet is the momentum arriving — a collision
redistributes speed, it does not manufacture it — so the isotropic travel is a
floor under every direction's, exactly as `g dt^2/h` is under the downward one,
with half again on top for the concentration the impulse also does (measured:
8.6 cells of displacement became 11.7 in the next step). It is zero at rest, so
it costs the calm case nothing.

**A. Overrun threads must exit. REVERTED — the band is not free.**

Implemented as asked, one helper per shader. It changes the answer. On the
tall-air dam the far-wall impact step reported **4.392 m/s against the
whole-domain control's 3.106**, identically at 1x, 4x and 8x, and disabling
nothing but that clip restored 3.104. It is not the window's size: forcing the
box to 40 cells — the size the solve-window report shipped — still spiked, and a
larger shell reach only masked it by enlarging the extension's tile set.

The conclusion is that the exact window box plus its padding does NOT contain
everything a step needs, and the host's lag allowance has been silently
supplying the difference all along. The hatched band is doing real work on real
cells. Until the padding covers the impact case the overrun runs, and the
overlay now says so. `activeId`, `activeVertexId`, `activeBaseId`,
`hierarchyActiveId` and `mgActiveId`'s domain-lattice arm are back to WP2's
behaviour exactly; `mgActiveId`'s window-lattice arm needs no clip at all,
because a window-local dispatch covers exactly its capacity.

### The floor above a calm surface

Tall-air container at 4x, tank-filled to a still eight-cell pool with 248 cells
of air above it, max speed 1.2e-3 m/s:

    pool top y = 8 · window top y = 28 · floor = 20 cells

Against the brief's expectation of 4 + 12 + alignment <= 3 = 19. The extra tile
is the union with the previous step's box and the 4h alignment. The remaining
floor is the two-level reach defaults, which Peter can move on the panel: the
padding's standing term is `(fineReach + shellReach) * 4`, so Shell reach 0
takes it from 12 to 8 and the floor to 16 — and, on this scene, takes the
capacity from 64x64x64 to 64x32x64, which is five levels and 1,442 planned
passes instead of six and 2,008. There is nothing between those two capacities:
64x40x64 and 64x48x64 both semi-coarsen and are worse than either.

On the moving fixture, B and C together take the window from 40 cells tall to 36
(4x: fraction 0.1562 -> 0.1406; 1x: 0.625 -> 0.5625). Volume is unchanged
(32767.56..32768.40 in every arm), violations stayed 0 in all eight windowed
arms across the two 70-frame captures, and the peak speed now matches the
whole-domain control in all of them.

The E3 live transport set was verified not to constrain the window, as the
addendum's claim required: a live tile outside the window is simply not
dispatched, its cells keep the V the dust-floor predicate already guarantees is
zero, and `uvNormalizeDonors` divides by the column sum over the rows that were
BUILT, so the gather moves every donor's V somewhere and never duplicates it for
any built set. The shell tiles DO constrain it, and are the standing term above.

## Oracles

**Hydrostatic** (`large-power-hydrostatic`, 60 frames, three arms: window off,
window + domain lattice, window + window lattice). Max speed identical in all
three to four digits (5.191e-1 final, 7.601e-1 peak); V = 1024.0005 in all
three. Both windowed arms took exactly one clipped step, at frame 3 — the
release transient, where the pad computed from a lagged box of liquid at rest
does not cover two steps of a collapse starting. `FLUID_UNIFORM_WINDOW_LAG_PAD=28`
(the old allowance) avoids it and produces the same answer, which is the point:
the dense fallback is exact, and the pad is not required to be a safety net.

**Forced violation** (`FLUID_UNIFORM_WINDOW_LAG_PAD=0`, 4x window + window
lattice, 30 frames), run because the containment logic changed — the finalize
now also checks the union box against `[O, O+C)` and raises bit 128. Two clipped
steps, 17 dense steps, zero validation errors, volume within 0.03 cells of the
unforced arm over the same frames and the same front position. A violated step
falls back to the whole-domain lattice and the whole-domain counts, and nothing
is created or destroyed.

## Not verified

- Scenes with a solid, terrain or a free surface in the window's halo. The
  in-domain-halo path builds them correctly by construction, but no fixture here
  puts one there.
- Multi-body scenes where the window is not one box. The capacity is a box; a
  second body far away grows the box and the capacity with it.
- Inflow sources outside the current window. The source scan adds them to the
  seed, and the lattice is planned from the same box, so they should follow —
  untested.
- Any domain whose axes are not multiples of 32 or 16. The planner falls back to
  the domain on such an axis; the fixture here is 64x64n x64 throughout.
- The paper (non-geometric) method. Every change is behind `geometric` or behind
  `pressureWindow`, and the paper arm's padding expression is untouched.

## Commands

    # CPU, no GPU
    npx tsc --noEmit
    node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/pressure-plan-census.mts

    # Dawn, under the repository WebGPU lease (the probe takes it itself)
    FLUID_TALL_MULTIPLES="1,8,8w,8wp" FLUID_TALL_FRAMES=70 \
      FLUID_TALL_OUT=docs/research/uniform-geometric-tall-air-2026-09-19/pressure-window-8x.json \
      node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts

    FLUID_TALL_MULTIPLES="1w,1wp,4w,4wp" FLUID_TALL_FRAMES=70 \
      FLUID_TALL_OUT=docs/research/uniform-geometric-tall-air-2026-09-19/pressure-window-1x-4x.json \
      node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/probe-tall-air-dawn.mts

Arm suffixes: `d` dense control, `w` solve window on, `p` window-local pressure
lattice (needs `w`). `FLUID_TALL_OVERRIDES` is a JSON object of extra param
values applied to the windowed arms, which is how the padding floor is moved
without editing a default.

## Overrun dependency

The host sizes each windowed dispatch from a box a couple of steps old, so the
last workgroups overrun the exact window. WP3 found that stopping those threads
at the published extent moved the far-wall impact peak from 3.106 to 4.392 m/s,
identically at 1x, 4x and 8x. Correctness rested on a heuristic lag pad. It no
longer does.

### The bisect

One lockstep capture at 1x with a per-group clip mask (host-set scaffolding,
since removed): bit 0 reference cells, 1 vertex lattice, 2 extension finest, 3
extension hierarchy levels, 4 pressure levels in domain-lattice mode, 5 volume
transport and sharpening.

| arm | f11 max speed |
| --- | --- |
| whole-domain control | 3.1056 |
| window, no clip | 3.1035 |
| clip 63 (all groups) | 4.3920 |
| clip 1 (reference cells) | 3.1056 |
| **clip 2 (vertex lattice)** | **4.3920** |
| clip 4, 8, 16, 32 | 3.1056 |

One group, on its own, reproduced the whole difference. A margin sweep on the
vertex lattice then showed **four vertices of margin removes all of it**, and 8,
12 and 16 add nothing — so the dependency is a READ REACH, not the hypothesized
growth into never-written cells. A cell entering the window for the first time
is not the problem; a vertex on the face of the box is.

### The mechanism

`uvAdvectPhi` writes the advected vertex phi over the box. `uvRedistancePhi`
then reads it back through `uvPhi` at the closest point, with `q` clamped to
`p ± 4` cells, `uvGradient` probing `± 0.25` around that and a trilinear tap one
vertex further. Reach: **6**. `uvAgreementShift`'s tent gather is 5. Writing
only the box means a vertex on its face redistances from phi four vertices
outside it — which the ping-pong partner still holds from the previous step.
The overrun band was rewriting exactly those vertices every step, and the host's
lag allowance decided how many of them it covered.

This is fix (b) from the brief: the reach was genuinely larger than the derived
padding, and the padding cannot express it, because the padding covers what a
CELL reads while these passes read *each other*.

### The fix

`activeVertexId` dilates the vertex lattice by `VERTEX_PHI_REACH = 6` on both
sides of the cell box (clamped to the domain), and the host's vertex group
counts size a superset of it. Every phi a phi pass reads was then written this
step. The derivation is in the comment above the constant.

With it in place, every windowed helper drops its overrun threads
unconditionally — `activeId`, `activeVertexId`, `activeBaseId`,
`hierarchyActiveId` and `mgActiveId` in domain-lattice mode. Level extents ride
in the per-level scratch words as a packed count with a usable flag; a level too
large to pack clips nothing, which is the safe direction.

### Clip-all vs no-clip

Impact step, 16-frame lockstep, all three quantities the brief asked for:

| arm | f10 | f11 | f12 | peak | front (m) | V at 16 |
| --- | --- | --- | --- | --- | --- | --- |
| 1x whole domain | 3.3520 | 3.1056 | 2.6157 | 3.4994 | 0.400 | 32768.073 |
| 1x window, no clip | 3.3520 | 3.1035 | 2.6166 | 3.4994 | 0.400 | 32768.076 |
| 1x window, clip all | 3.3520 | 3.1036 | 2.6161 | 3.4994 | 0.400 | 32768.096 |
| 8x whole domain | 3.3520 | 3.1056 | 2.6155 | 3.4994 | 0.400 | 32768.098 |
| 8x window, no clip | 3.3510 | 3.1038 | 2.6361 | 3.4994 | 0.400 | 32768.112 |
| 8x window, clip all | 3.3520 | 3.1036 | 2.6163 | 3.4994 | 0.400 | 32768.089 |

Re-measured after the re-plan work below, with the clip unconditional: 1x
3.3520 / 3.1035 / 2.6158, 8x 3.3520 / 3.1035 / 2.6163, peak 3.4994, front
0.400, V 32768.08 and 32768.10 -- unchanged.

Clip-all matches no-clip at both sizes, and at 8x it matches the whole-domain
control more closely than no-clip did (f12 2.6163 against 2.6155, where no-clip
read 2.6361). **Item A is on, unconditionally.** The hatched overrun band in the
solve-window overlay is now free: those threads exit at the window edge, and the
host's lag pad is a launch-size allowance with no bearing on the answer.

## Re-plan hitch

### Where the 75-81 ms went

Whole 64x512x64 instance, 4228 dispatches, every device call timed by wrapping
the device (no instrumentation left in the solver):

| part | calls | ms |
| --- | --- | --- |
| `createView` | 59192 | 51.6 |
| `createBindGroup` | 4228 | 47.2 |
| `createBuffer` (params) | 4230 | 34.2 |
| plan walk (JS) | — | 34.9 |
| `writeBuffer` (params) | 4229 | 9.0 |
| `createTexture` | 118 | 1.4 |
| **total** | | **178.4** |

Three of those four leaders were duplication. The plan names the same few dozen
textures thousands of times, and the same (entry point, resources, parameters)
triple recurs on every cycle of every level.

### What was cut

1. **One view per texture** — 59192 views become 100.
2. **Parameter buffers keyed by contents** (dimensions, level, control, gating)
   — 4230 buffers become 85.
3. **Bind groups keyed by entry point, parameters and the textures that entry
   point actually binds** — 4228 become 255. A bind group is immutable, so two
   dispatches naming the same resources need only one; the encoded command
   stream is unchanged.
4. **Allocation-free walk** — the per-binding texture table is an array, the
   sampled/writable split is precomputed per entry point, and the cache key is
   built without spreading a Map.

| capacity | passes | before | after |
| --- | --- | --- | --- |
| 64x512x64 (domain) | 4228 | 178.4 ms | 14.9-18.6 ms |
| 64x96x64 (window) | 2008 | ~80 ms (75-81 measured live) | 9.7-11.2 ms |

After the cut the walk is ~65% of what remains; the device calls are 3-4 ms.

### Prewarm

A growth re-plan is the one that cannot wait: the step that outgrows the
capacity needs the bigger lattice in the same step. So the plan walk was made
resumable — `buildPlanSteps` yields at every cycle boundary, since the only
state a cycle carries is the ping-pong parities and the boundary list — and the
solver builds the next capacity ahead of needing it:

- Trigger: the axis with the least headroom, once that headroom falls below
  `8 + 6 * travel` cells. A fixed eight cells is five steps of a calm surface
  but **one step of a far-wall impact**, which is exactly when the hitch lands;
  leading by the box's own measured travel is what makes the warning useful.
- The start-up steps prewarm too, from the CPU-known SEED box rather than the
  readback -- a reset seeds the geometric window at the whole domain, so the
  lagged box those steps see is the capacity already in hand.
- The step that allocates the lattice does no walking. Later steps spend up to
  2 ms each on cycles.
- `adoptPressureInstance` finishes whatever is left and swaps it in.

Measured at 8x, 40 steps:

| step | event | host ms |
| --- | --- | --- |
| 1 | prewarm 64x64x64 allocates, from the seed box | 1.23 |
| 2 | prewarm walks | 3.78 |
| 3 | **first seating, whole domain to 64x64x64** | **3.41** (was 9.6-12.9) |
| 4 | prewarm 64x96x64 allocates | 1.01 |
| 5-7 | prewarm walks | 3.37, 3.08, 2.15 |
| **12** | **capacity switches to 64x96x64** | **0.00** |
| 12-13 | prewarm 64x128x64 (guessed, never needed) | 1.02, 2.85 |
| 14 | headroom recovered, prewarm discarded | — |

The growth switch is free. The first seating is not, and cannot be: the window
only runs after `UNIFORM_WINDOW_LAG_STEPS` of domain capacity, so there are two
steps of warning by construction and the build needs about four. It is cut from
~10 ms to 3.4.

Per-step prewarm cost is 1.0-3.8 ms against a 2 ms budget, because a cycle is
the smallest resumable unit the walk has and the budget is checked between
cycles, not inside one; the first cycle of a plan is an F-cycle and is the
expensive one. Finer chunking would mean yielding inside `vCycle`, which shares
the parity arrays.

One cost is honest to name: a wrong guess holds a second lattice's textures
until the headroom recovers -- one step up on one axis, so ~1.5x the current
window's pressure allocation at worst -- and spends its walk for nothing.

### Can the domain instance be dropped? (note only)

Not as it stands, and the reason is ownership rather than memory. The
constructor's instance owns the compiled `GPUShaderModule`, the per-entry-point
bind-group layouts and the pipelines; every window instance borrows them through
`programs` and that is what makes a re-plan synchronous. It is also the fallback
the solver adopts whenever a step clips or the window planner rejects a
capacity. Dropping it would mean either holding the programs somewhere that
outlives any instance — they are dimension-independent, so this is a small
refactor of `programs` into a separate object created once — or paying a shader
compile on the first violation. The first is the right shape if its allocation
matters: at 8x the domain hierarchy is 13 textures a level over 9 levels with a
66x514x66 finest, which is a fifth of a gigabyte. Nothing else about the
instance is load-bearing.
