# E2 — the extension, the advection and the projection on 4h tiles

Uniform Geometric, `cm12-figure-7` (128³), headless Dawn/Metal. Follows
`e1-implementation-report.md`, which put a 4h velocity sampler behind a fine
tile map without shrinking a single dispatch. E2 shrinks the work.

One toggle turns all of it on: **Two-level velocity sampler → On**. The two new
selects (`Extension work`, `Far-air work`) ship at `tiles` and are the dense
control arms when set to `dense`.

## Headline

`cm12-figure-7`, per-advance GPU medians (probe, 70 advances per arm):

| regime | frames | off | on | change |
|---|---|---|---|---|
| free fall | 2–25 | 80.08 ms | 60.49 ms | −24% |
| impact | 26–40 | 98.30 ms | 83.36 ms | −15% |
| spread | 45–70 | 105.91 ms | 98.83 ms | −7% |
| whole run | 1–70 | 100.79 ms | 85.72 ms | −15% |

A second capture, with the arms in the opposite order and the machine in its
slower mode, reported off 158.86 ms → on 106.89 ms (−33%), free fall −35%,
impact −39%, spread −31%. **Read the stage deltas, not the totals**: this lane is
bimodal (memory: "CM12 stage-cost lane is bimodal"), and the two runs disagree
on the total by more than they disagree on any stage. The stage table below is
consistent across both.

## Stage deltas (medians, same probe, off vs on)

| seam | free fall off→on | spread off→on |
|---|---|---|
| Sec. 3.3 interface authority | 0.59 → 1.11 ms (**+0.52**, the tile classify) |  0.59 → 1.11 ms |
| Sec. 3.3 narrow-band FIM front | 18.42 → 5.96 ms (−68%) | 21.82 → 12.39 ms (−43%) |
| Sec. 3.3 hierarchy fill + transport shell | 1.83 → 0.85 ms (−54%) | 1.77 → 0.92 ms (−48%) |
| Velocity advection + body forces | 5.37 → 1.25 ms (−77%) | 3.54 → 1.64 ms (−54%) |
| Pressure projection | 2.10 → 0.39 ms (−81%) | 2.16 → 0.59 ms (−73%) |

Net of the classify: ≈ 19 ms/advance in free fall, ≈ 13 ms in spread. Fine tiles
run 3.5k–7.4k of 32,768 (11–23%), shell tiles 5.2k–9.3k (16–28%).

Attribution arms (forward capture, spread regime): extension-only saves the
extension but *raises* advection (3.74 → 8.59 ms) because the sampler's coarse
branch is taken for far-air samples that the dense advection still computes;
advection-only saves advection and the projection outright. The two belong
together, which is why both default to `tiles`.

## What changed

### 1. The 4h table is now published by the extension hierarchy
- `lib/methods/uniform/webgpu-uniform-velocity-extrapolation.wgsl.ts:593`
  `publishCoarseVelocityTable` writes the three face words of each 4h tile from
  the extrapolator's own `ceil(n/4)` hierarchy level.
- `lib/methods/uniform/webgpu-uniform-velocity-extrapolation.ts:78,287,444`
  `coarseVelocityLevel` is `hierarchyLevels[1].up` (its `.down` when it is the
  coarsest level); one extra dispatch of `ceil(n/16)³` workgroups at the end of
  `encode`, gated on `publishCoarseTable`.
- New binding: `@group(0) @binding(12)` `tileScratch`, the parent's
  `conditioningScratch`, bound read_write once (never both read-only and
  writable in one dispatch).
- E1's `uvTwoLevelRestrict` — an area-mean restriction of the fine transport
  shell — is gone, because that shell is exactly what E2 stops computing.
- The hierarchy's transfer maps the component axis by `(t+1)·S/T − 1`, which at
  `S/T = 4` is fine face `4t+3`: the sampler's own upper-face convention. The
  host therefore requires **every axis to be a multiple of 4** and a
  `ceil(n/4)` level to exist, else the whole two-level path is disabled
  (`webgpu-uniform-reference.ts:530`, `:657`).
- Wall treatment (approximation, see below): an unknown component publishes 0,
  and the last coarse layer on the component axis publishes 0 unless at least
  one of its 16 fine faces is open.

### 2. The tile classify moved to the head of the step and grew a second class
- `lib/methods/uniform/uniform-volume.wgsl.ts:411` `uvTwoLevelSeed` replaces
  `uvTwoLevelRestrict`: liquid above the dust floor, any solid/terrain share, a
  source this step, or a vertex with `phi < 4h`. The phi test is **one-sided** so
  that "every cell with negative centre phi is in a seed tile" is a theorem.
- `:438` `uvTwoLevelDilate` carries two radii in two bits through one separated
  three-axis scan: FINE at `k` tiles (bit 1, the sampler's set) and SHELL at
  `k + shellReach` (bit 2, the extension's set). Chebyshev balls compose, so
  SHELL is exactly FINE dilated by the reach.
- `webgpu-uniform-reference.ts:1498` encodes the four classify passes **before**
  `encodeVelocityExtrapolation`. Nothing between there and the transport stage
  writes V or phi, so the classes are the ones a post-extension pass would have
  produced; nothing clears words [N,2N) in between.

### 3. The extension's finest passes early-exit outside SHELL
`webgpu-uniform-velocity-extrapolation.wgsl.ts`: `shellAt(p)` (`:97`) gates
`seedActiveFront`, `updateActiveFront` (still indirect), `resolveConvergedFront`,
`packTransportShell`, and the finest `prolongUnknownVelocity`. Every RESTRICT and
every level at or below `ceil(n/4)` stays dense. The authority pass stays dense
too — it is 0.6 ms and its `surfaceA` output is read by velocity advection at
back-traced positions unrelated to tile classes.

**Stale state is closed at the readers, not by clearing.** `neighborDistance`,
`neighborValue` and `activeNodeConverges` consult `shellAt`; the finest
restrict's two sampling helpers consult `sourceKnownAt` (`:432`). An out-of-shell
face therefore reads as *unknown and infinitely far*, which is exactly what the
dense schedule writes wherever the band does not reach. There is no clearing
pass, no hysteresis, and no dependence on ping-pong parity, so a live toggle flip
or a change of `k` is correct from the first step after it.

### 4. E2b: advection and the projection take the far-air arm on non-FINE tiles
- `webgpu-uniform-reference.wgsl.ts:930` `semiLagrangianAdvection` — outside
  FINE, store `v = 0`, carry V and clear the pressure seed; skip three backward
  traces and the force term.
- `webgpu-uniform-reference.wgsl.ts:1085` `project` — outside FINE, store
  `v = 0`, zero the boundary face, carry V; skip the face data, the pressure taps
  and the ghost-fluid fractions.
- The argument: a cell outside FINE has no liquid, solid or source within `4k`
  cells, so it owns no pressure row and has no liquid `+axis` neighbour. Every
  branch of `project` then lands on the far-air arm, which writes zero — so the
  advected value it would have overwritten was dead work. Confirmed empirically:
  the extension-only and the both arms produce **identical** cell counts and
  extents through frame 30.
- `maccormack` transport is **not** shrunk (only `semiLagrangianAdvection` is);
  Uniform Geometric defaults to semi-Lagrangian.

### 5. UI (all live, no rebuild)
- `Extension work` (tiles|dense) and `Shell reach` (0–8 tiles) beside the E1
  controls on the velocity-extension stage, plus a `Shell tiles` readout; the
  chip reads `two-level · tiles p%`.
- `Far-air work` (tiles|dense) and a `Fine tiles` readout on **both** the
  velocity-advection and the pressure-projection stages, chip
  `far air skipped · p% tiles`. Both stages carry the one control because both
  are shrunk by the same map.
- `lib/methods/uniform/uniform-volume-pipeline.ts`,
  `lib/methods/uniform/uniform-volume-method.ts`.
- Tests: `tests/uniform-volume-initial.test.ts`, new case "E2 shrinks the
  extension and the far-air advection under the same sampler" (8/8 pass).

## The bug this cost most of its time to

The first working build diverged from the off arm by frame 10 and, by frame 40,
had the liquid piling into 15k over-filled cells instead of spreading across the
floor. Raising the shell reach to 8 tiles (84% of the domain) did **not** fix it,
which ruled out the margin.

Cause: `shellAt` sized the tile table from `baseDims()`, i.e.
`textureDimensions(densityIn)` — but `resolveConvergedFront` rebinds that slot to
an `(n+2)³` FIM scratch texture. That one pass therefore computed a table base of
`130³` instead of `128³` and read its SHELL bits out of unrelated memory, so it
resolved a random subset of the band every step. Fixed at
`webgpu-uniform-velocity-extrapolation.wgsl.ts:80` by sizing the table from
`faceOpenIn`, the one binding every group points at the parent's own lattice.
After the fix both arms reproduce the off arm's cell counts and extents exactly
through frame 40.

## Acceptance

Bit-identity was explicitly dropped as a gate for this step; these numbers are
for the record.

**(a) No validation errors, no non-finite values.** Zero `uncapturederror`
messages in every Dawn run. Zero non-finite values across V, velocity and phi at
eight samples of a 70-step run on each arm, and at three samples on
`minimal-power-dam-break-64`. All 26 geometric and 9 extrapolation entry points
compile and build pipelines on Dawn with zero WGSL messages.

**(b) 70-step trajectory, `cm12-figure-7`** (`dt` = scene, dust floor 1e-6, k=2,
shell reach 1):

| frame | V sum off / on | maxSpeed off / on | liquid cells off / on | fine / shell tiles (on) |
|---|---|---|---|---|
| 10 | 33463.9994 / 33463.9994 | 3.334 / 3.334 | 36796 / 36796 | 3484 / 5168 |
| 20 | 33463.9996 / 33463.9996 | 6.668 / 6.668 | 36656 / 36656 | 3552 / 5272 |
| 30 | 33464.0004 / 33464.0004 | 20.120 / 20.120 | 28760 / 28760 | 2156 / 3160 |
| 40 | 33463.9849 / 33463.9846 | 15.512 / 14.840 | 52073 / 52099 | 4180 / 5252 |
| 50 | 33463.9696 / 33463.9693 | 15.818 / 17.000 | 73344 / 74258 | 6284 / 7824 |
| 60 | 33463.9525 / 33463.9489 | 5.117 / 4.549 | 112874 / 111112 | 7435 / 9319 |
| 70 | 33463.9189 / 33463.9151 | 6.874 / 8.363 | 126669 / 126674 | 7161 / 9076 |

V sum is conserved to the same order on both arms (drift ≤ 2.5e-5 of 33464, i.e.
7e-10 relative). Liquid extent is identical through frame 60; at frame 70 the
on arm's water stands one tile higher (y 65 vs 60). The `maxSpeed` and cell-count
series track through the impact and stay the same order through the spread.

**(c) Same-input single step** (prefix run off, one step under each arm, against
an off/off float-CAS noise floor):

| prefix | arm | phi band max | V > dust max | velocity band max |
|---|---|---|---|---|
| 25 | off/off noise | 7.674e-7 | 2.866e-2 | 2.611e-5 |
| 25 | **on** | 8.196e-7 | 2.866e-2 | 4.482e-5 |
| 40 | off/off noise | 1.786e+0 | 1.315e+1 | 8.003e+0 |
| 40 | **on** | 1.329e-1 | 1.148e-1 | 3.427e-2 |

At prefix 25 the on arm sits at the noise floor. At prefix 40 the scene is past
first impact and its own float-CAS noise floor is enormous; the on arm's
difference is an order *below* it, so nothing is resolvable there.

**(d) `minimal-power-dam-break-64`** — clean: no validation errors, no
non-finite values, V sum 94208.018 on both arms at frame 20, cells 104611 / 103959,
extent identical. Live flips (`off → on` mid-run) exercised at two prefixes.
**No wall saving on this scene**: the pool fills the domain, so 73% of tiles are
FINE and there is almost nothing to skip. The win scales with how empty the
domain is.

## Approximations and caveats

Deliberate, in the spirit of "cheaper where it simplifies":

1. **Shell margin is generous, not minimal.** `shellReach` defaults to 1 tile
   past FINE (4 cells) — enough for the finest trilinear tap (1 cell) and the
   FIM's 2-cell accurate band, with room. It is a live 0–8 slider, so a wrong
   guess is a UI change, not a rebuild. On an anisotropic lattice the constructor
   derives `ceil(0.5·max h / min h)` instead.
2. **Wall treatment of the 4h field is coarse.** Unknown components publish 0,
   and the outermost coarse layer on each component axis publishes 0 unless one
   of its 16 fine faces is open. Interior closed faces get no test at all, on the
   argument that any tile holding a partially open cell seeds, so solids are
   always inside FINE and their faces are never read from the table. Untested
   against a scene with interior solids.
3. **E2b reuses FINE, not a liquid-only set.** Simpler, and FINE is already
   `seed ⊕ k` tiles, so the margin is 4k cells where 1 would do.
4. **The authority pass and the 2h prolong stay dense** — 0.6 ms and a
   staleness risk over 262k threads respectively.
5. **The 4h field is not the same field E1 sampled.** E1 area-averaged the fine
   transport shell; E2 reads the hierarchy's renormalized known-only restriction.
   Measured on its own (arm "table only"), the swap reproduces the off arm's
   cell counts and extents exactly through frame 40, so it is not a source of
   error — but it is a different field.

Unverified / open:

- **Totals are not a stable measurement on this lane.** Two probe captures of the
  same code disagree by 50% on absolute per-advance time. Only the stage deltas
  were reproduced across both, and the arms run sequentially in one process, so
  thermal drift biases whichever arm runs last. Both orders were captured; the
  conservative reading is the reversed one (−15% total).
- No scene with an interior solid, terrain, a rigid body, an inflow or a drop was
  run with the toggle on. The seed covers all of them by construction
  (`uvOpen < 1`, `dropSource`, `inflowSweptPlugSource`), but none was measured.
- `maccormack` velocity transport is not shrunk.
- `large-power-dam-break` (64×20×64) was not run; the axis-multiple-of-4 gate
  admits it, but nothing was measured.
- The `pressure-projection` chip and control were placed by stage id; they are
  asserted by the new test but have not been seen in the app.

## What to look at in the app

`cm12-figure-7`, Uniform Geometric, SIM tab:

1. Velocity-extension stage → **Sampler: Two-level**. That is the only switch;
   `Extension work` and `Far-air work` are already on `Shell tiles` / `Fine tiles`.
2. Watch the chips: the extension reads `two-level · tiles p%`, and both the
   velocity-advection and pressure-projection stages read
   `far air skipped · p% tiles`. Their per-stage costs should drop by roughly
   half (extension) and three-quarters (projection).
3. Flip `Extension work` and `Far-air work` to `Dense` and back while it runs —
   the water should not jump.
4. The thing to judge visually is the far field: with the sampler on, air
   velocity outside the fine tiles comes from a 4h field. If anything looks wrong
   at the fine-tile boundary, raise **Shell reach**, then **Fine reach**.
5. `minimal-power-dam-break-64` is the control for "no win": the pool fills the
   domain, so almost every tile is fine and the cost should be unchanged.
