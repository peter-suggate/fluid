# M5 design note: rigid bodies on the brick engine (lane P, 2026-11-17)

Status: **E accepted §8 on 2026-11-17** with the changes in §8.1. Implemented so far: the shared body WGSL
(lib/methods/narrow-band/bodies-shapes.wgsl.ts) and its R6/R7 tests (§10). No kernel yet. Scope is structural plan §2.8 / §4.2 M5 and the rollout
plan (brick-default-rollout-2026-11-17.md, lane P).

The physics is the old engine's, ported verbatim. All file:line references are to the old uniform tree on main at
c95f17aa. The structure follows the brick pool: stamp the geometry into brick channels once a step, then let every
kernel read the channels. §8 lists the ABI amendments E is asked to accept or change.

## 1. What is ported

| old engine | brick engine |
|---|---|
| `encodeBodies` mirror (uniform-mixed-solid.wgsl.ts:485), `bodyMask` (647), `umBuildTile` records (813-934), `encodeCoarse` (1153) | `bodyStamp`: one list-indirect launch over `F_BODY` (§3) |
| `umCellOpenGeometry` (1728), `umPressureFaceV` (1915), `umFaceOpen` (1879), `umSolidFaceVelocity` (1817) | the same functions, evaluated by `bodyStamp` into the cut slot's channels (§2) |
| `markTiles` / `umBodyMarkTile` (uniform-mixed-bodies.ts:189-217), `umBodyBoxTileGap` (uniform-body-tile-bound.ts:27-98) | `setMark`'s body-contact criterion, as a P function E's `setMark` calls (§4) |
| band RHS moving wall (uniform-pressure-arena.ts:1571-1631), K_CLOSED / sealed faces, project's closed face = u_s (3217), release (3368) | P's `operator` / `rows` / `project` read `FRAC`, `AP*` and `W*` (§5) |
| `couple` (uniform-mixed-bodies.ts:37-94), `rigidExchange` 12 words ×1e6 / ×65536 (webgpu-eulerian.ts:443-471) | `couple`: one list-indirect launch over `F_BODY` ∩ liquid, after `project` (§6) |
| `WebGPURigidBodySystem` (lib/core/webgpu-rigid-body.ts): tables, integrate, contacts, motors, fixtures | unchanged and shared, encoded at the tail (§7) |
| `nbRigidContact`, `nbParticleBlocked`, body windows (uniform-particle-contact.wgsl.ts:147-178, uniform-narrow-band-flip.ts:1454-1567) | F's move stage, with P's shared body WGSL module (§9) |
| `umWalk` / `umCellIsOpen` with bodies, `umReachSolidClear` off when bodies exist (uniform-mixed-surface.ts:459-552) | F's level-set paths through `FLAGS.OPEN`, which `bodyStamp` maintains (§2.3) |

**Not ported, and refused explicitly.** Each gives a `brickSceneSetup` gap, or a `FAIL.UNSUPPORTED` latch when it
arrives at runtime:
- **Rounded plates** (shape 5, the exact ray-interval path, uniform-rigid-intervals.wgsl.ts). No catalog scene
  uses one.
- **Unknown shape tags.** The old WGSL returns inside=false and distance=1e20 for these; the brick engine latches
  instead.
- **Interaction entities** (interactive balls). These are M6.
- **More than 24 bodies** (`GPU_RIGID_BODY_CAPACITY`).

Interior (inverted, `-(shape+1)`) bodies and body-local box cuts (`SolidBoxCut`, solid-cuts.ts) are ported. Both
are cheap and verbatim.

## 2. Channels: bodies share the cut slot

### 2.1 The cut pool

`solidStamp` (E, M3) writes the static channels `FRAC` and `AP0..2` into a brick's K (cut) slot.

Bodies use the **same slot and the same channels**. A brick near a body (class `NEAR_BODY`) holds a K slot even
when it has no static cut, and its `FRAC` / `AP*` are the **composite**: static and body together, by the old
formulas.

Three channels are added:

| pool | ch | name | layout | written by | read by |
|---|---|---|---|---|---|
| `cut` | 0 | `FRAC` | cell | `solidStamp`; `bodyStamp` on `NEAR_BODY` bricks | `operator`, F's solid paths, `couple` |
| `cut` | 1–3 | `AP0 AP1 AP2` | lower face | as above | as above |
| `cut` | 4–6 | `W0 W1 W2` | lower face: the moving-wall velocity u_s (m/s), 0 off bodies | `bodyStamp`; `solidStamp` writes 0 | `operator` (RHS), `project` (closed faces, release) |

After the K region, the cut pool holds **one word per K slot**, `BMASK`: the slot's body candidate mask (24 bits).
`bodyStamp` writes it, and `couple`, F's move stage and `project` read it.

This keeps every reader of body geometry on the existing `cut` binding. The cut pool's size becomes
`7 · KCAP · 520 + KCAP` words.

### 2.2 Who writes a NEAR_BODY brick

**`bodyStamp` owns a `NEAR_BODY` brick.**
- It needs the static solid at sample points as well as per cell, because the old formulas sample voxels and
  terrain inside body faces (umFaceOpen 1879-1908, umSolidFaceVelocity 1817-1871).
- It therefore **recomputes the static part from E's source**, never from a stored static channel.
- E provides three WGSL functions, the same ones `solidStamp` uses:
  - `nbStaticCellOpen(c) -> f32`, the cell's static open fraction;
  - `nbStaticFaceOpen(c, axis) -> f32`, the static aperture;
  - `nbStaticSolidAt(world) -> bool`, a point test over the voxel mask, terrain, procedural bricks, padding and
    the box.
- Where no body candidate reaches a cell or face, `bodyStamp` writes exactly `solidStamp`'s value: the same function
  on the same input, so the value is bitwise equal.
- `solidStamp` therefore skips `NEAR_BODY` bricks. It also leaves `W` at 0 in the bricks it writes.

**Static values are never stored separately.** No second copy of the static channels exists. When a brick stops
being `NEAR_BODY`, `bodyStamp` makes one more pass over it (class bit `BODY_WAS`, §3) and writes the static values
and `W = 0`. E's `setUpdate` then frees the K slot if the brick has no static cut. A slot-less brick is fully open,
as today.

### 2.3 Composite formulas (verbatim)

These are written per cell or face of a `NEAR_BODY` brick, with `cands` the slot's `BMASK` filtered by the old
reach tests:

- `FRAC(c)` = umCellOpenGeometry (1728-1741):
  - `clamp((1 − body)(1 − terrain), 0, 1)`, with `body` = umCellBodyFraction (uniform-rigid-coverage.wgsl.ts:10-41):
    8 corner samples at ±0.4h, a union across bodies, and the 1.01·length(0.4h) clearance certificate;
  - `min(box.y, min(1 − body, 1 − terrain))` on a boundary-record cell;
  - 0 on a voxel cell.
  - E's `nbStaticCellOpen` supplies the static branch.
- `AP_a(c)` = umPressureFaceV of the face between `c − e_a` and `c` (1915-1952):
  - a face no body reaches within length(0.5h) takes the static value;
  - a voxel-adjacent face takes `0.5 · (FRAC(c − e_a) + FRAC(c))`;
  - otherwise umFaceOpen: 4 transverse samples at ±0.35h, each solid if `nbStaticSolidAt` or a body holds it. The
    clearance certificate is `1.01 · 0.35 · length(h_a, h_b)`.
- `W_a(c)` = umSolidFaceVelocity of that face (1817-1871):
  - 8 dual-cell samples at ±0.4h;
  - the owner certificate;
  - static samples count as solid with velocity 0;
  - the result is the mean rigid velocity `v + ω × (x_face − x_body)` over the solid samples;
  - the fallback is the nearest body within one cell;
  - a voxel or wall face gets 0.
- `FLAGS.OPEN(c)` = `FRAC(c) > 1e-5` (the umCellIsOpen predicate). `bodyStamp` writes the OPEN bit of its bricks, so
  F's walk and solid tests see bodies without new code. FLAGS gains a writer: `bodyStamp`, OPEN only.
- **Body distance with cuts** follows solid-cuts.ts:15-17: `max(d, −solidBoxCutDistance)` for each of the body's
  cuts. The cuts are read from a P-owned `bodyCuts` buffer (§7.3), uploaded on change.

## 3. bodyStamp and the lists

**Classes.**
- `NEAR_BODY` (F_CLASS bit 4) is set by `setMark` from the body criterion (§4), not by `bodyStamp`. The ABI table
  names `bodyStamp`; that is amendment A2.
- `BODY_WAS` (F_CLASS bit 24, P) means `NEAR_BODY` was set last step. `setLists` keeps it from the old word.

**The list.** `F_BODY` (already in `setLists`' fLists) lists every `NEAR_BODY | BODY_WAS` brick. Only the K-slot
rule above makes `BODY_WAS` bricks need a list entry. E can choose a separate `F_BODY_WAS` list instead.

**Place in the schedule.** `bodyStamp` runs **after `setLists`**, before the level set. It cannot run at the head,
as §2.11's table has it, because entering bricks have no slot before `setUpdate`. E's `solidStamp` has the same
need for entering bricks, so one "stamp" stage after `setLists` holds both: `solidStamp` (gated), then `bodyStamp`.

The stages before it (the move) test bodies **analytically** from the pose records, as the old engine does:
- `nbRigidContact` and `nbParticleBlocked` use umBodyAt and umBodyDistance, never the cell records;
- so the move needs no channels, and reads this step's pose.

**Launch.** A workgroup per `F_BODY` brick, 128 lanes, 4 cells a lane. Per brick, it:
1. builds `BMASK`: each body whose bounding sphere, dilated by length(2h) plus the swept reach, meets the brick;
2. writes FRAC, AP and W for its 512 cells and 3·512 lower faces, then OPEN;
3. latches `FAIL.UNSUPPORTED` (info: the key) on:
   - a shape tag outside 0–4;
   - a rounded plate;
   - a `NEAR_BODY` brick with a tile that is not h-owned (§4).

**Reuse (the old umBodyMoved).** A brick whose candidates' poses are bitwise unchanged since its last stamp, and
which is not ENTERING, is skipped. This covers static bodies: rigid-hydrostatic, garden-hose, sphere-jet. The
built poses are a P buffer copied from the rigid state after the stamp (`copyBufferToBuffer`, 24 · 128 B). This is
the first optimisation, not part of the first port.

## 4. setMark: bodies stay at h

The old criterion is ported as `pbBodyMarkTile(tile)`, in P's `bodies-shapes.wgsl.ts`, for E's `setMark` to call
per 4³ tile:
- **Ball prefilter:** `dist(clamp(c, lo, hi), c) > r + |v| · dt + cell`.
- **Margin:** `(|v| + |ω| r) · dt + cell`.
- An inverted shape always marks.
- `lower = dist(body, middle) − length(2h)`, and for a box `max(lower, umBodyBoxTileGap)`. That is the
  separating-axis test: 3 + 3 + 9 axes plus the closest-point direction.
- The tile marks when `lower ≤ margin`.

**A marked tile is h-owned.** Its brick is `NEAR_BODY`, and the one-tile ring around it is h as well, so the "4h
owners never reach the helpers" rule holds: no C row ever has a body-cut face.

This replaces the old 4h moving-wall path (umCutWord bit 16, uniform-mixed-pressure-velocity.ts:254-278): with the
ring at h it cannot activate. A `NEAR_BODY` brick holding a non-h tile latches `FAIL.UNSUPPORTED`, and never
falls back silently.

**Pose timing.** setMark in part 2 of step n reads pose n; bodyStamp in part 1 of step n+1 reads pose n+1. The
margin's `|v|·dt` plus one cell covers that one-step lag.

**Liquid only.** Residency is still driven by liquid and particles. `NEAR_BODY` is a class on resident bricks; a
body far from liquid costs nothing.

## 5. Pressure: the body terms (P's files)

Units are the slot contract's: X = p dt/ρ, rows ×h² dt/ρ. The band rows (uniform-pressure-arena.ts:1571-1631) are
ported to `rows` and `operator`. A row is a cell of a brick with a K slot.

**Capacity.**
- `cap = FRAC(c)`.
- The volume-control term is `rate · h · cap`. It is already in the RHS formula; today `cap = 1`.

**Face weights.**
- `A_a = AP_a`. A face is K_CLOSED (coupling 0, fraction 0) when `A ≤ 1e-6`, or when it is sealed:
  - `liquid ∧ shut(c) ∧ ¬wet(n)`, or `¬liquid ∧ wet(n) ∧ shut(n)`;
  - `shut = ¬OPEN`.
- A lattice-wall face's coupling is `V/h²` scaled as today.

**θ** (verbatim):
- between liquid and air where either cell is shut: bPairTheta;
- where both cells are inside a solid with `FRAC < 0.1`: θ = 1;
- otherwise bFaceTheta.

**Moving-wall RHS.** For each face that is not a wall or an air face, with `cap ≠ A`, a liquid row in a `NEAR_BODY`
brick adds `sign · (cap − A) · W` to the divergence. So:

`RHS(c) = −h ( Σ_a [ (A·US + (cap(c) − A)·W)_{upper face} − (A·US + (cap(c) − A)·W)_{lower face} ] − rate·h·cap(c) )`

Note that `cap` is **the row's own** `FRAC` on all six faces (the old head capacity, bHeadCapacity), not the
neighbour's.

**The inside-solid kinds** (0x40000000 inside, 0x20000000 shut) are carried as row bits wherever the old rows read
them. The implementation checks every use before porting it.

**project:**
- A K_CLOSED face gets `U = W` (3217).
- An open face gets `U = US − ∇X` as today.
- The release test uses `U − W` (3368, FLAGS bits 8–10).

**MG.** The coarse levels' couplings are formed from the fine `K`, which already carries the apertures.
- The cut latches in pressure-composite.wgsl.ts:545/563 and pressure-mgpcg.ts:820 are replaced by the cut handling.
- C rows still latch on a cut, by the §4 invariant.

**As landed (e4e1e152, behind `opt.cuts`, default off).** The terms above, with these explicit UNSUPPORTED latches
(info word `tag | brick key`), each with an owner:
- **0xa2, a liquid row in a static solid cell** (`CUT_STATIC` brick, `FRAC <= 1e-5`): the old inside-solid rows and
  their complementarity machinery (umCellInsideSolid, 0x40000000 / 0x20000000) are not ported. Owner P, next after
  A2-A4 and the rigid stage (sooner if E's batch reaches voxel scenes first). Any scene that binds E's static mask
  with liquid able to enter a voxel cell hits it.
- **0xa3, a seam face with `A != 1`:** rowsC takes the seam faces at volume 1, so an aperture there would be lost.
  Only a static obstacle (voxels, a boundary record, terrain) that cuts through the h / 4h seam can produce it:
  bodies never do, since §4 keeps every body tile and its one-tile ring at h. So the scenes are the static-solid
  ones whose obstacle meets a coarse tile: tiered scenes with scenery or terrain below or beside coarse water.
  Owner P with 0xa2 (a cut seam needs rowsC to read the h face's aperture, or E's tiering to keep cut tiles' rings h).
- **0xa4, a face into an absent brick from a brick with a body candidate** (BMASK != 0): the absent side has no K
  slot, so its aperture is unknown. §4's h ring makes it unreachable while setMark's margin holds; it latches if not.
- A cut brick with a lattice tile not h-owned (the §4 invariant broken).

## 6. couple

**What it is.** A list-indirect launch over `F_BODY` bricks with an h-owned liquid cell.
- It runs **after `project`**, so X is final, and before the integrator.
- It is not fused into `project`. Its 12 × 24 workgroup atomics would cost `project`'s occupancy on every brick,
  not only the body bricks.

Per cell, verbatim (uniform-mixed-bodies.ts:37-94):

**Pressure:**
- `p = X · ρ / dt` (Pa).
- p = 0 when `FRAC(c) ≤ 1e-5` or the cell is not liquid (umBodyPressure 171-176).

**Per candidate body in `BMASK`:**
- It is near when `umBodyNear(c, length(h))`.
- It is clear when `|umBodyDistance| > 0.505 · length(h)`.
- **Face fraction:** umBodyFaceFraction (17-25), 4 transverse samples at ±0.35h, **bodies only**.
  - This is recomputed from the pose, not read from `AP`, because AP includes the statics.
  - The pose is the same pose `bodyStamp` used: the integrator runs after `couple`.
- **Reaction:** `p · (blocked⁺ − blocked⁻) · h_a h_b · dt`.
- **Torque:** `(x − x_body) × reaction`.

**Wet weight:**
- `wet = clamp(0.5 − 0.125 · Σφ8 / 4, 0, 1)`. That is umBodyOwnerWet with φ in h, so the old `/(4h)` becomes `/4`.
- The weight is `wet · fraction`, with fraction = umBodyFraction (0/1 when clear).

**Exchange:** workgroup sums into `rigidExchange`:
- words 0–5: `round(x · 1e6)`;
- word 6: `round(wet · 65536)`;
- word 10: an `atomicStore` of 1 when near.

The exchange buffer is cleared before `couple` (`clearBuffer`, as encodeCoupling 226-236). Words 7–9 and 11 stay 0,
as in the old couple.

**No double counting.** This reaction is the transpose of the RHS moving-wall flux. With word 10 set, rigidKicked
applies no empirical drag, buoyancy or added mass (webgpu-rigid-body.ts:692-719).

## 7. The rigid integrator, motors and scenery

### 7.1 The shared system

`WebGPURigidBodySystem` is constructed exactly as webgpu-uniform-reference.ts:932-946 does. It takes:
- the exchange buffer;
- the terrain texture;
- the SolidWorld;
- the procedural collision source.

**Tail order:** `uExtend` ×2, `uDelta`, then `couple`, then `rigidSystem.encode(encoder, dt, h³, 1, h)` (tables,
then integrate), then the built-pose copy (§3).

**Paddleboats motors** come from the motor records (rigid-motor.ts). Fixtures are attached by attachMotor. Thrust
is pressure on the paddles; nothing empirical.

**Prescribed motion and the roster** stay on the CPU:
- samplePrescribedRigidBody, then syncBodies, with the unchangedSinceSync fast path and the cached records of P2
  item 7.
- The write goes into the system's own state buffer before part 1 (amendment A4).

### 7.2 The pose binding

Kernels read the rigid state buffer itself as a read-only `bodies` binding: 24 × 32 f32, the 128 B record of
webgpu-rigid-body.ts:73-92.
- The readers are `bodyStamp`, `couple`, F's move, and `setMark` through pbBodyMarkTile.
- The integrator writes the buffer only at the tail, so every stage of a step sees one pose. No mirror is needed.
- The ring's words 64–191 (amendment A3) hold a header instead of pose records:
  - 64: body count (u32);
  - 65: flags (u32): bit 0 coupling on, bit 1 any cut, bit 2 any inverted;
  - 66: the stamp horizon (dt);
  - 67: cell volume;
  - 68: a pose generation (u32), bumped when the roster changes, which invalidates the reuse of §3;
  - 69–191: spare, P.

### 7.3 Body cuts

`bodyCuts` is a P buffer holding the atlas of solid-cuts.ts:32-38, packed by packRigidBodyCutAtlas. It is written
only when the signed description changes (the setBodyCuts cache).

It is bound read-only to `bodyStamp`, `couple`, and F's move (contact distance). The old engine required its
`plates` feature for cuts (uniform-mixed-solid.wgsl.ts:233-244). Here cuts need no feature flag, and plates are
refused.

### 7.4 No-slip

The old engine's no-slip changes only the viscous term's held taps (uniform-mixed-forces.ts:306-311, 408-420). It
is inert at μ = 0: paddleboats and every rigid scene are inviscid.

So no-slip needs no body work. F's `nbfVelocitySupported` already refuses μ > 0 with no-slip (velocity-formulas.wgsl.ts:24),
and brick-fluid-stages.ts:47 passes `noSlip: false`. It should pass the scene's `fluidWallMode`, so that the
refusal fires; that is F's file.

### 7.5 Scenery

Rigid bodies meet scenery only inside the integrator: the SolidWorld voxel mask, the container shell, procedural
collision and terrain. That is unchanged. The fluid sees scenery through E's static channels.

## 8. ABI amendments for E

| # | change | where |
|---|---|---|
| A1 | `cut` pool channels 4–6 `W0..2` and the per-K-slot `BMASK` word region; size `7·KCAP·520 + KCAP` | brick-abi.md §2.3, brick-abi.ts POOLS.cut |
| A2 | `NEAR_BODY` is set by `setMark` (P's pbBodyMarkTile), and `BODY_WAS` is F_CLASS bit 24 (P) | §4.1 F_CLASS |
| A3 | ring 64–191: the header above, not 8 × 16 pose records (paddleboats has 18 bodies; the cap is 24). Poses come from the `bodies` binding | §5.2 |
| A4 | a host write per step besides the ring: syncBodies into the rigid state buffer, only on change, before part 1 | §5.2, "no other host write" |
| A5 | a K slot for every `NEAR_BODY \| BODY_WAS` brick (E's `setUpdate`); KCAP is authored from the measured peak (no runtime growth); `solidStamp` skips `NEAR_BODY` bricks | §2.3, §9 pools |
| A6 | one stamp stage after `setLists`: `solidStamp` (gated), then `bodyStamp` | structural §2.11, brick-stages.ts |
| A7 | E exports `nbStaticCellOpen`, `nbStaticFaceOpen` and `nbStaticSolidAt`, the functions `solidStamp` itself uses | E's M3 solid module |
| A8 | `FLAGS.OPEN` gains `bodyStamp` as a writer (OPEN only, on its bricks) | §2.4 |

### 8.1 E's verdict (2026-11-17)

All eight accepted, with these changes:
- **A1.** Every K-slot pop gets `W = 0` from whichever stamp first writes the slot. bodyStamp's clean-up pass (BODY_WAS
  and not NEAR_BODY) also writes `BMASK = 0`, so couple and F's move skip the brick.
- **A2.** setMark owns both bits in one read-modify-write: `BODY_WAS := old NEAR_BODY`, then the new NEAR_BODY.
  setLists does not keep BODY_WAS. Entering bricks: setUpdate's entry path runs the brick-level ball prefilter on
  ENTERING bricks and pops a K slot, so the same step's stamp covers them (a brick entering inside a body is otherwise
  open for a step). setMark's body test is gated on ring word 3 bit 1 and gains the `bodies` binding.
- **A3.** The "bodies present" gate is E's ring word 3 bit 1, not header word 65.
- **A4.** syncBodies goes into the queue after step n's part 2 and before step n+1's part 1. P checks that a
  prescribed-motion change never rewrites a dynamic body's integrated pose, lists the brick scenes that need
  samplePrescribedRigidBody, and measures it at the step-2 gate (a per-step CPU stage: plan a GPU evaluation from the
  authored parameters for any scene that needs it).
- **A5.** F_CLASS bit 9 (E) `CUT_STATIC`, set at entry and by solidStamp. A brick holds a K slot while CUT_STATIC,
  NEAR_BODY or BODY_WAS is set. KCAP: static-cut bricks the fluid can reach plus the measured body-shell peak; an
  underestimate latches POOL_K.
- **A6.** The stamp stage goes at the end of the head, after residency, lists(1), `want`, the tiered transfer and the
  certificate, and before release.part1 (setInit and the transfer write FLAGS; the release port's reach words read
  OPEN). F_BODY is one list of NEAR_BODY or BODY_WAS.
- **A7.** `nbStaticCellOpen`, `nbStaticFaceOpen`, `nbStaticSolidAt`, and (accepted by E the same day) two more (the old composite
  branches on them, bodies-shapes.wgsl.ts `PbStaticHooks`): `nbStaticCellKind(c) -> u32` (0 plain, 1 boundary
  record, 2 voxel without a record: the FRAC min/product forms and the voxel-wall face) and
  `nbStaticFaceBoxed(c, axis) -> bool` (umBoundaryTouchesBox at the face's unit box: a body-free voxel-wall face
  takes the static value when boxed).

### 8.2 The shared module and a Metal miscompile

`bodies-shapes.wgsl.ts` ports, with the pose and cut atlas as storage arrays the kernel names and every query taking
its candidate mask: the pose queries (§9), `pbCellBodyFraction`, `pbStampCell` / `pbStampFace` /
`pbStampFaceVelocity` (§2.3, through the static hooks), `pbBodyFaceFraction` and `pbCoupleBody` (§6),
`pbBodyMarkTile` / `pbBodyBoxTileGap` (§4), `pbRigidContact` (§9) and `pbBodyBoxCandidates` (BMASK).

**The module writes no vector component through a dynamic index.** The verbatim couple loop
(`var low=id;low[axis]-=1;area[axis]=ff(id)-ff(low)`) gave wrong face areas on Metal in one test kernel while each
call alone was right (the submerged sphere's impulse: −6.71 N·s against +10.84). The old text compiled in the same
kernel was also wrong (+1.35). An `array<f32,3>` accumulator was right; a select-built callee with a vec3f accumulator
was not; a small repro was right. The trigger therefore depends on the kernel. Offsets are now `c · e_axis` (`pbE`,
exact). The old engine's production couple passes its own hydrostatic test (uniform-rigid-rest, "large floating
sphere", 2026-11-17), so the old kernel as built is not hit, but the pattern is fragile. Old-WGSL parity cannot see a
miscompile both texts share, so R6/R7 also hold a Float64 mirror (§10).

## 9. F's side (F's files: scratch first, F reviews)

P provides `bodies-shapes.wgsl.ts`:
- `pbBodyDistance`, `pbBodyInside`, `pbBodyVelocity` and `pbBodyNear`, with interior inversion and cuts;
- `pbBodyAtCandidates`;
- `pbRigidContact`, the nbRigidContact port.

F calls them in four places:
- **pTrace:** `pbRigidContact` at the move's start, and in every half-cell stride where the brick's `BMASK` is not
  0 (nbFinish 1493-1567). Procedural and terrain contacts inside a body are rejected.
- **pSolid:** a particle inside a body retires (nbParticleBlocked).
- **The level set:** umWalk's stop and the surface gradient's closed samples come through OPEN. umReachSolidClear
  is off when the body count is above 0 (uniform-mixed-surface.ts:491, 552).
- **brick-fluid-stages.ts:47:** pass the scene's `fluidWallMode` (§7.4).

## 10. AGENTS.md regressions

The stamped channels are derived data. The source geometry is the body descriptions: shape, dims, pose and
body-local `cuts`, plus E's static sources. These stay in the scene, so hit testing and editing keep going through
the shared path:
- visibleToolSurface (voxel-editor/surface.ts:44-57);
- sceneWithRoundedPlateCuts / sceneWithSolidStroke (rigid-surface.ts:28-47, solid-world.ts:1274-1283).

The brick engine adds no geometry that is rendered or edited. The tests are below; each is a CPU-oracle test unless
marked GPU.

| | test | asserts |
|---|---|---|
| R1 | nearest-surface hit | In a brick scene with a static body in front of a voxel wall and behind another, rays return the nearer surface. The cell holding the hit point has stamped `FRAC < 1` (GPU readback): the collision agrees with what is visible. |
| R2 | local edit opens both | (a) A voxel cut inside a `NEAR_BODY` brick opens the composite `FRAC`/`AP` exactly in the cut cells, against the CPU oracle of umCellOpenGeometry/umFaceOpen, and body coverage is retained. (b) A body-local box cut (a `SolidBoxCut` on a static body, through sceneWithSolidStroke where the body is editable) opens `FRAC`/`AP`/`W` in the cut, and the visual is the same cut (rigid-cut-render's path). One edit feeds both; there is no separate collision copy. |
| R3 | untouched fidelity | Every cut-pool word outside the edited cells and their face stencil is bitwise unchanged, and the scene's other bodies and voxels are unchanged. |
| R4 | save/load/undo | Saving the edited scene, loading it and re-stamping gives channels bitwise equal to the post-edit stamp. Undo gives bitwise the pre-edit stamp. |
| R5 | unknown kinds | Shape tag 9 and a rounded plate give a `brickSceneSetup` gap. The same codes arriving at runtime (a roster change) latch `FAIL.UNSUPPORTED` on the GPU. A body count above 24 throws. |
| R6 | oracle parity | `bodyStamp`'s FRAC/AP/W on a frozen state matches the old engine's umCellOpenGeometry / umPressureFaceV / umSolidFaceVelocity (GPU, the old WGSL run as the oracle), for sphere, box, capsule, cylinder, cup, inverted and cut bodies. |
| R6/R7 (done) | bodies-shapes.test.ts | bitwise against the old WGSL (from its generators) for FRAC/AP/W, all-bodies and per-brick BMASK, pbRigidContact and couple's per-cell words; 0 of 32768 stamp words off a Float64 mirror; submerged sphere impulse −1.06% of ρgV·dt |
| R7 | couple parity | Exchange words 0–6 and 10 match the old couple on a frozen pressure field (GPU), and the hydrostatic impulse is ρgV dt within 2% for a submerged sphere. |

**Editing policy.** A wet scene's sphere keeps the object tool (cloth-system.test.ts:445), so R2(b) uses an
editable body. Both forms of R2 use the shared edit path and nothing new.

## 11. Order, dependencies and gates

**Step 2: rigid-hydrostatic, rigid-float, rigid-sink.**
- Each is a 0.8 m tank at h = 0.05 (16³), dt = 1/30, closed top, free slip, with one sphere.
- **They need from E:**
  - `brickPools`;
  - 16³ padded to whole pages (static padding solid);
  - the tank-fill initial liquid;
  - the K slots of A5.
- **Gates.** Every run is one short GPU run, with no FAIL and no abort.

| scene | steps | asserts |
|---|---|---|
| hydrostatic | 15 (0.5 s) | vertical impulse/dt = ρgV within 2% at steps 5–15; peak |u| < 0.01 m/s (as uniform-rigid-rest.test.ts:8) |
| float | 60 (2 s) | y within 0.005 m of the old engine's equilibrium; peak |u| < 0.1 m/s |
| sink | 30 (1 s) | y(t) at steps 15 and 30 within h/2 of the old engine; rests at y = r on the floor |

All three are also checked for volume drift of at most 0.0025 at the end, and give a step-time figure (median of
steps 5–N, fenced).

**Step 3: paddleboats.**
- 18 bodies, motors, open top (E, M3), h = 0.025.
- `brickPools.k` is currently 0. It is authored from the measured `NEAR_BODY` peak.
- **Gates:** C0 and P1–P4 over 300 steps (tools/nbflip-gates.ts, nbflip-gate-bench.ts `--scene=paddleboats`), plus
  the hull test of paddleboats-scene.test.ts:60 with FLUID_PADDLEBOAT_GPU.
- **Step-time target:** 19 / 17 ms against 49.7 / 44 (structural M5).
- **Known cost:** the integrator's 24 single-thread workgroups (`integrate`, webgpu-rigid-body.ts:778) and the
  4.78 ms CPU advanceTo. These are measured at that gate, not changed here.

**Then** the remaining body scenes, in the order the rollout plan gives, after E's statics land.
