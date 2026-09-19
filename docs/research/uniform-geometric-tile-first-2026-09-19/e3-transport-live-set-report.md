# E3 — conservative volume transport on a live tile set

Uniform Geometric, `cm12-figure-7` (128³), headless Dawn/Metal. Follows
`e2-extension-tiles-report.md`, which shrank the extension, the advection and
the projection onto the same 4h classes. E3 shrinks the one stage E2 left
alone: Sec. 3.4's conservative volume transport.

One control turns it on: SIM tab → **Conservative volume transport** →
`Transport work: Live tiles` (the default). It needs the two-level sampler and
a non-zero dust floor, both of which are already the defaults.

## Headline

`cm12-figure-7`, per-advance GPU medians, ONE probe capture of three arms in
`dense, tiles, dense` order (70 advances each, 63–68 hardware samples):

| seam | dense | **tiles** | dense (again) | change |
|---|---:|---:|---:|---:|
| Sec. 3.4 transport coupling | 27.72 ms | **7.34 ms** | 27.53 ms | **−73%** |
| gather + phi capacity (`balance` seam) | 2.88 ms | **0.72 ms** | 2.88 ms | **−75%** |
| **the two together** | **30.60 ms** | **8.06 ms** | **30.41 ms** | **−22.5 ms/advance** |
| whole advance | 111.35 ms | **89.46 ms** | 115.21 ms | **−21%** |

Every other seam is unchanged inside the 65.5 µs tick (phi 2.36/2.29/2.29,
sharpening 2.62/2.49/2.82, extension authority 1.376 on all three, FIM front
1.64/1.57/1.77, hierarchy fill 0.852 on all three, advection 1.64/1.57/1.64,
projection 0.59/0.59/0.52). The whole-advance delta (−23.8 ms against the mean
of the two dense arms) closes against the stage delta (−22.5 ms), so the total
is not carrying an unattributed change.

### By regime (same capture, medians)

| regime | frames | coupling dense / **tiles** / dense | gather dense / **tiles** / dense | total dense / **tiles** / dense |
|---|---|---|---|---|
| free fall | 2–25 | 24.02 / **3.67** / 24.18 (**−85%**) | 2.79 / **0.59** / 2.82 | 76.55 / **51.51** / 76.94 (−33%) |
| impact | 26–40 | 27.43 / **7.86** / 26.67 (**−71%**) | 3.44 / **0.72** / 2.82 | 112.03 / **93.59** / 116.49 (−18%) |
| spread | 45–70 | 31.13 / **8.95** / 30.31 (**−71%**) | 2.88 / **0.95** / 2.95 | 124.65 / **98.04** / 121.96 (−20%) |

The two dense arms bracket the tiles arm and agree with each other to 3% on
every stage, so this lane's bimodality (memory: "CM12 stage-cost lane is
bimodal") is not what is being read here.

Free fall reaches the design table's target — 24.0 → 3.67 ms is a **6.5×** cut,
against the table's "~12 × fine⊕2 ≈ a tenth of the dense 33 ms". Impact and
spread reach 3.5×, because the live set is 19–24% of the domain there rather
than 11%.

### Live set

| regime | live tiles (of 32,768) | measured displacement D (cells) | reach used (tiles) |
|---|---|---|---|
| free fall | 3,542 (10.8%) | median 2.8, max 5.3 | 2 |
| impact | 6,328 (19.3%) | median 8.1, **max 27.2** | 4 (up to 8) |
| spread | 7,873 (24.0%) | median 4.6, max 13.2 | 3 |
| whole run | min 3,476 / med 6,328 / **max 16,404** | max 27.2 | 2–8 |

## The reach decision: measured, not configured

The brief offered (a) a separate authored reach, (b) reuse FINE and raise its
default, (c) derive it per step. **(c) shipped**, in a form better than the
brief hoped for: not last step's max speed, but *this* step's.

`uvTwoLevelSeed` already sweeps every fine cell of its tile at the head of the
step. It now also reads that cell's velocity and `atomicMax`es the domain
maximum of |v|·dt/h, in cells, into a counter word
(`uniform-volume.wgsl.ts:486`). The three dilation passes run in separate
dispatches *after* it, so `uvTwoLevelTransportReach` (`:516`) reads that number
back and dilates by `ceil((ceil(D)+1)/4) + margin` tiles — the predicate's own
`ceil(D)+1` cells, in whole tiles, because a seed cell is at most `4m` cells
from the boundary of its `m`-tile dilation. This is exact rather than
approximate: transport traces the field the extension propagates *from* this
same velocity, so its domain maximum bounds every backward displacement.

The data says a fixed reach would have been wrong either way. On figure 7 the
requirement ranges over **1 to 8 tiles within one run** — D peaks at 27.2 cells
on a single impact frame, where a reach of 2 (the brief's starting guess) or
even 4 would have starved the set, while 8 everywhere would have kept ~50% of
the domain live for the whole run instead of 11% in free fall.

`Transport margin` (default 1 tile) is therefore headroom on top of the
measured requirement, not the reach itself. Zero is the exact predicate.

**The readout** (`Reach`, beside `Live tiles` on the transport stage) prints
`N used · M required (D cells)` and appends ` · SHORT` when used < required,
which — since used is required + margin by construction — can now only mean the
shader's 16-tile cap bit. The chip appends ` · reach SHORT` in that case.

## What is actually conserved, and how that differs from the design note

The design note (`tile-first-design.md`, "Option B") proved that leaving cells
outside the live set with their **old V** creates volume: a donor outside the
set gives all of its V to the set *and* keeps it. **This implementation does
not do that** — `uvGather` writes `V = 0` and `γ = 0` outside the set
(`uniform-volume.wgsl.ts:294`). That changes the failure mode, and it is worth
stating plainly because it is the safer of the two:

- `uvNormalizeDonors` divides by a column sum accumulated over the rows that
  were **built**, so `Σ_{i∈F} w[i][d] = 1` for every donor any receiver in F
  samples, and `uvFallback` gives a donor nobody samples a self-edge. The
  gather therefore moves every donor's V somewhere and duplicates none of it —
  **for any built set F ⊇ seed**. The total is conserved unconditionally.
- What a set can still do is (a) **destroy** the V of a cell outside it, since
  zero replaces the old value; and (b) **refuse liquid** to a cell outside it
  that should have been wetted, which stalls a front while conserving the total.
- (a) is closed by the predicate "V = 0 outside the set", which is exactly what
  the dust floor buys: it zeroes |V| below the threshold wherever V is written,
  and every cell at or above the threshold seeds its own tile. **With the dust
  floor at zero the predicate fails, so the host forces the dense schedule**
  (`webgpu-uniform-reference.ts:1377`) and the chip reads
  `dense finest lattice`.
- (b) is closed by the measured reach above.

### Conservation numbers (`e3-oracle-fig7.json`)

Four arms from t=0, run sequentially in one process: `dense`, `dense2`
(identical config — the float-CAS noise floor), `tiles`, `starved` (the
negative control). V (`volumeA`) and γ (`gammaA`) captured at four frames and
differenced against `dense`.

Totals (reference 33464 cell volumes):

| frame | dense | dense2 | tiles | starved |
|---|---|---|---|---|
| 10 | 33463.9998 | 33463.9998 | 33463.9998 | 33463.9998 |
| 25 | 33464.0003 | 33464.0002 | 33464.0003 | 33463.9994 |
| 40 | 33463.9875 | 33463.9874 | 33463.9874 | 33463.9894 |
| 60 | 33463.9634 | 33463.9635 | 33463.9632 | 33463.9648 |

Per-cell differences against `dense`. `Σ Δ` is the signed sum, i.e. how much
volume the arm created or destroyed:

| frame | arm | max &#124;ΔV&#124; | Σ&#124;ΔV&#124; | **Σ ΔV** | max &#124;Δγ&#124; | liquid cells | extent |
|---|---|---:|---:|---:|---:|---:|---|
| 10 | dense2 (floor) | 9.48e-6 | 8.5e-4 | +1.4e-6 | 4.3e-6 | 36,796 | identical |
| 10 | **tiles** | 9.48e-6 | 7.5e-4 | −1.2e-6 | 1.6e-6 | 36,796 | identical |
| 25 | dense2 (floor) | 1.28e-3 | 4.7e-2 | −5.3e-5 | 8.6e-6 | 36,632 | identical |
| 25 | **tiles** | **1.23e-5** | 3.6e-2 | −4.1e-5 | 1.1e-5 | 36,632 | identical |
| 40 | dense2 (floor) | 0.213 | 3.70 | −5.5e-5 | 0.099 | 51,977 | identical |
| 40 | **tiles** | 3.82 | 56.6 | −5.1e-5 | 1.0 | 51,962 | identical |
| 60 | dense2 (floor) | 16.85 | 2159 | +3.6e-5 | 1.0 | 86,098 | y≤50 |
| 60 | **tiles** | 25.23 | 2284 | −3.0e-4 | 1.0 | 86,144 | y≤52 (dense 52) |

Reading: **nothing is created or destroyed anywhere** — the signed sum is
≤ 3e-4 of 33,464, i.e. 1e-8 relative, on every arm at every frame. At frame 10
the tiles arm sits *on* the noise floor; at frame 25 it is **100× below** it
(the dense/dense pair disagrees more with itself than tiles does with dense).
At frames 40 and 60 the scene is past first impact and the lane's own
reproducibility has collapsed — the dense/dense pair differs by 0.21 and 16.8
cell volumes on single cells — and the tiles arm's difference is the same order
(18× the floor at 40, 1.5× at 60) with liquid cell counts agreeing to 0.03% and
identical extents. Per the E2 precedent, nothing is resolvable there.

γ deserves a note of its own: writing 0 outside the live set instead of
evaluating `uvTarget` is the single largest saving in the gather (eight
trilinear phi probes = 64 texture loads per cell), and the frame-10/25 γ
differences of 1.6e-6 and 1.1e-5 are direct evidence that the closed form is
right. The argument: the live set contains every tile with a vertex inside the
4h band (the seed's phi test is one-sided), so outside it every corner sample
is positive, and both `uvTarget`'s fill count and its plane-box fraction are 0.

### Negative control

`starved` sets the margin to −8 (not reachable from the panel; see caveats),
which clamps the reach to 0 and starves the live set to the **seed tiles
themselves** — 3.5–8.4% of the domain against the honest 10.8–24.0%, and below
the measured requirement on every frame.

It diverges hugely and conserves exactly:

| frame | max &#124;ΔV&#124; vs floor | Σ&#124;ΔV&#124; vs floor | Σ ΔV | liquid cells (dense) | extent |
|---|---|---|---:|---|---|
| 10 | 3.75e-3 vs 9.5e-6 (**395×**) | 0.173 vs 8.5e-4 (**203×**) | +7.1e-6 | 36,796 (36,796) | identical |
| 25 | 1.003 vs 1.3e-3 (**784×**) | 1084 vs 4.7e-2 (**23,000×**) | −9.0e-4 | 37,332 (36,632) | **[43,4,43]..[84,51,84]** vs [44,0,44]..[83,51,83] |
| 40 | 8.06 vs 0.213 | 7,952 vs 3.70 | +1.9e-3 | 53,249 (51,971) | identical |
| 60 | 26.4 vs 16.8 | 14,850 vs 2,159 | +1.4e-3 | 88,343 (86,249) | identical |

Frame 25 is the picture the proof predicts: the starved column's liquid **has
not reached the floor** (`y` starts at 4, not 0) while the dense arm's has, and
it has spilled a tile sideways instead. That is a stalled front, not lost mass
— the signed sum is 9e-4 of 33,464. So the restricted path is genuinely
restricted, and shrinking it below the predicate mispositions liquid rather
than creating or destroying it.

## No-win control: `minimal-power-dam-break-64`

Runs clean, costs the same, as designed: the pool fills the domain, so almost
every tile is live and there is nothing to skip.

| arm | live tiles | coupling | gather | total | liquid V |
|---|---|---:|---:|---:|---:|
| dense | — / 4,096 | 3.539 ms | 0.393 ms | 38.67 ms | 94,207.830 |
| tiles | **4,081 / 4,096 (99.6%)** | 3.473 ms | 0.393 ms | 38.54 ms | 94,207.820 |

Zero validation errors, no non-finite values, D = 7.0–9.2 cells, reach 4 used /
3 required. `minimal-power-dam-break-32` likewise runs 512/512 tiles live.

## What changed

### 1. A third tile class, dilated by a measured reach
- `lib/methods/uniform/uniform-volume.wgsl.ts:486` — `uvTwoLevelSeed` reads each
  cell's velocity and `atomicMax`es the domain maximum backward displacement in
  cells into counter word 2.
- `:516` `uvTwoLevelTransportReach` reads that word back (one dispatch later,
  so the barrier is free) and returns `ceil((ceil(D)+1)/4) + margin`, clamped to
  [0, 16]. `params.twoLevel.w` carries the margin **biased by eight** so that a
  negative value can still mean "experiment off" (`:507` comments the bias).
- `:529` `uvTwoLevelDilate` carries a third radius in a third bit (4 = TRANSPORT)
  through the same separated three-axis scan; the scan range is now
  `max(shell, transport)` and every bit gets an explicit range test, so the FINE
  and SHELL bits are unchanged whenever transport ≤ shell — and the reach is
  forced to 0 when the experiment is off, which makes that unconditional.
- `:524` the seed plane reads back as 7 rather than 3 so bit 4 propagates.
- `:554` `uvTwoLevelDilateZ` counts live tiles into counter word 1.
- `uniform-volume.wgsl.ts:20-29` — `UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS` 1 → 3
  (shell count, transport count, displacement), all cleared at the head of the
  step.

### 2. Twelve transport passes early-exit on the class, whole workgroup
- `uniform-volume.wgsl.ts:176-185` — `uvTransportTiles` / `uvTransportTileAt` /
  `uvTransportSkip`. A 4×4×4 workgroup **is** one 4h tile, so the test is
  uniform across the workgroup.
- Early exits at `:188` (`uvBuildEdges`), `:200` (`uvSumDonors`), `:206`
  (`uvFallback`), `:212` (`uvNormalizeRows`), `:220` (`uvNormalizeDonors`),
  `:241` / `:252` (`uvBalanceLiquidRows` / `uvBalanceLiquidDonors`, off by
  default but restricted for when they are on).
- `:294` `uvGather` stores `V = 0, γ = 0` outside the set and returns before the
  nine-term gather and before `uvTarget`. The stores stay because `volumeOut`
  and `gammaOut` are ping-pong targets whose previous contents are two steps
  old — this is the one place the restriction cannot be a pure skip.
- **Level 2 (a compacted tile list + indirect dispatch) was measured to be
  unnecessary and is not built.** Free fall runs 10.8% live tiles and the
  coupling seam falls to 15.3% of dense — a residual floor of ~1.0 ms out of
  24 ms, for 12 × 32,768 launched-and-exited workgroups plus 4 full clears. An
  indirect list costs ~15–25 µs of setup per dispatch against ~3–6 µs direct,
  so at twelve dispatches it would spend a meaningful part of what it saves.
  The seam to build it later is in "Where the coarse arm plugs in" below.

### 3. Host gate, params and diagnostics
- `webgpu-uniform-reference.ts:1377` `transportTilesEnabled` — needs the class
  map (`twoLevelEnabled`) **and** `volumeDustThreshold > 0`.
- `:1062` `writeParams` publishes `twoLevel.w = margin + 8`, or −1 when off.
  That −1 is the whole gate: every tile test folds to "never skip" and the dense
  arm executes the identical instruction stream it did before E3.
- `:1556` the encoded latch; `:1899-1906` the readback (three counter words now
  copied into `statsReadback[196..208)`, which was already allocated).
- `lib/core/webgpu-eulerian.ts:433-439` — six new `GPUEulerianInfo` fields.

### 4. UI (all live, no rebuild)
- `uniform-volume-method.ts:52` `transportWorkMap` (`tiles` | `dense`, default
  `tiles`, runtime) and `:55` `transportReach` (label **Transport margin**,
  default 1 tile, 0–8, runtime).
- `uniform-volume-pipeline.ts:131-150` the two controls plus `Live tiles` and
  `Reach` readouts, attached to the **Conservative volume transport** stage
  beside the existing Dust floor control and Dust discarded readout.
- `:151-165` the chip: `dust floor 1e-6 · live tiles 24%`, or
  `· transport dense`, `· transport dense · no tile map` (sampler off), and just
  `dense finest lattice` when the dust floor is off (the floor's own chip
  already says it, so the stage does not repeat itself).
- Tests: `tests/uniform-volume-initial.test.ts`, new case "E3 runs the
  conservative transport on a live tile set the panel prices" (9/9 pass).

### 5. Verification harness (checked in beside the report)
- `docs/research/uniform-geometric-tile-first-2026-09-19/e3-oracle-fig7.mts` —
  the four-arm conservation oracle; raw output `e3-oracle-fig7.json`.
- `probe-fig7.mts` gained named arms (`FLUID_PROBE_VARIANTS=dense,tiles,dense`)
  and publishes the live-set counters; raw output `e3-probe-fig7.json`,
  `e3-probe-dam64.json`.

## Acceptance

- **No validation errors** in any of the five Dawn runs (`validationErrors: []`
  in every JSON). **No non-finite values**: the oracle counts them per sampled
  frame per arm; zero across 16 field captures of 2.1 M cells.
- **CPU tests green**: `tests/uniform-volume-initial.test.ts` 9/9.
- **`npx tsc --noEmit -p .`** — 15 errors, all pre-existing, none in a file this
  work touched (sparse-cm12 test files and `tools/` probes; the count is
  unchanged from before the change).
- **Existing Dawn tests** (`tests/uniform-volume-dawn.test.ts`,
  `tests/uniform-volume-tile-work-dawn.test.ts`, the three
  `tests/geometric-volume-*-dawn.test.ts`): 12 pass / 3 fail, **both failures
  pre-existing**:
  - `geometric-volume-phi-feedback-dawn` halts inside
    `webgpu-adaptive-mass-solver.ts` (`addWholeFrameUncoveredDonorFallbacks`) —
    a different method entirely; E3 touches nothing on that path.
  - `uniform-volume-dawn` → "mini32 retains its liquid region through far-wall
    impact" asserts `|representedVolumeDrift| < 0.05` at frame 30 and `< 0.10`
    at frame 90. Measured under BOTH arms, twice each:
    dense −0.0608 / −0.1186 and −0.0584 / −0.1156; tiles −0.0555 / −0.1133 and
    −0.0524 / −0.1139. **The dense arm — which is HEAD's instruction stream —
    fails it worse**, at frame 30 as well as frame 90, and E3 moves the number
    in the improving direction by about the same amount the dense arm differs
    from itself (0.003). On mini32 the live set is 512/512 tiles, i.e. E3 does
    no skipping at all there, so the difference is float-CAS ordering. The
    failing metric is the *phi*-derived volume; raw V drift is 2e-5 on every
    arm. **No bound was weakened.**
- **Live toggling** exercised on figure 7: dense 1–20 → tiles 21–35 → dense
  36–45 → tiles with margin 0, 46–55. V stays within 33463.76–33464.40 across
  every flip with no discontinuity at a flip frame, and the reach tracks D per
  step (at margin 0, used == required exactly). No validation errors.

## Caveats and what is not verified

1. **The dense arm is numerically identical but not free.** The classify now
   reads one velocity per fine cell and does one `atomicMax` per tile whether
   or not the live set is used. The class-map values are unchanged (the seed
   decision does not consult velocity, and the dilation range is unchanged when
   the reach is forced to 0), and the measured `Sec. 3.3 interface authority`
   and extension seams did not move — but the classify's own cost was not
   isolated. It is inside the extension seam, which reads 1.376 ms on all three
   arms.
2. **The four donor-sum clears are still full-lattice** — 4 × 8 MB per step at
   128³. They must be, because a receiver at the edge of the live set
   backtraces onto donors *outside* it and `uvNormalizeDonors` reads those
   columns. Making them tile-ranged needs a tile-major edge arena, which is the
   follow-on noted below. Their measured share was not isolated; the 32 MB of
   writes is ~0.16 ms at M-series bandwidth, well inside the ~1.0 ms residual
   floor.
3. **The edge arena stays dense-indexed** (168 MB at 128³,
   `UNIFORM_VOLUME_EDGE_BYTES = 80`). E3 shrinks the work, not the storage.
4. **A negative `Transport margin` is reachable from the solver options but not
   from the panel** (`webgpu-uniform-reference.ts:505` clamps to [−8, 8]; the
   param spec's min is 0). It exists so a verification run can starve the set
   below its own predicate; it is the `starved` arm above. If that offends, the
   clamp is one line.
5. **The 16-tile cap on the reach is untested.** It needs D > 63 cells, which
   figure 7 never reaches (max 27.2). A capped step would read `SHORT` in the
   panel; nothing was run that produces one.
6. **No scene with an interior solid, terrain, a rigid body, an inflow or a
   drop was run with the live set on.** The seed covers all of them by
   construction (`uvOpen < 1`, `dropSource`, `inflowSweptPlugSource` — the same
   seed E2 relies on), and sources are added inside `uvGather` in cells that are
   seeds by definition, but none was measured. Uniform Geometric forces
   `solidExcessCorrection` off, so the one other V writer is inert.
7. **`liquidCapacityBalancing` is restricted but not measured.** It is off by
   default; both its passes take the same early exit, and its own clears are
   full-lattice for the same reason as (2).
8. **Frames 40 and 60 measured nothing.** Figure 7's own float-CAS
   reproducibility at those frames is of the same order as the tiles/dense
   difference. The claim "tiles tracks dense" rests on frames 10 and 25, where
   it is at or 100× below the floor, plus the identical cell counts and extents
   at 40 and 60.
9. **The dust floor's "Dust discarded" readout will differ slightly** between
   arms, because cells outside the live set never call `uvDustFloor`. Under the
   predicate their gather sum is exactly 0.0, which `uvDustFloor` returns
   without counting, so the counts should match — but this was not checked
   against a capture.
10. **Turning the dust floor on mid-run, then the live set on in the same
    frame**, would let the live set zero whatever sub-threshold residue the
    previous (floor-off) step left outside the seed. Bounded by the threshold
    per cell, and the panel gates the control on the floor being on, but the
    ordering is not enforced.

## Where the coarse arm plugs in

The follow-on the brief asked to leave a seam for — *coarse mass + fine shape*,
`Transport authority: fine | coarse` — attaches at exactly three points, none
of which E3 has closed off.

The live set is already a per-tile predicate evaluated once per step and read
by twelve kernels through one helper (`uvTransportSkip`). A coarse arm needs a
*second* class inside it: "this tile transports at 4h". Bit 8 in the same word,
dilated by the same separated scan (the scan already carries three independent
radii, so a fourth is a line), is the whole classification. `uvTransportSkip`
then becomes a three-way `uvTransportAuthority(gid) -> skip | fine | coarse`,
and each kernel picks a branch instead of returning. Because the branch is
uniform across a workgroup — a workgroup is a tile — a coarse kernel can use
the same dispatch and the same bind group, writing one coarse row per tile into
the same arena at a tile-major offset.

The two things that would want to change first are the ones E3 deliberately
left: (i) the edge arena is dense-indexed, so a coarse row would waste 63 of
every 64 records — a tile-major compacted arena, indexed by a live-tile list,
is the prerequisite for both the coarse arm *and* for ranging the four
donor-sum clears; and (ii) the compacted tile list itself, which
`uvClassifySharpenTiles` already demonstrates (one pass, one atomic counter,
count at word 2N+7, map from 2N+8) and which the coarse arm would want anyway
so it can dispatch one workgroup per coarse tile rather than one per fine tile.
Neither is needed for E3's win, which is why neither was built.

## What to look at in the app

`cm12-figure-7`, Uniform Geometric, SIM tab, **Conservative volume transport**:

1. The chip should read `dust floor 1e-6 · live tiles N%`. Watch N: ~11% while
   the ball falls, ~19% through impact, ~24% as it spreads. The stage's cost
   should be a third to a sixth of what it is with `Transport work: Dense`.
2. Flip `Transport work` between `Live tiles` and `Dense` while it runs. The
   water must not jump, and the stage cost must move by 3–6×. Flip it back.
3. Read the `Reach` readout during impact: it should climb from `2 used ·
   1 required` in free fall to `4 used · 3 required` or higher, with the cell
   figure in brackets tracking the flow. It should never say `SHORT`.
4. Drag `Transport margin` to 0 (the exact predicate) and back to 1. Live tiles
   should drop a few percent; the water should not change. Negative margins are
   not reachable, which is deliberate — a short set stalls the front.
5. Set `Volume dust floor` to 0. The chip must fall back to
   `dense finest lattice` and the `Transport work` control must grey out: the
   live set is only lossless while every cell outside it holds V = 0, and that
   is what the floor guarantees. `minimal-power-dam-break-64` is the "no win"
   control — 99.6% of tiles are live and the cost should be unchanged.
