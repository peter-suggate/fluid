# Uniform Geometric: keeping small splashes alive

Status: TOGGLES IMPLEMENTED, 2026-09-23. Approaches A–F exist as live runtime toggles, inert when off. After
comparing them in the app, Peter made **cubic advection, drain ghost phi and airborne momentum the defaults**; the
rest stay off. §7 lists them with their panel locations and first single-run Dawn measurements. The splash-survival Dawn lane
(§5) is not built yet. Sections 1–6 are the original plan. They come from the literature in `docs/papers/`, earlier
in-repo research, and the `uniform-volume` shaders. Each claim is marked **code** (read from the source), **derived**
(worked out analytically) or **measured** (with its ledger).

Out of scope: pure-density methods. The volume + level-set design stays: V (volume) decides how much liquid there is,
phi (the level set) decides where it is (`docs/uniform-geometric-phi-volume-agreement-handoff.md`, Principles).

---

## 1. What kills a splash

| # | Mechanism | Where | Status | What it predicts |
|---|---|---|---|---|
| M1 | **Representation floor.** A cell owns a pressure row and is rendered only if its centre phi, the mean of its 8 corner values, is negative. For a two-sided sheet of thickness t, the centre phi is (h − t)/2 whatever its alignment, so any sheet with t < h has no liquid centre anywhere. A droplet needs radius r > 0.87h when centred in a cell and r > 1.12h when centred on a vertex, i.e. about 2.7–5.9 cell volumes. | `uvPublish`, `pressurePhi` | derived | Anything smaller has no row, no render and no velocity of its own. A droplet close to the limit blinks as it moves. |
| M2 | **Curvature ratchet.** The signed distance of a convex body is a convex function, so trilinear interpolation over-estimates it. The zero set therefore moves inward by up to h²/(4r) on every resample. There are two resamples per step: `uvAdvectPhi` for every moving vertex, and `uvRedistancePhi`. The redistance runs on every band vertex every step, even at rest: it writes exact distances to the already-shrunk trilinear contour, which the next interpolation shrinks again. A flat surface (κ = 0) is unaffected. | `uniform-volume.wgsl.ts:434, 448` | derived | Droplets and sheet rims erode at a rate set by their curvature. The bound is ~19 % of volume per step at r = 2h, ~5 % at 4h and ~1 % at 8h; the mean is about 2/3 of the bound. Pools don't erode, which is why only splashes look dissipative. Existing ledgers are bulk only: on the thin film, advection −290 against redistance −41 (half-cell-cliff memo). No droplet ledger exists. |
| M3 | **Relay drain.** Cells within 2.1h of phi's surface are admitted to sharpening. A phi > 0 cell with no phi fill becomes a *relay*, and flux runs only down phi's gradient. So once phi has lost a droplet, the droplet's V is poured into the nearest phi body. This is the mass transfer CM12 Fig. 3/4 criticises in MMTD07, except that here the gradient is phi's rather than rho's. | `uvPrepareSharpen`/`uvProposeSharpen` (`:672-760`) | code | A droplet just above the pool vanishes into the pool once its phi dies. |
| M4 | **Global surface-volume shift** (`totalSurfaceVolume`, default on). One scalar normal shift is solved so that phi's total fill equals ΣV. Every body gets the same δ, which costs a feature of radius r a volume fraction of about 3δ/r. When phi encloses more than V the shift is inward: WP0 of the agreement handoff measured +7.7 % from entrained voids. | `uniform-surface-volume.wgsl.ts:199-216` | code | Volume a droplet loses to M2 reappears on the pool. When the shift is inward, droplets erode fastest. |
| M5 | **Stranded V.** Gravity acts only where `surfaceOccupancy` is non-zero, i.e. phi < 2h. `project` zeroes faces between rowless cells, and the extension then overwrites them with the nearest phi-liquid's velocity. | `webgpu-uniform-reference.wgsl.ts:346-351, 1141-1150`; handoff "Dissipation" | code + measured (films) | V beyond the relay band hangs in the air, invisible, moving with the pool's surface velocity. |
| M6 | **Dust floor.** `uvDustFloor` deletes \|V\| < 1e-3 on every write. CM12 only zeroes rho < 1e-5, and hands that mass back (Eq. 17 / Alg. 2). | `:560`, param `volumeDustThreshold` | code; never measured on a splash | Stranded V that spreads under conservative transport is eventually deleted. |

**The chain as read from the code.**
1. M2 erodes the droplet's phi.
2. Below M1 the droplet drops out of the pressure rows and the render.
3. Its V is either relayed into the pool (M3) or stranded in the air (M5), and stranded V is slowly lost to the dust floor (M6).
4. The global shift (M4) puts the lost phi volume back onto the pool.

On screen, the splash shrinks, blinks out, and the pool rises very slightly.

---

## 2. WP0 — measure before building

### 2a. Synthetic droplet lane (new Dawn probe, later promoted to a lane)

**Scene**
- A 48³ box, not yet in `scenes.ts`.
- Droplets of r ∈ {0.75, 1, 1.5, 2, 3, 4}h, placed at least 12 cells apart at random sub-cell offsets, so that each one sits inside its own 8³ window.

**Motion arms**
1. **Rest:** zero gravity.
2. **Free fall:** gravity on, dropped from rest. Every step then has a non-integer displacement.
3. **Uniform translation**, optional: this needs `initialVelocity_m_s`, which its doc comment says only Sparse CM12 reads (`lib/core/model.ts:192`); check whether uniform honours it before relying on it.

**Measured per droplet per frame**
- ΣV in its window.
- phi fill, using the exact tetrahedral `fill()` from `uniform-surface-volume.wgsl.ts`.
- Liquid centres.
- Published cells ≥ 0.5.
- V centroid against the analytic trajectory.
- KE.

**Solver arms**
- HEAD.
- `redistance: off`.
- `totalSurfaceVolume: off`.
- Both off.

**Decisive predictions**
- **M2 at rest:** a resting droplet with r = 2h, redistance on, loses phi volume every step; with redistance off it is bit-still. One run settles whether the ratchet is real.
- **M1:** the smallest droplets have no liquid centre from frame 0.

### 2b. Splash ledger on production scenes

**Scenes**
- `cm12-figure-6` (crown splash; the paper needed Sec. 3.8 for it)
- `cm12-figure-5` (ball into a pool)
- `water-box-dam-break` (wall splash)
- `hero-garden-hose`

**Components.** Every 5 frames, read back V, vertex phi and published values. Label 26-connected components of cells with V above the dust floor or phi_c < 0. The largest component is the main body; everything else is "airborne".

**Measured for airborne components**
- ΣV, split into:
  - owned (phi_c < 0);
  - relay band (0 ≤ phi_c < 2.1h);
  - stranded (phi_c ≥ 2.1h).
- Rendered volume: published ≥ 0.5.
- KE.
- Lifetime: birth, when phi dies, when V merges into the main body, and whether that merge happened without any contact.

**Measured globally**
- Dust mass, from diagnostic words 5 and 6.
- The per-step global shift `state[0].x`, **with its sign**.

**Arms:** HEAD, `totalSurfaceVolume: off`, `redistance: off`.

**Output.** A budget of where airborne mass ends up: still owned, stranded, relayed into the pool, or dusted. This budget ranks the approaches below. Reuse the scaffolding of `docs/research/uniform-geometric-thin-film-2026-09-19/probe-surface-noise.mts` and `probe-energy.mts` (WGSL patch hook, `--set=key:value`, arms).

---

## 3. Approaches

Every change ships as a live toggle, default off, with the off value bit-identical to today. Peter judges each one in
the app.

### A. Redistance that does not move the surface (CM11b §3.4) — targets M2

The tall-cell paper, same authors and same dt = 1/30, stabilises exactly this step (`docs/papers/tallCells.txt:260-272`):

> do not modify φ values of grid points next to the surface in order to avoid moving it … clamp the value of φ next to the liquid surface to not exceed the grid spacing ∆x

It also runs the step only every ten frames.

- **Change** (`uvRedistancePhi`):
  - A vertex with an opposite-sign lattice neighbour keeps its advected value, clamped to |phi| ≤ h.
  - Every other band vertex is redistanced to the contour those frozen vertices define.
  - A later arm adds redistance every N steps.
- **Cost:** at most today's. Fewer Newton solves; the clamp is one `min`.
- **Risk:** |∇phi| drift at frozen vertices. CM11b's clamp answers that, and it reports "no significant problems or artifacts". Interaction with the global shift's gradient-scaled band (`metric`), and with contact vertices on walls.
- **Gates:**
  - A resting droplet with r ≥ 1.5h keeps its phi volume over 90 frames (it shrinks at HEAD, if M2 is right).
  - Free-fall droplet phi half-life at least 2× HEAD.
  - Dam break: roughness ≤ 0.026 and jitter within baseline noise.
  - Resting pool bit-still.
  - `uniform-volume-dawn`, `uniform-surface-volume-dawn` and `uniform-buried-phi-dawn` show no new failures.

### B. Keep a lost droplet's V where it is (sharpening) — targets M3, M6; prerequisite for D and E

This is CM12's own argument against MMTD07. Mass returned by sharpening should go to *its own* 0.5 contour, following
its own density gradient within D·dx (Algorithm 2, D = 2.1). It should not go to whichever surface is nearest
(`massConservingLiquids.txt:300-345, 386`).

- **B1 — no relay across empty air.** A relay cell accepts flux only if it already holds V above ε, or is face-adjacent to a phi-liquid cell. The bulk's skirt is contiguous, so it is unaffected. A detached droplet can no longer cross an empty gap into another body.
- **B2 — orphan V compacts in place.** Orphan V is V with no phi surface inside the admission band; it is not admitted today.
  - Donors: cells below ½.
  - Receivers: the face neighbour with larger V, where larger V is read from **start-of-stage V**, a fixed potential just as phi is fixed through the eight sweeps today. Receivers are capped at capacity.
  - This is Eq. 17 ("mass only moves from the air side to the liquid side") and Algorithm 2's trace up ∇ρ, reduced to face-local fluxes. It uses the existing propose/limit/commit kernels, so it needs no atomics.
  - The eight sweeps give a reach of 8 cells, which is at least D.
- **Cost:** only tiles holding orphan V, admitted by the existing classify pass. Zero on pools.
- **Gates** (splash ledger):
  - Airborne V merged into the main body without contact goes down.
  - The fraction of orphan V sitting in cells ≥ ½ goes up.
  - Dust mass goes down.
  - ΣV is exact.
  - Pools are bit-identical, since they have no orphan tiles.
  - The work-map bit-identity lane (`tests/uniform-volume-tile-work-dawn.test.ts`) stays green.

### C. Show V where phi offers none (CM12 §3.8, restricted) — the visual lever below M1

CM12 needed this for the crown splash, Fig. 6 (`massConservingLiquids.txt:452`). The pipelines exist, but they are
forced off for Geometric (`uniform-geometric-options.ts:37`) and they are dense. This is not renderer masking: it shows
real solver state (V) where phi has none.

- **Change** (in `uvPublish`), only where no phi-liquid centre is within 2 cells:
  - value = max(value, rho'').
  - rho'' = V / min(max(γ̄, θ), 1), where γ̄ is the mean of 2·min(V, ½) over 3³ cells, and θ = 0.01 as in CM12.
  - Add a cluster-mass cut (ΣV over 3³ at least ~¼ cell) so dust does not sparkle.
  - Alternative arm: splat an equal-volume sphere, r = (3ΣV/4π)^{1/3}, which is volume-consistent by construction.
- **Cost:** fused into publish, with a 27-tap gather on orphan tiles only.
- **Known artifacts** (CM12 and the reference doc): specks are amplified; a uniform film under ½ maps to exactly 0.5 and flickers; visible volume exceeds mass (CM12 lists "thinning" as future work). B's compaction removes most of the first two.
- **Gates:**
  - The published field is bit-identical on pool and dam scenes wherever phi-liquid is within 2 cells.
  - Visible airborne volume over airborne ΣV stays in [0.7, 1.5].
  - Visible component births and deaths per second are no worse than phi-only.
  - Peter's eyes on fig 6, fig 5 and the hose.

### D. Local volume for small isolated bodies — targets M2 and M4 on droplets

This is the one-shift-per-body constraint ("arm D"), without component labelling.

- **Change:**
  - For each band vertex, take the 8³ box W around it. If W's outer shell holds no V above dust and no phi-liquid centre, every body the vertex touches lies wholly inside W.
  - Shift phi by (ΣV − Σfill)/A over W, where A is the cut area. Use full gain, clamped to ±0.25 cell per step.
  - All other vertices keep today's global shift. The global solve excludes vertices that took a local shift.
- **Why it cannot pump:**
  - A body is moved only by its own residual (handoff Principle 4).
  - A pool never qualifies, so pools stay bit-identical.
- **Build order:** it needs B first. As WP1 showed, a residual means nothing while V is spread out.
- **Production form:** the separable box sums WP2 already planned, of V, fill, cut area and shell occupancy, over band tiles only.
- **Cost:** about six separable passes over band tiles, plus an apply fused into the phi advect. Measure it.
- **Gates:**
  - Droplet phi volume over ΣV stays ≥ 0.95 for 3 s, for r ≥ 1.5h.
  - No pumping: port `tests/geometric-volume-phi-feedback-dawn.test.ts` (a drop plus a disconnected resting pool) to uniform, with the pool height within 2 cm.
  - Pools bit-identical.
  - Airborne owned-V rises on fig 6.

### E. A seed that yields a liquid centre, and the missing drain — targets M1 flicker

- **Why today's seed cannot help:** it writes h(½ − V̄) with V̄ the mean of the vertex's 8 cells. That is negative only when V̄ > ½, and it fires only above ¼ (`uvSeedPhi`). A one-layer film of fill f gives V̄ = f/2 and a one-cell drop gives V̄ = ⅛, so the seed never creates liquid where phi could not already hold it. That is the ~1100 rows lost per second of flicker in the handoff.
- **Change:**
  - After B, orphan V has cores ≥ ½. Seed with the minimum over incident cells of h(½ − V_c/open), keeping the existing "no phi-liquid centre nearby" guard.
  - Add the drain: a phi-liquid region whose V̄ ≈ 0 for N steps becomes `max(phi, ½h)`. It is the first suspect for the unexplained over-growth at clamp 0.01.
- **Gates:**
  - Rows lost and seeded per second fall below `phiSeedFromVolume` today.
  - corner-brick-drop is at least as good as the all-three result over 10 s.
  - Seed firings beside healthy surfaces on the dam break are ≈ 0; count them.
  - Airborne liquid centres rise on fig 6.

### F. Airborne V keeps its own motion (restricted keepab) — targets M5

`keepab` proved velocity is the lever: film KE rose 4–5×. It tore films because ballistic faces on a *contact* film are
not divergence-free. Free flight is the physically correct model for liquid in the air.

- **Change:** select faces that have V above dust on a side, no phi-liquid centre within 2 cells, and no solid or terrain within 2 cells. On those faces:
  - keep the advected velocity plus gravity, dropping the 2h gravity gate for them;
  - skip `project`'s zeroing and the extension overwrite.
- **Gates:**
  - Centroid error of free-fall droplets against the parabola.
  - corner-brick-drop and dam-break liquid-cell counts unchanged.
  - fig 6 splash apex height.
- **Risk:** fast airborne V widens transport and extension reach, the fig7-256 root cause, so it needs its own timing gate.

### Quick checks inside WP0

- `totalSurfaceVolume: off` on the splash scenes. If M4 dominates, D moves ahead of A.
- `volumeDustThreshold` 1e-5 on one splash arm. If dust dominates, deposit it into the nearest admitted cell (MMTD07's postponed mass) instead of deleting it.

---

## 4. Not recommended now

| Idea | Why not |
|---|---|
| Particle level set (ENGF03) | CM12 Fig. 7: with 64 particles per cell at dt = 1/30, "most of the liquid disappears". |
| MacCormack / BFECC for phi | CM11b: "MacCormack causes noisier surfaces even if care is taken near the interface". It also widens the trace reach, which eats the tile-map margin (`tile-first-design.md`, "Where this is likely to fail" #4). |
| Per-cell CLSVOF / `volumePressureRows: all` | Every surface bubbled: roughness 0.021 → 0.059. |
| Stronger or longer sharpening | Saturated: 8 → 512 sweeps gave identical results; strength 2 gave no gain. |
| Fine 2× phi band (Aanjaneya 2017) | Effective, but 8× the band cost. |
| CM11b particle thickening for sheets | The only large-dt sheet result in the papers, but it brings Lagrangian state and a scatter-min. Revisit if sheets are still the gap after A–F. |
| Any global V→phi correction | Pumps disconnected pools (`correctWholeFrameVolumePhi`, removed in 54806ba6). |

---

## 5. Assessment protocol

**Survival metrics** (WP0 ledgers)
- Droplet lane: phi-volume half-life per r/h; V retention in the window; trajectory error.
- Splash ledger: owned, stranded, relayed and dusted fractions of airborne ΣV at fixed t; visible airborne volume over airborne ΣV; airborne KE.

**Non-regression**
- Dam break: `probe-surface-noise.mts` roughness ≤ 0.026 and band jitter within baseline noise; ±25 % is noise on a single run.
- `hydrostatic-power-large-offset` bit-still.
- The two-body no-pumping port.
- corner-brick-drop alive at 10 s.
- Lanes `test:dawn:uniform-volume`, `:uniform-garden`, `:uniform-geometric-boundaries`, `uniform-surface-volume-dawn` and `uniform-volume-tile-work-dawn`. Diff the failing names against the known reds: uniform-inflow-window, mini32 far-wall drift 0.0272 and volume-levelset-overlay.

**Performance** (must not clearly degrade)
- Tools and scenes:
  - `tools/profile-uniform-geometric-dawn.ts` on `cm12-figure-7-256`: stage means, impact window.
  - `tools/benchmark-uniform-long-dam-paging-dawn.ts` (HEAD 26.3 ms).
  - One run per arm, serially.
- Acceptance:
  - Each new stage's cost scales with splash or orphan tiles, and is zero on a resting pool.
  - Frame medians stay within run-to-run noise (about ±1–2 %).
- Before trusting any timing, check that the maxSpeed trajectory matches the control arm.

**Visual.** Live toggle per approach, default off. Peter's eyes are the gate: a Dawn metric is necessary, not
sufficient.

**New Dawn lane.** `tests/uniform-splash-survival-dawn.test.ts`, built from the droplet lane. Each approach's claim
must first go red on HEAD. No non-Dawn tests.

---

## 6. Order and decision points

1. **WP0.** One run per arm per scene. It decides everything below.
2. **A and C in parallel.**
   - A is cheap, removes work, and is the direct test of M2.
   - C is the visual lever with no simulation risk. Its look improves once B lands.
3. **B.** It keeps orphan V local and compact, which D and E need.
4. **E, then D.** E makes resurrected droplets own rows; D holds them at their own volume.
5. **F.** Momentum, last, because it widens tile reach.

**Re-order rules**
- If WP0 shows airborne V is mostly relayed into the pool, B1 goes first.
- If it is mostly dusted, fix the dust floor first.
- If M2 at rest is negligible, drop A.
- If the global shift is mostly inward during splashes, D moves up.

---

## 7. Implemented toggles and first measurements (2026-09-23)

Every toggle is `update: "runtime"`. Flipping one never rebuilds the solver or resets to t = 0, so arms can be compared
on the same running scene. They are Uniform Geometric only: `UNIFORM_GEOMETRIC_SPLASH_KEYS` keeps them out of the
Rust/2D lab parameters, and the paper `uniform` method does not carry them. Each has its own GPU lane, and every new
branch is guarded by it, so a stage that is off costs nothing. Since 2026-09-23 **cubic advection, drain ghost phi
and airborne momentum default on**, in the param specs and in the solver constructor (`!== false`). The long-form
tooltips live once in `UNIFORM_GEOMETRIC_SPLASH_HINTS`, shared by the param spec and its SIM panel switch.

**Consequence for Uniform 2D alignment.** `tools/wasm/uniform-geometric-parity.ts` runs the WebGPU dim-2 oracle on
the full 3D defaults, while Rust receives only the native keys. The oracle therefore now includes three stages Rust 2D
does not implement. Either port them to `rust/crates/fluid-core/src/uniform_geometric/`, or pin them off in the
oracle.

### Where they are (SIM pipeline panel, "Level set + volume" band)

| Approach | Stage → control | Param key | Values (default first) | Params lane |
|---|---|---|---|---|
| A | Vertex level set → Redistance surface | `redistanceSurface` | rebuild · preserve · sparse ("Every 10th") | `splash.x`; sparse is host-side (`encodedSteps % 10`) |
| — | Vertex level set → Cubic advection | `phiCubicAdvection` | **on** · off | `splashB.x` |
| D | Vertex level set → Isolated volume | `isolatedBodyVolume` | off · on | `splash.w`; also disables the Follow-V tent shift |
| E | Vertex level set → Seed from V cells | `phiSeedCells` | off · on | `splashB.y` |
| E | Vertex level set → Drain ghost phi | `phiDrain` | **on** · off | `splashB.z` |
| B | Volume sharpening → Orphan V | `orphanVolume` | relay · local · compact | `splash.y` (1 local, 2 compact) |
| C | Phi surface publication → Show orphan V | `orphanVolumeRender` | off · density · spheres | `splash.z` (1, 2); presentation only |
| F | Velocity extension → Airborne momentum | `airborneMomentum` | **on** · off | `splashB.w` |

"Cubic advection" was not in §3. It came from reading M2 again: clamped Catmull-Rom, applied only where |phi| < 2h. It
attacks the same inward bias of the trilinear resample, but in the advect step instead of the redistance.

### What each one does (code)

- **A preserve.** In `uvRedistancePhi`, any vertex that has a 26-neighbour of opposite sign keeps its advected value,
  clamped to ±1 cell. Newton redistance runs only off the surface. **sparse** adds redistancing only every tenth step
  (CM11b §3.4); on the other steps the redistance pass is a copy.
- **Cubic.** `uvPhiCubic` samples 4³ taps and clamps the result to the range of the 8 enclosing vertices, so it cannot
  overshoot.
- **B local.** An air cell receives sharpening flux only if it already holds V or is within one cell of phi's surface.
  **compact** is local plus one change: V more than `tuning.y`·h from phi flows up its own V gradient, from the
  lower-V to the higher-V neighbour, and full cells stop receiving. Classify admits tiles that hold such V.
- **C.** Applies only to cells whose centre phi exceeds 1.5h, i.e. orphan cells. It clusters their V over a 3³
  window, ignoring phi-owned cells.
  - density: publishes rho'' = V/gamma, gated at rho'' ≥ 0.6.
  - spheres: draws a sphere of the cluster's own volume, gated by the cluster's second moment so a smear is not drawn
    as a ball.
- **D.** The vertex's 16³ window must have a one-cell shell with no V and no gamma, so its 14³ interior holds a whole
  body. If it does, phi shifts by h·clamp((ΣV − Σgamma)/A, ±0.25), where A counts partial-gamma cells. Otherwise the
  shift is zero, so nothing near the main pool changes. Bodies wider than about 12 cells never qualify.
- **E seed.** At the departure point, it seeds phi = min h(0.5 − V/open) over the 8 incident open cells, but only
  where no vertex in the surrounding 4³ is already liquid.
  **Drain.** It raises a phi-liquid vertex to at least 0.5h when no cell in its 4³ holds V above 0.05.
- **F.** `uvAirborneCell`: V above `UV_LIQUID_EVIDENCE` (0.05, the drain's threshold), centre phi above 1.5h, and a
  ±2-cell box that is in the domain and fully open.
  - The floor was the dust floor until the planar hydrostatic lane caught a resting pool's 1e-5 V tail at 2.5h
    free-falling at 0.65 m/s.
  - The tail over that pool is about 5e-3 at 1.5h, so 0.05 has a tenfold margin.
  - Such cells count as extension sources (authority raised just above the 0.5 isovalue).
  - They keep gravity.
  - `project` keeps their predicted face velocity instead of zeroing it.
- **Far-air skip.** The lean phi advect's early exit for far-air vertices is bypassed when any of cubic, D, E-seed or
  E-drain is on and the departure point is in a shell tile. Without the bypass, those arms would never see the
  airborne vertices they target.

### First measurements (one Dawn run per arm, Metal, balanced)

These were measured with every toggle off except the arm under test. "default" means the pre-2026-09-23 defaults.

Probe: a scratch script, not a lane. Ledger definitions:
- **inside phi / band / orphan:** V in cells whose centre phi is < 0, in [0, 2.1h), or ≥ 2.1h.
- **phiFill:** Σ clamp(0.5 − centre phi / h, 0, 1).

**Resting droplets.**
- Setup: zero gravity; r = 1, 1.5, 2, 3, 4 cells, each alone in its own ±10-cell box; 90 frames.
- Result, **M2 confirmed:** V is exactly constant in every arm, so the loss is in phi.

| Arm | r = 1 | r = 2 phiFill (f30 → f90) | r = 3 | r = 4 |
|---|---|---|---|---|
| default | vanishes | 26.7 → 14.1 | 102 → 70 | 254 → 196 |
| A preserve / sparse | kept | 29.25 constant (bit-still) | constant | constant |
| cubic, compact, drain | as default | as default | as default | as default |
| D isolated | — | 33.5 / 34.4, tracks V | 112 / 113.75 | 250.9 / 268.9 (window too small) |
| all | — | phi ≈ V | phi ≈ V | — |

**cm12-figure-6 crown splash, frame 90.**
- Reference V: 94276.
- Frame medians are from frames 6–90.

| Arm | V inside phi | band | orphan | ΣV drift | frame median |
|---|---|---|---|---|---|
| default | 80425 | 7303 | 6061 | −487 | 48.4 ms |
| cubic | 84409 | 4794 | 4675 | — | within noise |
| B compact | 81125 | 8197 | 4757 | — | within noise |
| F airborne, dust floor (first cut) | **92003** | 924 | 1165 | — | 51.4 ms (+6 %, one run) |
| F airborne, 0.05 floor (shipped) | **90782** | 1191 | 2123 | — | 54.2 ms (one run, no paired default) |
| all | 90620 | 1375 | 2198 | **−83** | within noise |

Other fig 6 results:
- **A preserve** is close to default on the dynamic crown.
- **D** is bit-identical to default, because fig 6 has no isolated bodies inside a 14³ window.
- **C: gates are required.** Without them, density added about 32k wet cells and spheres about 52k: a uniform smear
  maps to 0.5, and every smear cell was "inside".
  - Frame 90 air cells drawn wet: density ≈ 11.9k, spheres ≈ 3.0k.
  - Density still draws part of the crown sheet as haze. Spheres is the cleaner visual.

Across all arms:
- Every arm compiles.
- No arm raises a GPU error or a non-finite value.
- Default is deterministic across runs (frame 90 ΣV 93788.77).
- Every median lies in 47–51.5 ms against 48.4 ms, and resting droplets run at about 40 ms.

### Suggested comparisons in the app

- Crown and sheet (cm12-figure-6, cm12-figure-5): **F airborne + cubic + B compact**. Add **C spheres** to see where
  orphan V still sits.
- Droplets and rain (hero-garden-hose, water-box-dam-break late splash): **A preserve + D isolated**. This pair is
  the only one that stops the resting shrink.
- **All on** conserved best (−83 against −487). It is the candidate default set to check by eye before timing it on
  `cm12-figure-7-256`.

### Not yet done

- The §5 Dawn lane.
- The §5 non-regression and timing matrix.
- The fig7-256 impact-window profile for F, since it widens transport reach.
