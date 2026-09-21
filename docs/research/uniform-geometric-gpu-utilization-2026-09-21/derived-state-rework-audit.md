# Derived-state rework audit — Uniform Geometric (`uniform-volume`)

Static read of HEAD (`codex/uniform-paging-2026-09-21`, 918fc273). No GPU, no edits,
no git operations. Scope: **derived per-cell / per-face / per-vertex quantities that
are recomputed** instead of compiled once when their inputs change. Activity/sparsity
structures (tile classes, page lists, windows) belong to the sibling report and are
referenced only where they gate a recompute.

Every claim marked **V** is read directly from the cited line. **I** is inference,
with the mechanism named. File shorthand: `ref.wgsl` =
`lib/methods/uniform/webgpu-uniform-reference.wgsl.ts`, `uv.wgsl` =
`lib/methods/uniform/uniform-volume.wgsl.ts`, `mg.wgsl` =
`lib/methods/uniform/webgpu-uniform-pressure-multigrid.wgsl.ts`, `host` =
`lib/methods/uniform/webgpu-uniform-reference.ts`.

Live set assumed: geometric shipping defaults per static-core.md §2 — semi-Lagrangian
advection, `twoLevelVelocity` on, `redistance` on, `totalSurfaceVolume` on,
`surfaceDeficitBalancing` on, `volumePressureRows` off, `densityPostProcessing` off,
`gammaDiffusionIterations` 0, no gamma diffusion, `activeRegion` false,
`pressureWindow` false (`uniform-geometric-options.ts:12–38`,
`uniform-geometric-parameters.ts:23,31,52,73,77`). **V**

## Corrections to the sibling reports

1. static-core.md §3 lists `copyField(vertexPhiScratch → vertexPhiField)` as a live
   per-advance copy. It is not live: the host branches
   `if (this.geometricRedistance) runVertex("uvRedistancePhi") else copyField(...)`
   and `redistance` defaults to `"on"` (`uniform-geometric-parameters.ts:23`). **V**
2. static-pressure.md states `divergenceAt` → 6×`pressureFaceData` (48 solid samples).
   That holds only for body-free, terrain-free scenes. `domainFaceSolidVelocity`
   early-outs on `(!checkSolid && !hasTerrain())` (`ref.wgsl:1232–1236`); with terrain
   (garden scenes) it does not, and `divergenceAt` becomes **12×**`pressureFaceData`
   (`ref.wgsl:1237–1251`). **V**
3. static-volume.md prices `uvOpen` at "10 buffer loads" as a typical case. It is the
   *only* case: `packSolidVoxels` writes the `0x53565731` magic unconditionally
   (`host:915–926`), so `staticSolidVoxelOccupied`'s cheap header early-out
   (`ref.wgsl:254`) never fires in any shipping scene. **V**

---

## 1. Inventory of recomputed derived quantities

Costs are storage-buffer word loads on the body-free, terrain-free interior path
(the dam/mini lanes). "+T" marks one extra `r32float` texel per terrain query.
Metal CSE column: **no** = a `read_write` storage binding is loaded and a *buffer*
store occurs in the same kernel, so Tint's un-`restrict`ed `device` pointers force
may-alias and the loads cannot be hoisted (**I**, mechanism; corroborated by the
repo memory `atomic-load-is-an-fp-barrier`). **maybe** = pure load chain with no
intervening buffer store, so hoisting depends only on inline budget.

| # | Quantity | Defined at | Inputs | Inputs change when | Cost / eval | Metal CSE |
|---|---|---|---|---|---|---|
| D1 | `staticSolidVoxelOccupied(p)` | `ref.wgsl:253–261` | `activeScratch` packed bitfield at `params.dropExtent.z` | never during a run; rewritten only on scene edit (`host:947–955`, `host:3128–3141`) | 5 loads + 5 `round()` of a uniform | no |
| D2 | `cellTerrainFraction(p)` / `cellInsideTerrain(p)` | `ref.wgsl:283–284` | `terrainIn` 2D `r32float` | uploaded once (`host:1607`) | 0 (no terrain) or 1 texel | yes (texture) |
| D3 | `cellSolidFraction(p)` | `ref.wgsl:686–691` | D1, `rigidBodies` | body motion | D1 + 12-iteration body loop (0 bodies → 0) | no |
| D4 | **`cellOpenFraction(p)` = V_i** | `ref.wgsl:364–367` | D1, D3, D2 | scene edit / body motion only | **10 loads (+T)** — D1 twice | no |
| D5 | `uvOpen(id)` | `uv.wgsl:296` | `valid` + D4 | as D4 | = D4 | no |
| D6 | `solidVelocityAtWorld(w)` | `ref.wgsl:699–708` | D1 at world point, terrain, bodies | as D4 | 5 loads (+T) | no |
| D7 | `faceSolidData(id,axis)` | `ref.wgsl:710–721` | 4×D6 | as D4 | 20 loads | no |
| D8 | `faceOpenFraction(id,axis)` | `ref.wgsl:722–744` | 2×D1, D7 | as D4 | **30 loads** | no |
| D9 | **`pressureFaceData(id,axis)` → `pressureFaceVolumeFraction` = V_face** | `ref.wgsl:745–784`, `:785` | 8×D6 (interior arm) | as D4 | **40 loads (+8T)** | no |
| D10 | `uvPhi(pos)` | `uv.wgsl:73–80` | `uvPhiIn` vertex texture, `d4Sum8` | every `uvAdvectPhi` / `uvRedistancePhi` / surface-volume apply | 8 `r32float` texels + 8-term D4 fold | yes |
| D11 | `pressureSurfacePhi(p)` | `ref.wgsl:393–425` | D4, D10 | per advance (phi) | D4 + D10; **the `open` local at :415 is dead when `params.physical.w=0`** | maybe |
| D12 | `pressurePhi(p)` | `ref.wgsl:429–440` | D4, D11 | per advance | **2×D4 + D10 = 20 loads + 8 texels** | maybe |
| D13 | `sampleVelocityComponent(p,c)` | `ref.wgsl:462–475` | `uvTwoLevelFineAt` (1 atomicLoad), `transportIn` | once per advance (`transportA` written by the extension) | 1 atomic + 8 `rgba32float` texels + D4 fold | no (atomic) |
| D14 | `sampleVelocity(p)` | `ref.wgsl:476` | 3×D13 at the **same p** | as D13 | 3 atomics (identical word) + 24 texels → **13.875 distinct texels mean** (8 min, 19 max) | no |
| D15 | `departurePoint(x,dt,h)` | `ref.wgsl:529–539` | D14 | as D13 | ≤32 substeps × 2×D14 | — |
| D16 | `uvTrace(p,dt)` | `uv.wgsl:89–98` | 2×D14 + per-step D4 | as D13 + D4 | 2×D14 + ~2×D4 | — |
| D17 | `divergenceAt(id,chk)` | `ref.wgsl:1237–1251` | D4, 6×D9, 6×`domainFaceSolidVelocity` | as D4 + velocity | 6×D9 = 240 loads (12×D9 with terrain) | no |
| D18 | `uvTarget(id)` | `uv.wgsl:408–420` | 8×D10 + D5 | per advance | 64 phi texels + 10 loads | maybe |
| D19 | `uvSurfaceDeficit(id)` | `uv.wgsl:763–…` | D4, D12, V, γ | per advance | 3×D4 + D10 | maybe |

**The central fact (V):** D1, D2, D3, D4, D6, D7, D8 and D9 are functions of *scene
geometry only*. Nothing in an advance writes their inputs. The solid bitfield is
uploaded at construction (`host:947–955`) and re-uploaded only from
`applySceneUniforms` on a scene edit (`host:3128–3141`); the terrain texture is
uploaded once (`host:1607`). In a body-free scene **every evaluation of D4 and D9
after the first is bit-identical to the first, for the entire run.**

### Structural redundancy inside D4 (V)

`cellOpenFraction` (`ref.wgsl:364–367`) returns `0.0` if
`staticSolidVoxelOccupied(p)`; if it did not return, it calls `cellSolidFraction(p)`
(`ref.wgsl:686`), whose **first line is `if(staticSolidVoxelOccupied(p)){return 1.0;}`
— a test already proven false one line earlier.** 5 of D4's 10 loads are
unconditionally dead on every call, in every kernel. The same doubling appears in
`faceOpenFraction`: lines 723–725 prove both `staticSolidVoxelOccupied(id)` and
`…(neighbor)` false, then line 726–727 re-evaluates both inside a `select` on the
invalid-neighbour arm.

### Per-cell evaluation counts per advance

Live kernels only; per cell (or per vertex, ≈1 per cell at 64³). Stage names follow
static-core.md §3.

| Stage | Kernel | D4/D5 | D9 | D10 (8-tap) | D14 |
|---|---|---|---|---|---|
| 3 two-level | `uvTwoLevelSeed` (`uv.wgsl:652`) | 1 | – | ~1.95 | – |
| 4 authority | `storeExtrapolationAuthority` (`ref.wgsl:790–794`) | 2 (via D12) | **3** | 1 | – |
| 4 extension ×20 | reads `faceOpenIn` texture — **no recompute** | 0 | 0 | – | – |
| 5a phi advect | `uvAdvectPhi` (`uv.wgsl:273–283`) | **~34–36** | – | 2–3 | 2–4 |
| 5b redistance | `uvRedistancePhi` (`uv.wgsl:285–295`) | 0 | – | ≤56 | – |
| 5c edges | `uvBuildEdges` (`uv.wgsl:333–344`) | **24** (16 of them the same `uvOpen(id)`) | – | – | 2 |
| 5 transport | `uvFallback`, `uvNormalizeRows`×3, `uvGather` | 1+3+3 | – | 8 (`uvTarget`) | – |
| 5 correction | `uvCorrectionCapacity` + `uvCorrectionTargets` (`uv.wgsl:47–56`) | 2 | – | 8 (`uvTarget` again) | – |
| 5 sharpen ×8 | `uvPrepareSharpen` + `uvProposeSharpen` (`uv.wgsl:506,530`) | 8 + **48** | – | 8+24 | – |
| 6 advection | `semiLagrangianAdvection` (`ref.wgsl:1124–1147`) | 0 | – | – | 3×D15 |
| 7 balance | `uvBalanceMeasure` (`uv.wgsl:775–…`) | **6** | – | 2 | – |
| 8 mg topology | `mgBuildFinestTopology` (`mg.wgsl:229–254`) | 3 | **3** | 1 | – |
| 8 mg rhs | `mgBuildFinestRhs` (`mg.wgsl:256–281`) | **9** | **6** (12 w/ terrain) | 2 | – |
| 9 project | `project` (`ref.wgsl:1302–1378`) | **~42** (via ~21×D12) | **6** | ~21 | 1 |
| 10 publish | `uvPublish` (`uv.wgsl:751–756`) | 1 | – | 1 | – |

**Totals per cell per advance, interior, body-free, at 64³ (V by summation):**
D4/D5 ≈ **185–190 evaluations** ≈ 1 850–1 900 storage-word loads of a value that is
constant for the whole run. D9 ≈ **18 evaluations** ≈ 720 loads, likewise constant.
D10 ≈ **135** 8-tap gathers.

### Does a cache already hold a bit-identical value? (V)

| Quantity | Existing field | Bit-identical? | Written where |
|---|---|---|---|
| D9 (3 axes) | `velocityD` = `faceOpenIn`, `.xyz` = `pressureFaceVolumeFraction(id,0/1/2)` | **Yes** — same function, same advance | `storeExtrapolationAuthority` `ref.wgsl:793`; bound at `host:1012–1022`, `host:1155–1158` |
| D12 (as ρ′) | `surfaceA` `.x = 0.5 − pressurePhi(id)/h` | No — phi changes at `uvAdvectPhi`, so the stored value is one stage stale by the time pressure setup runs | `ref.wgsl:792` |
| D4 | γ field `gammaB` ← `uvOpen(id)` | **Yes**, but written at the *tail* of the previous advance | `uvPublish` `uv.wgsl:755` |
| D4 | γ field `gammaB` ← `uvOpen(id)` | **Yes**, mid-advance | `uvCorrectionCapacity` `uv.wgsl:48` |
| D4/D8 | `uvEdges` weight[5], weight[6], donor[0..2], donor[4] | Yes, for sharpening only | `uvCacheSharpenCells/Faces` `uv.wgsl:478–504` |

---

## 2. Backtrace rework

**V.** Three independent characteristic families run per advance, all reading the
same `transportIn = transportA`, which is written once by the extension's
`packValue` (`webgpu-uniform-velocity-extrapolation.wgsl.ts:651–662`) and is not
written again before stage 9.

| Family | Integrator | Sample lattice | Traces / cell |
|---|---|---|---|
| Velocity advection | `departurePoint` (`ref.wgsl:529–539`): bounded loop ≤32 substeps, `stepSeconds = min(remaining, 1.5/max(rate,1e-6))`, 2×`sampleVelocity` per substep | cell centre + face offset, per component | 3 (`advectVelocityComponent` `ref.wgsl:625`) |
| Phi advection | `uvTrace` (`uv.wgsl:89–98`): single RK2, then a half-cell solidity walk calling `cellOpenFraction` per step | **vertex** lattice | 1, plus 1 more inside `uvClosedWallPhi` at domain-face vertices (`uv.wgsl:150–171`) and 2 `sampleVelocity` inside `uvEmbeddedAir` (`uv.wgsl:173–199`) |
| Volume transport | `uvTrace` again (`uv.wgsl:333`) | cell centre | 1 |

The three families differ in integrator (substepped vs. single RK2), in sample
position (face-staggered vs. cell-centre vs. vertex) and in the solidity walk, so no
two produce the same departure point. **I:** a single compiled departure field is
therefore not a drop-in for all three; the cell-centre `uvTrace` in `uvBuildEdges`
and the vertex `uvTrace` in `uvAdvectPhi` are the two that share an integrator and
differ only by the half-cell lattice offset.

**V.** `departurePoint`'s substep count is data-dependent and unbounded up to 32; at
CFL ≤ 1 it is 1. The loop's `rate` is recomputed from a fresh `sampleVelocity` each
iteration, so the 24 texels per sample are paid twice per substep.

---

## 3. Velocity sampling rework

**V.** `sampleVelocity(p)` (`ref.wgsl:476`) calls `sampleVelocityComponent` three
times with the **identical `p`**. Each call:

1. `uvTwoLevelFineAt(p)` (`uv.wgsl:612–615`) → one `atomicLoad` of the *same class
   word* for all three components.
2. 8 × `textureLoad(transportIn, base + o + vec3i(1))` → `d4Sum8`.

The three atomic loads are of one word. **I:** `atomicLoad` compiles to
`atomic_load_explicit`, which Metal will not CSE or reorder across, so all three are
issued (repo memory `atomic-load-is-an-fp-barrier`).

**V.** `transportIn` is `rgba32float`, so the three components live in one texel.
The three 8-tap boxes are offset by the component's half-cell stagger, so the 24
taps cover, by inclusion–exclusion over the eight independent δ cases, a mean of
**13.875 distinct texels** (8 when all three δ agree, 19 when all disagree). ~10 of
the 24 taps are redundant re-reads of a texel already in registers, but across
separate `textureLoad` calls with different indices that the compiler cannot prove
equal. **I:** Metal's texture-load CSE works on provably identical coordinates only.

**V.** The 4h coarse path (`uvCoarseVelocityComponent` `uv.wgsl:628–639`) issues
**8 `atomicLoad`s** of `conditioningScratch` per component — 24 per `sampleVelocity`
in far-air. The table is written by `publishCoarseVelocityTable`
(`webgpu-uniform-velocity-extrapolation.wgsl.ts:697–715`) with a **plain,
non-atomic** `tileScratch[slot+component] = bitcast<u32>(value)` store, once per
advance, in a pass that precedes every reader; the extension module already binds
the same buffer as a plain `array<u32>` (binding 12) and reads it non-atomically in
`shellAt` (`:109–113`). The atomic qualifier in `uv.wgsl` is therefore not needed by
any writer inside the advance. **I:** dropping it would let Metal CSE and cache the
8 loads.

---

## 4. Pressure setup rework

**V.** Per cell per advance the three pressure-setup kernels together evaluate:

- `cellOpenFraction` ≈ 54 times (3 + 9 + 42);
- `pressureFaceData` ≈ 15 times (3 + 6 + 6) = 600 storage loads;
- `pressurePhi` ≈ 24 times, each 20 loads + an 8-tap phi gather.

All 15 `pressureFaceData` evaluations return values already sitting in `velocityD`
(`faceOpenIn`), written this advance by `storeExtrapolationAuthority`
(`ref.wgsl:793`) and bound into the extrapolator at `host:1012–1022`. **V:** the
main bind-group layout (`host:1070–1099`) is at 10/10 storage buffers, but
`pressureInputLayout` (`host:1102–1104`) is mainEntries minus bindings 32 and 33, so
the pressure layout has one free storage slot; texture slots have headroom in both
(`lib/core/webgpu-device-limits.ts` requests the adapter maximum).

**V.** `project` (`ref.wgsl:1302–1378`) is the single worst site.
`geometricProjectedFace` (`ref.wgsl:1288–1299`) evaluates `pressurePhi(id)` and
`pressurePhi(q)` for the liquid test, again inside `ghostFluidFraction`
(`ref.wgsl:444`), and a third time inside `geometricPressureValue`
(`ref.wgsl:1285`) — **up to 6 `pressurePhi` per axis**, ×3 axes, plus 3 more in the
release loop, plus 3 more `pressureFaceData`. `pressurePhi(id)` is the same value
every time. **I:** there is no buffer store inside the axis loop
(`storeBoundaryVelocity` is at :1377), so Metal *may* CSE these; whether it does
depends on inlining a ~20-load function 21 times, and the `atomicLoad` in
`uvTwoLevelFineAt` at :1310 sits upstream of the loop.

**V.** `pressurePhi` (`ref.wgsl:429–440`) evaluates `cellOpenFraction(p)` for the
`>1e-5` gate and then `pressureSurfacePhi(p)` evaluates `cellOpenFraction(q)` again
at :415 for the `open` local — which, with `volumePressureRows` off
(`params.physical.w = 0`), is consumed only by a short-circuited `||` and is dead.
Half of `pressurePhi`'s 20 loads are for a value the shipping configuration never
reads.

---

## 5. Compile-once candidate record (facts only, unranked)

| Candidate | Value | Where it would be written | Where it is read today | Invalidated by | Storage at 64³ | Binding cost |
|---|---|---|---|---|---|---|
| C1 V_i field (D4) | `cellOpenFraction(id)` | once at construction + on scene edit / per body move | ~185 sites/cell/advance | scene edit, body motion | 1 MiB `r32float` (or 0.25 MiB `r8unorm`) | 1 texture slot; γ already exists as a field |
| C2 V_face field (D9) | `pressureFaceVolumeFraction(id,0/1/2)` | already written every advance to `velocityD` `.xyz` (`ref.wgsl:793`) | 15 sites/cell/advance recompute it | scene edit, body motion | already allocated | rebind `faceOpenIn` into the mg + project layouts |
| C3 face-open field (D8) | `faceOpenFraction(id,axis)` | with C1 | `uvProposeSharpen` ×24/cell/advance (`uv.wgsl:530–566`) | as C1 | 1 texture | `uvCacheSharpenFaces` already does this above 64³ |
| C4 solid-voxel binding | move the packed bitfield out of `activeScratch` (binding 30, `read_write`) into a `read` storage binding | construction | every D1 | never | none | needs a `read` binding; `activeRegion` (29) is already `read` |
| C5 shared departure field | cell-centre `uvTrace` result | after the extension, before stage 5 | `uvBuildEdges` (`uv.wgsl:333`); vertex variant in `uvAdvectPhi` | velocity | 1 `rgba16f`/`rgba32f` | 1 texture |
| C6 phi-at-cell field | `pressurePhi(id)` | after `uvRedistancePhi` + surface-volume apply | ~24 sites/cell/advance in stages 8–9 | phi transport | 1 `r32float` | 1 texture |
| C7 4h table de-atomicised | drop `atomic<u32>` on the coarse face table in `uv.wgsl` | — | 8 atomics/component in far-air | — | none | none |

**V** for every "where it is read today" count and every "written where"; the
invalidation columns follow from §1's input analysis.

---

## 6. Existing partial caches and why they are partial

| Cache | Holds | Why partial |
|---|---|---|
| `velocityD` / `faceOpenIn` (`host:1012–1022`, written `ref.wgsl:793`) | V_face for all 3 axes | **V:** bound into the *extrapolation* group only. The extension consumes it correctly (`openBaseFace`), but `mgBuildFinestTopology`, `mgBuildFinestRhs`/`divergenceAt` and `project` do not have it in their layout and recompute. This is the one clean success story and the one clean gap. |
| `surfaceA` authority ρ′ (`ref.wgsl:792`) | `0.5 − pressurePhi(id)/h` | **V:** written before `uvAdvectPhi` changes phi, so by stage 8 it no longer matches `pressurePhi`. Cannot substitute for C6. |
| γ field (`gammaB`) | `uvOpen(id)` | **V:** written by `uvCorrectionCapacity` (`uv.wgsl:48`) mid-advance and by `uvPublish` (`uv.wgsl:755`) at the tail, but `uvGather` (`uv.wgsl:389–407`) and `uvCorrectionTargets` (`uv.wgsl:51`) overwrite `gammaB` with `uvTarget`. The γ slot is time-multiplexed between two different quantities, so no reader can assume it holds V_i. |
| `uvCacheSharpenCells/Faces` (`uv.wgsl:478–504`) | `uvOpen(id)`, `uvOpen(q)`, `faceOpenFraction`, phi | **V:** gated on `uvPageWorkEnabled()`, which the host enables only when `Math.max(nx,ny,nz) > 64` (`host:974`). At and below 64³ the *uncached* arm of `uvProposeSharpen` (`uv.wgsl:529–566`) runs — the default lane recomputes what the large lane caches. Also scoped to sharpening only; no other stage reads these slots. |
| `faceOpenIn` in the extension (`webgpu-uniform-velocity-extrapolation.wgsl.ts` binding 8) | V_face | Working as intended across 20 passes. Cited as the existence proof that the pattern is already viable. |

---

## 7. Rework caused by ordering accidents

**Full-field copies per advance at 64³ geometric defaults (V):**

| Copy | Site | Eliminable by binding swap? |
|---|---|---|
| `output → vertexPhiField`, `(n+1)³` `r32float` | `lib/methods/uniform/webgpu-uniform-surface-volume.ts:~88` (`copyTextureToTexture`, or `fieldPages.copy`) | **I: yes** — `this.phi` is `vertexPhiField` (`host:1263–1265`); the module's 14 passes could ping-pong and hand back whichever texture is current, if every downstream reader takes the handle rather than a fixed binding. |
| `gammaB → gammaA`, n³ | `host:~2527` | **I: yes** — a parity flip on the γ read/write pair per advance. |
| `volumeB → volumeA`, n³ | `host:2795` | **I: yes** — same pattern. |

`copyField` is `host:1684–1687`. **V:** the `vertexPhiScratch → vertexPhiField` copy
listed by static-core.md is *not* in this set (see Corrections §1).

**Ordering accidents that force recomputation (V):**

1. `uvPublish` writes γ = `uvOpen(id)` into `gammaB` at the tail of advance *N*
   (`uv.wgsl:755`). At the start of advance *N+1*, `uvBuildEdges` needs exactly that
   value 24 times per cell — and `gammaB` still holds it at that point — but
   `uvBuildEdges` runs under `volumeDonorGroup`, whose γ *read* binding is `gammaA`
   (holding last advance's `uvTarget`). The correct value is resident in GPU memory,
   in the wrong slot of the wrong group. `uvGather` then overwrites `gammaB`.
2. `uvTarget(id)` is evaluated twice per cell per advance — `uvGather`
   (`uv.wgsl:406`) and `uvCorrectionTargets` (`uv.wgsl:51–56`). **Not** pure rework:
   the surface-volume module rewrites phi between them, so the second evaluation
   reads a different phi. The `uvOpen(id)` factor inside `uvTarget` (`uv.wgsl:419`)
   is unchanged between the two, and `uvCorrectionCapacity` wrote precisely that
   value into `gammaB` one pass earlier, only for `uvCorrectionTargets` to overwrite
   it.
3. `storeExtrapolationAuthority` runs at stage 4 (`host:2248`), before
   `uvAdvectPhi`. Its V_face output stays valid (geometry-only); its ρ′ output does
   not. Splitting the two writes across the phi update would make `surfaceA` usable
   by stage 8 — **I**.
4. `mgSmoothColour` (`mg.wgsl:448–468`) re-derives the row diagonal from 6
   `mgCoefficient` taps on every red and every black sweep, although
   `mgBakeCoefficients` (`mg.wgsl:419–428`) already ran once per cycle;
   `mgMeasureFineResidual` (`mg.wgsl:472–494`) gathers the same coefficients twice
   within one invocation. **V.**

**Deliberate non-findings:** the D4 `d4Sum6`/`d4Sum8` folds and the atomic read
order are bit-exactness load-bearing (repo memories `canonical-folds-are-the-regression`,
`atomic-load-is-an-fp-barrier`); any cache that changes the add order or replaces an
atomic read with a plain one is a numerics change, not a pure speedup.
