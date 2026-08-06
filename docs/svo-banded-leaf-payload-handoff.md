# Banded leaf payload — store the surface, reconstruct the rest

**Status.** The prize is measured and large: **930 MB → 80.0 MB** at refinement
depth 3, an 11.64x cull on top of SP20's shipped f16 lane. SP21's material-palette
work is merged in — the band record set and the identity palette are **one
structure**, not two savings to multiply. Reference encoder/decoder landed
(`lib/svo-banded-leaf-payload.ts`, `tests/svo-banded-leaf-payload.test.ts`, 15
tests green), with both record-set rules expressible and the conservative one as
the default.

**B0.5b has run and the reconstruction does not hold the hash.** `0xb081f6cf` →
`0x4318d060` on `hero-garden-hose`, both arms deterministic, with a diffuse
0.586/255 mean and 7/255 peak difference across 41 % of the frame — a *lower* peak
than the f16 arm that shipped, but not identity. The drafted fallbacks are both
ruled out by measurement: a wider radius buys 86.4 % → 81.4 % on occupied flips,
and an occupancy-mask normal gradient is identically zero over the same population.
**B1 and B2 are therefore not implemented**, and should not be until the bar is
settled — see "Why the occupancy-mask normal is not the fallback" for the three
options and which of them are Peter's call.

Getting that answer cost one lever and one shader function. Building the arena
first would have cost the allocator, the masks and the palette, and produced the
same sentence.

## The question

Two culls were running separately. One shrinks the *width* of a voxel:
`SparseBrickSceneGeometryFormat`, 8 bytes of geometry toward 2. The other — this
one — asks **how many voxels need storing at all**. A third, the width of the
material-owner word, turned out not to be a separate question at all.

The per-voxel payload arena is the whole memory problem at refinement depth 3:
the hero garden's 227 168 leaves hold 116.3 M voxels, so at 12 bytes a voxel
(8 geometry + 4 owner) the arena is **1.40 GB** before anything else.

## Two lanes, one structure

Measured through the geometry lane (`tmp/sp22/band-census.ts`, a CPU replay of
`rebuildDirtyBrickPayload` over a uniform seeded sample of 3 000 planned leaves
per depth, both dry garden scenes, `FLUID_SVO_REFINEMENT_MODE=surface`):

| scene / depth            | leaves | cell    | band   | interior | exterior | all-interior leaves |
|--------------------------|-------:|--------:|-------:|---------:|---------:|--------------------:|
| `hero-garden-hose` d0    | 10 369 | 6.25 mm |  3.2 % |   79.1 % |   17.7 % |                68 % |
| `hero-garden-hose` d1    | 22 834 | 3.13 mm |  4.6 % |   71.3 % |   24.1 % |                56 % |
| `hero-garden-hose` d2    | 86 006 | 1.56 mm |  4.2 % |   74.7 % |   21.1 % |                62 % |
| `garden-svo-lighting` d0 |  1 815 | 25.0 mm | 12.7 % |   42.7 % |   44.5 % |                18 % |
| `garden-svo-lighting` d1 |  7 843 | 12.5 mm | 11.0 % |   48.3 % |   40.8 % |                22 % |
| `garden-svo-lighting` d2 | 37 897 | 6.25 mm |  8.4 % |   58.2 % |   33.4 % |                35 % |

Measured through the owner lane, off the device rather than replayed
(`tmp/sp21/`): **83 % of hero leaves at depth 0 hold exactly one non-air identity
across all 512 voxels**, 68 % at depth 1; mean 1.457 distinct identities a leaf;
scene-wide alphabet 415 words against a 16 386 capacity.

These are the same fact — *a leaf is usually one solid* — seen through two lanes,
and that is why the two designs had to merge. Encoded separately, a band record
would carry an owner field that is a copy of the leaf's own in five leaves out of
six, and the two savings would have been quoted over overlapping bytes.

### Not the SP9 wall from the other side

SP9 found buried coarsening could not reduce page counts, because base pages are
a function of *claim volume* rather than of surface. Same fact, and it is what
makes this cull work: pages are claimed by volume, which is exactly why so many
are solid all the way through. This removes no leaf and no page — it removes the
6 144 bytes each solid leaf spends saying "solid" 512 times.

## The layout

One record set carrying geometry, one palette carrying identity, per leaf:

| part            | bytes | note |
|-----------------|------:|------|
| occupancy mask  |    64 | 1 bit a voxel. **The occupancy predicate.** |
| record mask     |    64 | 1 bit a voxel; record index by prefix popcount |
| header          |    16 | record base, record count, palette base, index width, flags |
| leaf palette    |   0-… | `u16` slots into the scene-global table; **one entry on most leaves** |
| identity index  |     0 | **zero bits when the leaf has one identity**; else 1/2/4/8 bits per *occupied* voxel, compacted by occupancy popcount |
| records         |   var | size-classed; **geometry only** — 8 B `f32x2`, 4 B `f16-unorm8`, 2 B `snorm8-unorm8` |

The scene-global identity table is charged once for the whole scene: 415 words is
1.7 kB against a gigabyte.

### What an interior voxel costs

**Two bits, both already inside the fixed masks** — one occupancy bit and one
record bit that is zero. On a single-identity leaf that is the entire cost: no
record, no index, no palette entry beyond the leaf's one slot. On a
multi-identity leaf it is those two bits plus the leaf's index width.

### Why the record set is wider than the band

`safeNormal` central-differences `solidDistance` at `local ± 1` on each axis,
clamped inside the leaf, so a band voxel's normal is exact only if its six axis
neighbours also carry a distance. The record set is the band **dilated by that
stencil**: 6.8-9.4 % of a hero leaf, 15.7-23.1 % of a lighting-study leaf. On the
hero the *median* leaf still stores nothing at all.

Voxels outside it keep a reconstructed distance: the saturated band value, signed
by occupancy, **sharing** `SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII` with the
`snorm8` format rather than copying it, so past the band the reconstruction *is*
the value that format would have stored. A test pins the two constants together.

Fraction and identity are **exact for every voxel by construction**.

### The one lossy channel — and SP20's lesson applied to it

`solidDistance` is reconstructed, not stored, outside the record set. The argument
was that those voxels are buried and occluded, so their normals cannot reach the
frame.

**SP20's device data is evidence against that argument, and it lands squarely on
this design.** `snorm8` was rejected because clamping flattens the central
difference over the deep interior — the same population this layout reconstructs,
by the same mechanism, over a *larger* share of it: snorm8 loses the gradient
beyond 4 cell radii, the stencil record set loses it beyond about one. If SP20's
28.5 % pixel diff is caused by interior normals rather than band quantisation,
then this layout inherits it and the frame hash cannot hold.

SP20's own recorded lesson is the reason this was checkable at all: *a
quantisation study must sample the population the field holds, not the population
whose answers are consumed.* Its synthetic study scored only voxels with coverage
strictly between 0 and 1 — a surface population, where nothing saturates — and
picked 4R on that basis. This design's census is band-centric in exactly the same
way, so the reconstruction was scored over the wrong population too.

`tmp/sp22/banded-normal-fidelity.ts` closes that gap. It follows SP20's device
probe — same world build, same arena readback, same replay of the shader's clamped
central difference — and adds the arms this design needs plus **three
populations**: `band` (SP20's, for comparability), `occupied` (every voxel
`safeNormal` is actually called on, which is the one that decides this) and `all`.

Measured on the real hero arena — 10 369 published leaves, 5 308 928 voxels, lane
`f16-unorm8` — with the shipped f16 lane as the control:

| rule    | stored  | band error / flips (n=167 825) | occupied flips (n=4 384 719) |
|---------|--------:|-------------------------------:|-----------------------------:|
| f16     | 100 %   |                 0.0000° / 0    |                            0 |
| stencil |  6.64 % |             **0.0000° / 0**    |          3 789 718 (86.4 %)  |
| 1R      |  4.35 % |                8.6411° / 15    |          3 846 086 (87.7 %)  |
| 2R      |  8.42 % |                0.3186° / 0     |          3 739 679 (85.3 %)  |
| 4R      | 15.55 % |                0.0000° / 0     |          3 567 349 (81.4 %)  |
| 6R      | 22.24 % |                0.0000° / 0     |          3 355 523 (76.5 %)  |

Both halves of the expectation were wrong, in opposite directions.

**The band is safe, and the stencil rule is why.** It is exact — zero angular
error, zero flips — because it is *defined* as the closure of the band under the
stencil `safeNormal` reads. No distance-radius rule respects that geometry: 1R
stores *less* than stencil and is 8.6° wrong with 15 flips. So the mechanism that
demonstrably moved SP20's pixels — flips in the band population, which is the only
population SP20 scored — is **identically zero here**. snorm8 also quantised the
band itself (8 bits over ±4R is a quantum of R/32, coarse where the central
differences live); this layout stores the band at f16 and inherits none of it.

**The interior cannot be fixed with a radius, so the fallback plan was futile.**
Going from 6.64 % stored to 15.55 % — 2.3x the records — moves occupied flips from
86.4 % to 81.4 %. Even 22.24 % stored only reaches 76.5 %. That is SP20's
structural finding restated: a rule keyed on distance-to-surface cannot help a
population that is nowhere near the surface, and the hero is mostly deep interior.
**The `{ radii }` arm is therefore not the fallback this document previously
proposed**, and `SVO_BANDED_DEFAULT_RECORD_SET` is `"stencil"` — the narrow rule,
chosen on evidence for being both exact where it matters and the cheapest.

### B0.5b: the interior does reach the frame

Run on device with `FLUID_SVO_BANDED_RECONSTRUCTION` (off by default; off is the
shipped arm). The lever changes **only** what `solidDistance` returns — no arena,
masks, palette or allocator — so it isolates the one lossy semantic.

`hero-garden-hose`, 800×460, settled frame:

| arm | frame hash | frame median |
|---|---|---:|
| off (shipped) | `0xb081f6cf` | 18.00 ms |
| on (banded reconstruction) | `0x4318d060` | 31.71 ms |

Each arm is internally deterministic (two consecutive settled frames agree), and
`gpu-validation` passes on both. **The hash moves.** The claim that occluded
interior normals cannot reach the frame was an argument, and it is now measured to
be false.

The amplitude, though, is not a broken frame:

| metric | banded reconstruction | SP20 f16 (accepted) | SP20 snorm8 (rejected) |
|---|---:|---:|---:|
| pixels differing at all | 152 099 (41.33 %) | 9 368 (2.55 %) | 104 802 (28.5 %) |
| > 1/255 | 38 172 (10.37 %) | — | — |
| > 4/255 | 1 956 (0.53 %) | — | — |
| > 16/255 | **0** | 0 | — |
| max channel delta | **7/255** | 11/255 | — |
| mean channel delta | 0.586/255 | — | — |

So the change is *everywhere and tiny* — a lower peak error than the f16 arm that
shipped, spread across 41 % of the frame instead of concentrated in 2.55 % of it.
That signature is consistent with buried-voxel radiance shifting the GI slightly
through the coarse pyramid rather than with any surface being drawn wrongly.
(Diagnosis, not proof: it was not isolated further.)

### Step 1 and 2: the skip closes the leak completely, but it is not a small change

`hero-garden-hose`, 800×460, `FLUID_SVO_BURIED_RADIANCE_SKIP` and
`FLUID_SVO_BANDED_RECONSTRUCTION` gated separately, both off by default.

| arm | hash | frame median | radiance pages | pyramid |
|---|---|---:|---:|---:|
| shipped | `0xb081f6cf` | 17.57 ms | 44 | 12.3 MB |
| skip only | `0xc96b00af` | 17.51 ms | 44 | 12.3 MB |
| skip + banding | `0xc96b00af` | 17.16 ms | 44 | 12.3 MB |

**Step 2 passes perfectly: banding on top of the skip is byte-identical** — same
hash, and the raw frames differ in 0 of 368 000 pixels at 0/255. The leak had
exactly one path, the skip closes all of it, and with it closed the banded
reconstruction is free. That is the result this whole sequence was after.

**Step 1 does not pass on its own merits.** The skip alone against the shipped
frame:

| metric | skip vs shipped | (banding, for scale) |
|---|---:|---:|
| pixels differing at all | 288 011 (78.26 %) | 41.33 % |
| > 4/255 | 170 661 (46.38 %) | 0.53 % |
| > 16/255 | **133 055 (36.16 %)** | 0 |
| max channel delta | **188/255** | 7/255 |
| mean | **22.22/255** | 0.586/255 |

So buried-voxel radiance is *not* a fiction in effect: it carries a large amount
of the hero's light. The likely mechanism is coarse leaves — `originCells` is
`cell + normal·1.5·scale`, and at a coarse level `1.5·scale` is tens of finest
cells, so a voxel metres inside the ground starts its visibility march *above* the
terrain and receives direct sun. Whether that is a bug worth keeping is a
judgement about the hero's look, not a memory question, and it is Peter's.

Before/after PNGs, 800×460:
`…/scratchpad/out/hero-before.png`, `…/scratchpad/out/hero-skip.png`
(`hero-skip-banded.png` is byte-identical to `hero-skip.png`.)

The two bonus hypotheses are **negative**: the skip does not make the build
cheaper (17.57 → 17.51 ms, within noise) and does not reduce resident radiance
pages (44 in every arm, pyramid 12.3 MB in every arm). Radiance pages live at
level ≥ 3 whether or not their level-0 seeds are computed, so skipping seeds
removes arithmetic that was never the cost.

### Step 3: the march-origin root cause, fixed and measured

The shipped radiance seed offsets every gather origin by `1.5 * scale` finest cells
along the normal, so at a coarse level a voxel metres inside the ground began its
visibility march *above* the terrain. The offset cannot merely shrink — a band
voxel needs an offset of order its own extent or its gather self-intersects the
voxel it started in, which is self-shadow acne and is worse at coarse levels. What
separates the cases is not size but whether there is anywhere outside to move to,
and a fully solid voxel has none. `FLUID_SVO_SOLID_MARCH_OFFSET` therefore removes
the offset for fully solid voxels only, expressed on the fraction (`>= 1.0`)
because that is the exact question and needs no stencil.

| arm | hash | frame median |
|---|---|---:|
| shipped | `0xb081f6cf` | 17.57 ms |
| offset fix | `0x08c3d5ea` | 17.19 ms |
| offset + banding | `0xa46942dd` | 16.49 ms |
| offset + skip + banding | `0x5aa057cc` | 16.32 ms |

`terrain-coverage-solid` passes in **every** arm: 4 061 223 of 4 061 223 buried
voxels carry full scene coverage. So does `gpu-validation`, `radiance-black-pages`
(0 of 44) and `frame-carries-radiance` (256/256).

**The offset fix is small and defensible on its own**: 47.00 % of pixels differ,
mean **1.16/255**, max 30/255, and only 119 pixels (0.03 %) exceed 16/255 — against
the skip's 22.22/255 mean and 188/255 peak. No acne: band voxels keep the shipped
`1.5 * scale` untouched, so the construction cannot reintroduce it, and the image
confirms it (`hero-offset.png` — clean surfaces, no speckle).

**But it does not make banding free, and the two routes did not converge.**

- banding on top of the offset fix still moves the frame: 39.42 % of pixels, mean
  0.54/255, max 7/255 — statistically the same as banding on top of shipped
  (41.33 %, 0.586/255, 7/255). The offset was about 1/19th of the leak.
- the offset frame and the skip frame differ by mean **21.29/255** — they are not
  the same picture. Fixing the origin does **not** make a buried voxel
  self-occlude: with the origin left inside the solid, `directVisibility` still
  samples the *coarse* pyramid four times at 2.5-54 coarse cells and multiplies
  `1 - coverage` per step, which never reaches zero through partially-covered
  blocks. That residual is where the other 21/255 lives, and it is a second
  mechanism this work has not addressed.

So the hypothesis that the origin fix would retire the skip is **falsified**. The
decision is unchanged in shape: byte-identical banding still requires the buried
skip and its 22/255 relighting, or banding is accepted with its own 0.54/255.
The offset fix is worth landing regardless, as a correctness fix with a small
diff and every oracle clean.

## Mechanism two, measured and closed (SP27)

`FLUID_SVO_SOLID_DIRECT_OCCLUSION`, off by default, returns zero transmittance
from `directVisibility` before any step when the receiver's own fine fraction is
`>= 1.0`. Same predicate as `FLUID_SVO_SOLID_MARCH_OFFSET`, threaded from the seed
through `directIncident`; the step schedule is untouched in both arms, so a band
voxel's march cannot pick up acne from this. The localisation below was correct
and the fix is the one it proposed.

**The lane is now bit-reproducible across processes.** Two `baseline` runs in the
same batch produced byte-identical PNGs and the same `0x1ab2434b`, so the noise
floor is exactly zero and every delta below is signal. That contradicts the
warning in `tools/run-svo-dry-render-smoke.ts`'s header (0x7eec7076 vs
0xd84be85c across processes) — worth re-reading that paragraph before trusting it
again. `per-brick-candidates` fails identically in *every* arm (busiest brick 73
against the 64 target, 4 of 3922 bricks over), which is the pre-existing tree
state, not any of these arms.

`hero-garden-hose`, 800x460, one lock acquisition, all seven arms:

| arm | hash | vs baseline: % px | mean/255 | max | mean luma |
|---|---|---:|---:|---:|---:|
| baseline (shipped, offset fix on) | `0x1ab2434b` | — | — | — | 116.99 |
| `SOLID_DIRECT_OCCLUSION=on` | `0x5b393c3d` | 70.10 | **12.30** | 91 | 105.70 |
| `BURIED_RADIANCE_SKIP=on` | `0x62749865` | 70.97 | 12.98 | 111 | 104.81 |
| both | `0x73744ac9` | 73.38 | 14.65 | 117 | 103.27 |
| occlusion steps in finest cells | `0x08e0ac7d` | 45.55 | 1.42 | 40 | 115.77 |
| that + direct occlusion | `0x28bde2d8` | 70.08 | 12.29 | 91 | 105.71 |
| direct occlusion, `coverage>=1` too | `0x44926961` | 56.72 | 5.26 | 74 | 112.18 |

**The two routes converge.** `direct` and `skip` are now 3.21/255 apart (they were
21.29/255 apart with only the offset fix), so *the darker frame is the correct one,
reached independently twice*. Its remaining 3.21 is two-sided — 43 % of pixels
darker, 14 % brighter — which is two different populations of the same
mechanism, not a third mechanism.

**The occlusion march does not need its own step scaling.** Taking the same four
steps in *finest* cells (8x shorter; first sample 1.6 cm rather than 12.5 cm)
recovers only 1.42 of the 12.30, and on top of the guard it adds **0.013/255**.
The guard subsumes it. The pyramid holds one average coverage per floor-level
block and cannot separate "just below the surface" from "just above" at any step
length in that family.

**A stricter receiver test was measured and rejected.** Also requiring the
floor-level texel's own opacity to be full closes only 5.26/255, leaves the contact
shadow at luma 71.0 where both independent routes put it at 50.8/51.2, and
converges with neither. The level-3 opacity texel is a filtered average that a
genuinely buried block does not reach 1.0 on, so it under-corrects rather than
being safer.

**It is not over-darkening.** Sky is byte-identical in every arm (luma 98.2).
`terrain-coverage-solid` 4 187 294 / 4 187 294, `radiance-black-pages` 0 of 44 and
`frame-carries-radiance` 256/256 pass in every arm. No pixel that was above luma 32
falls below 8, and the change is a darkening in 69.99 % of pixels against 0.11 %
brighter with a maximum brightening of 2 luma. Bucketed by the baseline pixel's own
brightness, the darkening is concentrated where it should be: mean −25.2 luma in the
64-95 bucket against −6.6 in 128-159. The bonsai's contact shadow goes 82.5 → 50.8;
lit ground goes 134.2 → 116.9, which is its GI bounce no longer being fed by sunlit
subsoil, and the skip route independently puts it at 117.9.

**No frame-time claim.** The two identical `baseline` runs measured 24.635 ms and
21.506 ms `medianSubmitToFence`, a 13 % spread on a byte-identical frame, so this
lane's timing is noise-dominated even where its pixels are exact. Every arm here
falls inside that spread. Time the guard on a timing lane if it matters.

### Scored against the plate: the guard loses, and why that is the finding

`npm run gate:hero-fidelity` turns the "it looks better" judgement into a number.
Four numbers were needed rather than two, because **the shipped grade is fit to the
bug**: exposure was solved against a frame whose subsoil was leaking sun, so
scoring the guard against that grade measures the stale compensation. Each arm was
therefore also re-solved with `tools/solve-hero-grade.ts` off its own scene-linear
raw. That solve is CPU-only, and the harness was verified exact — re-encoding a raw
under the *shipped* grade reproduces the smoke lane's own PNG byte-for-byte.

| arm | grade | mean ΔE₀₀ | grad octaves |
|---|---|---:|---:|
| committed baseline (Aug 5 frames) | shipped 0.2200 | 12.81 | 1.792 |
| **today's tree, lever off** | shipped 0.2200 | **19.44** | 2.422 |
| lever on | shipped 0.2200 | 22.01 | 2.657 |
| lever off, re-solved 0.3980 | own solve | **13.46** | 2.456 |
| **lever on, re-solved 0.4138** | own solve | **14.77** | 2.651 |

1. **The committed baseline does not reproduce, and not because of this work.**
   Today's tree scores 19.44 with the lever off. An independent gate run from the
   real tree (`artifacts/hero-fidelity/frame-rep0.png`, 09:47) scores 19.23, within
   the 0.25 floor of the snapshot these arms rendered from — and this harness
   reproduces **12.81 / 1.792 exactly, every region to 2 dp**, on the Aug-5
   baseline's own retained frames. So the scene regressed against the plate during
   a day of other work, and `docs/hero-fidelity-baseline.json` is stale. Most of it
   is the `ground` region: ΔE₀₀ 2.43 → 17.80 and gradient energy 0.3439 → 0.0273, a
   12x loss of detail in the backdrop sweep, which also makes it the worst gradient
   region in the frame at 4.05 octaves. That belongs to whoever moved the scene.
2. **The stale-grade regression is real and predicted:** +2.57 ΔE₀₀ at the shipped
   exposure, because the frame is now under-exposed by the share of the grade that
   was compensating for the leak.
3. **Re-solving does not rescue it.** Each arm at its own optimum still leaves the
   guard 1.31 ΔE₀₀ and 0.195 octaves worse, and worse in *every* region: coping
   +0.80, stones +1.85, canopy +3.45, pond +0.47. Note the off arm's own re-solve
   lands at exposure 0.3980 against the shipped 0.2200 — **0.86 stops** — so the
   grade is already stale for today's tree independently of this lever.
4. **No canopy win.** ΔE₀₀ 20.31 → 23.76 is the single largest regression. The
   lightness ratio the grade note calls inverted stays inverted: canopy − coping is
   −24.2 L\* off and −23.9 L\* on, against the plate's **+5.7**. A 0.3 L\* move on
   a 30 L\* gap has measured nothing, and the note's verdict — fill and bounce, not
   grading — stands.

**The interpretation matters more than the verdict.** A physically correct removal
of light makes the frame match the plate uniformly worse. That is the signature of
a leak that was doing the job of a term the scene does not have: global fill and
bounce. The plate is 55 % water this frame does not render at all, and the grade
note already has canopy fill as open work. So the guard is *right* and it should
land **with** a legitimate fill source rather than before one — it does not create
an under-lit scene, it reveals one. Until then the lever stays off, which is where
it defaults.

PNGs: `…/scratchpad/out/hero-baseline.png`, `hero-direct.png`, `hero-skip.png`,
`hero-direct-skip.png`, `hero-strict-receiver.png`, `hero-probe-steps-only.png`.
By eye the frame reads better rather than worse: the baseline's shadows are milky
and the terrain relief is washed out, and in the `direct` arm the tree gets a real
contact shadow and the contours come back. **But the plate disagrees** — see the
scored section above, which is the arbiter and says the guard loses 1.31 ΔE₀₀ even
with the grade re-solved. Contact sheets:
`…/scratchpad/out/score-off-resolved/contact-sheet.png` and
`…/scratchpad/out/score-on-resolved/contact-sheet.png`.

## Mechanism two, localised exactly (as found)

`directVisibility` takes four samples at `cellScale * {2.5, 6, 18, 54}` where
`cellScale = max(mappingCellSize) * exp2(radianceFloorLevel())`. The hero reports
`radiance 44 pages at level >= 3`, so the floor level is 3 and **the first sample is
2.5 x 8 = 20 finest cells away — 12.5 cm at 6.25 mm cells.**

For a voxel just under the terrain, every one of those four samples is already in
open air above the ground. The march never samples the ground between the voxel and
the sun, `1 - coverage` is 1 at every step, and solid ground receives *full* direct
sun. That is the residual ~21/255.

It is not an oversight so much as a mis-scaled trade, and the shader says so at
`lib/webgpu-svo-live-derived-builder.ts:998-1002`: "the four gather steps are
expressed in *receiver* widths, not in finest voxels ... a 2.5-cell first step from
an eight-cell block lands inside the block that emitted it". Preserving the ratio
of step to sample width is right for **radiance gathering**, where the hazard is
self-sampling; it is wrong for **occlusion**, where the hazard is skipping the
occluder. The same scaling fixes one and breaks the other.

The exact fix is also the cheapest: a fully solid voxel sees no sun, so
`directVisibility` should return 0 before taking any step. That is physically exact
rather than a tolerance, and it uses the fraction test already in scope.

**Note where that lands.** Applying it reproduces, for the direct-sun component,
what `FLUID_SVO_BURIED_RADIANCE_SKIP` did wholesale — so the two routes are
expected to converge, and the 22.22/255 frame is then the *correct* one, arrived at
independently. That is the answer, not a disappointment: the hero has been reading
over-exposed, and removing sun that solid ground was never entitled to should move
it toward the reference plate. Gate it on its own hash, diff and PNG.

## Two durable findings

Recorded because they cost six device runs and should not be rediscovered.

1. **The deep interior's distance gradient is irreducible.** No record-set radius
   recovers it — 2.3x the records moves occupied normal flips from 86.4 % to
   81.4 %, and 3.3x reaches only 76.5 % — and no occupancy-mask gradient recovers
   it either, because occupancy is uniformly 1 over exactly that population and its
   gradient is identically zero. The information is genuinely absent from anything
   stored nearby; only storing the distance itself reproduces it.
2. **Buried-voxel radiance is not a fiction in effect.** Whatever it ought to be,
   the hero's shipped frame depends on it for 22.22/255 of mean image value. It
   arrives by two paths, and both are now fixed at source: a gather origin that
   escaped the solid at coarse levels (`FLUID_SVO_SOLID_MARCH_OFFSET`, ~1.16/255,
   shipped) and a four-step coarse visibility march that never reached zero
   transmittance inside solid (`FLUID_SVO_SOLID_DIRECT_OCCLUSION`, 12.30/255
   against the post-offset baseline, gated).
3. **The occluder-skipping and self-sampling hazards need different scalings.** A
   march schedule expressed in receiver widths is right for gathering and wrong for
   occlusion, and no single schedule serves both — the measurement is 1.42/255
   recovered by rescaling against 12.30/255 by answering the question from the fine
   fraction instead. When a coarse structure cannot represent the distinction a
   query needs, tune the *query*, not the step length.

### Why the occupancy-mask normal is *not* the fallback

The drafted fallback — reconstruct the *normal* from the occupancy mask's
6-neighbour gradient instead of reconstructing the distance — cannot close this
gap, and the reason is the same one that killed the radius fallback.

The 86.4 % of occupied voxels that lose their normal are *deep* interior: the true
distance field there has a smooth non-zero gradient (distance to a surface several
cells away), but **occupancy is uniformly 1**, so a mask gradient is exactly zero
and yields the same `vec3f(0,1,0)` the saturated distance already yields. Same
flips, same leak, same hash. The mask carries no information the saturation did not.

The honest conclusion is stronger and more useful than another fallback: **the deep
interior's distance gradient is not redundant.** Nothing reconstructable from
nearby stored data recovers it, because the information genuinely is not there.
Only storing it does — and storing it for every occupied voxel collapses the cull
to roughly 1.3-2.4x.

Which leaves the actual choice, and it is Peter's, not this document's:

1. **Accept the difference.** 11.64x memory for a 0.59/255 mean, 7/255 peak shift
   whose physical meaning is nil in both arms — the radiance of a buried voxel is a
   fiction either way, and the f32 answer is the incumbent rather than the correct
   one. This needs the byte-identical bar relaxed to a pixel-diff bar, explicitly.
2. **Remove the leak at its source, not in storage.** If the radiance build skipped
   fully-interior voxels — coverage 1 with every neighbour occupied — their radiance
   would be zero in *both* arms and the reconstruction could not leak. That would
   make the banded layout hash-identical *and* is defensible on its own terms, but
   it changes the shipped frame by itself, so it is a separate proposal needing its
   own gate.
3. **Store the whole occupied set** and take ~1.3-2.4x instead of 11.64x.

A useful side result: the device puts the stencil share at **6.64 %** where the
CPU census predicted 6.78 % at the same depth. The replay and the arena agree to
2 %, which is the first cross-check either measurement has had.

## What it delivers

By the shipped encoder (`bandedLeafBytes`), size-class rounding, leaf palette,
identity index and dense escapes all charged. Each geometry width is quoted
against **its own matched dense baseline** — a dense voxel is geometry plus a
4-byte owner word, so `f32x2` faces 12 B a voxel, `f16-unorm8` faces 8 and
`snorm8-unorm8` faces 6. The last column is the honest total change from today.

| scene / depth            | `f32x2` B/leaf | vs 12 B | `f16-unorm8` | vs 8 B | `snorm8-unorm8` | vs 6 B | **vs today's 12 B** |
|--------------------------|---------------:|--------:|-------------:|-------:|----------------:|-------:|--------------------:|
| `hero-garden-hose` d0    |            463 |  13.28x |          310 | 13.21x |             234 | 13.14x |          **26.28x** |
| `hero-garden-hose` d1    |            586 |  10.49x |          375 | 10.91x |             270 | 11.37x |          **22.73x** |
| `hero-garden-hose` d2    |            539 |  11.40x |          352 | 11.64x |             259 | 11.88x |          **23.75x** |
| `garden-svo-lighting` d0 |          1 188 |   5.17x |          676 |  6.06x |             421 |  7.30x |          **14.61x** |
| `garden-svo-lighting` d1 |          1 085 |   5.66x |          622 |  6.58x |             391 |  7.86x |          **15.71x** |
| `garden-svo-lighting` d2 |            859 |   7.15x |          508 |  8.06x |             333 |  9.23x |          **18.45x** |

**Zero dense escapes at every depth on both scenes.** Unifying the palette
removed the escapes the split design had (48 leaves on the lighting study at
depth 0) *and* cut bytes a leaf by 28-30 % at depth 2, because the owner field it
deleted from records was the thing pushing crowded leaves past their dense form.

Two things the shape of this table says. On the hero the ratio barely moves with
record width (13.28x / 13.21x / 13.14x at depth 0) because the fixed 144 bytes
dominate a scene whose median leaf holds no records at all — narrowing geometry
buys the hero almost nothing once the band cull has run. On the lighting study it
climbs steeply (5.17x → 7.30x) because records genuinely dominate there. The two
culls therefore help *different scenes*, which is a better outcome than either
helping the same one twice.

Leaf palette sizes at depth 2: 70 % of hero leaves that hold anything hold
exactly **one** identity (79 % on the study), and no sampled leaf needed more
than a 4-bit index. SP21's device census saw one leaf at 45 identities, which the
8-bit rung covers.

### The composed depth-3 figure, honestly

SP21's lane-only ladder (465 MB → 233 → 121 → 62 MB) and this cull's earlier
~84 MB are **overlapping claims about the same bytes** and must not be
multiplied. Under the unified layout there is no separate owner lane to project:
identity lives in the leaf palette and index, which `bandedLeafBytes` already
charges. So there is one number, `leaves × bytes-a-leaf`:

**The record is 4 B, not 2 B.** `snorm8-unorm8` was measured on device by SP20 and
rejected, and its failure is structural: a solid scene is mostly deep interior, so
4.5 M of the hero's 5.3 M voxels clamp at any band radius, their central
differences go to zero and `safeNormal` falls back to `vec3f(0,1,0)` — 3 299
normal flips against 5 for f16, and 104 802 of 368 000 pixels differing (28.5 %)
against f16's 9 368 (2.55 %, max 11/255). The shipped lane is one u32 a voxel:
f16 distance in bits 0-15, unorm8 fraction in 16-23, word-aligned. The 2 B arm
stays reachable so the older measurements reproduce, but it is not the default
here either.

Re-derived from the census rather than interpolated (`encodedBytesPerLeaf4`,
which is the f16 record width):

| arena                                | dense              | banded, f16 records |
|--------------------------------------|-------------------:|--------------------:|
| hero d2, measured (86 006 leaves)    | 352 MB (8 B/voxel) |         **30.3 MB** |
| lighting d2, measured (37 897)       | 155 MB             |         **19.3 MB** |
| **hero d3, extrapolated (227 168)**  |     **930 MB**     |         **80.0 MB** |

352.0 B a leaf at depth 2, so 227 168 × 352 B = **80.0 MB — a measured 11.64x**
against today's shipped 930 MB, and 17.45x against the 1.40 GB that stood before
SP20 landed. (My earlier "~84 MB" and SP21's "62 MB" were overlapping claims about
the same bytes; under the unified layout there is no separate owner lane to
project, so this is the one number.)

The 144-byte fixed part is 227 168 × 144 B = **32.7 MB, 41 % of the 80 MB**. At
this record width the masks are approaching the arena, so narrowing *them* — a
coarser occupancy summary, or deriving the record mask from the occupancy
boundary — is the next question after this one.

## The blocker: the occupancy predicate

Occupancy is defined today by `fraction > 0`. The voxeliser writes a material
identity only where the fraction is positive
(`lib/webgpu-sparse-scene-proxies.ts`, `rebuildDirtyBrickPayload`), so
`material != 0` and `fraction > 0` are **one predicate**. An interior voxel that
stops carrying an explicit fraction stops being occupied.

`bandedLeafOccupied` is where it moves to. Consumers that must move with it:

- `rebuildDirtyBrickPayload` — writes the identity, gated on `primitiveFraction > 0.0`.
- `finalizeDirtyBricks` — builds the per-leaf occupancy word from `sceneMaterial == 0u`.
- `lib/webgpu-svo-live-derived-builder.ts` — `leafOpacityAt` and the radiance
  build, through the `sceneCoverage` / `sceneIdentity` lane accessors.
- `lib/webgpu-svo-dry-scene.ts` — the primary and visibility marchers. **Five
  sites read the owner half**, including `dryLodCellSolid`, where it is the LOD
  tier's only solidity test, and `traceLeafPayload`, where it becomes
  `primitiveIndex = owner - dry.metadata.y`.
- `lib/webgpu-svo-brick-occupancy.ts`.

**Nothing may truncate the lane word.** It is a packed pair
`(ownerId << 16) | materialId`, not a material id. The palette stores the pair
whole and hands out an index; a test asserts the owner half survives the round
trip for owners up to `SPARSE_BRICK_NO_OWNER`.

`voxelOffset` is exactly `leafIndex × 512` (SP21 verified 693/693 aligned, sole
GPU writer `lib/webgpu-sparse-brick-topology-mutation.ts:200`), so
`leafSlot = voxelOffset >> 9` addresses the per-leaf structures at all read sites
with no signature changes.

## Sequencing

Every consumer file is currently modified by SP20, and the layout needs a row in
`SPARSE_BRICK_PAYLOAD_PROFILES`.

- **B0 — landed.** Reference encoder/decoder and tests. Layout, popcount
  indexing, global and leaf palettes, index widths, size classes and **both
  record-set rules** are pinned and executable, and the census computes its
  headline numbers *through* the encoder so the quoted prize and the code cannot
  drift. `SvoBandedRecordSet` defaults to `{ radii: 4 }`, the conservative arm —
  `"stencil"` is opt-in, because a lever that defaults to the arm nobody has
  cleared is how SP24's "baseline" hash became SP20's snorm8 arm.
- **B0.5 — the gate on B2's shape, not on its code.** The record-set rule must be
  settled on device before the arena is built around it, because the two rules
  differ by roughly a factor of two in records a leaf and the narrow one is the
  one SP20's data implicates. `tmp/sp22/banded-normal-fidelity.ts` is that
  measurement. **B2 must not be implemented before it reads clean.**
- **B0.5b — isolate the reconstruction from the structure.** The normal probe
  gives the *mechanism*; only a frame hash gives the outcome, and there is a way
  to get one without building the arena first. Keep the dense arena exactly as it
  is and change only `solidDistance` in the derived builder: return the stored
  value where the voxel or an axis neighbour is band, and the signed saturation
  otherwise. That is the banded layout's reconstruction, byte for byte, with none
  of its allocator, masks or palette. If the hash holds, the narrow record set is
  cleared and B2 is pure engineering; if it moves, B2's shape is wrong and no
  amount of allocator work would have found that out. Cheap, decisive, and it
  should run before either.

  Note this also separates the two mechanisms SP20's snorm8 result conflates.
  snorm8 quantised the *band* as well — 8 bits over ±4R is a quantum of R/32,
  coarse where the central differences live — so its 28.5 % pixel diff is band
  quantisation *plus* interior clamping. This layout stores the band at f16, so it
  inherits only the second. Whether that half alone moves the frame is exactly
  what B0.5b measures, and it is not answerable by reasoning from SP20's total.
- **B1 — ready to start, independent of B0.5.** The occupancy mask as a per-leaf
  side buffer and the predicate move, no storage saving yet, behind
  `FLUID_SVO_LEAF_OCCUPANCY_MASK` defaulting **off**. Frame hash must be
  identical; the arena grows 64 B a leaf (1 %). This does not touch the
  reconstruction, so it is unaffected by whichever record set wins.
- **B2 — after B0.5.** The banded record arena and the identity palette,
  `voxelOffset` becoming a leaf slot, and the suballocator. Behind
  `FLUID_SVO_LEAF_PAYLOAD=dense|banded` defaulting to `dense`, both arms
  measured, with a cross-decode provenance probe on the model of
  `tmp/sp20/format-provenance.ts` so a lever that did not take effect is provable
  rather than assumed — mandatory here, because this moves occupancy, fraction,
  distance *and* identity to a new decode path at once.
- **B3.** Retire the dense arm if the hash holds.

### Three edits inside SP20's region (SP21's flags, folded in)

1. `CHANNEL_FORMAT_BITS` (`lib/sparse-brick-octree.ts:77`) needs a `u16` member.
2. `sparseBrickLaneStrideBytes` (`:106`) floors at one byte and must support
   sub-byte lanes, since identity indices are 1/2/4 bits.
3. `sparseSceneProxyVoxelizationShaderFor`
   (`lib/webgpu-sparse-scene-proxies.ts:1336-1340`) makes the payload binding
   atomic **only** when `sceneGeometryFormat === "snorm8-unorm8"`. Sub-byte
   identity indices share words at any geometry width, so atomicity must become
   independent of that format.

The allocator is the largest remaining engineering risk. The size-class ladder is
finer in the middle than a doubling one because that is where the leaves are; the
alternative — a bump arena recompacted on the topology epoch, which the tree's
`BRICK_RELOCATING` lifecycle already contemplates — avoids free lists entirely
and is worth costing first, since the dry render path's scene is static between
editor drags.

## Verification bar

`hero-garden-hose` frame hash `0xa949ae24`, byte-identical, on both arms, with
arena and frame time reported for each — the precedent the `dry` payload profile
set (155 MB → 116 MB, 32.6 ms → 14.1 ms).

**Budget for this gate rather than assuming it.** Neither this work nor SP21's
has yet measured a frame hash, and the composed structure changes how *every*
voxel in the world is read: occupancy, fraction, distance and identity all move
to a new decode path at once. A hash that moves will need bisecting between the
four, which is why B1 (predicate only) is separated from B2 (storage) at all.

## Honest caveats

- No number here has touched a GPU. The geometry census replays the voxeliser,
  reusing `sparseScenePrimitiveSignedDistance` and
  `sparseSceneNeedsConservativeSampling` — the same CPU functions the exactness
  contract in `tests/webgpu-svo-primitive-exact.test.ts` holds against the WGSL —
  with the terrain fold mirrored by hand from the shader. SP21's owner census
  *did* read the published lane off the device, which is why the two are quoted
  as separate evidence rather than blended.
- The two censuses do not agree on leaf counts everywhere. Hero depth 0 matches
  exactly (10 369 both); hero depth 1 is 22 834 here against 22 900 there. The
  lighting study reports 693 published leaves at *both* depths in SP21's device
  run against 1 815 / 7 843 / 37 897 planned here, so its refinement lever
  appears not to have taken effect on the device. **The identity statistics from
  the lighting study should be re-taken before they are relied on**; the hero's,
  which agree at depth 0, are the sound ones.
- The replay binds *every* overlapping primitive where the GPU binds at most
  `candidatesPerBrick()` and drops the surplus, so in a crowded leaf it reports
  slightly more band than the device produces — conservative against the prize.
- 3 000 leaves of 86 006 at hero depth 2 is a 3.5 % sample. The depth-3 arena
  figures are an extrapolation from depth 2 at depth 2's bytes-a-leaf rate; the
  227 168 leaf count is the one this work was briefed with, and depth 2's
  measured 86 006 implies a consistent 2.64x per-level growth.

---

# B1/B2 storage: what landed, and the one number that is still unmeasured

**Read this section before the ones above.** The plan above is unchanged and still
correct; this records where the code actually is, and corrects one thing the plan
implies: the 80 MB depth-3 figure **still has never existed**. It is a census
projection. No depth-3 arena has been measured, by this work or any other.

## State, honestly

| item | state |
|---|---|
| Three prerequisite edits (`u16`/sub-byte `CHANNEL_FORMAT_BITS`, sub-byte `sparseBrickLaneStrideBytes`, atomic payload independent of geometry format) | **landed**, 11 CPU tests |
| Lane machinery for the banded arena (5 new lanes, per-leaf and suballocated elements, `productBytes`) | **landed** |
| **B1** — occupancy mask lane + the predicate move | **landed, ungated.** Allocates and writes; the predicate is moved at `finalizeDirtyBricks`. **No frame hash taken.** |
| **B2 writer** — encoder pass, bump allocator, masks, header, leaf palette, identity index, records | **landed, unverified on device beyond compilation.** |
| **B2 readers** — the eleven sites that read the owner and geometry lanes | **not started.** Enumerated below. |
| Cross-decode provenance probe | **written, never completed a run.** `tmp/sp26/banded-provenance.ts` |
| Measured depth-3 arena | **unmeasured.** |

Every lever defaults to the arm that shipped, verified by assertion:
`FLUID_SVO_LEAF_PAYLOAD=dense`, `FLUID_SVO_SCENE_GEOMETRY=f16-unorm8`,
`FLUID_SVO_BANDED_RECONSTRUCTION` **off**. The reconstruction default was
deliberately *not* flipped: the instruction to flip it is conditioned on B2's
storage being landed and measured, and it is landed but not measured. Flipping it
now would pay the 0.54/255 for a saving that has not been demonstrated to exist —
which is exactly the ordering error the comment on
`svoBandedReconstructionEnabled` warns against.

## One lever, three rungs

`FLUID_SVO_LEAF_PAYLOAD=dense|occupancy|banded` replaces the plan's two levers
(`FLUID_SVO_LEAF_OCCUPANCY_MASK` and `FLUID_SVO_LEAF_PAYLOAD=dense|banded`),
because two levers can express `banded` *without* the occupancy mask and that is
not a legal state — occupancy is what makes a voxel storable nowhere.
`occupancy` is the bisection rung: mask allocated and written, predicate moved,
both dense lanes retained, nothing saved.

## The only numbers measured on device

`hero-garden-hose`, environment refinement depth 0, 20 maintenance frames,
**11 395 published leaves / 5 834 240 voxels** (the plan's census says 10 369 at
this depth; the tree has moved since):

| mode | payload arena | B/voxel |
|---|---:|---:|
| `dense` | 60.5 MiB | 8.0000 |
| `occupancy` | 61.5 MiB | 8.1250 |

+1.65 % for the mask, against the +1.56 % arithmetic predicts (64 B on a 4 096 B
leaf) — the difference is 256-byte lane padding. **The `banded` row of that table
was never produced**: the probe's first run hit a per-voxel `new Uint32Array` in
its decode loop and was killed at the ten-minute mark; the second was terminated
when the session ended. The allocation is hoisted now and the probe is ready to
run.

**All 16 (profile × geometry format × leaf payload mode) shader arms compile on
Dawn** — `tmp/sp26/compile-arms.ts`, which throws if it compiles fewer than 8.
That is the whole of B2's device verification so far.

## What the storage actually is on device

Five lanes, all byte ranges of the one payload arena, so no consumer gains a
binding:

| lane | elements | stride | per leaf |
|---|---|---:|---:|
| `sceneOccupancy` | voxels | **1/8 B** | 64 B |
| `sceneRecordMask` | voxels | **1/8 B** | 64 B |
| `sceneBandedHeader` | **leaves** | 16 B | 16 B |
| `sceneBandedBlob` | **leaves** | 96 B capacity, bump-allocated | ~16 B used |
| `sceneBandedRecords` | voxels × 0.25 | 4 B, bump-allocated | ~136 B used |

The fixed part is **144 B a leaf exactly**, pinned against
`SVO_BANDED_LEAF_FIXED_BYTES`. `voxelOffset >> 9` is the leaf slot at every read
site, so no signature changed.

### Four deliberate departures from the reference encoder, each priced

1. **Leaf-palette entries are the packed `(ownerId << 16) | materialId` word
   inline (4 B), not a `u16` slot into a scene-global table.** The interning is
   the *only* part of the layout that needs a GPU hash map, and it buys 2 B per
   palette entry: at the measured 1.457 entries a leaf that is 2.9 B a leaf,
   **0.66 MB of the projected 80 MB, 0.8 %**. The decode is unchanged in shape and
   the word is never truncated. Every arena figure charges the 4 B.
2. **A bump allocator, no size classes.** Size classes exist to stop a *free list*
   fragmenting; a bump arena has none, so records cost their exact count and the
   measured arena should come in *under* the census projection, which charges the
   class. The cursors reset when a revision rebuilds every leaf
   (`dirtyCount >= publishedLeaves`), which covers every full build. A **partial**
   rebuild leaks its old block — deliberately, because leaves this chunk is not
   touching still address it — and the leak shows as the high-water mark climbing
   until the overflow flag fires. Recompacting on the topology epoch, which
   `BRICK_RELOCATING` already contemplates, is the fix.
3. **`banded` requires `f16-unorm8`.** `snorm8-unorm8` would put two records in one
   word written by two threads of one workgroup, and it is measured-and-rejected on
   its own terms; `f32x2` is the *geometry* axis's rollback and composes with
   `dense`. Carrying both would double the arms to verify for a rollback nothing
   needs.
4. **The encode pass reads the dense lanes back rather than re-voxelising.** This
   is the one that has a real cost, and it is why `banded` currently allocates the
   *provenance* shape (dense lanes retained) rather than the product shape. See
   below.

### One workgroup owns a whole leaf

`rebuildDirtyBrickPayload` runs 512 voxels across two workgroups and **no part of
this encoding is expressible that way**: the band's stencil dilation, the leaf
palette, the identity ranks and the record ranks are all whole-leaf reductions.
`encodeBandedLeaves` is 256 threads, two voxels a thread, one workgroup a dirty
brick, with a fourth indirect-dispatch triple at state word 17.

Owning the leaf also removes **every atomic** from the mask writes: 32 voxels share
a mask word and all 32 are lanes of this workgroup, so the word is assembled in
workgroup storage and stored once. (The *dense* rebuild pass still needs atomics
for its per-voxel occupancy bit — that is prerequisite 3.)

Reading the dense lanes back has one property worth keeping even after the fusion:
the band predicate is computed from the **stored** fraction, so the encoder and
every reader agree about which voxels are band. A voxel whose true coverage is
0.999 stores 255/255, reads back as 1.0, and is correctly *not* a band voxel on
both sides. A fused encoder must reproduce that rounding or the two will disagree.

## The provenance shape, and why the arena is reported twice

`banded` today allocates dense lanes **and** banded lanes
(`retainDenseLanesForProvenance`), so:

- every existing consumer keeps reading the dense lanes and the frame is untouched;
- the banded bytes can be **cross-decoded** against the dense ones — the same voxel
  read both ways, in occupancy, fraction, distance and identity, over millions of
  voxels. That is a stronger check than a frame hash *for the storage*: a hash
  proves the pixels agree, this proves the bytes do, channel by channel. It is also
  mandatory here, per the plan, because four channels move at once.
- `tree.payloadProductBytes` is what the same measured scene costs once the dense
  lanes go, resolved by the *same function* with the flag cleared rather than by
  subtracting lane sizes.

The cross-decode is exact by construction on three of the four channels: the
record write is a `packSceneGeometry(sceneDistanceOf(...), sceneFractionOf(...))`
round trip through the same codec, so recorded voxels are **bit-identical**, and an
unrecorded voxel's fraction is exactly 0 or 1 in both. Only `solidDistance` outside
the record set differs, and that is B0.5b's already-gated population.

## Remaining work, in order

**R1. Run the probe.** `DEPTHS=0,1,2 npx tsx tmp/sp26/banded-provenance.ts` under
`tmp/sp6/gpu.sh`. It throws on: any overflow flag, a record-mask popcount that
disagrees with the allocator's handout, a zero population, a leaf count that
matches across two depths, a readback that overruns, and any cross-decode
mismatch. It prints the **measured** arena as `fixed + blob + records` at the
published leaf count. That is the figure this task was for, and it is one run away.

**R2. The reader cutover — eleven sites.** All of them read the owner lane as
`payload[base + voxel]` and become `bandedIdentity(voxel)`; the geometry reads go
through `bandedRecorded`/`bandedRecordWord` plus the reconstruction.

- `lib/webgpu-svo-live-derived-builder.ts` — 4 identity sites
  (`payload[params.laneOffsets.w+voxel]`), plus `lane.solidDistance` and
  `lane.sceneCoverage` in `derivedLaneAccess`, which is 2 expressions covering
  every geometry read in both the build and the feedback shader.
- `lib/webgpu-svo-dry-scene.ts` — 5 sites (`materialOwners[...]`, lines ~2604,
  2869, 2897, 5008, 5176), 3 bind sites, and the `arrayLength(&materialOwners)`
  guards, which must become a voxel-count compare. The dry scene reads **only the
  owner lane**, never scene geometry — that is the single biggest reason this is
  tractable.
- `lib/webgpu-svo-brick-occupancy.ts:194` — 1 site.
- `lib/webgpu-octree-sparse-bricks.ts:1203` — the inspection shader.
- `lib/webgpu-voxel-debug.ts:339` — a binding.

The mechanical part is rebinding those slices to the **whole payload buffer** and
carrying the five lane word offsets in each shader's own uniform block; the dry
scene's is `SVO_DRY_SCENE_PARAMS_LAYOUT`, which is memoized and the fiddliest part
of the job. Under `dense` every binding and every expression must stay exactly what
it is.

**R3. Fuse the encoder into `rebuildDirtyBrickPayload`** so the dense lanes can be
dropped. Same arithmetic, staged in the workgroup storage `encodeBandedLeaves`
already declares; no new algorithm. Until this lands, `banded` cannot be the
product shape.

**R4. Gate.** `occupancy` against `dense` (must be byte-identical — a hash that
moves is a bug in the predicate move and nothing else), then `banded` against
`dense`-with-`FLUID_SVO_BANDED_RECONSTRUCTION=1`. The second pairing is the
important one: with the reconstruction on in *both* arms, the storage A/B is pure
and any difference is a storage bug rather than the known fidelity change. Flip
`FLUID_SVO_BANDED_RECONSTRUCTION` on by default with that hash, not before.

## Four WGSL facts that each cost a compile round trip

Recorded because they are not obvious and the encoder pass hit all four.

1. **A module-scope `const` cannot reference `var<uniform>`.** Every lane base in
   the shared codec is a `fn ...Base()->u32` for this reason, and a test asserts no
   `const BANDED_*BASE` exists.
2. **An early `return` on a storage-derived bound makes every later
   `workgroupBarrier` non-uniform control flow**, even when the bound is provably
   the same for all lanes — the analysis cannot see through a storage load. The
   encoder carries an `encoding` flag and every barrier sits at the top level. This
   is load-bearing, not style.
3. **`active` is a reserved keyword.**
4. **Making the payload binding atomic for one lane forces every other lane's store
   through `atomicStore`.** The `f32x2` and `f16-unorm8` geometry stores were plain
   assignments; they now go through `writePayload` and the `dense` expansion is
   unchanged to the character.

## Two test-visible sizes that moved, with the arithmetic

`SparseSceneProxyVoxelizer.allocatedBytes` 532 → **592** on every arm including
`dense`, and the buffer sizes `[96, 304, 36, 96]` → `[96, 320, 48, 128]`: the
parameter uniform grew 32 B for the banded lane offsets and their capacities, the
maintenance state block grew 16 B for a fourth dispatch triple, and the argument
buffer grew 12 B to carry it. A fixed-capacity arena whose size depends on a lever
is a size no test can pin, so all three are allocated on every arm.

`SPARSE_SCENE_MAINTENANCE_STATE_WORDS` also moved: `bandDispatch` is word 17,
contiguous after the other three triples because all four are copied in one
`copyBufferToBuffer` whose offsets are differences from `binDispatch`, which pushed
`chunkCursor` 17 → 20 and `maximumBrickCandidates` 18 → 21. Every reader uses the
named constants.

## Pre-existing CPU failures at this commit, so they are not attributed here

Baselined at `git archive HEAD`: `tests/sparse-brick-octree.test.ts` #9 (needs a
device), `tests/svo-brick-occupancy.test.ts` #8 (needs a device),
`tests/sparse-brick-dry-payload.test.ts` "the census reproduces the measured
hero-garden distribution" (staged-new, concurrent), and
`tests/webgpu-svo-live-derived-builder.test.ts` #1 plus
`tests/pond-vessel.test.ts` and `tools/shape-lab-canopy.ts` type errors — all
three from concurrent edits to files this work never touched.
