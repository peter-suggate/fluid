# Sparse CM12 Figure 6 paging investigation handoff

## Status

The investigation changes were reverted at the user's request. This document is
the only retained artifact. No proposed solver or test change described below is
currently applied.

The original failure was reproduced in Dawn. In the default activity-policy
configuration, the Figure 6 liquid ball falls from approximately `maxY=120` to
`maxY=68`, then remains at about `y=56..68` through step 80 instead of crossing
the unauthored rows and reaching the shallow pool.

## Reproduction

Figure 6 is a 128 cubed scene with a shallow pool at `y=0..8` and an analytic
sphere centred at `y=104` with radius 17. Use Dawn/Metal and the normal balanced
activity configuration:

```text
brickFineResolution: 8
resolutionMode: adaptive
selectorMode: activity
gammaDiffusion: on
surfaceFineRings: 1
timeStep: paper
```

A useful regression test should:

1. Construct `cm12-figure-6` with `adaptiveMassMethod`.
2. Read `globalFineLevelSetSource` at generation zero.
3. Advance 30 paper steps.
4. Assert that runtime topology pages were published.
5. Assert accepted cell and row indirect counts remain non-zero.
6. Assert the maximum liquid Y falls by more than 20 fine cells.
7. Assert published negative-phi samples exist in every row `y=8..23`, proving
   that the falling liquid crossed the page rows immediately above the pool.
8. Assert no insertion faults, capacity faults, or uncaptured WebGPU errors.

The experimental version of that test completed in about 21 seconds on Metal.
Run Dawn serially and under the repository lock; do not overlap it with another
Dawn or browser process.

## Primary cause of the original stall

SparseWorld runtime frontier growth is rooted in accepted B8 leaves. Activity
mode deliberately applies `initialSurfaceCoarseningBiasRings: 1`, so the outer
support of the authored Figure 6 sphere starts at B4. Once the ball reaches the
edge of its generation-zero authored support, there is no accepted B8 frontier
root from which to demand and publish the next dry world page. The apparent
floor under the ball is therefore a topology/paging boundary, not a gravity or
pressure failure.

Forcing `initialSurfaceCoarseningBiasRings: 0` proved this diagnosis: Figure 6
published runtime pages, crossed rows `y=8..23`, and passed the 30-step drop
test. A global force-to-zero is not an acceptable fix, however. It changed the
generation-zero topology of unrelated planar scenes and failed the canonical
gate:

- `hydrostatic-adaptivity`: expected resolution ladder `[1,1,2,4]`, observed
  `[1,2,4,8]`.
- `long-dam-far-wall`: expected 276 generation-zero tiles, observed 208.
- `mini64-performance`: atlas construction rejected a face between bricks 460
  and 461 for exceeding the 2:1 grading contract.

A promising narrow policy is to retain the existing one-ring activity bias for
broad planar/box liquid, but use zero bias for compact curved analytic initial
volumes (`sphere`, `hemisphere`, or `cylinder`). This was sketched but not fully
validated before the revert. It should be expressed as a topology/frontier
requirement rather than a Figure 6 scene-name special case.

## Transaction defect exposed after paging was enabled

Once the sphere had B8 frontier roots, the first large page-growth transaction
exposed a separate, concrete failure in `reserveShadowWorklistRange` and
`validateAndAuthorizeShadowTopology`.

Many leaf workgroups concurrently reserve ranges in the shadow worklists. The
reservation used a weak compare-exchange loop capped at 256 attempts. Under the
first large Figure 6 growth event, a valid reservation could exhaust that retry
budget despite sufficient capacity. It set the shadow phase to failure (`3`)
and left a partial list. Later, `validateAndAuthorizeShadowTopology` did not
require the phase still to be buildable (`1`) before authorization, so it could
authorize and commit the partial counts and zeroed triplet lists.

Observed receipt sequence around the failure:

- Eight runtime pages were published.
- Accepted header counts reported roughly 232,192 cells and 485,964 rows.
- Accepted cell/row triplet work counts were zero.
- The following transport step therefore ran with live packet lists but without
  the matching accepted-clear work and reused stale scratch; density/mass then
  exploded.

The experimental repair used one `atomicAdd` ticket per reservation, retained
the existing pre-write capacity check, and made authorization return unless the
shadow phase was exactly `1`. Afterward, accepted indirect work remained
non-zero through page growth. This is a general paging/tiling fix worth landing
independently, with source-contract tests for both properties.

## Other concrete defects found

These were individually plausible and testable but should be landed and gated
separately from the initial-topology policy.

### Dynamic compact-row semantic mapping

The compact dynamic row page physically stores:

```text
packed, metadata, distance, dual, center-x, center-y, center-z
```

The semantic row planes are:

```text
packed, metadata, dual, area, distance, exterior-phi, center-x, center-y, center-z
```

The existing pair of generic decrement rules maps semantic dual and distance
incorrectly for dynamic one-sided frontier rows. The explicit mapping tested in
the experiment was:

```text
semantic 2 (dual)     -> stored 3
semantic 4 (distance) -> stored 2
semantic >= 6         -> stored plane - 2
```

This is general dynamic-page ABI corruption and is a strong candidate for other
paging symptoms.

### Dynamic TEI selector-bank lifecycle

Frontier pages bypass the host topology-delta replay. The existing frontier TEI
compiler writes only the currently accepted bank and skips inactive pages.
After a selector flip or leaf-ID recycle, the other bank can therefore expose a
stale active record for the page's old coordinate.

The experimental change mirrored every complete dynamic page into both TEI
banks and wrote inactive descriptors for retiring pages before ID reuse. This
needs a runtime regression that forces at least two selector flips and a page
retirement/reuse cycle, not only a shader-source test.

### Gamma signed fixed-point range

Two signed `i32` overflow paths were found while following later density spikes:

- Horizontal D4 averaging sums up to eight gamma values. At the legal gamma
  maximum, `4096 * 65536 * 8 == 2^31`, which wraps the signed accumulator.
- Gamma diffusion uses the physical-mass fixed-point scale even though gamma is
  an intensive quantity with a legal range up to 4096. A coarse B1 cell can
  overflow the signed receipt before accumulation.

The experiment used one-eighth of the normal fixed scale for D4 gamma, a
gamma-specific physical receipt scale divided by `CM12_GAMMA_MAX`, and clamped
gather/finalize gamma to `[0, CM12_GAMMA_MAX]`. Gamma then remained around 2.5
instead of blowing up. These changes should be reviewed against precision and
conservation requirements and tested independently.

### Authored/dynamic face seam

Dynamic pages expose unit B8 faces. `hostExteriorRowSuperseded` will not replace
a coarse authored exterior row with a dynamic B8 row, while ordinary 2:1
closure can leave the authored neighbour at B4. The experiment forced an
authored unit-span neighbour touching an accepted dynamic page to B8. This is a
reasonable representability rule, but it did not by itself explain the
Figure 6 stall and needs a targeted seam-connectivity test before landing.

## Recommended implementation sequence

1. Add the 30-step Figure 6 Dawn regression first and confirm the original
   `maxY` stall.
2. Land the shadow reservation/authorization fix independently. Test that a
   failed phase cannot be authorized and that contention cannot fail a valid
   in-capacity reservation.
3. Correct the compact dynamic-row semantic mapping with exact plane tests.
4. Add a TEI two-bank flip plus page-retirement/reuse runtime test, then correct
   the frontier compiler.
5. Introduce the narrow compact-volume generation-zero B8-root policy and run
   the Figure 6 test plus `hydrostatic-adaptivity`, `long-dam-far-wall`, and
   `mini64-performance` immediately.
6. Investigate the gamma arithmetic as a separate numerical hardening change.
7. Run the complete mandated gate:

```bash
npm run test:dawn:sparse-cm12
```

## Gate receipt from the exploratory branch

The full gate was stopped after 180 seconds because the global bias experiment
was already known invalid. Before stopping, these lanes passed:

- symmetric expansion
- mini32 correctness
- min8 region surface
- mini32 performance (median about 28.8 ms, under the 40 ms limit)
- mini64 min8 surface
- tall-cells hills far wall
- live rigid-body coupling
- live liquid injection

The three semantic failures listed above were caused by the over-broad initial
bias change. The final outside-tank symmetric-collapse lane exceeded its timeout
by about 48 ms while the suite itself was already over budget; treat that as an
unconfirmed timing result, not a diagnosed correctness regression.

