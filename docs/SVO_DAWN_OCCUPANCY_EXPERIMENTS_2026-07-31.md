# SVO occupancy experiments — Dawn / Metal / M1 Max

Date: 2026-07-31

## Controlled baseline

- Scene: `garden-svo-lighting`
- Resolution: 660×662
- Renderer: split shading, cone scale 0.25, full-resolution relight
- Timing: Dawn submit-to-`onSubmittedWorkDone`, two encodes per sample
- Samples: 4 warmups, 20 measured cycles
- Baseline: **12.369 ms median**, **13.492 ms p95**, **12.396 ms
  min/max-trimmed mean**
- Configured image hash: `0x80bb2b26`
- G-buffer hashes: `0x741d3571`, `0xa37539f9`, `0x827d767e`

The source tree was dirty before this work. These are controlled same-source
A/Bs, not comparisons with the stale checked-in image fingerprint.

## Results

Positive percentages are faster than the 20-cycle baseline. A p95 dominated by
an external GPU stall is marked rather than interpreted as shader behavior.

| Experiment | Median ms | Trimmed mean ms | Median change | Output | Finding |
|---|---:|---:|---:|---|---|
| Baseline | 12.369 | 12.396 | — | reference | Control |
| One-cone-per-lane fan-out | 15.071 | 14.829 | −21.9% | exact | More dispatches and intermediate texture traffic cost more than the removed loop divergence |
| Inline incoherent cone receivers | 15.262 | 14.702 | −23.4% | exact | 42.32% of receivers are boundaries; inlining makes most 8×8 groups pay retraversal and raises coherent-kernel register pressure |
| Blit-clear boundary queue | 12.357 | 12.340 | noise | exact | Removes the one-thread reset dispatch, but reset is not a material share of the frame |
| Strip reduced-split counters | 12.138 | 12.341 | +1.9% | exact | Removing declarations and writes shortens private live ranges, but alone does not cross a large occupancy step |
| Narrow f16 accumulators | 12.852 | 12.718 | −3.9% | small f16 delta | Immediate f16→f32 expansion preserved the five long-lived f32 vectors and added conversions |
| Wide f16 lighting state | 12.223 | 12.221 | +1.2% | small f16 delta | Keeping long-lived values in f16 finally reduces state; one p95 sample was an unrelated 48.63 ms stall |
| Drop private GI page cache | 12.607 | 12.546 | −1.9% | exact | Saved private state is exchanged for dependent texture fetches; neutral until combined with the register diet |
| Compact 16-byte hierarchy | 12.600 | 12.611 | −1.9% | exact | Halving topology bytes does not shorten the dependent traversal and adds compact-node decode work |
| Compact + counter diet + wide f16 + no GI cache | 12.142 | 12.230 | +1.8% | small f16 delta | The register diet helps, but this remains 4.4% slower than the canonical winner |
| 16-entry traversal stack | 12.208 | 12.361 | +1.3% | exact | Safe for this depth-5 scene (worst deferred-sibling bound 15), but insufficient alone |
| 16-entry stack + counter diet | 12.008 | 12.242 | +2.9% | exact | Register reductions interact, confirming an occupancy threshold rather than additive instruction savings |
| 8-entry stack + counter diet | 13.715 | 13.969 | −10.9% | incorrect | Real stack overflow changes every image/G-buffer hash; a fallback population would be too large |
| Counter diet + 16 stack + wide f16 | 11.757 | 11.849 | +5.0% | small f16 delta | The combined private-state reduction reaches the occupancy step |
| **Above + no private GI cache** | **11.625** | **11.739** | **+6.0%** | small f16 delta | **Winner: +6.6% p95 and +5.3% trimmed mean** |

The winner preserves all three G-buffer hashes. Its configured radiance hash is
`0xaee3ba9d`; relative-luminance error changes from baseline mean
`0.01137399` / p95 `0.04870982` to mean `0.01137948` / p95 `0.04870834`.
The incremental mean delta is `0.00000549`.

## What each proposed direction taught us

1. **Cut private/register state.** This is the successful mechanism, but only
   after combining dead-counter removal, a depth-bounded stack, wide f16 state,
   and removal of the GI cache. Small isolated reductions mostly sit below an
   occupancy allocation boundary.
2. **Use f16.** Narrow arithmetic-only f16 is slower. The useful form keeps the
   long-lived private vectors in f16 and converts only at f32 consumers.
3. **Trade latency for bandwidth.** The already-retained direct `r32uint` page
   table removes up to 24 dependent directory probes and previously saved about
   5%. Removing the final private cache is useful only in the winning register
   stack. Compact 16-byte topology remains neutral: this scene's topology is
   only about 100 KiB and shrinking it does not shorten the dependent load
   chain. Authored child records are already contiguous.
4. **Repair the compact cone pass.** Inlining the boundary queue doubles down
   on eliminating its reset/consumer dispatches and is much slower because
   boundary work is not rare. A blit clear proves the one-thread reset is
   negligible. The queue is therefore useful compaction, not accidental
   overhead.
5. **Remove diagnostics and submission gaps.** Reduced split shaders now omit
   every private diagnostic counter and write. Full-rate/inline diagnostic
   shaders remain byte-identical, so overlays still work. Timing-only xctrace
   captures no longer request pass-encoder isolation; counter-attribution
   captures still do. On current Dawn, batching four frames per submission
   changes median cost from 12.596 to 12.228 ms/frame: about 0.37 ms/frame, not
   the older trace's alleged 9 ms gap. Encoder-isolation blits add about
   0.09 ms median (12.690 versus 12.596 ms).
6. **Reduce divergence.** Layered cone fan-out and boundary inlining both lose.
   They replace local divergence with extra whole-screen scheduling/texture
   traffic or contaminate coherent groups with expensive retraversal.

## Retained implementation

- The exact reduced-split counter diet is the renderer default. It is
  structurally disabled for full-rate and inline diagnostic shaders.
- All experimental arms remain independently selectable in the Dawn benchmark.
- The complete >5% occupancy stack remains opt-in because it requires the
  optional `shader-f16` device feature and uses a scene-depth proof for the
  16-entry stack:

```bash
FLUID_SVO_DRY_FRAME_STRIP_DIAGNOSTICS=1 \
FLUID_SVO_DRY_FRAME_SHORT_STACK=1 \
FLUID_SVO_DRY_FRAME_F16=1 \
FLUID_SVO_DRY_FRAME_DROP_GI_PAGE_CACHE=1 \
node --import tsx tools/run-webgpu-exclusive.ts \
  --import tsx tools/benchmark-svo-dry-frame-gpu.ts
```

Raw reports are in `/tmp/svo-occupancy-20260731/`.

## Independent scene repeat

The same complete stack was repeated on `hose-tank` (660×662, depth 6, 12
cycles, two encodes/sample):

| | Baseline | Occupancy stack | Change |
|---|---:|---:|---:|
| Median | 17.575 ms | 16.268 ms | **+7.4%** |
| P95 | 20.907 ms | 16.471 ms | **+21.2%** |
| Trimmed mean | 17.950 ms | 16.260 ms | **+9.4%** |

Reference and all three G-buffer hashes are exact. The configured f16 radiance
hash changes from `0xe7d07c0a` to `0x436a8738`; relative-luminance error moves
from mean `0.00823941` / p95 `0.03061019` to mean `0.00825876` / p95
`0.03061135`. This scene also proves that the actual depth-6 traversal does not
overflow the 16-entry stack, despite the conservative worst-case sibling bound
being 18.
