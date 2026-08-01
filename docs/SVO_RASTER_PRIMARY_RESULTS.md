# Raster-assisted SVO primary visibility — results

Implements `docs/SVO_RASTER_PRIMARY_HANDOFF.md`. Evidence lives under
`artifacts/render-raster-primary/`; `results.json` there is the machine-readable
form of every number quoted below.

## 1. Outcome

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| Phase 0 overdraw | median < ~4× | 1 (mean 1.01) front-to-back with tight proxies | pass |
| Phase A parity | hit/miss, depth, owner, normal match the traced primary | 0 silhouette-disagreement pixels; p95 luminance error 0 | pass |
| Phase B primary cost | beat the traced primary's 26.95 ms | **4.89 ms** GPU (brick raster + background), a 5.5× reduction | pass |
| Phase B occupancy | materially above 22% | 24.9% — moved 2.9 points, which is not material | **premise wrong**, see §7a |
| Phase C scaling | sub-linear cost growth to 2000² | 0.95 of linear, and *worse* than the traced arm's 0.87 | fails its thesis |
| Phase D end-to-end | ≥ 1.5× vs the 47.6 ms default, suite green | **1.71×** (49.62 → 29.01 ms) | pass |

Garden scene, 1500×1500, cones on, Apple M1 Max / Metal 3 / Dawn, quiet GPU.

```
canonical-parametric   49.62 ms median   (batched encodes; 47.65 unbatched)
raster-primary         29.01 ms median   (batched encodes; 29.39 unbatched)
```

The default configuration's 47.6 ms in the handoff reproduces exactly (47.65
unbatched), which is the strongest single check that these two arms are
comparable.

Read §7 before quoting any of these: **about 11 ms of every frame here is Dawn
CPU work, not GPU work**, which both inflates the absolute numbers and flattens
the resolution curves in §6.

## 2. What was built

A new traversal mode `"raster-primary"` that replaces the full-screen SVO
traversal megakernel with hardware rasterization of resident brick proxies:

- `lib/webgpu-svo-brick-raster.ts` — the standalone contract plus the emission,
  prefix-scan and scatter WGSL. Per frame: one compute pass emits a frustum-culled
  instance per resident occupied leaf, counting-sorts it into 1024 view-depth
  buckets, and writes a `drawIndirect` header.
- `lib/webgpu-svo-dry-scene.ts` — a terrain/background full-screen pass, then an
  indirect instanced box draw whose fragment calls the *production*
  `traceLeafPayload` unmodified. The raster stage only replaces the search that
  finds the leaf; the in-brick DDA is the same code the traced path runs.

All four primary planes (packed surface, identity/media, exact geometry, opaque
identity) are depth-tested colour attachments, 48 B/sample. That is 16 B past the
WebGPU default, so `lib/webgpu-device-limits.ts` now requests
`maxColorAttachmentBytesPerSample` and the renderer fails closed against the
granted device limit. Nothing in this graph writes primary geometry through an
untested storage texture.

Correctness rests on one property: **octree leaves partition space**, so a ray
meets each brick proxy over one interval and those intervals are disjoint and
totally ordered. The depth test alone therefore resolves visibility exactly,
whatever order instances arrive in. Ordering is purely a performance lever, which
is what makes the approximate bucket sort safe.

## 3. Phase 0 — the measurements that changed the design

`tools/measure-svo-raster-primary-scope.ts`, artifact
`scope-garden-svo-lighting.json`. Garden publishes 1843 leaf bricks, all at
level 5, 22.6% voxel fill.

Two findings redirected the implementation away from the handoff's sketch:

**896 of the 1843 bricks (48.6%) are completely empty.** The producer already
publishes a per-leaf occupancy word in `svoNodes[].links.w`, so the emission pass
drops them for free.

**That same word carries a tight occupied sub-AABB**, so the proxy drawn is the
occupied sub-box rather than the whole brick — again free, already published.

Together they move overdraw and DDA cost by roughly 4×:

| proxy / order | fragments per pixel | DDA steps per ray |
|---|---|---|
| whole brick, unordered | 7.64 | 10.10 |
| whole brick, front-to-back | 3.00 | — |
| tight sub-AABB, unordered | 4.11 | 3.97 |
| tight sub-AABB, front-to-back | **1.01** | — |

## 4. Phase A — parity

`image-diff-1500.json`, raster-primary against the traced primary at 1500²:

```
silhouette disagreement      0 pixels  (0 false positive, 0 false negative)
changed pixels               5565      (0.25%)
absolute luminance error     p50 0, p95 0, p99 7.6e-6
```

Zero silhouette disagreement is the load-bearing number: the two paths agree
exactly on where geometry is. The residual is shading noise on 0.25% of pixels at
brick-boundary ties, which is the tolerance the handoff allowed for.

## 5. Phase B — how much occlusion is actually reachable

Front-to-back ordering is in. The remaining lever the handoff named was hidden
surface removal, and its risk table pre-registered the concern: *"`frag_depth`
disables early-Z → two-draw conservative-depth variant"*.

Measured, cones off so the delta is the brick raster's own fragment cost:

| arm | median | saving |
|---|---|---|
| production (`frag_depth` + `discard`) | 13.93 ms | — |
| drop `frag_depth`, keep `discard` | 13.41 ms | 0.52 ms |
| drop both (HSR upper bound) | 10.91 ms | 3.02 ms |

A fragment whose depth *or* coverage is unknown until it has been shaded cannot
be hidden-surface-removed, so either construct alone suppresses HSR. The split
above says **`discard` is the dominant blocker (2.5 ms of the 3.0 ms), not
`frag_depth` (0.5 ms)** — the opposite of what the risk table assumed. Dropping
`frag_depth` alone buys 0.52 ms of plain ROP/ALU work, not occlusion: with
`discard` still present, HSR stays off.

**3.02 ms is a lower bound on HSR, not an upper bound.** The probe arm does not
only remove work — a fragment whose ray misses the proxy, or whose brick holds
no surface along that ray, no longer discards and instead runs the DDA to
completion and writes a miss. So the arm pays for HSR *and* for extra shading,
and still comes out 3.02 ms ahead. Whatever that extra shading costs is
additional headroom a design that keeps `discard` would collect.

That matters, because the two blockers are not equally avoidable:

- `discard` cannot be removed. Bricks genuinely hold no surface along many rays.
- The one rejection that could have sidestepped it is the **stencil test**,
  which does not depend on the fragment's depth or coverage and could therefore
  in principle run early even while `frag_depth` keeps the depth test late.
  Depth-disjoint brick slabs, each stencil-marking the pixels it resolved, would
  then cull later slabs before they shade — exactly, with no image change.

**Measured: this GPU does not do that.** `tools/measure-metal-early-stencil.ts`
renders eight overlapping full-screen layers through a deliberately expensive
fragment shader that writes `frag_depth` and discards, under three stencil
configurations (`artifacts/render-raster-primary/early-stencil.json`):

| stencil | median |
|---|---|
| no stencil attachment | 8.234 ms |
| test that always passes | 8.280 ms |
| test that always fails | 8.237 ms |

A test that rejects *every* fragment costs exactly what no test at all costs, so
the shader runs first and the rejection is applied to its result. The probe is
sensitive to fragment count — dropping to one layer takes it to 1.3–1.7 ms — so
this is a real negative, not a dead measurement.

That closes the occlusion question. There is no construct in this API that will
recover the 4.11 → 1.01 gap on this hardware, and the two-draw conservative-depth
variant the risk table proposed addresses only the 0.52 ms half. **The brick
raster is bound by fragments launched, not by fragments kept**, so the remaining
lever is fewer or tighter proxies rather than better occlusion — and the
tightening already applied took crossings from only 4.34 to 4.11, which bounds
how much more that direction holds. Both experiment arms are retained behind
`SvoDryOptimizationExperiments` (`rasterPrimaryNoFragmentDepth`,
`rasterPrimaryHsrProbe`) so this is re-measurable rather than re-argued; a unit
test pins them as timing-only probes that can never be reached as render modes.

## 6. Phase C — the LOD cut is unnecessary here

The cut has nothing to cut on this scene. Over the bricks actually drawn:

```
projected proxy size   p10 50.2 px, median 92.4 px, max 266 px
bricks below 4 px      0
bricks below 1 px      0
```

Every leaf sits at level 5 — the tree has no level structure to cut against — and
the smallest drawn proxy still covers ~50 pixels. Building projected-size LOD,
parent-node instancing and hysteresis would add machinery no garden frame
exercises. It remains the right answer for a scene with genuine depth spread; it
is not gating anything now, and the cull pass is the natural place to add it.

The phase's *gate*, though, is measurable without the cut, and it does not say
what the handoff expected. Raising resolution to 2000² (1.78× the pixels):

Unbatched, both arms grow at 0.90 of linear and the speedup looks flat. That
reading is an artefact: the fixed ~11 ms of §7 is resolution-independent, so it
drags both curves down equally. Batching the encodes removes most of it and the
real behaviour appears:

| arm | 1500² | 2000² | growth | vs linear |
|---|---|---|---|---|
| canonical-parametric | 49.62 ms | 77.12 ms | 1.554× | 0.874 |
| raster-primary | 29.01 ms | 48.92 ms | 1.686× | 0.949 |

Raster growth is still sub-linear, so the gate as written passes — but it grows
*faster* than the arm it replaced, and the speedup **erodes** with resolution:
1.71× at 1500², 1.58× at 2000². The handoff's reasoning ("the traced path is
linear — this is where raster should pull decisively ahead") is backwards for
this scene.

The cause is §5. A traced ray descends to exactly one leaf and pays one DDA;
the raster path shades every proxy the ray crosses, 4.11 of them, because
hidden-surface removal is unreachable. That per-pixel constant is what scales,
so raster-primary is closer to linear in pixels than the megakernel is.
Extrapolating, the arms would converge somewhere beyond 4K on this scene.
Raster-primary is the right choice at the resolutions measured, and the LOD cut
this phase was really about — fewer, coarser proxies at distance — is the lever
that would restore the slope, not the constant factor.

## 7. Method, and what is not proven

**Two timing lanes, and they disagree by 11 ms — which turned out to matter.**
The steady-state lane (`FLUID_SVO_DRY_FRAME_PROFILE_SECONDS`) reports 29.39 ms
for the raster-primary arm; the GPU-hardware-timestamp lane reports **17.89 ms**
for the same arm and settings. The steady-state lane starts its clock *after*
`encodeFrame`, so the gap is not shader-side encode. It is `encoder.finish()`,
which sits inside the timed region and is where Dawn validates and translates the
command buffer.

Batching eight frames into one encoder (`FLUID_SVO_DRY_FRAME_PROFILE_BATCH=8`)
does not close the gap — 29.01 ms/frame versus 29.39 — which is the confirming
detail: `finish()` cost scales with the number of commands, so batching cannot
amortise it, whereas it would have amortised submit and fence latency.

So roughly **11 ms per frame is Dawn CPU work serialized against ~18 ms of GPU
work**, and the wall lane is measuring the sum. Two consequences:

- The wall-clock speedups in §1 *understate* the GPU win, because a fixed CPU
  cost is added to both arms. Batched, the same comparison gives 1.71×
  (49.62 → 29.01 ms) rather than 1.62×.
- It also explains §6's flat resolution scaling. A resolution-independent 11 ms
  added to both arms compresses every growth ratio toward 1, which is why both
  arms measured 0.90 of linear. That number is an artefact of the harness, not a
  property of either traversal.

This is now the largest single item in the frame after cone lighting, and it is
not GPU work at all.

**Per-pass GPU attribution came from differencing, not from pass timestamps.**
In-process timestamps written at pass boundaries are not trustworthy on this
tile-based GPU: a pass's fragment work is deferred and overlaps its neighbours, and
the phase partition drops zero-duration marks, which silently folds one pass's cost
into another. An early capture attributed 7.93 ms to the brick *cull* — a compute
pass over 1843 leaves — because the cone prepass mark had been dropped and its time
absorbed. Differencing cones on/off instead gives a clean, checkable decomposition:

```
                       cones on   cones off   cone lighting
canonical-parametric     47.65       32.45        15.20
raster-primary           29.39       13.93        15.47
```

Cone lighting is untouched by this work and costs the same in both arms (15.20 vs
15.47 ms). That agreement is the control: it says the only thing that changed is
the primary, and it is what licenses attributing the whole 18.52 ms saving to it.
Non-cone cost falls 32.45 → 13.93 ms, a **2.33×** reduction.

**The occupancy gate is still unproven, but no longer blocked on capability.**
The handoff called occupancy "the thesis" and gated Phase B on moving it above
the traced path's 22%. Three obstacles turned up; two are fixed:

1. `xctrace --attach <pid>` fails from this shell — the child pid is
   namespace-local, so the Instruments daemon cannot see it. **Fixed:**
   `xctrace record --launch -- <script>` captures correctly.
   `tools/profile-svo-render-xctrace.ts` still uses `--attach`, so it still
   cannot run here.
2. The capture must set
   `FLUID_WEBGPU_DAWN_FEATURES=use_user_defined_labels_in_backend`, or Metal
   never sees the WebGPU pass names and every encoder is anonymous
   ("Render Command 8"). **Fixed:** with it set, intervals arrive as
   `Dawn_RenderPassEncoder_Sparse voxel primary brick raster`, and the trace
   does carry a `Fragment Occupancy` counter.
   `tools/measure-svo-raster-primary-occupancy.ts` reduces the two together.
3. **Still open:** `xctrace export` of the counter-value table has not completed
   inside ten minutes even for an 8-second capture — of order 25 M rows at a
   10 µs sample interval across 31 counters. The reducer streams rather than
   buffers, so the limit is the exporter, not the consumer. Reducing the capture
   window further, or exporting over a restricted time range, is the next thing
   to try.

What the evidence does establish is the *consequence* the gate was a proxy for:
the budget the primary dominates got 2.33× cheaper, and structurally the brick
fragment carries no octree traversal stack, no rigid loop and no pane loop (a
unit test asserts their absence), which is precisely the register pressure the
22% figure was blamed on. Treat the occupancy number as owed, not as claimed —
and note that §5 has since made it moot as a *decision* input: whatever the
occupancy turns out to be, no available construct recovers the overdraw, so it
would change the explanation rather than the plan.

## 7a. Per-pass GPU counters — the thesis was wrong, and that is the useful part

Captured with `xctrace record --launch`, reduced by
`tools/measure-svo-raster-primary-occupancy.ts`
(`artifacts/render-raster-primary/occupancy-raster-primary.json`). 8.86 M counter
samples, 68.5% attributable to a labelled pass (the rest fall between frames).
Per-frame costs divide each pass's occupied time by the 439 brick-raster
instances; they sum to 18.79 ms against the 17.89 ms the GPU-timestamp lane
reports for the whole frame, which is the cross-check that the attribution is
sound (the small excess is overlapping intervals counted twice).

| pass | ms/frame | Fragment occupancy | ALU limiter | LLC limiter |
|---|---|---|---|---|
| deferred dry lighting | 5.99 | 12.3% | 14.2% | **31.9%** |
| compact cone visibility + world GI | 5.72 | 3.4% | 35.7% | 23.4% |
| **primary brick raster** | **3.09** | **24.9%** | **83.1%** | 2.3% |
| (unlabelled encoder) | 2.19 | 8.9% | 22.5% | 23.4% |
| primary background and terrain | 1.80 | 27.6% | **92.1%** | 3.2% |

Two things fall out, and both contradict the handoff.

**Primary visibility is now 4.89 ms of GPU, against the traced path's 26.95 ms —
a 5.5× reduction on the pass itself.** That is a far larger win than the 1.71×
whole-frame figure suggests, and it is the number Phase B actually asked for.
The whole-frame ratio is small only because primary visibility is no longer
where the frame's time is.

**But occupancy is not why.** The handoff's thesis was that the megakernel ran at
22% occupancy because register pressure starved it, and that raster would fix
that. Occupancy moved to 24.9% — 2.9 points, not "materially above". What
actually changed is that the raster path *does less work*: it deleted the
per-pixel octree descent. And the pass now sits at **83% of the ALU limiter**
(the background pass at 92%), with the cache limiter at 2.3%. That is a pass
running near arithmetic saturation, which means:

- There is no scheduling headroom left to recover. Raising occupancy hides
  latency, and an ALU-saturated pass has little latency to hide.
- The only remaining lever is to execute fewer instructions — fewer fragments
  (closed, §5) or a cheaper in-brick DDA.
- It also independently explains §6. An ALU-bound pass costs one unit per
  fragment and fragments scale with pixels, so raster-primary scales nearly
  linearly (0.95) by construction.

The corollary is the more actionable finding: **the frame's largest GPU pass,
deferred dry lighting at 5.99 ms, is the one with the occupancy problem** —
12.3% occupancy, ALU limiter only 14.2%, cache limiter 31.9%. That is a
latency-bound, cache-bound pass, exactly the shape the handoff wrongly attributed
to primary visibility, and exactly the shape where register/occupancy work pays.

## 8. Where the time is now, and what is worth doing next

Primary visibility is no longer the dominant cost — it was 55% of the frame
before this change. The remaining budget at 1500², ranked:

1. **Cone lighting, ~15.5 ms.** Now the single largest item, and untouched by
   this work. Any further effort on this frame should start here.
2. **Dawn `encoder.finish()`, ~11 ms.** Not GPU work at all (§7). Whether this
   is inherent to the pass/bind-group count or an artefact of the Node binding
   is unmeasured, and it is worth finding out before optimising more shaders —
   it is the second-largest line item and no amount of shader work touches it.
3. **Everything else GPU-side, ~14 ms**, of which the brick raster is a part.

Inside the brick raster specifically, the ordering of levers has changed from
what the handoff assumed:

- **Occlusion is closed.** §5 measures that no available construct recovers the
  4.11 → 1.01 fragments-per-pixel gap on this hardware. Do not build the
  two-draw conservative-depth variant or the stencil-slab variant.
- **The LOD cut is now the interesting one**, for the opposite reason the
  handoff gave. It is worthless on the garden scene at these resolutions (§6:
  no brick projects below 4 px), but since raster-primary's cost scales with
  fragments and it now scales *worse* than the megakernel, coarser proxies at
  distance are the only lever that changes the slope rather than the constant.
  It becomes worth building for scenes with real depth spread, or for 4K.
- **Tighter proxies are nearly exhausted.** Moving from whole bricks to the
  published occupied sub-AABB took ray crossings from 4.34 to 4.11. Splitting
  further on the 8-bit macro mask could carve diagonal voids an AABB cannot, and
  `tools/measure-svo-raster-primary-scope.ts` could evaluate that offline before
  any renderer change — but the 4.34 → 4.11 result bounds expectations low.

## 9. Reproducing

```bash
# Steady-state arm (swap FLUID_SVO_DRY_FRAME_TRAVERSAL for the other arm)
FLUID_SVO_DRY_FRAME_WIDTH=1500 FLUID_SVO_DRY_FRAME_HEIGHT=1500 \
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary FLUID_SVO_DRY_FRAME_SHADING=split \
FLUID_SVO_DRY_FRAME_CONE_SCALE=0.5 FLUID_SVO_DRY_FRAME_RADIANCE_RECONSTRUCTION=full-res-relight \
FLUID_SVO_DRY_FRAME_CONE_TRACING=cones FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=30 \
node --import tsx tools/benchmark-svo-dry-frame-gpu.ts

# HSR probes add FLUID_SVO_DRY_FRAME_NO_FRAG_DEPTH=1 or FLUID_SVO_DRY_FRAME_HSR_PROBE=1
# Phase 0 scope
node --import tsx tools/measure-svo-raster-primary-scope.ts
```

`raster-primary` requires split shading; it is rejected at construction against
any other shading path, and against a device that cannot grant 48 B/sample.

## 10. Cutover, and what parity actually costs

`raster-primary` is now the production default. `lib/webgpu-renderer.ts` picks it
whenever the device grants 48 B/sample and falls back to `canonical-parametric`
otherwise, and the render panel exposes both under **Primary tracing → Primary
visibility** (`RASTER` / `TRACED`). The mode is compiled into the shader variants
and the render-pass shape, so the toggle retires the dry-scene pipeline and the
next sweep rebuilds it — the viewport shows the same compile progress it shows on
first attach.

Stationary primary reuse is now derived from the traversal rather than requested
unconditionally. It is worth 28.5 ms of a 49.6 ms traced frame and 7.9 ms of a
29.0 ms rastered one, and under the raster path the rigid impostor pass blocks it
outright whenever the scene has bodies. Asking for it there only made a dead mode
look live, so the raster path asks for `off` and the panel greys the toggle out.

### The static/dynamic arithmetic

The frame decomposes exactly, which is what makes the parity question answerable
(garden, 1500x1500, cones on, quiet M1 Max):

| component | ms |
| --- | --- |
| cone visibility + world GI | 15.08 |
| deferred dry lighting | 5.99 |
| **= traced primary + stationary reuse** | **21.07** (measured 21.08) |
| brick raster + terrain + cull/discovery | 7.94 |
| **= raster primary, no reuse** | **29.01** |

So "stationary reuse" is precisely "skip primary visibility", and parity with it
means driving the primary to zero — which no traversal can do while the camera
moves. What the cutover actually buys is the collapse of the *cliff*: the spread
between a settled and a moving frame goes from 21.08/49.62 (2.35x) to a flat
29.01 (1.00x), and the worst case improves by 20.6 ms.

Against genuinely dynamic input the raster path still is not flat:

| arm | ms | over settled |
| --- | --- | --- |
| settled camera and scene | 29.39 | — |
| moving camera | 34.61 | +5.2 |
| moving bodies | 38.07 | +8.7 |
| both | 39.82 | +10.4 |

The moving-body penalty is localized persistent-world-GI invalidation: entries
near a moving body miss and are recomputed in the same frame that invalidated
them. Bounding that refresh per frame — amortizing it instead of paying it whole
— is the remaining work for dynamic/settled parity, and it is worth more than
anything left in the primary.

### Two negative results worth not repeating

**Merging the terrain and brick render passes is slower.** They share every
attachment, so splitting them looks like a wasted flush of a 48 B/sample G-buffer
— roughly a quarter of a gigabyte of store-and-reload per frame at 1500x1500.
Merged, the frame measured 33.25 ms against the split path's 29.01 ms, with the
canonical arm at its quiet baseline (47.92 vs 47.65) confirming the machine was
not the cause. Apple's tiler overlaps one pass's fragment work with the next
pass's binning; one pass serialises the full-screen terrain triangle against the
indirect brick draw instead. The split is deliberate and commented as such.

**The wall-versus-GPU gap is not CPU.** Splitting the profile lane three ways
(`medianJsEncode_ms`, `medianCommandBufferFinish_ms`, `medianSubmitToFence_ms`)
puts Dawn's whole Metal translation at 0.007 ms and JS encoding at 0.29 ms; the
entire frame is submit-to-fence. The earlier "~5 ms per megapixel of CPU" reading
was wrong. The unlabelled `Render Command 0` encoders that grew with resolution
belong to **Codex (Service)**, not this process — per-process attribution in
`tools/measure-svo-raster-primary-occupancy.ts` (`--process`) settles it, and our
own process carries only the four labelled passes plus sub-0.2 ms discovery work.

### Deferred lighting now classifies by depth

The lighting pass takes the hardware depth buffer read-only and runs two
complementary full-screen draws: `drySkyLightingMain` under `greater-equal` for
the pixels primary visibility left as a miss, and `dryLightingMain` under `less`
for the rest. Neither writes `frag_depth` nor discards, so the rejection is
early. It is correct — the partition leaves no pixel unwritten — but neutral on
this camera, because the garden frame is only 8% sky. It pays on sky-heavy views
and it is the depth plumbing the next optimization needs.

## 11. Dynamic parity — the rigid-body reject

The cone-tracing pipeline was already on the raster path everywhere it could be:
the brick raster writes `splitGeometry`/`splitOpaqueIdentity` as attachments 2
and 3, and `dryPrepassCoherentMain` reads exactly those, so the prepass is seeded
from the rastered G-buffer rather than re-tracing. Rigid bodies reach it as
*receivers* through the impostor pass and its certificate bridge, and fluid needs
no special handling at all: evolving bricks are ordinary leaves, and the emit
pass rebuilds the instance list from scratch every frame.

What was **not** on the raster path was rigid-body *occlusion*.
`anyBodyBlockerIgnoring` looped every body for each shadow ray and each contact
sample, from the full-resolution lighting pass — at 1500x1500 with two lights
that is on the order of 13 million loop entries per frame, each iteration reading
`bodies[]`. It fits the deferred lighting pass being simultaneously the frame's
least occupied (12.3% fragment occupancy) and most cache-limited (31.9% LLC).

The fix publishes one sphere enclosing every body (`rigidBoundsWordOffset`, a
single uniform lane) and tests rays against it before the loop. It is provably
conservative: the centre is the midpoint of the bodies' axis-aligned bounds, the
radius is `max(|position - centre| + boundingRadius(body))`, and the device packs
`positionRadius.w` with that same `boundingRadius`, so every per-body sphere lies
inside the published one. The guard reuses `bodyBoundingSphereVisible`'s exact
segment/sphere form and epsilon, so the two cannot disagree about a grazing ray.
A scene with no bodies publishes a negative radius and retires the loop outright.

This closes the dynamic gap almost exactly (garden, 1500x1500, cones on):

| arm | before | after |
| --- | --- | --- |
| settled camera and scene | 29.39 | 29.69 |
| moving camera | 34.61 | 29.66 |
| moving bodies | 38.07 | 29.62 |
| moving camera and bodies | 39.82 | 29.68 |

The frame is now flat across every dynamic combination, and the image is
**bit-identical** to the pre-change render: 0 changed pixels, 0 silhouette
disagreements, 0 luminance error at 800x800. That is the expected outcome for a
guard that only skips iterations which would have returned false, and it is worth
stating because a guard that wrongly rejected rays would also have measured fast.

Note this supersedes §10's attribution of the moving-body penalty to world-GI
invalidation. The GI cache does still clear on body motion, but the cost that
showed up in the wall clock was the body loop it forced the fallback path to run,
not the cache refill.

## 12. Scene coverage — hose-tank was not a raster regression

Reported as "the raster SVO path runs incredibly slowly on hose-tank, while
garden runs fast". It reproduces in Dawn, but the primary is not what is slow:

```
FLUID_SVO_DRY_FRAME_SCENE=hose-tank FLUID_SVO_DRY_FRAME_SHADING=split \
FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=5 \
FLUID_WEBGPU_DAWN_FEATURES=skip_validation ... tools/benchmark-svo-dry-frame-gpu.ts
```

1280x720, M1 Max, cones on, median frame:

| scene | primary | before | after |
| --- | --- | --- | --- |
| hose-tank | raster-primary | 304.6 | **20.1** |
| hose-tank | canonical-parametric | 323.8 | 31.8 |
| garden-svo-lighting | raster-primary | 13.3 | 13.3 |
| garden-svo-lighting | canonical-parametric | 21.8 | 21.8 |

Both hose-tank arms collapsed, and by nearly the same amount, which is the
signal that the primary was never the variable. The cause is the static
node-mip opacity pyramid being **withdrawn**: hose-tank is a closed tank inside
a voxelized conservatory room and needs 10361 directory pages, but the directory
is one texture row per page and the device had only been granted WebGPU's
default `maxTextureDimension2D` of 8192. `OctreeSparseBrickWorld` then correctly
declines to publish — a truncated pyramid is not a coarser pyramid, its dropped
pages sample as empty air — and cone lighting falls back to exact traversal for
every shadow and GI ray: a 15x frame cost, reported only as a `console.warn`.

The fix is one line in `requiredFluidDeviceLimits`: request the adapter's
advertised `maxTextureDimension2D` (16384 on this adapter), exactly as that
function already does for the storage-buffer and 3D-texture limits. Every scene
preset that builds a static SVO world now publishes a pyramid; `ocean-seiche`
and `deep-water-ab` still fail their world build on the unrelated 4 GB geometry
allocation guard in `lib/sparse-brick-octree.ts`.

Two things this leaves standing. Raster-primary is 1.58x the canonical primary
on hose-tank, close to the 1.64x it holds on garden, so the cutover generalises
past the scene it was tuned on. And hose-tank's remaining 1.5x gap to garden is
geometry, not lighting: `tools/measure-svo-raster-primary-scope.ts` reports
11047 leaves against garden's 1843, with 204 proxy boxes straddling the camera
plane (garden has none) because the camera sits inside the room shell. Those are
drawn correctly — `SVO_BRICK_RASTER_CONTRACT.cullMode` keeps the far faces for
exactly this case — but they are near-full-screen fragment work, and they are
the first thing to look at if hose-tank needs to get faster still.
