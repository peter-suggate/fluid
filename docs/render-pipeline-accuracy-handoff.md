# Render panel frame pipeline — accuracy review and plan

Date: 2026-08-11. Review only; nothing here is implemented.

> **2026-08-24 — WP1.4 and WP5/SOURCE landed.** The frame now has a stage ABI
> (`lib/core/render-frame-stages.ts`) modelled on `SPARSE_CM12_RESIDENT_STAGES`:
> encoders close seams by *stage id*, the panel looks up label, phase, band and
> prose from one registry keyed by that id, and every stage is assigned to
> exactly one row by a table that is exhaustive over the ABI (`STAGE_NODE`), so
> an unowned or renamed stage is a `tsc` error rather than a row that quietly
> reports someone else's number. Alongside it the encoder publishes a per-frame
> **manifest** of what each stage encoded (`RenderFrameSeamRecorder`), which is
> what finally separates *encoded nothing* from *encoded render passes nobody
> can time*. `tests/render-frame-stage-partition.test.ts` pins the partition,
> the per-encoder seam order, and the attribution rule.
>
> The bug that motivated it: **Sparse world build read 27.9 ms on frames where
> the world encoded no pass at all.** The panel's rule "if exactly one row in
> this band has no measurement, give it the band's fence-partitioned wall" made
> the silent row the sink for the whole source band. It now reads 0.70 ms while
> maintenance runs and a true zero when the world is settled, and the band's
> unattributed residual stays on the collar, where the tooltip says what it is.

The question asked: does the render panel's frame pipeline reflect the passes and
work the frame actually does, are we measuring all the stages we should, can the
switches meaningfully buy performance, and is shader splitting warranted.

Short verdict: the *shape* of the graph is close to right, but the numbers on it
are not the frame. Only compute passes publish trusted costs, so the panel
prices the cheap stages and dashes the expensive ones (the primary alone is 73%
of a depth-3 frame and shows no figure). Attribution-by-seam silently charges
several whole pass chains to the wrong rows. Two node states report "on" for
passes the production frame never encodes. And roughly half of the switches
withhold less work than their rows claim, because the work is either not gated
(water surface extraction), or compiled into a shader as a uniform branch that
keeps its register price when "off" — the exact mechanism the unified-voxel
purge measured at 13.25 → 8.96 ms for deleting a branch the frame never took.

Ground truth below was built from three full sweeps: the encode path
(`webgpu-renderer.ts:2843-3084`, `webgpu-water-pipeline.ts:2007-2169`,
`webgpu-svo-dry-scene.ts:10032-10493`), the timing machinery
(`performance-trace.ts`), and every consumer of `disabledStages`.

---

## Part 1 — Findings

### A. The measurement cannot see most of the frame

**A1. Render passes have no trusted cost.** `GPUPassTimestampRecorder`
(`performance-trace.ts:1108`) instruments every pass, but `trusted = kind ===
"compute"` (`:1245`): on Apple GPUs a render pass's timestamp pair brackets the
tiler window, not the pass (documented in-file: 403 ms reported for 10.7 ms of
deferred lighting; 200x on the primitive overflow pass). `readSemanticTrace`
(`:1287-1319`) drops any semantic group containing an untrusted pass. Result:
**primary megakernel, cone prepass, seam closure, deferred lighting, reduced
shade, water interfaces, caustics, composite, present — every render pass —
publishes nothing.** The panel measures the compute minority of the frame and
shows an em dash for the majority.

**A2. The percentages are shares of the wrong total.** `RenderPipeline` gets
`stageTrace.total_ms` = the **sum of trusted compute passes laid end-to-end**
(`performance-trace.ts:1352-1376`, context suffixed `:compute-pass-sum`), so
every node/band share is "share of the compute-pass sum", not of the frame. The
`ms / frame` header is a different number again: queue-wall from
`performance.now()` at submit to `onSubmittedWorkDone` (`webgpu-renderer.ts:3079`),
and because solver advances are submitted to the same queue *before* the fence
begins, **that total includes up to two in-flight simulation advances**
(`webgpu-renderer.ts:3080-3082`). Meanwhile the honest "GPU busy" numbers —
`span_ms` and `overlap` (`performance-trace.ts:1036-1057`; measured compute
over-count up to 1.9x from pass overlap) — are computed and never reach the
panel.

**A3. Attribution is by pass-count seam, and several chains land on the wrong
row.** `completePhase` closes a label over *every pass since the previous seam*
(`performance-trace.ts:1192-1199`). Any pass not followed by its own
`tracePhase` is charged to whatever label closes next:

- SVO fluid coverage (1 fill + one compute pass per mip level,
  `webgpu-renderer.ts:2921`, `webgpu-svo-fluid-coverage.ts:215-239`) → charged
  to **Water surface extraction**. No node, no switch.
- The two peeled rear water interfaces (`webgpu-water-pipeline.ts:2144-2145`) →
  charged to **Layered optical composite**.
- Pixel-trace probe, brick raster probe, probe readbacks, decoration overlay
  (`webgpu-renderer.ts:3037-3055`) → encoded after the inspection seam, charged
  to **Final upscale + present**.
- Fluid-cell-trace gather (`webgpu-renderer.ts:3015-3022`) never sets
  `inspectionOverlayEncoded`, so on a probe-only frame the *Inspection overlays*
  label is never emitted at all while its passes ran — the row reads `idle`.
- Query resolve + copies after the last seam are dropped from the partition.
- ~20 untraced `clearBuffer`/`copyBufferToBuffer` calls (incl. a full clear of
  the 262 144-entry world-GI cache on toggle) ride inside neighbouring labels.

**A4. Averaging under-reports intermittent stages.** `averagePerformanceTraces`
tracks a per-key count (`performance-trace.ts:373`) and then ignores it,
dividing by the number of observations (`:377-380`). A phase present in k of n
sampled reports (caustics on mesh-move frames, world maintenance on edit frames)
reads k/n of its true per-encode cost, with nothing marking it as intermittent.

**A5. The toggle delta is one sample against one sample on a contaminated
basis.** `useStageAblation` (`VisualPanel.tsx:54-118`): N=1 per setting, no
variance, and the basis is the queue-wall total from A2 — so a delta measured
while the solver runs carries the sim's jitter. The signature covers only node
on/off states, not tuning/resolution/scene, so a stored `off` sample can be
differenced against an `on` sample taken under different tuning. (The 900 ms
settle and per-context trace reset are good; the rest is weak by construction.)

### B. The node table has drifted from the encoder

The docstring in `render-pipeline-graph.ts` claims the diagram, timing, and taps
"cannot disagree" because they project one list. But the list is hand-maintained
and the encoder has moved (the unified-voxel rewrite among others):

- **`reduced-shade` lies under the production default.** Production tuning is
  `coneRadianceReconstruction: "full-res-relight"`; the encoder gates the pass
  on `reconstructReducedRadiance` (not a relight mode) so it is **never
  encoded**, while `state()` tests only `reduced(context)` and reports **on**
  (`render-pipeline-graph.ts:380` vs `webgpu-svo-dry-scene.ts:10401`). Lamp on,
  row idle, forever.
- **`cone-visibility` claims a label production never emits.** "SVO
  cone-lighting prepass" exists only on the `!useSplit` inline arm; production
  is pinned to split shading (`webgpu-renderer.ts:1239`). The production cone
  cost arrives as "SVO compacted cone lighting" + "SVO cone sample fan-out".
- **`voxel-light-cache` state is wrong on wet scenes.** The real gate also
  requires `voxelLightUserEnabled && pageCount>0 && !fluidCoverage &&
  shadowsEnabled` (`webgpu-svo-dry-scene.ts:8859-8866`) — any scene with fluid
  coverage disables the cache while the panel says on.
- **`"Water interfaces"`** (`render-pipeline-graph.ts:447`) is a
  PerformancePanel display name; no encoder emits that string. Dead entry.
- **`water-interfaces` claims `"Water surface extraction"` but its switch does
  not gate extraction** (`webgpu-water-pipeline.ts:2024-2032` consults only
  revision/throttle). Because `measureLabels` wins before the withheld check
  (`render-pipeline-graph.ts:629-636`), the node reports a live measured cost
  while switched **off**. This is the single largest switch/measurement
  mismatch in the system.
- **Five labels can never appear in a production trace** — the raster-arm tiers
  (brick cull, scene-primitive, near-field band, rigid discovery, thin-glass
  discovery) plus the inline megakernel and prepass labels — because
  `resolveSvoPrimaryTraversal` answers `traced` unconditionally
  (`svo-render-options.ts:143-151`). The panel renders them `unavailable`
  (honest), but three permanent `unavailable` rows occupy the primary band.
- **`inspection-overlays`' tip overclaims**: "Off withholds all of them at once
  — including the ◨ taps and the probes below." The pixel-trace probe and
  fluid-cell-trace gather are not gated by `inspectionWithheld`
  (`webgpu-renderer.ts:3015-3021, 3037-3050`).
- **`sparse-world-build`'s one label** covers a whole maintenance subtree —
  topology publish/mutate, proxy voxelization, finalize + fence, derived
  lighting planner/builder, radiance feedback — for **two** producers, under a
  tip that says "voxelizes the authored scene".

### C. Switch honesty — what "off" actually buys

Three failure classes, in decreasing order of frame impact:

**C1. Work not gated at all.** The water surface-extraction compute chain (the
largest unswitched block in a wet frame), the fluid-coverage volume, the
compact-cone stage + fan-out (switchable only by collapsing the whole cone
mode), probes, the brick cull, and every untraced clear/copy.

**C2. Work gated in the shader, not the encoder.** There are **zero**
pipeline-overridable constants and zero per-feature shader variants in the
dry-scene pipeline; every lighting feature is a dynamically-uniform branch on
one uniform word (`webgpu-svo-dry-scene.ts:1095-1103`). GI composition
(`:5074, :4459, :3414`), the seam sample (`:4529`), cone code under
`coneLightingRequested`, voxel-light consumers, shadows/AO — all stay compiled
in when "off", keeping their register/occupancy price. The unified-voxel purge
proved this price is real: deleting a never-executed branch from the deferred
kernel was −30%, byte-identical. Every one of these switches under-reports what
the feature truly costs, and "off" saves less than it should.

**C3. Residual defects.** `thin-glass` off skips the pass *including its
clears*, so downstream samples stale glass keys (the one deviation from the
"keep clears" contract in `render-stage-switches.ts:19-23`). `rigid-impostor`
off leaves `splitRasterRigidVisibilityPipeline` selected
(`webgpu-svo-dry-scene.ts:10063-10065`) — the dearer variant, now reading a
stale plane. Toggling `primary-traversal`/`world-gi-cache` pays a full GI-cache
clear the following frame (fine, but untraced — see A3).

---

## Part 2 — Plan

Five work packages. WP1+WP2 are small and purely additive; WP3 makes render
passes measurable; WP4 is the performance axis (and the shader-splitting
answer); WP5 is the panel restructure that the rest earns.

### WP1 — Attribution: every pass owned, every label true (do first)

1. **Seam every unattributed pass.** New phases: fluid coverage
   (`svo-fluid-coverage`, after `webgpu-renderer.ts:2921`); rear interfaces fold
   into the front/back interface labels (move the seam from
   `webgpu-water-pipeline.ts:2143` to after `:2155`); probes + decoration move
   before the inspection seam and set `inspectionOverlayEncoded`; a final
   `trace-resolve` seam after `webgpu-renderer.ts:3077` so the tail is not
   dropped. Group adjacent untraced clears/copies under the phase they serve.
2. **Enforce it with a test.** Extend
   `tests/render-pipeline-stage-switches.test.ts`: run a representative encode
   against a mock device and assert the emitted pass list is *partitioned* by
   `tracePhase` seams — no pass between a label's seam and the previous one that
   the label's node does not own. This is the invariant the graph docstring
   already claims; make it checked instead of asserted.
3. **Fix the false states.** `reduced-shade` gates on the reconstruction mode;
   `voxel-light-cache` gets the real gate; delete the dead `"Water interfaces"`
   label; `cone-visibility` drops the prepass label on the split arm (or keeps
   it — harmless — but its tip stops implying the prepass is production).
4. **LANDED (2026-08-24). Structural fix so it cannot drift again: the encoder
   publishes a per-frame pass manifest** — `{label, kind, switchOutcome: encoded|withheld|gated-off}`
   per phase, assembled where `tracePhase` is called (near-zero cost, it is a
   push per seam). Node *state* for pass-owning nodes derives from the manifest;
   the static table keeps only prose, taps, ordering, and switch wiring. The
   panel then reports what the frame did rather than re-deriving predicates that
   go stale. (`armed`/`idle` fall out naturally: enabled but no manifest entry.)
   As built, the manifest counts compute and render passes per stage rather than
   carrying a `switchOutcome`, because the count answers both questions the
   panel asks — *did this stage encode* (a true zero, `withheld`) and *is what
   it encoded priceable* (`unpriced`, and therefore the only kind of row allowed
   to absorb a band's wall). `idle` is gone as a cost kind; it was the state
   that let a silent row look like a missing measurement.

### WP2 — Measurement truth (small)

1. **Publish `span_ms` + `overlap` to the panel.** Header shows three labelled
   numbers: *frame (queue wall, incl. sim)*, *presentation GPU busy (span)*,
   *compute-pass sum*. Shares become shares of span. The ⊂ and withheld
   semantics are unchanged.
2. **Fix k/n averaging** (`performance-trace.ts:377`): divide by the per-key
   count, and carry `encodedFraction = k/n` so the panel can annotate
   intermittent rows ("×3/12 frames") instead of silently diluting them.
3. **Harden the ablation delta**: include the renderer's `presentationContext`
   key in the ablation signature so tuning/scene changes invalidate stored
   samples; keep ≥3 samples per side and report the median; suppress the delta
   (or badge it) while the solver is running, since the basis includes sim time.
   Optional upgrade: an explicit "price this node" action that alternates the
   switch N times and reports median ± spread — strictly better than the
   passive pairing, and it is what the number is for.

### WP3 — Timing the render passes (the missing majority)

Metal render-pass timestamps are unusable (A1) and the boundary-chain model is
documented-dead on this path (`performance-trace.ts:1064-1107`). Two viable
routes, complementary:

1. **Fence-partitioned sampling frames.** One frame in N (N≈16, only while the
   panel is open), split the presentation encode at band boundaries into K
   submits, `performance.now()` at each `onSubmittedWorkDone`. This yields real
   wall costs per *band* — SOURCE / PRIMARY / LIGHTING / SHADING / OUTPUT — and
   in particular finally prices the primary megakernel and deferred lighting.
   The submit-to-fence metric is already this program's trusted lane
   measurement, so the figures agree with how everything else here is measured.
   Costs: serialization on sampled frames (they are excluded from the rolling
   frame-time mean, marked as sampling frames), and band-level rather than
   per-pass grain. Per-pass grain inside a band comes from route 2 or xctrace
   (`gpu-frame-profile` skill remains the deep lane).
   **LANDED**: `lib/webgpu-frame-band-sampler.ts` (`FencePartitionedFrameSampler`,
   a `FrameBandPartitioner` threaded through `waterPipeline.encode` and
   `svoDryScenePipeline.encode`; a baseline drain fence keeps in-flight solver
   advances out of the first band). Bands in encode order: source /
   water-surface / svo-primary / svo-lighting / svo-shading /
   composite-present; the inline arm crosses one seam after its megakernel,
   attributed to `svo-primary` like its trace phase. A sampling frame runs
   neither the pass-timestamp recorder nor the queue-wall recorder, so the
   frame mean never contains a partitioned frame. Published as
   `RendererFrameMetrics.presentationBands` → `PerformanceReport` → the
   panel's BAND ms strip. Unit-tested in
   `tests/webgpu-frame-band-sampler.test.ts`.
2. **Make convertible passes compute.** A compute pass times trustworthily.
   Deferred lighting (fullscreen triangle, read-only depth, writes HDR color)
   and the composite are compute-shaped; the primary must stay a render pass
   (it writes `hardwareDepth` as a real depth attachment). This overlaps WP4.1
   — where a pass is being split anyway, prefer compute for the split pieces,
   and each piece becomes measurable for free. Validate against the tile-memory
   caveat with an A/B on the hero lane before committing (a fullscreen fragment
   draw and a compute dispatch are not automatically the same price on TBDR).

### WP4 — Splitting shaders and passes: make "off" mean off

This is the performance axis, and the answer to "consider splitting shaders" is
yes — but split by *feature specialization*, not by fragmenting the megakernel.

1. **Split the deferred lighting pass.** Today one pass, two-three draws (sky +
   optional relight + full deferred), sharing one label and one row. Make sky
   and deferred shade separate passes (compute candidates per WP3.2): each gets
   its own row, cost, and switch. Sky is a known ~0.85 ms; the panel currently
   cannot show it.
   **LANDED**: the split arm now encodes "Sparse voxel deferred sky lighting"
   (clear + miss-pixel sky draw, depth greater-equal) and "Sparse voxel
   deferred dry lighting" (load + surface draws, depth less) as separate
   render passes with their own seams; new `sky-lighting` switch and panel
   node ("miss pixels · own pass"). Cost: one extra HDR attachment load/store.
   Not yet priced by an interleaved hero-lane A/B.
2. **Feature-specialized pipeline variants.** Introduce a variant key —
   (GI on/off, seam-sample on/off, cone mode cones/exact/off, voxel-light
   consumer on/off, shadows/AO) — and compile the deferred + prepass kernels
   with disabled features *absent*, cached per key exactly like the existing
   per-cone-scale split bundles (`webgpu-svo-dry-scene.ts:7830-7842`). Use
   WGSL-level composition or `override` constants (Dawn dead-code-eliminates
   const-false branches). Order by expected value:
   a. **GI composition** — the clearest never-taken-branch case; today its
      dispatch-level saving is only the world-GI cache pass while the gather
      stays compiled into the deferred kernel.
      **LANDED**: `SvoDryOptimizationExperiments.globalIlluminationAbsent`
      stubs `dryGlobalIllumination` to the exact uniform-flag-off return, so
      Dawn DCEs the cone gather; the split bundle cache is keyed
      `${scale}|gi` / `${scale}|no-gi`, GI flips recompile through the
      existing ensure paths, and the stale-but-correct bundle renders during
      the recompile. Both variants compile clean on Dawn/Metal at x1 and x4
      (~3 KB of gather WGSL removed). Imagery is unchanged by construction
      (stub = flag-off value); the price is not yet measured by an
      interleaved hero-lane A/B.
   b. **Cone/exact selection in the deferred kernel** — `dryLightVisibility`
      selects on a uniform; this is the same structure the unified purge
      measured at −30% for the analytic case, called out in that handoff as
      "generalises much further: the same uniform-selected structure exists for
      AO, GI and the prepass shortcuts".
   c. **Voxel-light consumer bindings** — off currently keeps bindings and
      sampling in four consumers.
   d. **Shadows / AO flags** — currently pure uniform branches with UI toggles
      and no honest price.
   Each variant lands with an A/B on the hero lane (interleaved, per the 30%
   drift trap) and the toggle's measured delta becomes its real price.
3. **Fix the residuals.** `thin-glass` off keeps its clears; `rigid-impostor`
   off falls back to the non-rigid pipeline variant.
4. **New switches for the ungated work**, each a normal `RenderStageSwitchId` so
   the panel machinery applies unchanged:
   - `surface-extraction` — gate `updateSurface`
     (`webgpu-water-pipeline.ts:2024`) and stop `water-interfaces` claiming its
     label. Off freezes the mesh at last extraction (retained-surface behaviour
     already exists), interfaces keep drawing the retained mesh.
   - `fluid-coverage` — gate `svoFluidCoverage.encode`
     (`webgpu-renderer.ts:2921`).
   - `cone-stage` granularity: keep the mode switch, but let compact-cone and
     fan-out be withheld individually (fan-out is already an experiment flag
     with no UI). **Resolved without a new switch**: `coneFanout` is a
     constructor flag enabled only by `tools/benchmark-svo-dry-frame-gpu.ts`
     and `tools/profile-svo-render-xctrace.ts` — the browser renderer never
     constructs the fan-out arm, so a panel switch would be a lamp over a path
     the app frame cannot take (the same anti-pattern as the three
     permanently-unavailable raster tiers). In the app the mode switch already
     withholds exactly the compact-cone march; in the tools the flag itself is
     the switch. Revisit only if fan-out ever gains renderer plumbing.
   - Probes/cell-trace under `inspection-overlays` (close the C3/B leak).

### WP5 — Panel restructure (earned by the above)

- **SOURCE**: split `sparse-world-build` into *world build* and *derived
  lighting publish* rows (separate seams first — WP1.1); add *fluid coverage*.
  **LANDED (2026-08-24)**: four rows — world build (topology publish + proxy
  voxelization), derived lighting publish (plan · build · feedback), rigid pose
  mirror, fluid coverage — each owning its own stages of the ABI.
- **PRIMARY**: collapse the three permanently-`unavailable` raster tiers into
  one "raster arm" row that expands only when `rasterPrimaryActive` — three
  rows of `unavailable` is diagram space spent on a path the frame cannot take.
  **LANDED**: the three tier nodes carry `collapseGroup: "raster-arm"`
  (`RENDER_PIPELINE_COLLAPSE_GROUPS`); while every member is `unavailable`,
  `RenderPipeline` renders one collapsed placeholder row and the members
  reappear the moment the raster primary makes any of them reachable.
- **SHADING**: deferred splits into *sky* and *deferred shade* rows (WP4.1); GI
  composition keeps its row but its cost becomes real once it is a variant, not
  a branch.
- **OUTPUT**: *surface extraction* becomes its own node (it is compute, already
  measured — it just needs to stop living inside `water-interfaces`); probes
  and decoration appear under inspection.
- Band collars annotate their measurement basis: bands measured by
  fence-sampling read as wall-clock band costs; compute rows keep per-pass
  figures; the two never mix in one share denominator.

### Ordering and acceptance

1. **WP1 + WP2** — small, no behaviour change, immediate accuracy win.
   Oracle: the seam-partition test; reduced-shade/voxel-light rows read
   correctly on the default and a wet scene; shares sum to ≤100% of span.
2. **WP4.3 + WP4.4** (residuals + new switches) — small encoder changes.
   Oracle: `water-interfaces` off shows extraction still priced on its own
   row; new switches produce non-zero measured deltas.
3. **WP3.1** (fence sampling) — the primary and deferred rows get numbers.
   Oracle: band sums vs queue-wall total on a paused frame agree within noise;
   figures cross-checked once against xctrace.
4. **WP4.1 + WP4.2** — the big one, staged per variant, each gated by
   interleaved A/B + settled-hash on the hero lane.
5. **WP5** — last, once the data it presents exists.

### Traps carried forward

- The lane's run-to-run drift reached 30%; single-run deltas are noise
  (memory: svo-dry-lane-noise-floor). Any WP4 A/B must interleave.
- A toggle changes `presentationContext`, resetting the trace window — deltas
  need the 900 ms settle they already have, plus the context-keyed signature.
- Fence-sampling frames serialize the queue; they must be excluded from the
  frame-time mean and never coincide with the solver's in-flight advances if
  the presentation-only wall is the goal (pause-aware sampling first).
- `webgpu-lighting.ts` encodes nothing — it is the shared shading contract.
  Any plan that says "add timing to webgpu-lighting" has the wrong file.
