# Pixel Picker for the Raster Primary — Implementation Handoff

Status: **implemented**. Companion to `docs/SVO_RASTER_PRIMARY_HANDOFF.md`
(the change this document reacts to) and `docs/SVO_RASTER_PRIMARY_RESULTS.md`.

Three things changed shape during implementation; §7 records them and why.

Scope: the "Ray work under the pointer" diagnostic — `lib/svo-pixel-trace.ts`
(ABI + presentation), `lib/webgpu-svo-pixel-trace.ts` (the GPU probe),
`components/PixelTraceHud.tsx` (readout), and the decoration layers the overlay
draws. The fluid cell picker (`components/FluidCellTraceHud.tsx`) reads published
solver state, not the render path, and is untouched by any of this.

## 1. Why — the picker now describes a shader that no longer runs

`4c1e68c` made `raster-primary` the production default
(`lib/svo-render-options.ts:24`, `lib/webgpu-renderer.ts:783`). Primary
visibility is a compute cull plus a hardware-rasterized, depth-tested brick-proxy
draw. The probe was not touched: `dryProbeMain`
(`lib/webgpu-svo-pixel-trace.ts:532`) still calls `probeTraceStatic`
(`:207`), a hand-written mirror of the *traced* megakernel's near-to-far stack
walk.

So on the default configuration the picker asserts, in the viewport and in words,
work that the GPU did not perform for that pixel:

| What the HUD shows | What actually happened |
|---|---|
| "Descend the hierarchy — N nodes, M children rejected" (`lib/svo-pixel-trace.ts:906`) | No per-pixel descent. One compute thread per *leaf* emitted a proxy; the search was the rasterizer. |
| Octree boxes opened / child boxes rejected by slab tests (`hierarchy`, `rejected` layers) | Never computed for this pixel. Frustum rejection happened once per leaf, per frame, camera-wide. |
| "front-to-back order let the first occupied brick finish the ray" | Front-to-back is a *performance* lever only; the depth test resolves visibility, and ordering is a 1024-bucket counting sort on min view depth (`lib/webgpu-svo-brick-raster.ts:227`). |
| "N bricks crossed without a surface before the hit" (`emptyBrickSkips`) | Bricks that find no surface `discard` (`lib/webgpu-svo-dry-scene.ts:1600`); they are *concurrent* fragments, not a sequence the ray walked through. |
| One primary ray drawn as an arrow chain through the bricks it walked | The winning fragment ran a DDA in exactly one brick. Every other covering proxy also ran one and lost. |

Two further defects fall out of the same neglect:

- The probe module is compiled with `shadingPath: "inline"` and the live
  `traversalMode` (`lib/webgpu-svo-dry-scene.ts:4730`). With
  `traversalMode = "raster-primary"`, `rasterPrimary` is false inside
  `createSvoDrySceneFragmentWGSL` (`:1158`), so `secondaryTraversalMode` stays
  `"raster-primary"`, misses both the canonical and compact branches (`:1198`),
  and the probe silently compiles the **wide-fanout** traversal for its secondary
  rays — while production secondaries run `canonical-parametric` (`:1192`). The
  reported shadow work is from the wrong traversal even on its own terms.
- `blockedReason` still calls the raster path "the raster fallback"
  (`components/PixelTraceHud.tsx:47`). Raster is the default; the traced path is
  the fallback.

The traced path remains selectable and is the parity oracle, so this is not a
rewrite: it is a second mode for the diagnostic, chosen by the same option the
frame was rendered with.

## 2. What actually produces a pixel now

The ladder a single pixel's colour passes through, with the pass labels the
encoder emits (`lib/webgpu-svo-dry-scene.ts:3554`, `:4919`):

1. **`Sparse voxel brick instance cull`** (compute; `svo-brick-cull`).
   One thread per resident leaf: reject empty bricks from the published occupancy
   word, frustum-reject the rest, quantize min view depth into 1024 buckets,
   prefix-scan, scatter into a front-to-back list, publish the indirect draw
   count. Frame-wide work, not pixel work — but it decides what the pixel *can*
   hit. Its real counters live in `SvoBrickSortState`
   (`lib/webgpu-svo-brick-raster.ts:149`): `resident`, `empty`, `culled`,
   `candidateCount`, `drawInstanceCount`.
2. **`Sparse voxel primary background and terrain`** (full-screen fragment).
   Owns the G-buffer clear and the exact miss encoding; runs the terrain secant
   march. Every pixel pays this.
3. **`Sparse voxel primary brick raster`** (`drawIndirect`, back-face-drawn boxes,
   reversed-Z). Every proxy whose box covers the pixel produces a fragment; each
   fragment clamps its own ray/AABB interval and runs the production
   `traceLeafPayload` DDA inside that one brick, then either `discard`s or writes
   four G-buffer planes plus `frag_depth`. Because leaves partition space, the
   depth test alone is exact — **the pixel's surface is the winner of a
   tournament, and every entrant cost real ALU.**
4. **`Sparse voxel analytic rigid primary discovery`** + certificate bridge
   (when bodies exist).
5. **`Sparse voxel raster thin-glass discovery`** — pane key per pixel.
6. Reduced-rate cone lighting, when the scale is not 1: prepass geometry /
   visibility / shade, the compact cone-visibility compute, optional fan-out, and
   the persistent world-space GI cache.
7. **`Sparse voxel deferred dry lighting`** — two complementary depth-tested
   draws. `drySkyLightingMain` takes the miss pixels, `dryLightingMain`
   (`:1718`) takes the rest: it *reads the G-buffer* (`drySplitGeometryAt`,
   `drySplitIdentityAt`) and shades from it. Shadow rays, cone marches, AO and GI
   are here and are essentially unchanged by the raster switch.

Everything from step 7 the current probe already mirrors faithfully. Steps 1–5
are what it gets wrong.

## 3. Target design

### 3.1 Two probes, one record stream

Split the capture along the seam the renderer itself now has — *finding* the
surface versus *lighting* it.

**Probe B — primary (new, standalone compute).** A single workgroup, dispatched
once per frame after the cull pass, in its own module with its own bindings
(the pattern `createSvoBrickRasterCullWGSL` already establishes). It binds the
camera uniform, the `SvoMapping` prefix of `DryParams`, the published topology,
`structural.materialOwners` and the primitive buffer, plus **the frame's real
`brickInstanceBuffer` and `brickSortStateBuffer`**.

That last point is the design's whole value. The expensive, register-heavy part
of the old probe existed to *re-derive the search*. Under raster there is nothing
to re-derive: the frame already published its sorted instance list and its cull
counters as GPU buffers. The probe reads them. What it mirrors is only the cheap,
bounded part — the same ray/AABB clamp and the same `traceLeafPayload` the brick
fragment runs.

Per requested pixel it emits:
- the cull summary, read verbatim from `SvoBrickSortState` (measured, not mirrored);
- one record per **covering proxy**: instance index in draw order, sort bucket,
  proxy AABB, the leaf's full AABB (so the occupancy tightening is visible),
  `[tEnter, tExit]`, DDA cells stepped, surface found or not, resulting depth;
- the winner's index, and for the winner the per-cell DDA and exact-test records
  the existing `brickCell` / `exactTest` kinds already carry.

Lanes scan the instance list strided, append to a workgroup-local buffer under a
workgroup atomic, then one lane writes them out ordered by instance index. No
device atomics, so the r32uint record texture stays the transport and the host
decode stays one buffer.

**Probe A — lighting (existing fragment probe, retargeted).** Keeps
`probeConeMarch`, `probeLightVisibility`, `probeContactVisibility`,
`probeGlobalIllumination` unchanged — those still mirror shipping code. Two
changes:
- **Stop re-tracing the surface.** Read the G-buffer at the requested pixel
  exactly as `dryLightingMain` does. The picker's "hit" then *is* the pixel's
  surface rather than a second opinion about it, which also removes the entire
  `probeTraceStatic` cost from the raster path.
- Compile as `split`, which means relaxing the gate at
  `lib/webgpu-svo-dry-scene.ts:1140` from "probe requires inline" to "probe
  requires inline **or** the split composition the frame is using", and adding
  the split lighting bind group to the probe pipeline layout (`:4759`).
  Production strings stay byte-identical: the probe entry is still appended only
  when `pixelProbe` is true.

In `traced` mode Probe B does not run and Probe A keeps `probeTraceStatic`. The
traced picker is unchanged, deliberately — it is the reference the raster mode is
read against.

Fix the traversal-mode leak while here: compile the probe module with
`canonical-parametric` when the production mode is `raster-primary`, so its
secondary rays match the secondaries production actually runs (`:1192`).

### 3.2 ABI

`SVO_PIXEL_TRACE_ABI_VERSION` → 4, magic → `0x53565404` (a stale mapped buffer
must never decode). Header grows 40 → 64 words; records stay 12 words, so bases
remain `64 + 12n`.

New record kinds, numbered after the existing thirteen so old numbers keep their
meaning:

| # | Kind | Meaning |
|---|---|---|
| 14 | `brickProxy` | One instance whose proxy box covers this pixel — one fragment the rasterizer produced. `a`/`b` are the proxy AABB, `level` the sort bucket, `detail` the instance index in draw order, `tEnter`/`tExit` the ray interval. |
| 15 | `leafBounds` | The winning proxy's full leaf AABB, so the occupied sub-box the emitter published (`lib/webgpu-svo-brick-raster.ts:331`) is legible against it. |
| 16 | `rigidProxy` | A rigid impostor quad covering this pixel, plus its analytic test outcome. |
| 17 | `glassPane` | The pane key the discovery pass wrote here. |
| 18 | `prepassTexel` | The coarse cone-lighting texel this pixel's visibility came from, or a marker that it was a boundary pixel resolved inline. |

New flags: `depthWinner`, `depthLoser`, `discarded` (DDA found no surface, so no
depth was written at all — a different cost and a different outcome from losing),
`hsrEligible` (an instance a tile-based hidden-surface pass could legally have
killed before shading; the picker must not claim it definitely ran).

New header words: `primaryMode` (raster | traced — the picker's own honesty
switch), `residentLeaves`, `emptyBricks`, `frustumCulled`, `candidatesEmitted`,
`instancesDrawn`, `coveringProxies`, `winnerInstanceIndex`, `winnerSortBucket`,
`ddaCellsAcrossProxies`, `terrainSteps`, `prepassState`.

`decodeSvoPixelTrace` returns the same shape plus a `raster?: {...}` block,
present only when `primaryMode` is raster. Absent block ⇒ the traced narrative,
unchanged. Nothing downstream has to branch on a version number.

### 3.3 Layers

Layers are what the reader toggles, and they are declared as visualizations
beside the pass that produces them (`lib/svo-pixel-trace.ts:1112`, the convention
from `085b21e`). Two sets, selected by the trace's own `primaryMode`, so a pinned
traced ray keeps traced layers while live raster traces show raster layers.

Raster set (pass names match the encoder's labels, so the legend and the profile
lane agree):

| Layer | Pass | Draws |
|---|---|---|
| `primary-ray` | — | The camera ray: solid to the winning surface, faint beyond. |
| `proxies` | brick raster | Every covering proxy box, ramped by draw order. This *is* the overdraw picture. |
| `proxy-losers` | brick raster | Covering proxies that shaded and lost: dashed if they `discard`ed (no surface), dimmed-solid if they were simply behind. |
| `winner` | brick raster | The winning proxy, bright, plus its leaf AABB as a hairline. |
| `cells` | brick raster | DDA cells the winning fragment stepped. Unchanged meaning. |
| `exact` | brick raster | Analytic surface tests. Unchanged. |
| `terrain` | background and terrain | Secant-march brackets. Promoted to its own layer: it is a separate pass every pixel pays. |
| `rigid` | rigid discovery | Impostor quads covering the pixel and their analytic tests. |
| `prepass` | cone lighting | The coarse texel the visibility was read from, drawn as its footprint on the surface. |
| `shadow-rays`, `cones`, `gi-cones` | deferred lighting | Unchanged. |

`hierarchy` and `rejected` are absent from the raster set — not renamed, absent.
Nothing in a raster frame corresponds to them, and a layer that draws nothing is
better than a layer that draws a fiction. Frame-scale cull work (empty bricks,
frustum rejects) is reported as HUD counters, never as geometry: those boxes were
rejected camera-wide, not for this pixel, and drawing them in the scene would
re-tell the old lie in a new colour.

### 3.4 The visual language

The traced picker's central image was a *path*: one ray threading boxes in
order. The raster picker's central image is a **tournament**: several boxes
covering one pixel, all shaded, one winning. Three devices carry that.

**Depth ladder (the headline).** Along the camera ray, one horizontal tick per
covering proxy at its `tEnter`, with a bar spanning `[tEnter, tExit]` — the
interval that proxy's fragment actually clamped to. Bars are ordered along the
ray, which is also depth order, which is also (approximately) draw order, so the
counting sort's quality is visible: a well-sorted pixel reads as a monotone
staircase, a badly-sorted one as a jumble. The winner's bar is filled and carries
the hit marker; losers are hollow. Rendered both in 3D along the ray and as a
compact strip in the HUD, sharing one colour ramp.

**Order ramp.** Proxy boxes are coloured by draw order, not by a flat layer
colour — cool for the first-drawn, warm for the last. One glance answers "did
the sort put the winner near the front?", which is the raster path's only real
performance question at a pixel. The existing mip ladder proves the pattern
(`lib/svo-pixel-trace.ts:423`): colour-by-index is how this codebase already
makes a hierarchy legible.

**Proxy versus leaf.** The winning instance is drawn as two nested boxes: the
occupied sub-AABB the emitter published, solid, inside the leaf's full AABB,
hairline. The gap between them is the occupancy word doing its job — the reason
fewer fragments were produced than a naive leaf-box instancing would produce. It
is free to draw and it explains a real optimization that is otherwise invisible.

Two rules the implementation must hold to:
- **Discarded is not rejected.** A proxy that ran its DDA and found nothing cost
  a full fragment invocation. It gets a distinct dash pattern and its own
  counter; folding it in with the frustum-culled instances would understate the
  pass's real cost.
- **Never claim a fragment ran when the tiler may have killed it.** Anything
  flagged `hsrEligible` is drawn at reduced intensity and the HUD says
  "≤ N shaded" rather than "N shaded". The `rasterPrimaryHsrProbe` experiment
  (`lib/webgpu-svo-dry-scene.ts:1031`) exists precisely because that number is
  not directly observable.

### 3.5 HUD

Narrative steps become the pass ladder of §2, in encoder order:

1. **Emit and cull bricks** — `resident → empty → frustum-culled → drawn`, and
   which of the 1024 buckets the winner landed in. Badged as frame-wide, not
   per-pixel; this HUD has an existing habit of badging gathered versus scheduled
   figures separately and never adding them (`components/VisualPanel.tsx`), and
   the same discipline applies here.
2. **Draw the background** — terrain steps, hit or miss, the depth the bricks
   then competed against.
3. **Rasterize brick proxies** — "N proxies covered this pixel; K found a
   surface; the depth test kept the nearest." The one number worth reading.
4. **Walk the winning brick** — cells stepped, exact tests issued.
5. **Rasterize rigid impostors** / **Discover glass panes** — when active.
6. **Shade from the G-buffer** — which of the two complementary lighting draws
   claimed this pixel.
7. **Query visibility** / **March the coverage pyramid** / **Gather global
   illumination** — as today.

The headline number changes. "Units of work for this one pixel" currently sums
counters that no longer exist; replace it with a **stacked stage bar** — cull
(badged frame-wide), terrain, brick raster (Σ DDA cells over *all* covering
proxies, not just the winner), rigid, glass, lighting. Under raster the
interesting fact is usually that the primary is now a small slice and lighting
dominates, which is exactly what `4c1e68c` measured at the frame level
(26.95 → 4.89 ms primary; 49.62 → 29.01 ms whole frame) and what a per-pixel
readout should corroborate.

Also fix the copy: `blockedReason`'s `path-inactive` text
(`components/PixelTraceHud.tsx:47`), the footnote's "mirrors the shipping
traversal, brick walk, and cone step law" (`:201`), and the control-group note in
`components/VisualPanel.tsx:324` which describes octree descent in detail.

## 4. Phases and gates

Branch discipline per repo policy: no stash/checkout/reset in this worktree.

### Phase 0 — truthful degradation (½ day)
Ship the honesty before the feature. Detect `primaryMode = raster`, suppress the
`hierarchy` / `rejected` / empty-skip narrative, and say plainly that the primary
is rasterized and not yet traced by the picker. Fix the probe's traversal-mode
leak (§3.1) and the stale copy.
**Gate:** no raster frame renders a layer or a sentence describing per-pixel
octree descent. The traced mode is untouched — verified by
`tests/svo-pixel-trace.test.ts` passing unmodified.

### Phase A — ABI and decode (1 day)
Version 4, new kinds/flags/header, `raster` block, HUD reading it. No new GPU
work: hand-built fixtures only.
**Gate:** round-trip tests for the new records; a v3 buffer decodes to the traced
shape; a mismatched magic still returns `undefined` rather than a half-believable
trace.

### Phase B — Probe B, cull and coverage (2–3 days)
The standalone compute probe: cull counters read from the real sort state,
covering proxies from the real sorted instance buffer, per-proxy DDA, winner
election.
**Gate — the parity oracle.** The probe's elected winner must agree with the
frame's own G-buffer at that pixel: same depth to within the reversed-Z epsilon,
same owner id, same material id. Both are readable, so this is a genuine oracle
and not a self-consistency check. It runs on the garden scene across a spread of
pixels — surface, silhouette, sky, terrain-over-brick — and belongs in CI.

### Phase C — Probe A retargeted (2 days)
G-buffer-sourced hit; split composition; prepass provenance records.
**Gate:** on a lit pixel the probe's cone/shadow/GI counters are unchanged from
the traced path on the same camera and scene (lighting did not change, so the
numbers must not either), and the reported surface is bit-identical to the
G-buffer sample.

### Phase D — the visuals (2–3 days)
Depth ladder, order ramp, proxy-versus-leaf nesting, layer registry split by
mode, stage bar, narrative rewrite.
**Gate:** on a pixel with known overdraw the drawn proxy count equals the decoded
`coveringProxies`; discarded and depth-lost proxies are visually distinct; the
traced mode's rendering is pixel-identical to before the change.

## 5. Risks and pre-planned answers

| Risk | Signal | Answer |
|---|---|---|
| Probe B's coverage set disagrees with what the rasterizer produced (near-plane instances take the full-screen-triangle path, `lib/webgpu-svo-dry-scene.ts:1562`) | Winner parity fails with the camera inside a brick | Mirror the same margin test; it is four lines and it is already written down |
| Scanning every instance is slow on a dense scene | Probe frame time spikes on garden | Strided workgroup scan; stop after the bucket containing the winner's far depth, and report the truncation rather than hiding it |
| HSR makes "N proxies shaded" an overstatement | Unknowable directly | `hsrEligible` flag + "≤ N" phrasing; the existing HSR experiment quantifies the gap when someone wants the number |
| Two probes, one record buffer, one of them not run | Half-populated trace decoded as complete | `primaryMode` plus per-stage presence bits in the header; a stage that did not run is drawn as absent, never as zero |
| Split-composition probe pipeline drifts from the production split shader | Silent divergence, the exact failure this document exists to fix | Probe A reads the G-buffer instead of recomputing it, so there is almost nothing left to drift; what remains is the lighting mirror, which the Phase C gate pins |
| The picker becomes raster-only and the traced oracle rots | Traced mode breaks unnoticed | Both modes stay in the layer registry and both are exercised by tests; the traced path is the parity oracle for the renderer too |

## 6. What changed during implementation

**Probe A reads the production tracer, not the G-buffer.** The plan had the
lighting probe read the G-buffer at the requested pixel, which meant compiling it
under the split composition and binding the split lighting group. That turned out
to buy less than it cost: the raster probe already elects a winner independently
from the frame's own instance list, so the two probes are *already* two routes to
one answer. Rather than collapse them into one, the merge compares them and
reports disagreement (`primaryParity`) — the plan's Phase B "parity oracle" gate,
but live in the HUD on every pixel instead of only in CI. Probe A in raster mode
therefore calls the uninstrumented production `traceStaticSolidScene` and emits no
traversal records at all; the surface it reports is corroborated, not asserted.

**Terrain reports an interval, not a step count.** Counting secant-march steps
honestly needs a counter inside the shared `terrainField`, and the scale-1
production string is gated byte-for-byte by the frame fingerprint
(`lib/webgpu-svo-dry-scene.ts:1095`). The probe mirrors only the march's *bracket*
setup — six lines of pure arithmetic, not the search — and records the interval it
was allowed to look in plus how it ended. Terrain is consequently absent from the
stage-cost bar: a segment sized by an invented number would make every other
segment a lie by comparison.

**The cone prepass is stamped by the host.** The probe is composed inline at full
rate by construction, so from inside it the reduced-rate pass beside it is
invisible. `withSvoPixelTraceConePrepass` adds the fact after decode, from the
renderer, which knows exactly. The per-pixel question — was *this* pixel a
boundary pixel resolved inline — is knowable only to the reduced shader, so the
`boundaryResolved` bit the plan proposed was dropped rather than guessed at.

Two defects the plan predicted were confirmed and fixed: the probe module's
traversal-mode leak (`raster-primary` matched neither the canonical nor the
compact branch and fell through to the wide-fanout cursor — the fix keys on the
traversal mode alone, not on whether the composition emits raster entries), and
the stale "raster fallback" copy.

One defect the plan did not predict, found while building: the raster record kinds
`terrainStep` and `rigidTest` had been in the ABI since version 1 and **no probe
had ever emitted either**. Terrain and rigid bodies were entirely absent from the
picker in both modes. Both are now recorded — terrain as its bracketed interval,
rigid as every body whose bounding sphere the ray pierces, which is the impostor
pass's own coverage test.

## 7. Files

Changed: `lib/svo-pixel-trace.ts` (ABI v4, mode-split layers, raster narrative,
stage costs, order ramp, merge + parity), `lib/webgpu-svo-pixel-trace.ts` (Probe A
mode branch, terrain and rigid records), `lib/webgpu-svo-dry-scene.ts` (probe
composition, traversal-mode fix, Probe B lifecycle and dispatch, merged readback),
`components/PixelTraceHud.tsx`, `components/VisualPanel.tsx` (copy),
`app/globals.css` (stage bar, depth ladder, frame-wide badge).

New: `lib/webgpu-svo-brick-raster-probe.ts` (Probe B module + host buffers),
`tests/webgpu-svo-brick-raster-probe.test.ts`.

Unchanged: `lib/svo-picking.ts`, `lib/webgpu-svo-picking-readback.ts`,
`components/FluidCellTraceHud.tsx`, `lib/webgpu-renderer.ts`.

## 8. Verification

- `tsc --noEmit` clean across the touched files.
- All 293 unit tests in every suite importing the changed modules pass.
- Real-device WGSL validation (Dawn/Metal, `WEBGPU_NODE_MODULE`): both probes
  compile and build their pipelines, including the raster probe against a device
  pinned to the guaranteed 16 KiB workgroup-storage minimum — which is a real
  gate, not a formality: the first version of the per-proxy record was 96 bytes
  and overran it at 18,480 bytes.
- The pixel-trace probe now compiles under `raster-primary` as well as the five
  traced compositions.

Not verified: no frame has been rendered against this. The parity check between
the two probes is the intended first signal when one is.
