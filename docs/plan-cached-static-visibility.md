# Plan: factored direct-light visibility (static cached × dynamic traced)

**Goal:** shadow rays dominate per-pixel ray work (~85%, ~39 node visits/ray, cost linear in
light count). Replace per-pixel exact traces against the *static* world with a persistent
world-space cache; keep dynamic occluders exact but cheap.

**Core factorization:** `visibility(pixel, light) = staticCache[brick, channel] × dynamicTrace × fluidCone`.
Nothing that moves is ever cached; the cached term never invalidates while static topology holds.

## Phase 0 — Measure (de-risk, ~half day)
Extend the pixel-trace visibility counters (`svo-pixel-trace.ts`, `webgpu-svo-pixel-trace.ts`)
to classify each exact trace per light: trivially-lit / blocked-early / partial transmittance.
The trivially-lit+blocked fraction is the expected cache hit rate. If partials dominate, stop
and rethink (per-cell bitmasks become mandatory).

## Phase 1 — Static visibility cache
- New buffer keyed (leaf brick × light channel), channels per the fixed ABI in `svo-light-abi.ts`.
  Entry: 2-bit state — LIT / SHADOWED / MIXED / UNKNOWN. (Per-cell bitmask for MIXED bricks only
  if Phase 0 says it's needed.)
- Cache key/versioning: reuse the publication hashing in `svo-static-shadow-field.ts`
  (node-mip topology hash + light revision). Topology or light republish ⇒ whole cache resets to UNKNOWN.
- Shader path: on shadow ray, read entry. LIT/SHADOWED ⇒ done. UNKNOWN ⇒ exact trace against
  static-only geometry, write result back (lazy warm-up; optionally budget N writes/frame).
  MIXED ⇒ exact static trace, no write-back.
- Exact trace must be filterable to static-only occluders (skip rigid-body/fluid bricks or use
  the static publication's topology directly).

## Phase 2 — Dynamic occluder term
- Per shadow ray, test against dynamic rigid bodies only: brute-force AABB + primitive tests
  over the (small) mover list from `svo-scene-primitives.ts` / `svo-primitive-motion.ts`.
  No hierarchy descent. Multiply into cached static visibility.

## Phase 3 — Fluid occlusion term
- Replace exact fluid shadowing with cone march over the existing coverage pyramid
  (`svo-fluid-coverage.ts`, same mip path the GI cones use). Soft fluid shadows are acceptable.
- If exactness is later required: invalidate via the solver's dirty-page set → per-light shadow
  frusta → mark intersected bricks (same mechanism as Phase 4).

## Phase 4 — (only if MIXED write-back or exact fluid needed) causal invalidation
Per frame, per light: swept AABB (prev ∪ curr pose) of each mover → light frustum → mark
intersected bricks UNKNOWN. Skip entirely if Phases 1–3 suffice — the factorization is designed
so they should.

## Verification
- Pixel-trace HUD: visibility units on a warm cache should drop from ~255 to <30 for a
  static-scene pixel; confirm on hose-filled-tank (seed 2011), the pinned-pixel workflow.
- Parity oracle: rendered image with cache vs. cache disabled must match exactly for LIT/SHADOWED
  bricks (static scene, no movers). Add a toggle + diff test.
- FPS on hose-filled tank with 8 lights: expect it to approach the current 1–2-light number.
- Regression: moving a rigid body through a light beam must darken stationary surfaces it
  shadows (the case screen-space caches fail); add a scenario test.

## Non-goals
Screen-space/temporal shadow caching (can't causally detect moving-occluder shadows on static
surfaces); GI changes (cones already amortized); light culling/importance sampling (separate win).
