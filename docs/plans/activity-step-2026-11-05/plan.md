# Step cost proportional to activity (2026-11-05)

Goal (user): halve the NB-FLIP frame time on a large hydrostatic all-BULK16
tank without fast paths, by a general work reduction that scales with what
moves. The next scene, a small disturbance in one corner, must cost the
floor plus work in proportion to the disturbance.

## Measurements (main cd6b085f, one run each, pipelined)

`tools/probe-reservoir-bulk.ts --pure --cell=0.01` (lib/core/bulk-rest-tank-scene.ts).

| tank | mode | median ms | dispatches (direct+indirect) | regions COLD/WARM/HOT |
|---|---|---|---|---|
| 16x8x8 (128 surface columns, 896 BULK16 pages) | default (`coldSurface:"warm"`) | **18.9** | 364+37 | 1024/1024/0 |
| 16x8x8 | `coldSurface:"still"` (global idle) | 8.05 | 114+37 | 2048/0/0 |
| 16x8x4 | default | 11.7 | 363+37 | 512/512/0 |
| 16x8x4 | still (global idle) | 4.4 | 114+37 | 1024/0/0 |
| 16x8x4 + `--drop=8` (8^3 cube over the -x/-z corner) | default | ~3x rest | 416+69 | 17 HOT, 487 h tiles, 55k particles |

Per-kernel attribution of the 16x8x8 tank (kernel timer, distorted; sums):
- **Pressure is the floor in both modes, about 7.5 of 9.5 ms when idle.**
  - The arena solve takes 5.2 ms and pTail alone 1.38 ms. pTail is the
    single-workgroup 32h tail; it grows with the box, not with activity.
  - Project takes 1.16 ms, bulkSelect 0.41 and buildRhs 0.36.
  - At rest the solve runs 1 V-cycle (each with 3 tail cycles) on most
    steps. The accepted residual hovers at 0.02 to 0.05 against the 0.05
    target, where an exactly hydrostatic warm start should read round-off.
- **The warm-only remainder is about 12 ms over 1024 still surface regions:**
  - extension 1.85, census 1.3, local speed 0.6 and momentum 1.0 ms;
  - surface advect 1.0, regional air 0.48, layout 0.45 and forces 0.41 ms;
  - and so on.

  The stages stride the pool's resident list (uniform-live-lists.ts), not
  the regions that can change.
- "Global surface volume" (about 1.2 ms) is the probe's diagnostic only.

## Diagnosis

1. The only proof that nothing changed is global (UniformColdFrame quiet /
   NarrowBandFlip.idle). The first non-COLD region anywhere encodes every
   stage over every resident page.
2. `coldSurface:"warm"` (the default) never certifies a still surface, so the
   tank's surface layer is transported every step at 4h with zero velocity.
3. The pressure solve's work is sized by the domain, not by the residual's
   support:
   - fine levels smooth every brick and page;
   - the coarse tail runs in one workgroup over every page;
   - a converged solve still launches its passes.

## Plan: two lanes

### Lane P: pressure work follows the residual

1. **Exact hydrostatic fixed point.** Find why the still tank's initial
   residual is above round-off. Likely suspects:
   - the BULK16/4h interface's gravity against pressure-gradient
     discretisation;
   - the freeze accumulate;
   - the warm start's copy.

   A well-balanced discretisation keeps rest exactly; this is correctness,
   not a gate. Expected result: 0 cycles at rest.
2. **A converged solve launches nothing.** The measure of the start (and of
   each cycle) writes indirect arguments, and the cycle passes take them,
   so a met residual launches 0 workgroups. The work-proportional-fluid
   notes measured 2.2 ms of closed passes at 0 cycles.
3. **Fine-level smoothing over the residual's support.** The measure lists
   bricks and pages whose residual exceeds a fraction of the target, plus a
   one-brick halo. The h, 2h and 4h smoothers stride that list; the coarse
   levels (16h, 32h) stay whole and small. pTail becomes parallel, or the
   existing superblocks tail is measured against it.

### Lane S: surface-region stages follow the active regions

1. **Per-region still certificate as the default.** The "still" rule is
   already on main (ca2d251b, e70d0a3a): omitted transport within f32
   round-off a step and 0.01 h summed, and no moving neighbour. Its history:
   letters diverged at frames 31/32 (tiered-adaptivity handoff §5 A1). Fix
   that on its cause; don't gate around it.
2. **Stages launch over the active region list.** Region heat already
   publishes it (UNIFORM_REGION_HEAT_ACTIVE_WORD: live and not deep COLD).
   Each stage that strides the resident list today strides the non-COLD
   regions' blocks instead (indirect, or a fixed grid over that count).
   COLD regions hold their outputs, as the global idle already proves they
   can. The global idle then becomes the empty-list case of the same
   mechanism, not a separate path.

## Acceptance

- 16x8x8 default: at most 9.4 ms median (from 18.9), with no `coldSurface`
  or other override.
- 16x8x4 `--drop=8`: report the step time against rest at the same size.
  The excess must track the disturbance (HOT/WARM regions, h tiles,
  particles), not the tank.
- Every commit: types, unit tests, and the short cross-scene check with
  drift and step time:
  - seep 300;
  - letters 120;
  - the tank, 60 steps.

## Test protocol (short: user, 2026-11-05)

- One run per configuration, 60 steps, `--every=60`, `--pipelined`, through
  `tools/gpu/fluid-gpu-run.sh timing`.
- A run whose shaders are cached constructs in about 25 s. A new shader set
  takes about 3.5 min to compile, once.
- Kernel attribution only when needed: `--kernels=40 --steps=60`.
- No repeats; no 180-step runs.
