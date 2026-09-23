# Airborne momentum in symmetric expansion (2026-09-23)

## Airborne-off symmetry fixes

The corrected default Uniform Geometric solver completes 30 frames with **zero
D4 column-height differences at every frame**. Mean conservative-volume D4
error at frame 30 falls from **0.0237686 to 0.0000294501**, an **807×** reduction.
No simulation pass was added and no field was averaged with its reflection.

| Frame | Original mean volume D4 error | Corrected | Corrected height D4 error |
| ---: | ---: | ---: | ---: |
| 2 | 9.0619e-6 | 3.1782e-8 | 0 |
| 20 | 0.0246292 | 8.8798e-7 | 0 |
| 30 | 0.0237686 | 2.9450e-5 | 0 |

Four causes were corrected:

1. **Source provenance lost tied contributors.** The hierarchy interpolated
   equally near source velocities, but retained the first contributor's single
   coordinate. Subsequent levels measured the interpolated value's distance
   from that arbitrary corner. The fix carries lower/upper bounds of the
   represented source patch, in two slabs of the existing provenance textures.
   Aligned MAC stencils also exclude the zero-weight positive plane, and the
   cell-centred fallback is restricted to the vertical component. Distances
   are compared in cell units to preserve geometric ties on cubic grids.
2. **A pressure dual-cell fraction was mistaken for an open wall.** Closed
   positive tank faces have pressure fraction 1/2. Extension accepted those
   faces, filled them with velocity, and propagated them as sources. There is
   no corresponding stored negative wall face. Extension now excludes closed
   positive domain faces while retaining an authored atmospheric +Y boundary.
3. **Red/black pressure order changes under reflection.** On an even grid a
   reflection swaps colours. Uniform Geometric now updates pressure
   simultaneously using the same two dispatches per sweep. Fused small-level
   visits and the coarse solve take an iteration snapshot within their
   existing dispatch. The reference CM11a path keeps its original schedule.
4. **Redistancing accepted steps away from the contour.** Clipped Newton
   steps crossed distance ridges and could turn tiny input differences into
   different distant roots, or convergence on only one side. The existing
   iteration now rejects a step whose absolute phi residual does not decrease;
   unsuccessful measurements retain the advected value, as before.

The first-frame pressure reflection error dropped from 0.111 Pa to about
0.000244 Pa. The original second-frame extension amplified roughly 0.000104
m/s of incoming error into 0.144 m/s. With the corrected hierarchy, second-frame
velocity errors remain around 5e-7 m/s.

Small residual differences remain. At frame 30 the maximum volume
mismatch is 0.037 cell volumes. Phi still has isolated same-sign distance
errors inside liquid (maximum 0.0955 m), but **no reflected sign mismatches**
at that frame. These remaining redistance errors should be distinguished from
surface displacement.

A broad exact-source tree was tested and rejected because its 256³ search
cost was excessive. The retained support-bounds transfer examines the same
eight local candidates as the prior hierarchy and retains its pass count.

The new `uniform-extension-symmetry-dawn` test covers tied sources and the
half-open pressure dual faces. `uniform-symmetric-expansion-dawn` checks all
30 frames against mean volume D4 error <1e-4 and exactly matching column
heights. The coarse-solver manufactured fixtures exercise both update schemes,
including grids larger than 256 rows.

## Original A/B reproduction

`tools/probe-uniform-airborne-expansion-dawn.ts` runs the 32 × 16 × 32
`symmetric-expansion` scene on Uniform Geometric for 45 steps at 1/30 s. Both
arms use the current default parameters; only `airborneMomentum` differs. It
reads conservative volume, velocity, phi, and stage audit fields at every step.
It measures raw-volume D4 error, liquid-weighted kinetic energy, and a
liquid-weighted squared horizontal velocity second difference as a local
velocity-noise proxy. These quantities are diagnostic, not a perceptual image
metric. The run was sequential under the repository WebGPU lease.

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
  node --import tsx tools/probe-uniform-airborne-expansion-dawn.ts
```

The original A/B below was measured before these fixes, against a working tree
that already contained unrelated Uniform Geometric changes. Its frame numbers
and toggle-specific effects are historical measurements, not claims about the
corrected solver.

## Original A/B observations

| Step | Time | Airborne off | Airborne on | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.033 s | D4 volume error 0 | same | Authored symmetry is exact. |
| 2 | 0.067 s | D4 mean error 0.0000091, maximum 0.0085 | identical | Symmetry begins drifting before airborne momentum participates. |
| 20 | 0.667 s | D4 mean error 0.0246 | identical | The substantial early symmetry disturbance is common to both arms. |
| 24 | 0.800 s | Projected face velocity 0 | −0.494 m/s at (16, 1, 16), Y | First A/B difference. Volume, phi, extrapolation authority, and velocity prediction still match. |
| 30 | 1.000 s | Kinetic proxy 312.3 | 313.4 | A/B difference is still small in total energy. |
| 45 | 1.500 s | Kinetic proxy 221.3; velocity-noise proxy 12.05 | 267.8; velocity-noise proxy 16.58 | Airborne on is 21% higher in kinetic proxy and 38% higher in local velocity-noise proxy. |

At step 24, cell (16, 2, 16), above the first changed face, holds 0.261 cell
volumes. Its interpolated phi is +0.07786 m. The airborne cutoff is 1.5 cells,
or +0.075 m, and its minimum volume is 0.05. This makes a partial-volume tail
immediately above the central body eligible for airborne treatment. The
neighbouring cell (16, 2, 15) similarly holds 0.177 cell volumes and has phi
+0.07787 m. The pressure projection is the first stage that differs: it keeps
the pre-existing downward predicted velocity on those faces when the toggle is
on, while the off arm clears them to zero. Gravity is not the first source of
the A/B difference, because velocity prediction is identical at this step.

On the following step the retained face velocity changes volume and phi. The
airborne cells then become extension sources: the authority field at
(15, 2, 15) is +0.501 with airborne on and −1.058 with it off. The later
differences in velocity advection and projection follow this feedback. This is
the largest identifiable toggle-specific route to disturbance in this scene:
a hard 1.5-cell/0.05-volume classification followed by preserving an entire
face velocity, then promoting the partial-volume cell to extension authority.

The 45-step run does **not** show a uniformly rougher free-surface height with
airborne on. At step 45, the four-neighbour column-height roughness proxy is
0.219 on versus 0.232 off; the D4 column-height error is 0.359 on versus
0.418 off. The velocity proxies show added motion, but its visible effect
depends on scene and time. The earlier D4 drift is a separate, larger source of
asymmetry that deserves its own stage-local investigation.

## Corrected airborne A/B

With these fixes, the on/off arms again agree through frame 23. The first
activation at frame 24 is now in four corresponding corner plumes: upward Y
faces at (2,7,2), (29,7,2), (2,7,29), and (29,7,29). The on arm applies one step
of gravity (about -0.327 m/s) and retains approximately +1.527 m/s at these
faces; the off projection clears them. Their initial spatial pattern is nearly
symmetric. Both arms still have zero D4 column-height error at frame 30.

By frame 45, kinetic proxy is 840.24 on versus 321.74 off, while the local
velocity second-difference proxy is 19.95 versus 17.60. Mean volume D4 error is
0.00478 on versus 0.00197 off. Airborne therefore still adds substantial late
motion; the corrected baseline makes its activation and subsequent amplification
separable from the former frame-2 hierarchy defect. Airborne classification and
momentum retention were not changed in this baseline repair.

## Validation

The 30-frame expansion regression passes with zero column-height symmetry error.
Native and paged pressure layouts agree. The compact source hierarchy takes
4.26–4.33 ms in warmed 256³ frames (the rejected exact tree took seconds).
The focused source-symmetry and isolated-drop extension tests pass. Both
coarse-pressure update schemes pass the manufactured constrained systems,
including 81×5×5 rows, and the authored wet garden advances. The geometric
separating-contact test passes. The numerical-invariants suite passes its first
ten subtests but retains its existing mini32 mass-conservation failure; the
same assertion is recorded in the pre-change working-tree log at
`tmp/ug256/tests-uniform.log` (20:21, before these fixes were enabled).
TypeScript reports existing errors in unrelated sparse tests and probe tools;
none refer to the files changed for this repair.

## Airborne-specific follow-up

Trace the projection and extension authority near the 1.5-cell and
0.05-volume thresholds, and distinguish partial-volume tails
connected to the main body from isolated droplets. Measure the same stage
fields and symmetry errors in a smooth-flow scene as well as this collapse.
Do not infer that a stronger V threshold alone is safe for small droplets.
