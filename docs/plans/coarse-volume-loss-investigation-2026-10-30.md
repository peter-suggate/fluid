# O6 investigation: no production fix adopted

The bounded fixes do not meet the handoff's goal. Per the user's stop rule,
abandon the production fix rather than expand this into a new surface method
or change Coarse-region semantics. O6 remains open. This change adds diagnostic
tools, explicit failing regression TODOs, and a passing construction/corner
test; it does **not** make coarse start safe to enable by default.

## Corrections to the starting assumptions

- The app's default `surfaceAdvection: balanced` uses **trilinear at 4h**,
  despite the geometric solver's standalone cubic default. See
  `lib/methods/uniform/uniform-narrow-band-controls.ts:60`.
- At h = 0.05 m, a 0.3 m slab is six **h** cells, only **1.5 coarse cells**.
  The handoff's table calls it six 4h cells. Both thicknesses were tested.
- Forced 4h owners **can have particles**. `seedCells` enumerates h cells
  within the geometric band, including 4h owners
  (`uniform-narrow-band-flip.ts:1626`). In the stage run, the slab had 11,476
  particles and the drop 11,791 at step 10, with zero fine tiles. Their coarse
  reconstruction is therefore not merely a copy of advected phi.
- The construction `readStats()` value is pressure fill until a requested
  phi gauge has run. Subsequent `volumeCellSum` uses the gauge. These are
  different quadratures, not time samples of one quantity. New regressions
  measure the same phi quadrature independently at construction and finish.

## Scope and reproduction

Tested the working checkout based on `dd9217ce`, including the pre-existing,
uncommitted detached-mass/free-flight changes. Other agents' work was preserved.
No production solver, scene, preset, capacity, or default was changed here.

The fixture is a 3.2 m box, h = 0.05 m, fixed dt = 1/60 s, no embedded solids,
volume control off. An all-box Coarse region holds every simulation tile at
4h. The h comparison removes that region and uses the production adaptive
census; it is not a forced uniform grid. All GPU jobs went through
`tools/gpu/fluid-gpu-run.sh check`, with one simulation at a time per slot.

The handoff does not provide slab bounds. These fixtures are explicitly:

- Thin upright slab: x = [-0.6, -0.3], y = [0, 1.2], z = [-0.6, 0.6] m.
- Resolved slab: x,z = [-0.6, 0.6], y = [0, 1.2] m, six coarse cells thick.
- Drops: centre (0, 2.2, 0) m, radii 0.2, 0.3 and 0.5 m.
- Film: x,z = [-0.6, 0.6], y = [0, 0.08] m.

The following commands reproduce the principal comparisons (set
`WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js`):

```sh
tools/gpu/fluid-gpu-run.sh check env O6_REST=1 O6_CASES=film,slab,resolved-slab node --import tsx tools/probe-coarse-surface-volume.ts
tools/gpu/fluid-gpu-run.sh check env O6_CASES=slab,resolved-slab,drop,small-drop O6_VARIANTS=baseline,no-smoothing,cubic,cubic-no-smoothing,fine node --import tsx tools/probe-coarse-surface-volume.ts
tools/gpu/fluid-gpu-run.sh check env O6_STAGE_STEP=10 O6_CASES=slab,drop O6_VARIANTS=baseline O6_STEPS=10 node --import tsx tools/probe-coarse-surface-volume.ts
```

Raw results are in `artifacts/coarse-volume-o6/`. Shader substitutions belong
only to the diagnostic device. They do not patch production source on disk.
Results vary with particle ordering; they are individual measurements, not
repeat-spread timing or statistical claims.

## Rest: smoothing changes the interface

These percentages compare step 1 to the final step, avoiding the initial
stats source change. Film runs for 60 steps; slabs for 30, gravity off.

| Fixture | Baseline | No smoothing | No redistance/retirement | Cubic |
|---|---:|---:|---:|---:|
| Film | -3.870% | 0.000% | -3.870% | -3.870% |
| Thin slab | -21.789% | 0.000% | -21.789% | -21.789% |
| Resolved slab | +0.431% | 0.000% | +0.431% | +0.431% |

The film's directly measured construction phi volume is 689.152071 cells;
after 60 steps it is 658.781280, **-4.407%**. Its first pressure-fill stats
value, 686.425781, is not an appropriate reference for this comparison.

The responsible operation is `umVertexFoot`'s smoothing contribution at
`lib/methods/uniform/uniform-mixed-surface.ts:978`, computed by
`lib/methods/uniform/uniform-surface-smoothing.wgsl.ts:18` and applied with
the advected value. It has no separate smoothing dispatch. Disabling that
term removes the rest drift; changing advection or redistance does not.

The added GPU stencil cases verify that a physical cubic is left unchanged
at both h and 4h. The coefficient and stencil are doing what their comment
says. A signed-distance field around a thin rim or spherical surface is not
that polynomial. Removing the filter would also remove its intended damping
of stationary lattice noise. A replacement that preserves these interfaces
and still controls that noise requires a different smoothing model; merely
zeroing epsilon does not solve motion.

## Motion and convergence

Again these are step 1 to final-step drift; 30 steps for slabs, 12 for drops.
The new regression tests instead compare directly measured construction phi
to final phi (see the following section).

| Fixture | Baseline 4h | No smoothing | Cubic | Cubic, no smoothing | h surface | Absolute drift ratio 4h/h |
|---|---:|---:|---:|---:|---:|---:|
| Thin slab | -81.095% | -83.646% | -78.292% | -72.175% | +1.484% | 54.63 |
| Resolved slab | -22.653% | -23.597% | -6.287% | -7.895% | +4.303% | 5.26 |
| r = 0.5 drop | -21.555% | -21.083% | -8.422% | -8.787% | +0.452% | 47.70 |
| r = 0.2 drop | -94.441% | -94.961% | -87.896% | -52.294% | -8.356% | 11.30 |

These are **absolute drift ratios**, not ratios of losses when h gains
volume, and not a formal convergence order: the h case also changes particle
coupling and the adaptive layout. All exceed the handoff's 4x heuristic.
The resolved slab's h gain also means a 3% slab gate is not universally
solved by removing the Coarse region (consistent with the separate O4 work).

Independent construction-to-final regression measurements:

| Fixture | Initial phi volume | Final phi volume | Drift |
|---|---:|---:|---:|
| Thin slab, 30 steps | 2893.333336 | 557.773784 | -80.722% |
| Resolved slab, 30 steps | 13120.000002 | 10507.108870 | -19.915% |
| r = 0.5 drop, 12 steps | 3855.333213 | 3055.359005 | -20.750% |
| Film, 60 steps | 689.152071 | 658.781280 | -4.407% |

There is no accepted after-fix column: no candidate met the full goal.
The handoff's r = 0.2 result (132 to 3 by step 10) is not deterministic;
this sweep measured 138.827 to 7.717 from steps 1 to 12. Neither result
supports relying on that feature at 4h.

The r = 0.3 m check also reproduces severe loss: construction phi volume
713.403956, step 1 volume 710.089355, step 12 volume 189.945770
(**-73.375% from construction**, versus the handoff's approximately -65%).
It has 2,765 particles by step 12 and zero fine tiles (`medium.jsonl`).

## Pass attribution, step 10

The probe inserts read-only spatial volume measurements into the existing
encoder. Full coarse owners use exactly the scalar gauge's six-tetrahedron
quadrature. The partial advection scratch is not a finished field until
deferred work completes; its intermediate volume must not be interpreted as
a separate conservative state.

| Boundary | Thin slab volume | r = 0.5 drop volume |
|---|---:|---:|
| Before advection | 1717.018365 | 3283.782402 |
| After main advection, deferred work pending | 1766.091733 | 3300.578141 |
| After deferred advection and smoothing | 1639.611578 | 3217.880666 |
| After scratch vertex resolve | 1639.611578 | 3217.880666 |
| After particle reconstruction | 1625.117796 | 3183.459455 |
| After clear-air | 1625.117796 | 3183.459455 |
| After retirement | 1625.117796 | 3183.459455 |
| After redistance | 1625.117796 | 3183.459455 |

Thus advection plus smoothing loses 77.407 slab cells and 65.902 drop cells;
reconstruction loses another 14.494 and 34.421 respectively. Reconstruction
is `couple` in `lib/methods/uniform/uniform-narrow-band-surface.wgsl.ts:291`.
Examples of affected h-coordinate coarse origins:

- Slab reconstruction: (20,4,32) loses 3.310 cells; (20,8,36) loses 2.981.
- Drop reconstruction: (32,48,28) loses 4.118; (28,48,32) loses 4.113.

The JSON contains the worst twelve cells at each boundary. The independent
spatial probe agrees with final `volumeCellSum` in every regression case.
This is not a pressure-fill-only reporting artifact. Classification and
pressure can affect the later velocity, but the measured phi loss occurs
before them in this step.

## Remaining local alternatives tested

With cubic and smoothing disabled, bypassing the certified cheap/Euler
transport shortcuts still gives -4.241% on the resolved slab and -9.778%
on the drop, measured from construction. Removing the cubic extrema limiter
gives -5.712% and -8.772%. Neither is a fix; the limiter is not the dominant
cause. See `alternatives.jsonl`.

The coarse particle-coupling ablation keeps advected phi at coarse vertices
instead of applying particle reconstruction. It is diagnostic, not a valid
replacement for NB-FLIP surface coupling:

| Fixture | No coarse coupling | Cubic, no smoothing, no coarse coupling |
|---|---:|---:|
| Thin slab | -88.405% | -40.391% |
| Resolved slab | -20.917% | +0.763% |
| r = 0.5 drop | -15.804% | +0.235% |

These percentages are construction-to-final. Although the combination helps
the two larger features, it removes the samples' control of the surface and
still loses 40% of the thin slab. Adopting it would hide part of O6 while
changing the method and leaving the required thin-slab test red.

## Construction and method implications

The passing construction test checks the actual canonical GPU phi corners
against the analytic r = 0.2 sphere to 1e-6 m. The grid's six-tet restriction
of those analytic corners encloses about **142.765 cells**, versus the
continuous sphere's **268.083 cells**. This initial deficit is a resolution
effect, not the earlier corner-load bug. No volume has been restored or
injected to make the test pass.

[Enright et al. 2002](https://physbam.stanford.edu/papers/stanford2001-04.pdf)
explicitly discuss loss of thin features on a level-set grid and use marker
particles to retain interface information. [Ferstl et al. 2016, sections
3 and 3.2](https://research-explorer.ista.ac.at/download/1415/4940/IST-2016-611-v1%2B3_CW_nbflip_postprint_2016.pdf)
place FLIP particles at the liquid surface and couple their surface with the
advected interior field. [Losasso et al. 2004](https://physbam.stanford.edu/papers/stanford2004-02.pdf)
refine the octree near the liquid interface. None justifies restoring lost
volume by a scene-specific correction.

The consistent next project is to guarantee adequate surface representation
when the census wants h: surface-priority capacity/reservation and explicit
failure when that capacity cannot be met, plus a deliberate decision about
Coarse regions over a moving surface. `settleOf` currently keeps a refused
claim at 4h (`uniform-mixed-layout-builder.ts:749`). Adding a fatal alone
would expose insufficient capacity; it would not make forced-Coarse surfaces
conserve volume. Changing Coarse-region precedence is expressly reserved for
a ruling in the handoff. This investigation does not make that decision.

## Verification and gates

- GPU regression run: construction and cubic-nullspace checks pass; four
  volume regressions execute and fail as explicit O6 TODOs. They are not
  counted as fixed. Existing merge TODOs remain open.
- `npm run check:types`: passes, zero errors (`types.log`).
- `npm run test:unit`: 1,353 pass, zero failures, 221 GPU tests skipped
  (`unit-final.log`). This does not imply that the O6 GPU TODOs pass.
- Existing free-fall acceleration and film-rest checks: both pass. The
  existing film-volume TODO still fails (`free-fall-film.log`: two pass,
  one TODO, zero non-TODO failures).
- Seep 1200 (3% and every 150-step receiving window), letters 240 (3%), rest
  ladder 1/1.5/2/2.5 m (zero), pipelined timing 3+3, and the full GPU sweep
  **were not run for an accepted candidate**, because there is no candidate.
  No cross-scene or timing pass is claimed. Diagnostic toggles must not be
  promoted on the strength of these small fixtures alone.
- Production construction bytes are unchanged by this patch. The spatial
  probe allocates temporary diagnostic buffers only when explicitly invoked.
- Scene definitions, presets, default start, M5 and the slot-accessor cost
  are unchanged. The user's pre-existing changes remain uncommitted as found.
