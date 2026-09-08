# Retained density: visual and physical A/B — 2026-09-08

**Both arms are unacceptable during motion.** Verified all-fine native cells
do not preserve the initially smooth sphere. The coarse arm additionally
develops strong physical asymmetry and detached box-shaped surface components.
This evidence does not justify a surface smoothing patch or a weaker tolerance.

## Captured production comparison

The actual `coarse-first-pool-impact-quarter` document is simulated using the
shipping balanced method: `32 × 24 × 32`, `h = 0.05 m`, 128 pressure iterations,
the authored `1/30 s` step, and checkpoints 0, 6, 15 and 30
(`0, 0.2, 0.5, 1 s`). The coarse arm retains the literal URL region
`0_0_0_25_66.6667_100_8_8`, snapped to
`[-0.8,0,-0.8] … [-0.4,0.8,0.8]`. The fine arm replaces it with a whole-domain
min1/max1 region. Every active fine-arm leaf has native width 1 at every
checkpoint, including signed frontier leaves.

The root task ran the two arms serially under the GPU lease. They completed in
29.13 and 28.41 seconds; logs are `/tmp/fluid-visual-quarter-coarse-1.log` and
`/tmp/fluid-visual-quarter-fine-1.log`. Capture files are under
`artifacts/retained-visual-ab/quarter/{coarse,fine}/step-N/`.

- [Matched mesh contact sheet](../artifacts/retained-visual-ab/quarter/mesh-comparison.png)
- [Physical front sections](../artifacts/retained-visual-ab/quarter/section-comparison.png)
- [Measured-checkpoint animation](../artifacts/retained-visual-ab/quarter/motion-checkpoints.gif)
- [Machine-readable comparison](../artifacts/retained-visual-ab/quarter/comparison.json)

These views render freshly emitted shipping GPU triangles without rebuilding,
smoothing or projection. Their flat QA shading exposes triangle facets and is
not the app's water optics. The front sections establish geometric distortion
independently of shading. The animation holds measured checkpoints; it does not
interpolate unseen states. The orange pre-impact sphere follows continuous
ballistic motion; finite-step temporal error must be separated from shape error.

## Observations

At reset both arms have a flat pool and spherical component. At 0.2 seconds,
before the approximately 0.23-second ballistic impact time, both spheres have
flat/stepped portions in their actual triangle cross-section. The coarse sphere
is also lopsided. At 0.5 and 1 second, the coarse arm has a weak, one-sided
depression and rebound; the fine arm is substantially more symmetric but still
has unacceptable geometric degradation. Min1/max1 is therefore a useful
diagnostic reference, not an accepted solution.

Native density and velocity readbacks establish a physical difference as well:

| Time | Coarse density X-mirror L1 / amount | Fine density X-mirror L1 / amount | Kinetic energy coarse / fine |
| --- | ---: | ---: | ---: |
| 0 s | 0.03787 | 0 | 0 / 0 J |
| 0.2 s | 0.05925 | `1.81e-7` | 113.50 / 125.91 J |
| 0.5 s | 0.14120 | `2.81e-5` | 86.86 / 135.13 J |
| 1 s | 0.19740 | `3.39e-5` | 38.88 / 92.65 J |

The coarse reset baseline already includes asymmetry from expanding unequal
native cell averages onto a fine diagnostic array; it is not evidence that the
initial continuous source is asymmetric. Its subsequent increase, together with
the energy difference and horizontal center of mass, shows evolving native
physics differs. At 1 second the coarse X center of mass is −9.297 mm, versus
−0.000021 mm for the fine arm. At 0.2 seconds their maximum liquid speeds are
nearly equal, 1.9617 and 1.9613 m/s: gravity and velocity advance while shape
quality is lost.

## Surface amount and detached components

At 0.2 seconds both meshes are closed and manifold. Their signed enclosed
volumes are 1.075487 m³ (coarse) and 1.084684 m³ (fine), compared with native
density amounts 1.089777 and 1.089778 m³. At reset, mesh volumes are already
1.087422 and 1.087505 m³. A half-density enclosed volume is not the integral of
a diffuse density; these quantities must not be equated or used as a new
acceptance threshold. The coarse suspended component shrinks from 0.063422 to
0.052952 m³ before impact; the fine component shrinks from 0.063505 to
0.060655 m³.

The coarse detached boxes deserve a separate field/publication investigation:

| Time | Component bounds in metres | Mesh enclosed volume | Expanded native amount in same box |
| --- | --- | ---: | ---: |
| 0.2 s | X [.025,.075], Y [.525,.575], Z [.725,.8] | `1.875e-4 m³` | `3.576e-10 m³` |
| 0.5 s | X [.225,.275], Y [.825,.875], Z [−.375,−.325] | `1.25e-4 m³` | `2.384e-10 m³` |
| 0.5 s | X [.225,.275], Y [.825,.875], Z [.325,.375] | `1.25e-4 m³` | `2.384e-10 m³` |

All intersected native samples in those boxes have density at most
`1.9073486328125e-6` (`2^-19`). The box integral uses physical overlap volumes,
not nearest-cell sampling. It is an integral of the diagnostic M0 expansion,
not a direct measurement of retained subcell extrema. Capturing current
coefficients and retained/published phi is required to distinguish retained
authority from publication or ownership errors. These box locations are
preserved for that investigation.

No interior open edges or nonmanifold edges were found in this pair. Some later
frames have open edges on the authored tank boundary; signed triangle volume
for such an open mesh is only a diagnostic quantity.

## Readback scope and next step

`readDiagnosticFields(true)` decodes dynamic leaves but clips output to the
authored dimensions. Amount, center of mass, momentum and kinetic energy above
are therefore authored-domain quantities. Captured outside-domain active
quarter leaves have mean density zero. `stats.volumeCellSum` and
`representedVolumeCellSum` remain at their initialized value in these receipts;
they are not used as independent evolving global-mass evidence.

The planned mini32 and rigid-tank pairs are paused following the user's
instruction to begin with the full-fine root cause. The next bounded diagnostic
is prepared in `tools/capture-retained-imposed-flow-dawn.ts` (commit `713ee123`).
It captures a 32³ isolated sphere at reset and after one and two imposed-flow
steps. Before each normal advance it prescribes both native velocity banks,
both face banks and the effective transport plane to `(0.75,0,0) m/s`, gamma
to one and pressure to zero. Density and retained coefficients remain untouched.
The measured VEX output must verify that velocity on every support in the swept
sphere's padded donor region. These are imposed-flow diagnostics, not a claim
that an initially uniform velocity survives shipping pressure projection.

Readbacks at conservative transport and scalar publication precede body forces
and pressure. Independent translated-sphere integrals measure native transport
error; raw retained coefficients measure pointwise density and one-sided jumps
across support faces. Actual packed GPU phi is compared separately with the
analytic sphere and the captured coefficient interpretation. This separates
scalar transport, retained shape and publication errors without meshing.
Run-start provenance records HEAD, source hashes and working-tree filenames.

The independent CPU oracle passes two tests: physical integral partitioning and
translation against a radial shell integral, and a negative control rejecting
the old seed on newly occupied support. **The GPU diagnostic has not yet run**
as of this preparation entry. Run serially under the shared GPU lease with:

```bash
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js FLUID_WEBGPU_BACKEND=metal \
  node --import tsx tools/capture-retained-imposed-flow-dawn.ts --assert-continuity
```

Its optional continuity assertion rejects a density jump above `1e-4` after
all requested evidence is saved. Native finite-volume translation errors are
reported independently; this is not an exact-M0 acceptance claim or a weakened
surface threshold. No production representation or renderer repair was made by
this investigation.

Reproduce one arm with `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js
FLUID_WEBGPU_BACKEND=metal node --import tsx
tools/capture-retained-visual-ab-dawn.ts --scene=quarter --arm=coarse`; use
`--arm=fine` for the other. The script owns the normal GPU lease. Render offline
with a Python environment containing NumPy, Matplotlib and Pillow using
`python tools/render-retained-visual-ab.py --input=artifacts/retained-visual-ab/quarter`.
