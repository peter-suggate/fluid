# Full-fine imposed flow: three separate scalar failures

The actual production scalar stages lose smoothness after the first prescribed
half-cell translation. Native transport introduces diffusion; the retained
affine lift introduces density discontinuities; its implicit companion also
publishes false zeros. None of these observations requires a mesh.

The root task ran `tools/capture-retained-imposed-flow-dawn.ts
--assert-continuity` serially on Dawn/Metal. All seven requested snapshots were
saved in 28.631 seconds, then the continuity assertion failed as intended for
this negative production result. The log is
`/tmp/fluid-retained-imposed-flow-2.log`; artifacts are in
`artifacts/retained-imposed-flow/sphere-full-fine/`.

![Captured coefficient field and independent GPU phi](../artifacts/retained-imposed-flow/sphere-full-fine/retained-field-comparison.png)

The top row evaluates the captured GPU coefficients inside their owning fine
supports, without smoothing. The middle row displays actual packed GPU phi
samples and their linearly interpolated zero contour. The magenta contour is
the exact translated quadratic sphere. Orange crosses identify actual phi-zero
samples whose retained density is not one half. The bottom row plots density
directly. These are separate authority and publication observations.

## Valid imposed-flow fixture

The sphere has radius 0.25 m, transition width 0.05 m and initial center
`(-0.15,0.8,0)` in a 32³ domain with 0.05 m finest cells. Both steps use the
authored `1/30 s` duration and imposed velocity `(0.75,0,0) m/s`, so displacement
is exactly 0.025 m per step. Gravity, viscosity, surface tension, gamma diffusion
and sharpening are disabled. Before each normal advance, the probe prescribes
both native velocity banks, both face banks, the effective transport plane,
gamma one and pressure zero. Density and retained coefficients are untouched.

All active native cells have actual width one. Every one of the 1,732 supports
in the swept transition sphere plus its donor margin is present and has exact
prescribed effective velocity at VEX publication, gamma one and zero pressure.
The margin includes every trilinear donor. The fixture does not depend on
pressure preserving an initially uniform velocity.

The post-gather effective plane is a new momentum product: dry output supports
are assigned zero velocity. This explains its later maximum deviation of
0.75 m/s. Wet output velocities remain prescribed within `1.79e-7 m/s`. More
decisively, the captured native density agrees with the independent half-cell
linear remap below to float precision.

## 1. Native transport diffuses cell averages

After one step the measured native means equal
`0.5*rho0[i] + 0.5*rho0[i-x]` within `2.98e-8`; after two steps they equal the
repeated remap, with weights `[0.25,0.5,0.25]`, within `4.47e-8`.
The analytic reference independently translates the whole diffuse sphere and
integrates it over each native cell. It uses exact vertical integration plus
adaptive x/z quadrature, with a summed estimated amount error below
`5.8e-10 m³`; this estimate is not a certified bound.

| Metric | Reset | One step | Two steps |
| --- | ---: | ---: | ---: |
| Maximum native mean error against exact translation | `1.83e-7` | 0.107704 | 0.161866 |
| Native L1 amount error / sphere amount | negligible | 4.709% | 7.942% |
| Maximum retained pointwise density error | 0 | 0.407353 | 0.496379 |
| Maximum one-sided density jump across support faces | 0 | 0.361273 | 0.415243 |

Total native amount changes by only `2.62e-9` relative to reset after two steps.
Thus conservation passes while translated shape and native distribution fail.
The L1 error is distribution error, not lost total mass.

## 2. The affine lift preserves wrong means but loses continuity

After scalar publication, both stored retained integrals and integrals inferred
from `a*seedMean+b*openFraction` agree with native means within `8.66e-8`.
That accurate restriction does not make the field smooth.

After one step the two traces at
`(0.15,0.75625,-0.04375) m` are 0.407353 and 0.046080 across an X support face.
The coefficients are respectively
`(a,b)=(0.592646658,0.407353342)` and
`(0.953919768,0.046080209)`. The original seed is zero at that point, so each
trace equals its independently updated offset `b`. A fixed seed multiplied and
offset per support cannot carry a smooth advancing interface into initially
dry support.

## 3. Phi zero is not always retained density one half

After the first step, 144 actual GPU samples have phi exactly zero. **All 144
have retained density different from one half** by more than `1e-4`; some are
completely dry. For example, at support center `(-0.175,0.675,-0.275) m`,
captured coefficients are exactly `a=0.5,b=0`, the independent seed density is
zero, and therefore retained density is zero. The GPU publishes phi zero.
The support's seed mean is `0.000184695702`, reflecting liquid elsewhere inside
the integration box, not at this sample point.

The endpoint branch `a+b<=0.5` returns `width*(0.5-a-b)` for the entire support.
At exact equality it therefore emits zero even where the seed is below one
and retained density is below one half. The analogous `b>=0.5` endpoint has
the opposite false-zero case. This is a semantic error in the implicit
companion, distinct from both transport diffusion and the discontinuous lift.
It is also distinct from a true positive-volume region of density exactly one
half, which itself lacks a unique surface.

Packed phi agrees with an interpretation of the current branch formula within
34.2 and 59.5 micrometres after the two steps. **That agreement does not prove
phi faithfully represents the half-density set.** Likewise, the maximum error
of the scalar phi value against analytic phi is not a surface-distance metric.
The earlier coarse phantom boxes are consistent with this defect, but their
coefficients were not captured and this investigation does not establish their
individual causes.

## Reproduction and provenance

The capture/oracle is committed in `713ee123`. Run-start provenance records
HEAD `ff983b6e`, working-tree filenames and full SHA256 hashes in
`provenance.json`; the resident source hashes begin `369429c2b42a` (host) and
`016ebcd1b74b` (WGSL). The shared production files subsequently changed, so the
current checkout must not be described as the exact captured implementation.

Run the CPU inspection with a Python environment containing NumPy and
Matplotlib:

```bash
python tools/render-retained-imposed-flow.py \
  --input=artifacts/retained-imposed-flow/sphere-full-fine
```

It reproduces the figure, section profiles, independent native-operator
comparison and false-zero semantic audit. The two independent CPU integration
and translation oracle tests pass. This entry records a **failing production
motion diagnostic**, not an acceptance pass or a renderer repair.
