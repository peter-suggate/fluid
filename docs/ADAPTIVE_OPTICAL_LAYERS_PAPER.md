# Adaptive Optical Layers: Technical Reference

This is an implementation-oriented Markdown reference for:

> Fumiya Narita and Takashi Kanai, "Adaptive Optical Layers: Efficient Tall
> Cell Grids for Liquid Simulation," *Computer Graphics Forum* 45(2), e70357,
> 2026. DOI: 10.1111/cgf.70357.

- [Publisher article and full text](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70357)
- [Author project page](https://graphics.c.u-tokyo.ac.jp/projects/Adaptive-Optical-Layers/)
- First published: 14 April 2026
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Venue: Eurographics 2026

This document is a faithful technical paraphrase, not a verbatim
transcription. Equation and algorithm numbers match the paper. It records the
details needed to implement and audit the method without repeatedly navigating
the two-column article.

## 1. Purpose and contribution

A tall-cell liquid solver represents the dynamically important region near the
free surface with ordinary cubic cells. Deeper liquid is compressed into
vertically elongated cells with two pressure samples. The cubic region is the
**optical layer**.

Earlier tall-cell schemes use one optical-layer thickness everywhere. That is
wasteful: calm regions receive the same vertical resolution as impacts,
splashes, jets, and steep velocity fields. The paper makes the layer thickness
vary per horizontal column and per time step:

1. estimate how badly one tall cell would approximate the local velocity;
2. turn that error into a bounded surface-dilation distance;
3. dilate the layer farther where motion is poorly represented by a tall cell;
4. conservatively smooth the tall-cell boundary; and
5. project on the resulting grid.

It also extends variational two-way rigid coupling to objects that intersect a
tall cell below a locally thin optical layer.

The intended effect is fewer pressure unknowns without a visible loss relative
to a conventional fixed-thickness tall-cell solve. The reported scenes use
roughly 29%-40% as many cells, make projection about 2.5x-3x faster, and make a
whole step about 1.5x-2x faster. These are paper results, not portable acceptance
thresholds for this repository.

## 2. Scope and dependencies

The adaptive layer is not a replacement fluid model. The authors use it with:

- inviscid Eulerian liquid simulation;
- a rectangular `(N_x, N_y, N_z)` domain;
- `j` as the vertical grid coordinate;
- Extended Narrow Band FLIP (EXNBFLIP) for surface detail;
- the variational tall-cell pressure formulation of Narita et al. 2025; and
- ICCG for the symmetric positive-definite pressure system.

FLIP or APIC could replace EXNBFLIP. The essential dependency is a tall-cell
representation with two pressure samples per deep column and a projection that
can be rebuilt after the column split heights change.

The present repository's retained `tall-cell` mode is a restricted packed
tall-cell implementation. The new mode must be treated as a third mode, not as
a silent change to that implementation. Where this paper refers to details
in the 2025 variational tall-cell paper, those details need to be reconciled
explicitly with the existing WebGPU operators.

## 3. Notation

| Symbol | Meaning |
| --- | --- |
| `(i,j,k)` | Cubic-grid index, with `j` vertical |
| `(i,k)` | Horizontal column; independent of `j` |
| `Delta x` | Cubic grid width used by the paper |
| `e(i,j,k)` | Local error caused by collapsing a column to one tall cell |
| `E(i,k)` | Vertically accumulated column error |
| `d(i,k)` | Optical-layer dilation distance for the column |
| `d_min`, `d_max` | Lower and upper dilation limits |
| `h(i,k)` | Height from the ground to the optical-layer boundary, in cells |
| `A` | Diagonal open-flux area fractions near solids |
| `F` | Diagonal fluid area/volume fractions near the free surface |
| `V` | Diagonal face-cell volumes |
| `[nabla]` | Discrete gradient; `-[nabla]^T` is divergence |
| `J` | Pressure-to-rigid generalized-force coupling matrix |
| `M_s` | `6 x 6` rigid mass/inertia matrix |
| `w*` | Predicted rigid linear and angular velocity |

The paper assumes an isotropic cubic spacing. A non-cubic implementation must
derive the metric factors instead of substituting one scalar `Delta x`.

## 4. Constructing the adaptive optical layer

The layer is rebuilt in two stages. The first chooses a desired local
thickness from liquid motion. The second smooths the deep tall-cell boundary
so that neighboring tall cells do not change height abruptly.

### 4.1 Estimate error from the tall-cell approximation

For every horizontal column, temporarily consider one tall cell that spans
from the ground to the ground-connected liquid surface. This diagnostic tall
cell is separate from the tall cells later used for projection.

Convert the original cubic-grid velocity in that span to the one-cell
representation used by tall-cell methods:

- obtain the horizontal velocity by a least-squares linear fit; and
- obtain the vertical velocity by averaging.

Interpolate the fitted tall-cell velocity back to all original grid positions.
At cubic cell `(i,j,k)`, sum the absolute before/after difference over its
velocity faces:

```math
e(i,j,k) = \sum_f \left|v_{f,\mathrm{after}}-v_{f,\mathrm{before}}\right|.
```

This measures violation of the linear/averaged velocity assumptions that make
a deep tall cell useful. It responds directly to liquid motion rather than only
to surface geometry. The authors avoid curvature as the principal signal
because curvature can produce extreme values.

Accumulate the error vertically from ground to the ground-connected surface:

```math
E(i,k)=\sum_{j=j_{\mathrm{ground}}}^{j_{\mathrm{surface}}}e(i,j,k).
```

### 4.2 Convert error to a dilation distance

Equation (1) maps accumulated error to a bounded dilation distance:

```math
d(i,k)=f(E(i,k))
      =\operatorname{clamp}\!\left(\alpha E(i,k)\Delta x,
                                    d_{\min},d_{\max}\right).
\tag{1}
```

The paper uses the same parameters for all its scenes:

```text
alpha = 0.5
d_min = max(4, N_y / 64)
d_max = N_y / 8
```

Distances are expressed in cubic-cell units after applying the grid-width
factor shown in Equation (1). Integer conversion must be defined and tested;
the article states the formula and integer grid construction but does not give
a separate rounding rule for `d`.

If every surface location uses `d_max`, the grid is the conventional uniform-
thickness tall-cell grid. This is the strongest differential oracle for the
new implementation.

### 4.3 Variable-radius dilation

Dilate outward from each liquid-surface grid cell by its column's `d(i,k)`.
The implementation in the paper uses Manhattan distance. The union of all
dilated regions is the initial optical layer.

This is a spatially varying distance transform, not a per-column vertical band
alone: a high-error surface point can thicken the layer in nearby columns.

### 4.4 Conservatively smooth the tall-cell boundary

Using the raw dilation result directly can produce abrupt changes in tall-cell
height, which the authors report can destabilize or fail a simulation.

For each column:

1. scan vertically and record `h_before(i,k)`, the height from the ground to
   the boundary of the optical layer;
2. convert the 2D height field to floating point;
3. smooth it while enforcing `h_after(i,k) <= h_before(i,k)`;
4. cast the result back to integer cell coordinates; and
5. if `h_before - h_after > 0`, add all cells between those heights to the
   optical layer.

Lowering the tall-cell top only adds optical-layer cells. Smoothing is therefore
conservative with respect to detail: it may spend more cells, but it must not
erase cells selected by the error-driven dilation.

The paper's concrete filter is:

```text
2D moving average
window: 9 x 9 columns
iterations: 5
constraint: h_after <= h_before at every column
```

The method is not tied to that exact filter, but reproducing the paper should
use it first. The final projection grid is constructed from the smoothed layer.

### 4.5 Liquid above the ground-connected surface

Columns may contain disconnected liquid from droplets, splashes, or jets. The
ground-connected scan alone does not cover it. The paper handles all additional
surface components as follows:

1. find the ground-connected liquid surface and its `d(i,k)`;
2. an additional surface within `D_offset` of that surface inherits `d(i,k)`;
3. a farther surface receives the constant distance `d_air`; and
4. run dilation for all of these surfaces to obtain the final layer.

The fixed values are:

```text
D_offset = max(4, N_y / 32)
d_air    = N_y / 16
```

The deliberately generous `d_air` allows an isolated splash or jet to deform
while airborne. Nearby falling liquid inherits the target surface's resolution
before impact.

## 5. Pressure projection

### 5.1 Variational fluid-only system

The paper adopts the 2025 variational tall-cell formulation. Its fluid-only
projection is Equation (2):

```math
- [\nabla]^T[V][A][F][\nabla]\{p\}
=- [\nabla]^T[V][A]\{u^*\}.
\tag{2}
```

`A`, `F`, and `V` are diagonal. With consistent gradient/divergence pairs, the
system is symmetric positive definite after the usual null-space/boundary
treatment. The paper solves it with ICCG to relative residual `10^-4` in the
reported experiments.

If there are `n` liquid cells in the optical layer and `m` liquid tall-cell
columns, there are `n + 2m` pressure samples:

```math
\{p\}=
[p_{o_1},\ldots,p_{o_n},
 p_{t_1,\mathrm{top}},p_{t_1,\mathrm{bottom}},\ldots,
 p_{t_m,\mathrm{top}},p_{t_m,\mathrm{bottom}}]^T.
```

### 5.2 Monolithic two-way rigid coupling

For a coupled rigid body, Equation (3) is:

```math
\Delta t\left(
  [\nabla]^T[V][A][F][\nabla]
  +[J]^T[M_s]^{-1}[J]
\right)\{p\}
= [\nabla]^T[V][A]\{u^*\}-[J]^T\{w^*\}.
\tag{3}
```

The coupling matrix stacks translational and rotational blocks:

```math
[J]=[J_{\mathrm{trans}}^T\;J_{\mathrm{rot}}^T]^T.
```

Its sign convention is fixed by Equations (4) and (5):

```math
- [J_{\mathrm{trans}}]\{p\}
=\mathbf f_{\mathrm{trans}}
=\iint_S p\mathbf n\,dS,
\tag{4}
```

```math
- [J_{\mathrm{rot}}]\{p\}
=\boldsymbol\tau_{\mathrm{rot}}
=\iint_S(\mathbf x-\mathbf x_{\mathrm{com}})\times p\mathbf n\,dS.
\tag{5}
```

Following Batty et al., the surface integrals are converted to volume-fraction
differences. Let `A' = 1-A`, `[X,Y,Z]=x_com`, and `(x,y,z)` be the center of
cell `(i,j,k)`. The six rows of `J` at that cell are:

```math
J_1^{i,j,k}=\Delta x^2
\left(A'_{i+\frac12,j,k}-A'_{i-\frac12,j,k}\right),
\tag{6}
```

```math
J_2^{i,j,k}=\Delta x^2
\left(A'_{i,j+\frac12,k}-A'_{i,j-\frac12,k}\right),
\tag{7}
```

```math
J_3^{i,j,k}=\Delta x^2
\left(A'_{i,j,k+\frac12}-A'_{i,j,k-\frac12}\right),
\tag{8}
```

```math
J_4^{i,j,k}=-\Delta x^2(z-Z)
\left(A'_{i,j+\frac12,k}-A'_{i,j-\frac12,k}\right)
+\Delta x^2(y-Y)
\left(A'_{i,j,k+\frac12}-A'_{i,j,k-\frac12}\right),
\tag{9}
```

```math
J_5^{i,j,k}=-\Delta x^2(x-X)
\left(A'_{i,j,k+\frac12}-A'_{i,j,k-\frac12}\right)
+\Delta x^2(z-Z)
\left(A'_{i+\frac12,j,k}-A'_{i-\frac12,j,k}\right),
\tag{10}
```

```math
J_6^{i,j,k}=-\Delta x^2(y-Y)
\left(A'_{i+\frac12,j,k}-A'_{i-\frac12,j,k}\right)
+\Delta x^2(x-X)
\left(A'_{i,j+\frac12,k}-A'_{i,j-\frac12,k}\right).
\tag{11}
```

For a non-cubic grid, replace the repeated `Delta x^2` factors with the
appropriate face areas and use metric-correct moment arms.

### 5.3 A rigid body intersecting a tall cell

A thin local optical layer allows a body to intersect a deep tall cell. The
ordinary one-cell/one-pressure mapping no longer applies because the tall cell
has two pressure unknowns.

The paper embeds a virtual uniform grid inside the tall cell:

1. for every virtual cubic cell `(i_0,j_0,k_0)` intersecting the body, compute
   its six-component `J^{i_0,j_0,k_0}` from Equations (6)-(11);
2. compute a normalized vertical position `s` within the tall cell,
   `0 <= s <= 1`;
3. add `(1-s)J^{i_0,j_0,k_0}` to the tall cell's top-pressure column; and
4. add `sJ^{i_0,j_0,k_0}` to its bottom-pressure column.

The weights sum to one, preserve the virtual-cell contribution, and place its
generalized force between the two pressure degrees of freedom. Endpoint and
midpoint behavior must be unit tested; the implementation must follow the
paper's `s` orientation rather than infer it from variable names.

## 6. Simulation loop

Algorithm 1 is:

```text
1. Save the current grid and variables.
2. Construct the adaptive optical layer.             [Section 3.1]
3. Advect velocity and level set.
4. Advance rigid bodies.
5. Convert cells into tall cells.
6. Solve pressure.                                    [Section 3.2]
7. Map pressure onto cubic cells.
8. Update velocity and rigid bodies.
```

Only steps 2 and 6 differ materially from the authors' conventional tall-cell
loop. The ordering is important: layer construction uses the current motion,
and the pressure result is mapped back before updating both phases.

## 7. Reported evaluation

All reported tests use:

```text
pressure solver: ICCG
relative-residual target: 1e-4
CFL number: 2
surface method: EXNBFLIP
alpha: 0.5
d_min: max(4, N_y/64)
d_max: N_y/8
D_offset: max(4, N_y/32)
d_air: N_y/16
smoothing: constrained 9x9 moving average, 5 iterations
```

The layer visualization maps thickness zero to blue and `N_y/8` to red. It
shows the thickness before smoothing.

### 7.1 Per-step timings from Table 1

Times are seconds per time step on the CPUs named in the article. `Tall` is the
fixed-thickness baseline and `Adaptive` is the proposed method.

| Scene | Resolution | Method | Layer build | Advection | Tall build | Projection | Extrapolation | EXNBFLIP | Total |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Animals Drop | `256^3` | Tall | - | 2.123 | 1.049 | 15.576 | 0.623 | 2.480 | 22.439 |
| Animals Drop | `256^3` | Adaptive | 1.077 | 2.004 | 1.246 | 5.112 | 0.542 | 2.166 | 12.656 |
| Breaking Dam | `512x256x128` | Tall | - | 1.396 | 0.583 | 12.672 | 0.373 | 1.261 | 16.652 |
| Breaking Dam | `512x256x128` | Adaptive | 0.548 | 1.441 | 0.775 | 4.402 | 0.390 | 1.300 | 9.241 |
| Pouring Water | `320x256x320` | Tall | - | 2.738 | 1.179 | 20.985 | 0.757 | 1.825 | 30.082 |
| Pouring Water | `320x256x320` | Adaptive | 1.140 | 2.745 | 1.402 | 7.725 | 0.699 | 1.835 | 18.111 |
| Lucy in the Rain | `512x256^2` | Tall | - | 3.464 | 1.519 | 24.309 | 1.003 | 3.070 | 34.545 |
| Lucy in the Rain | `512x256^2` | Adaptive | 1.622 | 3.592 | 1.937 | 9.248 | 1.067 | 3.263 | 21.988 |

### 7.2 Projection details from Table 2

| Scene | Method | Cells | Matrix assembly (s) | Poisson solve (s) | Residual | Iterations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Animals Drop | Tall | 2,306K | 2.232 | 12.047 | `1.3e-5` | 264 |
| Animals Drop | Adaptive | 841K | 0.594 | 3.392 | `1.5e-5` | 225 |
| Breaking Dam | Tall | 2,306K | 0.988 | 10.942 | `1.0e-5` | 343 |
| Breaking Dam | Adaptive | 884K | 0.370 | 3.340 | `1.1e-5` | 227 |
| Pouring Water | Tall | 3,590K | 2.899 | 14.351 | `2.7e-4` | 200 |
| Pouring Water | Adaptive | 1,057K | 0.668 | 5.130 | `2.7e-4` | 199 |
| Lucy in the Rain | Tall | 5,739K | 2.598 | 19.803 | `1.0e-4` | 232 |
| Lucy in the Rain | Adaptive | 1,941K | 0.983 | 6.471 | `1.0e-4` | 213 |

### 7.3 What each scene tests

- **Animals Drop:** complex disconnected shapes, repeated impacts, splashing,
  and dynamically moving high-error regions.
- **Breaking Dam:** large-scale motion, sharp initial features, and static
  obstacles.
- **Pouring Water:** an elevated jet, impact splashes, airborne liquid, and a
  floating armadillo with two-way coupling. The paper reports density values
  `rho_water=1.0` and `rho_object=0.3` in its chosen units.
- **Lucy in the Rain:** multiple pouring sources plus many small droplets and
  fine surface ripples.
- **Water Drop comparison:** demonstrates that a uniformly thin layer using
  one fifth of the cells loses the crown and Worthington column, while adaptive
  placement preserves them.
- **Body inside a tall cell:** verifies that the new weighted coupling retains
  buoyancy where the prior optical-layer-only coupling does not.

The paper also plots cell count and kinetic energy over time. The adaptive
method substantially lowers optical-layer cell count while following the
fixed-tall kinetic-energy curve closely in the showcased scenes.

## 8. Limitations stated by the paper

### 8.1 Numerical damping

Kinetic energy can decay more quickly than in the fixed-thickness tall-cell
baseline, including in the simple water-drop case. The authors suggest better
motion indicators or adaptive energy reinjection as future work.

### 8.2 Whole-step speedup is smaller than projection speedup

The method reduces pressure work but does little to advection and other
full-grid operations. It is compatible with horizontal adaptive grids, but the
paper leaves that hybrid for future work.

### 8.3 Heuristic parameters

The method has several heuristic values, although one set was used for all
paper scenes. `alpha` is the main user-facing tradeoff: increasing it approaches
the fixed-thickness tall-cell result; decreasing it emphasizes performance.

### 8.4 Fully submerged rigid bodies

Tall-cell schemes are not intended to resolve rigid interactions dominated by
deep submerged flow. The reported behavior remains stable and visually
plausible, but buoyancy or rise speed can be less accurate.

## 9. Repository integration contract

This paper will be implemented as a new mode while preserving both current
modes:

```text
uniform                 retained full-depth cubic comparison mode
tall-cell               retained fixed/restricted tall-cell mode
adaptive-optical-layer  new mode implementing this paper
```

The current mode seam is `GPUGridMethod` in `lib/webgpu-eulerian.ts`, selected
in `WebGPURenderer.ensureGPUFluid` and exposed by `components/FluidLab.tsx`.
Adding the new mode must not reinterpret the existing `tall-cell` string or
change its defaults, kernels, diagnostics, or UI labels.

The safest architecture is a shared tall-cell projection/remap core with an
explicit layer-construction strategy:

```text
fixed tall strategy       -> existing remesh behavior
adaptive optical strategy -> error, variable dilation, constrained smoothing
uniform solver            -> separate retained solver
```

The adaptive mode needs additional observable state:

- pre-smoothing and post-smoothing layer thickness;
- `E(i,k)` and `d(i,k)` fields;
- counts for optical cells, tall columns, and pressure unknowns;
- layer-construction GPU time;
- min/max/mean dilation and thickness;
- constraint violations and non-finite counts; and
- kinetic energy, volume drift, projection residual, and divergence already
  used by the simulator.

These fields are not decorative diagnostics. They are required to distinguish
a visually plausible result from an implementation that is always clamped to
`d_min` or `d_max`.

## 10. Verification plan

Verification should be differential, invariant-based, and scenario-based. A
video that looks reasonable is insufficient.

### 10.1 Exact unit and property tests

1. **Tall-fit error**
   - a velocity field exactly representable by the fit gives `E=0` within
     floating-point tolerance;
   - increasing a controlled perturbation never decreases accumulated error;
   - horizontal least-squares and vertical averaging are tested independently.

2. **Equation (1)**
   - output is always in `[d_min,d_max]`;
   - output is monotone in `E` and `alpha` before clamping;
   - zero error returns `d_min`;
   - explicit cases exercise both clamps and the rounding rule.

3. **Variable dilation**
   - compare the GPU result with a small CPU Manhattan-distance oracle;
   - a single impulse produces the exact expected diamond/octrahedral stencil;
   - overlapping seeds form the union of their variable-radius regions.

4. **Constrained smoothing**
   - `h_after <= h_before` for every column;
   - smoothing never removes a raw optical-layer cell;
   - constant fields are unchanged;
   - output is integer, bounded, deterministic, and free of unsafe jumps.

5. **Airborne surfaces**
   - a component within `D_offset` inherits the ground-surface distance;
   - a farther component receives `d_air`;
   - exact boundary cases are fixed by tests.

6. **Tall-cell rigid map**
   - `(1-s)+s=1` and the two mapped vectors sum to the virtual-cell `J`;
   - endpoint and midpoint cases map to the correct pressure samples;
   - force and torque signs match Equations (4)-(11);
   - non-cubic metrics, if supported, use correct face areas.

7. **Linear system**
   - small assembled systems are symmetric within precision;
   - `x^T A x > 0` for non-null test vectors after boundary/null-space handling;
   - matrix-free GPU products match a CPU assembled oracle;
   - the coupled system conserves equal-and-opposite generalized impulse.

### 10.2 Non-regression of existing modes

Before enabling the new selector, capture deterministic diagnostics for both
existing modes. After integration:

- `uniform` must choose the same solver and allocation;
- `tall-cell` must choose the same fixed strategy and initial layout;
- their fixed-seed short-run diagnostics must remain within the existing
  floating-point tolerances; and
- all current unit, build, native, and shell-contract tests must still pass.

This protects against accidentally implementing adaptation by changing the
meaning of the current tall-cell mode.

### 10.3 Strong differential oracles

1. **Uniform-thickness limit:** force `d(i,k)=d_max` at every surface point.
   Adaptive mode must produce the same layer, pressure-unknown mapping, and
   projected velocity as the fixed tall-cell reference, subject only to an
   explicitly documented solver difference.

2. **Quiet-flow limit:** use a representable hydrostatic/linear velocity field.
   The layer should remain at `d_min`, residual and divergence must stay within
   solver tolerance, and the result must remain stable.

3. **Small-grid cubic oracle:** on a resolution where uniform simulation is
   affordable, compare divergence, volume, kinetic energy, free-surface
   position, and rigid impulse across all three modes.

4. **CPU/GPU layer oracle:** compute `E`, dilation, and smoothing on the CPU for
   small randomized fields and compare every column to WebGPU output.

### 10.4 Paper-inspired scenario suite

Add deterministic, fixed-duration versions of:

- water drop/crown;
- dam break with pillars;
- elevated pour with a light floating body;
- droplets/rain over a pool; and
- a light body crossing from optical cells into a tall cell.

For every run record a machine-readable time series containing:

```text
cell and pressure-unknown counts
pre/post-smoothing thickness statistics
projection time and total GPU step time
pressure residual and iterations/cycles
divergence before and after projection
volume drift
kinetic energy
rigid linear/angular impulse and momentum-closure error
non-finite and boundary-penetration counts
```

Use the retained fixed tall-cell mode as the principal quality baseline. The
paper's absolute timing ratios should be reported for context, not made hard
cross-hardware pass/fail criteria. Hard gates should cover invariants,
stability, conservation, residuals, and bounded deviation from a checked-in
baseline. Performance gates should compare modes on the same adapter and run,
using medians after warm-up.

### 10.5 Visual verification

Expose a scientific overlay for raw and smoothed optical-layer thickness using
the paper's blue-to-red convention. Capture synchronized views of fixed tall
and adaptive modes for the paper-inspired cases. Review:

- crown and Worthington-column retention;
- splash and droplet resolution;
- whether high-error regions track impacts and jets;
- temporal flicker in the adaptive boundary;
- buoyancy as a body crosses the layer boundary; and
- damping relative to fixed tall cells.

Visual evidence complements, but does not replace, the numeric gates above.

## 11. Traceability checklist

| Paper item | Required implementation evidence |
| --- | --- |
| Tall approximation error | CPU reference, GPU kernel, representable-field test |
| Equation (1) | parameter object, clamp/rounding tests, diagnostics |
| Manhattan dilation | CPU/GPU exact comparison |
| 9x9, five-pass constrained smoothing | invariant and constant-field tests |
| `D_offset`, `d_air` | disconnected-component tests |
| Equation (2) | operator derivation and residual/divergence evidence |
| Equations (3)-(11) | coupled operator tests and impulse closure |
| `(1-s)J`, `sJ` mapping | endpoint/midpoint and buoyancy-crossing tests |
| Algorithm 1 ordering | encoded-pass trace or command-order test |
| Tables 1-2 claims | same-adapter A/B benchmark report |
| Kinetic-energy limitation | checked-in energy time series |
| Existing modes preserved | deterministic uniform and fixed-tall regressions |

## 12. Known specification gaps to resolve explicitly

The paper leaves some low-level choices to its cited tall-cell implementation.
They must not be filled in silently:

- exact least-squares sample positions and weights for the horizontal fit;
- the integer rounding convention for `d(i,k)` and smoothed `h(i,k)`;
- dilation tie/boundary behavior at the domain edge;
- the precise conditional moving-average update rule within each iteration;
- the orientation used to compute `s` inside a tall cell;
- cut-cell fraction construction and null-space handling inherited from the
  2025 variational solver; and
- metric corrections if the repository keeps anisotropic cell sizes.

Each resolved choice should point to either the cited precursor derivation, an
author artifact, or an explicit tested convention. This prevents a plausible
but non-reproducible approximation from being mistaken for the paper method.

