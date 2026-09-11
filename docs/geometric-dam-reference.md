# Sparse Geometric dry-bed dam reference

The UI scene **Sparse Geometric Ritter dam break** (`sparse-geometric-ritter-dam-break`, factory `createSparseGeometricRitterDamBreakScene`) defaults to `adaptive-volume`, balanced quality, the production coarse-first B8 profile and its 1/30 s paper step. It is a dedicated diagnostic scene, not a replacement for the long-dam residency ladder.

## Authored geometry

The tank is 4 × 0.2 × 0.1 m, with a flat floor and closed free-slip walls. Its lattice is 320 × 16 × 8 at Δ = 0.0125 m. Initially resting liquid occupies 2.4 × 0.1 × 0.1 m from the negative-x corner. Initial volume is 0.024 m³. Gravity is (0, −9.81, 0) m/s²; viscosity, surface tension, inlet and rigid bodies are absent. The scene runs for 2 s so the measured wall impact is visible at the production step.

World coordinates are x ∈ [−2, 2], y ∈ [0, 0.2], z ∈ [−0.05, 0.05]. The dam is x_world = 0.4 m, or fine-grid face 192. Reference coordinate x = x_world − 0.4 = x_fine Δ − 2.4. The reservoir rear wall is x = −2.4 and the downstream wall x = 1.6. Sensors at reference x = 0.4, 0.8, 1.2 are world x = 0.8, 1.2, 1.6 (fine-grid faces 224, 256, 288).

## Analytic window and threshold arrivals

Ritter's inviscid, hydrostatic, dry-bed shallow-water solution has c₀ = √(g h₀), h₀ = 0.1 m. In the fan −c₀t < x < 2c₀t,

    h(x,t) = (2c₀ − x/t)² / (9g)
    u(x,t) = 2(c₀ + x/t) / 3.

Upstream h = h₀, u = 0; downstream h = 0. The mathematical zero-depth tip is x_tip = 2c₀t. These are shallow-water references, not an exact solution to the full three-dimensional Euler initial-release transient. See the [WOLF Ritter example](https://wolf.hece.uliege.be/tutorials/wolfgpu_ritter.html).

Here c₀ = 0.9904544412 m/s. The zero-depth tip first touches the downstream wall at **0.8077100438 s**. Stop all unreflected Ritter profile/arrival scoring strictly before that time. The rear rarefaction reaches the rear wall only at 2.4231301313 s, but this does not extend the downstream validity window: impact has already occurred. Finite-lock reflected waves require a different solution; [Hogg (2006)](https://people.maths.bris.ac.uk/~maajh/PDFPapers/JFMLockRelease.pdf) describes that distinction.

A finite wetness threshold η = h/h₀ has x_η = (2 − 3√η)c₀t and point arrival t_η = x / [(2 − 3√η)c₀]. The same threshold must be applied to simulation and reference. Do not compare the first positive numerical tail to the mathematical dry tip.

| Reference sensor x (m) | η = 0.01 (s) | η = 0.05 (s) | η = 0.10 (s) |
| --- | --- | --- | --- |
| 0.4 | 0.237562 | 0.303838 | 0.384142 |
| 0.8 | 0.475124 | 0.607676 | 0.768284 |
| 1.2 | 0.712685 | 0.911513 | 1.152426 |
| 1.6 (wall) | 0.950247 | 1.215351 | 1.536568 |

Primary arrival metrics are η = 0.05 and 0.10 at x = 0.4 and 0.8. The 1% threshold is tail sensitivity. The last two thresholds at x = 1.2 and **every finite-threshold wall time** are post-impact, hence diagnostic observations rather than exact Ritter targets. Wall first contact should be reported as a threshold-dependent time bracket; a low threshold is not an exact zero-depth detector. Avoid defining pass tolerances from the result being measured. Profile comparisons should emphasize approximately 0.2–0.8 s, alongside earlier transient samples.

## Accepted-volume observation

Use `solver.readDiagnosticFields(true)` after `awaitFrameCompletion()`. This reads the accepted bank and replicates each accepted cell's stored ρᵢ = Vᵢ / fullCellVolume across its fine-grid footprint; it does not sample a rendered interface. Returned `velocity` has four entries per fine voxel and is already in m/s. Index a voxel as j + nx(k + ny l), with j along x, k along vertical y and l along z.

For this contained flume, compute h_j = Δ Σ_(k,l) ρ_jkl / nz and q_j = Δ Σ_(k,l) ρ_jkl u_x,jkl / nz. Set depth-averaged velocity q_j/h_j only where positive observed liquid exists. Preserve the signed density sum for volume checks; if tiny negative roundoff is excluded from velocity observation weights, report that excluded amount separately. Sum all densities times Δ³ and require agreement with `readAcceptedGeometricVolumeQA()`; its outside-authored volume must remain zero. A discrepancy means the dense diagnostic no longer covers the authority and must not be scored.

Compare against the analytic depth **averaged over the same x bins**. Integrate the piecewise constant/fan/dry formula across each bin, then use the same spatial interpolation and threshold-crossing procedure for reference and observed arrivals. This avoids attributing bin-width bias to transport. Track the downstream threshold crossing connected to the dam, and report detached droplets separately rather than silently choosing the furthest speck.

## Energy observations and dissipation attribution

A consistent shallow-water observation is

    K_SW = ½ ρ_water W Σ_j Δ h_j ū_j²
    P_SW = ½ ρ_water g W Σ_j Δ h_j².

Before impact/rear reflection the continuum Ritter reference satisfies K = (1/5)ρ_water W g h₀² c₀t and P = E₀ − K, with E₀ = ½ρ_water W g h₀² L_left. Compare like-for-like bin-averaged analytic observations as well as these continuum totals; averaging itself changes quadratic energy.

Also report collocated K_cell = ½ρ_water Σ_i V_i |u_i|² from accepted volume and velocity (the replicated fine field computes the same quantity). Cell-centroid P_cell ≈ ρ_water g Σ_i V_i y_i is an **estimate**, not an exact PLIC centroid integral; mixed-cell positioning limits its interpretation. A fine-bin replication of constant cell density has the same centroid integral. Positive velocity-observation weights avoid signed-roundoff cancellation artifacts while signed extensive volume remains untouched.

Energy decline alone does not locate dissipation. Instrument matching volume/velocity epochs around velocity advection, projection, bounded volume transport and topology transfer before changing an algorithm. Report limiter reductions, time-step/substep counts, resolution distribution, source/outflow and stage energy changes. Distinguish vertical/shear kinetic energy from depth-averaged kinetic energy, and physical impact/vertical acceleration from numerical loss. This scene supplies a reference and observations; it does not establish that any existing transport stage conserves mechanical energy.

## Measured dissipation investigation (12 September 2026)

The production face predictor previously averaged opposite face velocities into
cells and interpolated those cell values back onto faces on every step. On a
uniform grid its zero-time-step component stencil is [1/4, 1/2, 1/4]. A Fourier
component therefore loses amplitude by cos²(kΔ/2), even as Δt approaches zero.
The manufactured Dawn vortex measured energy retention 0.728553410 at Δt =
1e−7 s, matching the predicted 0.728553391. Sampling the source staggered face
field instead retained 0.999999996, with relative L2 error 1.09e−8.

The production predictor now uses the extended collocated field for RK2
trajectories and the source staggered field for the advected component. Mixed
rows select actual opposing-cell overlap patches. Moving cut faces, unsupported
air samples and uncertified partial one-sided exterior patches still use the
existing velocity extension; the identity result is not a claim about those
fallback classes. The authored outer timestep remains unchanged.

The matched adaptive dam observations at 0.7 s are:

| Observation | Before | Source face advection | Ritter, same bins |
| --- | ---: | ---: | ---: |
| 5% front distance (m) | 0.578529 | 0.591250 | 0.921588 |
| 5% arrival at x = 0.4 m (s) | 0.521054 | 0.499281 | 0.303740 |
| 10% arrival at x = 0.4 m (s) | 0.541219 | 0.519898 | 0.384073 |
| Depth-averaged horizontal kinetic energy (J) | 0.880178 | 0.888765 | 1.357801 |

Arrival values are interpolated inside recorded step brackets, not measurements
with microsecond accuracy. Both runs complete 21 accepted 1/30 s steps with
bounded volume and unchanged source fingerprints. These results establish a
real averaging defect and a modest dam improvement, not complete elimination
of the flow lag or surface errors. Depth L1 error does not improve in this pair;
L2 decreases slightly. See `artifacts/analytic-motion/dam-before.json`,
`dam-staggered.json`, and `dam-staggered-comparison.png`.

At 1/60 s with the same finest lattice and production adaptivity, the 5% front
reaches 0.813673 m and horizontal kinetic energy reaches 1.289359 J. The 5%
sensor arrival is 0.339593 s. The fixed-finest 1/30 s control still lags (front
0.606105 m, arrival 0.486228 s), so adaptive coarsening alone does not explain
the dominant timing error. These are diagnostic configurations, not new
production defaults. Low-order limiter reductions also fall at the smaller
outer step. The velocity, pressure membership and free-surface coefficients are
currently computed once per outer step, then held through the volume
microsteps. Those microsteps enforce bounds and geometric transport CFL; they
do not establish temporal accuracy of coupled momentum and pressure. Updating
the coherent physical state within an outer step remains a separate algorithmic
requirement, including consistent time, banks, moving capacities and source
ledgers. Increasing only the volume microstep count does not perform that work.

An attempted positive-volume velocity-seed change was **withdrawn**. It repaired
the existing isolated sub-isovalue translation regression, but degraded the dam:
5% front 0.222595 m (0.234629 m together with staggered sampling). Gravity acts on
open air-side rows that the liquid pressure solve does not project. Treating
all positive-volume cells as velocity seeds promotes those unprojected
velocities into the liquid's transport field. The accepted source-face fix
therefore retains the original isovalue-based velocity-seed convention;
the isolated sub-isovalue-motion regression remains known red. No volume bound,
pressure threshold or test tolerance was weakened to keep the change.

The fixed-finest 1/60 s control reaches a 5% front of 0.837533 m and a
0.340350 s arrival at the 0.4 m sensor. Its depth L2 error is 0.003504 m³⁄²,
versus 0.007221 for fixed-finest 1/30 s. This confirms substantial timestep
sensitivity even with topology fixed. Raw pressure-supported positive-volume
fractions and volume-weighted vertical velocities are now included per column
for subsequent investigations.

The 2 s production-step run completes 60 accepted steps. Wall-adjacent depth
first crosses 1% and 5% of the initial head in [1.633333, 1.666667] s, and 10%
in [1.666667, 1.700000] s. Final volume is 0.023999998995 m³ with zero invalid
cells. These wall observations are explicitly post-impact diagnostics; they
are not scored as exact Ritter finite-threshold arrivals. Receipt:
`artifacts/analytic-motion/dam-staggered-wall.json`.

Additional Dawn checks preserve near-zero-step face values on uniform cell
widths 2 and 4 and a fixed 1:2 interface, with maximum error bounded by the
probe's 1e−5 criterion on certified supported interior faces. The mixed case
uses a phase-shifted field with nonzero interface-normal energy (14 across
56 interior mortar faces); its maximum interior error is 5.96e−8. Receipt:
`artifacts/analytic-motion/face-advection-mixed12-nonzero.json`. Boundary
trajectory clamps and unsupported fallback faces are outside that identity
criterion. A separate
large-step run accepts two 0.2 s outer steps: actual flux Courant numbers
35.83 and 49.63 use 72 and 100 volume microsteps, zero invalid cells, and
maximum relative volume balance error 6.74e−7. This is bounds/clock support
for large steps, not a claim of temporal accuracy at those steps.

The unchanged canonical Dawn gate completed in 454.3 s (480 s suite budget),
with 8 of 17 lanes passing. Symmetric expansion and min8 region surface
publication failed their existing correctness thresholds. Topology page budget,
mini32 correctness, mini32/mini64 performance, both far-wall lanes and
outside-tank symmetric collapse exceeded their lane timeouts. These are not
waived or counted as passes. The earlier paused gate already recorded symmetry
and min8 surface failures; it did not provide a complete matched timing baseline.
Receipt: `artifacts/analytic-motion/staggered-dawn-regression.json`.
