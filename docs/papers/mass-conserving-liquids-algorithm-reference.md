# Mass-Conserving Eulerian Liquid Simulation: Complete Algorithm Reference

**Implementation, invariants, edge cases, correctness arguments, validation tests, and dependency audit**

Prepared from Chentanez and Mueller (2012), its operational dependencies, and the supplied local paper corpus.  
Reference date: 12 August 2026  
Status: implementation-oriented reconstruction; source ambiguities are called out explicitly.

---

## Executive answer: are any source papers missing?

The initial corpus was missing five operational dependencies. Those five primary-author PDFs and layout-preserving TXT extractions have now been added to `docs/papers`. The remaining absent direct provenance papers are HW65 (the MAC grid) and LC87 (marching cubes); both algorithms are fully described at the implementation level in this reference, but their original full texts are not yet local.

### Locally present and directly used

| ID | Source | Role in the executable method | Local status |
|---|---|---|---|
| CM12 | Chentanez and Mueller, *Mass-Conserving Eulerian Liquid Simulation* (2012) | Main method: density advection simplification, local sharpening correction, solid-density handling, pressure inputs, post-processing | Present as TXT and PDF |
| CM11b | Chentanez and Mueller, *Real-Time Eulerian Water Simulation Using a Restricted Tall Cell Grid* (2011) | Two-zone velocity extrapolation: accurate narrow band plus hierarchical far-field fill | Present as `tallCells.txt` and PDF |
| JRW07 | Jeong, Ross, and Whitaker, *A Fast Eikonal Equation Solver for Parallel Systems* (2007) | Fast Iterative Method used as the GPU-friendly near-interface propagation mechanism | Present as TXT |
| CM11a | Chentanez and Mueller, *A Multigrid Fluid Pressure Solver Handling Separating Solid Boundary Conditions* (2011) | Variational cut-cell pressure matrix, projected red-black Gauss-Seidel, multigrid cycles, separating wall condition | Present as TXT |

### Newly downloaded operational dependencies

| ID | Added source | Why it matters | Local files |
|---|---|---|---|
| LGF11 | Lentine, Gretarsson, and Fedkiw, *An Unconditionally Stable Fully Conservative Semi-Lagrangian Method* (2011) | Defines the original backward-clamp/forward-remainder conservative transport construction and collision-aware boundaries | `LGF11_...pdf` and `.txt` |
| LAF11 | Lentine, Aanjaneya, and Fedkiw, *Mass and Momentum Conservation for Fluid Simulation* (2011) | Defines the cumulative row-sum correction and diffusion that suppress coarse-grid incompressibility artifacts | `LAF11_...pdf` and `.txt` |
| MMTD07 | Mullen et al., *A Variational Approach to Eulerian Geometry Processing* (2007) | Introduces surface density, the 0.5 interface, Godunov-style sharpening, and global correction | `MMTD07_...pdf` and `.txt` |
| BBB07 | Batty, Bertails, and Bridson, *A Fast Variational Framework for Accurate Solid-Fluid Coupling* (2007) | Derives the kinetic-energy-minimizing cut-cell pressure projection and separating-wall complementarity | `BBB07_...pdf` and `.txt` |
| ENGF03 | Enright et al., *Using the Particle Level Set Method and a Second Order Accurate Pressure Boundary Condition for Free Surface Flows* (2003) | Supplies the ghost-fluid free-surface pressure treatment referenced by CM12 | `ENGF03_...pdf` and `.txt` |

The extraction command used Poppler's `pdftotext -layout`, preserving columns and equations as far as PDF text encoding permits. SHA-256 checksums and primary URLs are recorded in `mass-conserving-liquids-source-manifest.md`.

### Direct dependencies that are also absent, but are sufficiently specified here

The regular MAC grid [HW65] and marching cubes [LC87] are direct dependencies. Their full papers are not in `docs/papers`. The solver also uses standard trilinear interpolation, forward Euler integration, Gaussian convolution, finite differences, prefix sums or atomics for scatter, and linear-system residual calculations without separately cited implementation papers.

### Papers cited by CM12 that are not executable dependencies

The related-work citations for particle level sets, VOF/CLVOF, BFECC, MacCormack, derivative advection, Lattice-Boltzmann methods, SPH, mesh tracking, anti-diffusion, and volume-control methods are comparisons, alternatives, or future work. They are not required to implement CM12 as published. In particular, KLL+07 connected-component redistribution is discussed as an inadequate mitigation for MMTD07 sharpening; CM12 does not use it. SHA11 anti-diffusion is future work. EMF02 and the Mokberi/Faloutsos PLS library are used only for comparison.

## 1. Scope and fidelity rules

This document extracts every algorithm on the execution path described by CM12, including the specific algorithms imported from CM11a, CM11b, JRW07, LGF11, LAF11, MMTD07, BBB07, ENGF03, HW65, and LC87. It also records algorithms that the paper leaves underspecified.

Three labels are used:

- **Source-specified** means the paper gives enough equations or pseudocode to implement the step.
- **Reconstructed** means the implementation follows directly from the cited numerical method, but one or more engineering choices are not fixed by CM12.
- **Specification gap** means multiple materially different implementations satisfy the prose. Such a choice must be documented and tested rather than silently assumed.

The surface density `rho` is not the physical liquid density `d`. The physical density is constant for the incompressible pressure solve. `rho` is a cell-centered occupancy-like conserved scalar whose 0.5 isosurface defines the visible free surface. Confusing these two meanings breaks both pressure scaling and mass accounting.

## 2. State, geometry, notation, and invariants

### 2.1 Grid and fields

CM12 uses a uniform three-dimensional staggered MAC grid:

- `rho[i]`: surface density at cell center `i`.
- `p[i]`: pressure at cell center.
- `u`, `v`, `w`: velocity components at the centers of faces perpendicular to x, y, and z.
- `V[i]` in `[0,1]`: non-solid volume fraction of cell `i`; it includes liquid and air.
- `Vf[face]` in `[0,1]`: non-solid area fraction of a face.
- `u_s[face]`: solid velocity sampled at faces or reconstructed where the cut-cell discretization needs it.
- `phi_s`: signed distance to the solid, used to move excess density away from solids.
- `gamma[i]`: cumulative destination-fill correction for conservative density advection. This is unrelated to the post-processing field also called gamma in CM12 section 3.8; implementations should use distinct names such as `advect_gamma` and `detail_gamma`.

The discrete advection operator is written `rho_next = A rho`. With CM12's indexing, `A[i,j]` is the fraction of donor cell `j` delivered to destination cell `i`.

- `beta[j] = sum_i A[i,j]` is the column sum: the fraction of donor `j` transported. Exact global conservation requires `beta[j] = 1` for every closed-domain donor.
- `gamma[i] = sum_j A[i,j]` is the row sum: the total interpolation weight arriving at destination `i`. A constant field is preserved exactly when `gamma[i] = 1`.
- A doubly stochastic advection matrix has every `beta` and `gamma` equal to one. CM12 enforces `beta = 1` exactly up to arithmetic error and keeps `gamma` near one.

### 2.2 Core invariants

An implementation should check these after every relevant kernel, not only at the end of a frame:

1. **Finite state:** every active `rho`, velocity, pressure, weight, and fraction is finite.
2. **Nonnegative surface mass:** `rho[i] >= 0`, allowing only a small floating-point tolerance before repair.
3. **Closed-domain global mass:** `sum_i rho[i] * cell_volume` changes only through explicit sources, sinks, or open boundaries.
4. **Solid exclusion:** a completely solid cell ends density handling with `rho = 0`; ordinarily `rho <= V` near partial solids after excess ejection.
5. **Advection donor conservation:** `abs(beta[j]-1)` is near roundoff after the conservative transport step.
6. **Pressure feasibility:** solid-adjacent pressure satisfies the separating constraint `p >= 0` at constrained nodes.
7. **Projection:** the volume-weighted divergence residual is below the chosen solver tolerance.
8. **Rendering isolation:** density post-processing operates on a copy and never changes the conserved simulation field.

### 2.3 Initialization

Initialize `rho` from the initial liquid volume. CM12 does not prescribe the rasterizer. A robust implementation computes subcell occupancy or supersamples the liquid indicator so that boundary cells start in `[0,1]`, rather than assigning only zero and one at cell centers. Initialize `advect_gamma = 1` in all transportable cells. Rasterize solids into `V`, face fractions `Vf`, and solid velocities. Initialize pressure to zero.

Edge cases:

- An empty domain must remain a valid no-op: zero mass, no pressure unknowns, and no marching-cubes surface.
- A full liquid domain has no air interface; free-surface ghost coefficients must not be manufactured.
- A cell with `V = 0` must never be used as a donor or interpolation target unless a step explicitly extrapolates an auxiliary value into solid cells for the pressure stencil.
- If the initial `rho > V`, run the solid excess-ejection step before the first pressure solve.

## 3. Complete frame algorithm

### 3.1 Source-specified order

CM12 Algorithm 1 is:

```text
AdvanceOneFrame(rho, velocity, solids, dt):
    1. Extrapolate velocity from liquid into air.
    2. Conservatively advect rho; sharpen rho locally and globally.
    3. Advect velocity; add external forces.
    4. Enforce incompressibility with cut cells, a ghost free surface,
       excess-density divergence, and separating solid boundaries.
```

The ordering is semantically important. Density characteristics require an air-side velocity field. Sharpening must act on the newly advected surface density. Pressure projection must see the updated density and updated solid geometry.

### 3.2 Expanded implementation pipeline

```text
AdvanceOneFrame(state, dt):
    validate_finite_inputs()
    rasterize_or_update_solids(V, Vf, u_s, phi_s)

    # Velocity support for large-CFL characteristics
    identify_liquid_and_interface_from(rho > 0.5)
    near_velocity = FIMVelocityExtension(rho_or_interface_proxy, velocity, band=2 cells)
    velocity_ext = HierarchicalFarFieldExtrapolation(near_velocity)

    # Conserved surface transport
    rho_adv, advect_gamma = StreamlinedConservativeAdvection(
        rho, advect_gamma, velocity_ext, dt, solids)
    advect_gamma, rho_adv = GammaDiffusion(advect_gamma, rho_adv, 1..7 iterations)

    # Interface sharpening without long-range mass transfer
    delta = ComputeGodunovSharpeningIncrement(rho_adv, Vf, fictitious_dt=3*dt)
    delta = LimitSharpeningIncrement(delta, rho_adv, epsilon=1e-5)
    rho_sharp = rho_adv + delta
    rho_sharp = ReturnRemovedMassLocally(rho_sharp, -delta, D, solids)

    # Moving/cut solid consistency
    rho_clean = EjectExcessDensityFromPartialSolids(rho_sharp, V, phi_s, S=1)

    # CM12 does not fully specify this kernel; record chosen scheme.
    velocity_star = AdvectVelocity(velocity_ext, dt, solids)
    velocity_star += dt * external_acceleration

    # Pressure inputs and projection
    rho_eff = EffectiveFillDensity(rho_clean, V)
    extrapolate rho_eff one cell into V==0 where pressure stencil requires it
    phi_liquid = -(rho_eff - 0.5) * dx
    divergence_rhs = CutCellDivergence(velocity_star, V, Vf, u_s)
    divergence_rhs += ExcessDensityDivergence(rho_eff, lambda=0.5, eta=1)
    p = SeparatingBoundaryMultigrid(phi_liquid, V, divergence_rhs)
    velocity_next = ApplyCutCellPressureGradient(velocity_star, p, V, Vf, dt, d)

    assert_invariants(rho_clean, velocity_next, p)
    return updated state
```

### 3.3 Time-step size and stability

The paper deliberately uses large CFL numbers, up to 32 in its examples, with `dt = 1/30 s`, `dx = 0.05 m`, and gravity `10 m/s^2`. “Unconditionally stable” refers to the absence of a semi-Lagrangian CFL blow-up, not to unconditional accuracy, monotonicity of every optional interpolation, pressure convergence, or preservation of visible volume. Large `dt` increases characteristic-tracing distance, collision complexity, interpolation error, solid-crossing risk, and the need for far-field velocity extrapolation.

## 4. Characteristic tracing and interpolation primitives

### 4.1 Backward semi-Lagrangian trace

For destination position `x_i`, integrate the characteristic backward through the velocity field:

```text
x_depart = IntegrateCharacteristic(x_i, velocity, -dt)
rho_next[i] = TrilinearSample(rho, x_depart)
```

CM12 uses backward and forward traces with trilinear interpolation. It does not state whether the characteristic integrator is Euler, midpoint/RK2, or higher order. This is a specification gap. Straight-line Euler tracing is consistent with the first-order transport described in the dependencies; RK2 generally reduces trajectory error but must be used consistently for forward and backward weight construction.

### 4.2 Trilinear weights

For a point inside a grid cell, let fractional coordinates be `(fx,fy,fz)` in `[0,1]^3`. The eight weights are products of `(fx or 1-fx)`, `(fy or 1-fy)`, and `(fz or 1-fz)`. They must be nonnegative and sum to one before solid masking.

Solid-aware interpolation:

1. Stop a characteristic at the first solid crossing.
2. Set weights of solid or occluded stencil points to zero.
3. Renormalize the remaining weights.
4. If no weight remains, use a documented fallback: retain the donor value, sample the closest visible point, or mark the trace invalid. Never divide by zero.

### 4.3 Scatter versus gather

Backward semi-Lagrangian evaluation is a gather. Computing donor sums and distributing forward remainders are scatters. On a GPU, scatter requires atomics, sorting/reduction, or prefix-scan compaction. CM12's main advection contribution reduces scatter passes from five to three and avoids explicitly storing sparse matrix `A`.

Correctness checks for every interpolation or scatter kernel:

- All emitted weights are finite and nonnegative for the first-order scheme.
- Each unmasked trilinear stencil sums to one within tolerance.
- A masked forward scatter renormalizes to one across visible targets.
- An invalid trace produces a deterministic fallback and a counter in diagnostics.
- Periodic, closed, inflow, and outflow boundaries are treated explicitly rather than by accidental clamping.

## 5. Velocity extrapolation into air

CM12 uses CM11b's two-zone scheme: JRW07/FIM close to the interface and a grid hierarchy far away. This is necessary because a large-CFL backward trace may originate many cells outside the current liquid.

### 5.1 Mathematical extension problem

For each velocity component `q`, CM11b extends values away from the interface in fictitious time `tau`:

`partial q / partial tau = -(grad phi / |grad phi|) dot grad q`.

At steady state, the normal derivative is zero, so the interface value is copied along outward normals. CM12 tracks a density rather than a signed-distance level set. It does not state exactly which `phi` is supplied to this step. A practical reconstruction uses a local interface proxy such as `phi = -(rho-0.5) dx`, or computes a narrow-band signed distance from the 0.5 surface. This choice must be recorded because raw `rho` gradients are not distance gradients and may vanish or become noisy.

### 5.2 Fast Iterative Method for the eikonal equation

JRW07 solves `|grad U| = 1/f`, with positive speed `f`, using a synchronous active list rather than a heap.

#### Local Godunov update

For each axis `p`, take the smallest valid neighbor value `a_p`. On an anisotropic grid with spacing `h_p`, solve

`sum_p (max(U-a_p,0)/h_p)^2 = 1/f^2`.

A robust local solver is:

```text
SolveGodunov(neighbor minima a[p], spacing h[p], speed f):
    require f > 0
    discard axes whose two neighbors are unavailable/infinite
    sort remaining (a[p], h[p]) by a ascending
    s = 1/f
    for k = 1..number_of_axes:
        A = sum_{r<=k} 1/h[r]^2
        B = -2 * sum_{r<=k} a[r]/h[r]^2
        C = sum_{r<=k} a[r]^2/h[r]^2 - s^2
        disc = max(B*B - 4*A*C, 0)   # only roundoff may make it negative
        candidate = (-B + sqrt(disc))/(2*A)
        if k is last or candidate <= a[k+1]:
            return max(candidate, a[k])
    return infinity
```

#### Pointwise FIM

```text
FIM(sources, epsilon):
    U = infinity everywhere
    set source/boundary values in U
    active = one-neighbors of sources
    while active is not empty:
        synchronously compute U_new[x] = min(U[x], SolveGodunov(...))
        converged = abs(U[x]-U_new[x]) <= epsilon and local residual is small
        commit U_new
        remove converged nodes from active
        for each removed node x:
            add any downwind one-neighbor y not active when U[y] > U[x]
    return U
```

Newly activated nodes are updated on the next iteration, not the current one. This preserves synchronous semantics. A node can be revisited; the algorithm's practical efficiency, not single-visit optimality, is its advantage.

#### GPU tile FIM

JRW07 groups the domain into `8^3` tiles:

1. Update all nodes in each active tile for a fixed number of local iterations.
2. Encode node convergence, then reduce each tile to one active flag in `log2(8)=3` 8-to-1 reductions.
3. Update neighboring tiles once to see whether a converged tile influences them.
4. Reduce neighborhood results and rebuild the tile active list.

Modern CUDA should use boolean masks and stream compaction rather than sign bits in the solution field, but must preserve the same fixed-point semantics.

#### FIM edge cases and checks

- `f <= 0` is invalid for the eikonal equation; treat obstacles as unavailable rather than zero-speed nodes.
- An isolated component with no source remains infinite and must not seed velocity extension.
- Clamp a slightly negative quadratic discriminant to zero only within a roundoff-scale tolerance; a materially negative discriminant indicates bad inputs.
- Use a relative-plus-absolute convergence tolerance. With large distances, absolute-only tests can stall or terminate early.
- The update must be monotone nonincreasing from infinity toward the solution.
- Validate with constant-speed point-source distances, planar sources, anisotropic spacing, multiple sources, obstacles, and disconnected domains.

### 5.3 Reconstructing near-interface velocity extension from FIM

CM11b says it applies the JRW07 method in a two-cell band but does not publish component-level pseudocode. The implementable reconstruction is:

1. Identify liquid cells and interface-adjacent air cells.
2. Compute or approximate distance/order values outward from the interface with FIM.
3. Process active air nodes synchronously.
4. For each component, average already-known upwind neighbors whose distance is smaller, using the upwind discretization of the normal-extension PDE.
5. Mark a node known when the local update converges.
6. Stop accurate extension at two air cells, as in CM11b.

Do not extend normal velocity by simply copying the nearest cell in index space; this introduces grid-direction bias. Do not let a trace interpolate through a moving solid. For staggered CM12 velocities, either extend each face component on its native face lattice or consistently transfer to a collocated temporary field and back. CM11b's statement that all components can be processed together relies on its collocated grid and does not automatically apply to CM12's MAC grid.

### 5.4 Hierarchical far-field extrapolation

CM11b constructs levels from finest `L` to coarsest `1`, with each coarse spacing twice the finer spacing.

```text
HierarchicalExtrapolation(q_fine, known_fine):
    for level = finest down to coarsest+1:
        for each coarse node c:
            gather corresponding/interpolated finer values marked known
            if at least one valid value exists:
                q[level-1][c] = weighted average with renormalized weights
                known[level-1][c] = true

    for level = coarsest+1 up to finest:
        for each unknown fine node x:
            q[level][x] = trilinear interpolation from known coarse values
            known[level][x] = true if a valid stencil exists

    return q at finest level
```

The down sweep makes progressively cruder values available over large distances; the up sweep fills holes on finer levels. At boundaries, ignore unavailable nodes and renormalize. If all coarse interpolation nodes are unavailable, leave the node unknown until a documented fallback; do not silently set a physically meaningful velocity to zero unless the outer boundary condition demands it.

Validation:

- A constant interface velocity must remain constant throughout every connected extrapolation region.
- A planar linear tangential field should show bounded, grid-refining error.
- No value may leak through a solid separator.
- Every departure point needed by the maximum characteristic length should land in a valid extrapolated region.
- Compare the two-zone result against a full narrow-band FIM solve on small grids.

## 6. Conservative density advection

This is the center of CM12. Three generations of the algorithm must be distinguished: LGF11 guarantees conservation, LAF11 additionally controls destination fill, and CM12 reorders the same ideas for fewer GPU scatters.

### 6.1 Standard semi-Lagrangian operator

For each destination `i`, trace backward to a departure point and compute trilinear weights from donor cells `j`. Set `A[i,j] = w_minus[i,j]`. Ordinary semi-Lagrangian interpolation makes every row sum one, so a constant field is reproduced, but donor column sums are arbitrary. Some donors are oversampled (`beta > 1`) and others undersampled (`beta < 1`), so total mass changes.

### 6.2 LGF11 conservative backward-clamp/forward-remainder scheme

LGF11 visits each donor after the backward gather:

```text
LGF11ConservativeAdvection(rho):
    construct backward weights A from characteristic tracing
    beta[j] = sum_i A[i,j]

    for each donor j with beta[j] >= 1:
        for each destination i receiving from j:
            A[i,j] /= beta[j]

    for each donor j with beta[j] < 1:
        x_forward = trace cell center j forward by dt
        f[k,j] = solid-aware trilinear weights around x_forward
        A[k,j] += (1-beta[j]) * f[k,j]

    rho_next = A * rho
```

For every donor, final weights sum to one. Therefore

`sum_i rho_next[i] = sum_i sum_j A[i,j] rho[j] = sum_j rho[j] sum_i A[i,j] = sum_j rho[j]`.

This proof assumes a closed domain and exact unit-sum forward stencils. With open boundaries, mass carried into ghost cells must be recorded as outflow; ghost-cell donors contributing inward must be recorded as inflow.

#### LGF11 boundaries and interpolation

- Backward and forward rays stop at solids.
- Invisible interpolation points get zero weight.
- Remaining forward weights are renormalized so the remainder is not lost.
- Outflow requires enough ghost layers to receive the longest characteristic.
- LGF11 describes higher-order quadratic weights, but negative weights complicate positivity and conservation. CM12 uses trilinear interpolation, so the reference implementation should remain first order unless higher order is validated separately.

#### LGF11 failure mode in incompressible graphics flow

LGF11 fixes `beta = 1` but lets `gamma` vary widely. In CM12's comparison, `gamma` ranged from 0.271 to 9.793, creating empty-looking gaps and compressed regions on coarse grids. Conservation alone is therefore insufficient for a convincing incompressible scalar transport.

### 6.3 LAF11 approximate doubly stochastic correction

LAF11 seeks `beta = 1` and `gamma` close to one. It carries cumulative `gamma` across frames:

1. Initialize cumulative `gamma = 1` on the first frame.
2. Advect cumulative `gamma` backward with the ordinary semi-Lagrangian method.
3. Build backward `A`.
4. Scale each destination row by its `gamma`.
5. Compute `beta`.
6. For donors with `beta < 1`, forward-scatter `(1-beta)` into `A`.
7. Recompute `gamma` from row sums.
8. Scale rows by `gamma` again.
9. Recompute `beta`.
10. Normalize each donor column by `beta`, making `beta = 1` exactly.
11. Recompute `gamma`; evaluate `rho_next = A rho`.
12. Diffuse cumulative `gamma` while moving the corresponding amount of `rho` so donor sums remain unchanged.

Alternating row and column normalization resembles matrix balancing. LAF11 stops after a small number of normalizations and uses gamma diffusion because fully iterating to a doubly stochastic matrix is too expensive and can move the discrete operator farther from the original characteristics.

### 6.4 CM12 streamlined three-scatter algorithm

CM12 avoids explicitly storing `A`. The paper's exact sequence is:

```text
StreamlinedConservativeAdvection(rho_n, gamma_n, velocity, dt):
    1. gamma_adv = SemiLagrangianBackward(gamma_n)
       # gamma_n is one on the first frame.
    2. beta = 0
    3. For each destination i and each backward donor l:
           beta[l] += w_minus[l,i] * gamma_adv[i]

    4. For each destination i and backward donor l:
           scale = gamma_adv[i] / max(1, beta[l])
           rho_next[i]   += scale * w_minus[l,i] * rho_n[l]
           gamma_prime[i] += scale * w_minus[l,i] * gamma_n[l]
    5. gamma_next = gamma_prime

    6. For each donor j with beta[j] < 1:
           trace j forward; compute visible normalized w_plus[j,k]
           rho_next[k] += rho_n[j] * (1-beta[j]) * w_plus[j,k]

    7. Concurrently, for each donor j with beta[j] < 1:
           gamma_next[k] += gamma_n[j] * (1-beta[j]) * w_plus[j,k]

    8. GammaDiffusion(gamma_next, rho_next)
    return rho_next, gamma_next
```

Steps 3, 6, and 7 are scatter passes. Steps 6 and 7 can share the trace and weights and may be fused into one kernel that atomically updates two arrays. The paper counts them as distinct logical scatters and reports three rather than LAF11's five.

`max(1,beta)` is essential. For an oversampled donor it scales the backward contribution down. For an undersampled donor it leaves the backward contribution untouched and forward-scatters the missing fraction. Replacing it with `beta` for every donor eliminates the residual logic and changes the operator.

### 6.5 Gamma diffusion

After conservation, neighboring destination-fill weights are equalized dimension by dimension. For adjacent cells `i` and `j`, assume `gamma[j] > gamma[i]`. Transfer

`m = rho[j] * (gamma[j]-gamma[i]) / (2 gamma[j])`

from `rho[j]` to `rho[i]`, then set both gamma values to `(gamma[i]+gamma[j])/2`. Reverse the direction when `gamma[i] > gamma[j]`.

Implementation:

```text
GammaDiffusion(gamma, rho, iterations):
    repeat iterations times:              # CM12/LAF11 use 1..7
        for axis in x,y,z:
            old_gamma = gamma             # Jacobi within an axis
            old_rho = rho
            for each disjoint or atomically handled neighbor pair (i,j):
                if old_gamma[j] > old_gamma[i] and old_gamma[j] > 0:
                    m = old_rho[j]*(old_gamma[j]-old_gamma[i])/(2*old_gamma[j])
                    rho[j] -= m; rho[i] += m
                else if old_gamma[i] > old_gamma[j] and old_gamma[i] > 0:
                    symmetric transfer
                gamma[i] = gamma[j] = (old_gamma[i]+old_gamma[j])/2
```

The source describes Jacobi updates within a dimension and Gauss-Seidel behavior between dimension sweeps. A GPU implementation can use parity-colored nonoverlapping edges to avoid atomics, but the update ordering then becomes an implementation variant that must be regression-tested.

Correctness:

- Each pair transfer adds and subtracts the same `m`, so global mass is unchanged.
- Averaging two gamma values preserves their sum.
- Column sums `beta` are not changed by this post-operator redistribution.
- When gamma is uniform, no `rho` diffusion occurs.

Edge cases:

- Do not divide by zero gamma. A nonpositive gamma is already an invariant violation or disconnected-cell marker.
- Clamp transfer `m` to `[0,rho[donor]]` within floating-point tolerance.
- Do not diffuse across solid walls, closed boundaries, or disconnected fluid regions.
- A naive in-place parallel sweep is race-prone and nondeterministic.
- Too many iterations over-smooth the correction and moves mass farther than the characteristic-based operator intended.

### 6.6 Conservation and accuracy evaluation

CM12 reports, for its Figure 2 case:

| Method | Minimum gamma | Maximum gamma | Interpretation |
|---|---:|---:|---|
| CM12 | 0.627 | 2.403 | Similar fill variation to LAF11, fewer scatters |
| LAF11 | 0.627 | 2.502 | Reference cumulative balancing method |
| LGF11 | 0.271 | 9.793 | Large visible compressibility artifacts |

Required tests:

1. Constant scalar in a divergence-free translational field: mass and constant value should remain constant.
2. Single-cell impulse under translation: sum is invariant, all values remain nonnegative, centroid follows the characteristic within interpolation error.
3. Zalesak disk or notched shape rotation: mass roundoff-level invariant; shape error measured separately.
4. Divergent synthetic velocity: compare conservative-form behavior to a high-resolution conservative finite-volume reference.
5. Large-CFL obstacle flow: no donor crosses the obstacle, no alternating gaps, mass balance includes boundary flux.
6. Empty and full fields: exact fixed points.
7. Randomized weights: verify every donor's combined backward-plus-forward coefficient sums to one.
8. GPU determinism: bound run-to-run mass differences from atomic addition order; use double-precision diagnostic reduction even if storage is float.

### 6.7 Important limitations

- First-order trilinear semi-Lagrangian interpolation is diffusive even when mass is exact.
- “Unconditionally stable” does not mean the departure point is accurate for arbitrarily large `dt`.
- Exact global surface-density mass does not imply exact volume of the 0.5 isosurface.
- Gamma diffusion repairs an algebraic fill defect heuristically; it is not a proof of discrete divergence-free transport.
- Floating-point atomics conserve only up to arithmetic error. Summation order matters.

## 7. Density sharpening

Advection blurs the 0.5 interface. CM12 starts from MMTD07's normal-flow sharpening increment and replaces its global mass correction with a local gradient-directed return.

### 7.1 MMTD07/CM12 Godunov-style sharpening increment

For a fictitious sharpening time `DeltaT`, compute mass changes caused by unit coordinate velocities. Along x:

`delta_x_plus[i]  = -(rho[i]-rho[i-x]) * dx * DeltaT`

`delta_x_minus[i] = -(rho[i+x]-rho[i]) * dx * DeltaT`.

Compute y and z analogues. CM12 uses `DeltaT = 3*dt` in all examples.

The maximum magnitude of a mass increase caused by any unit velocity is

```text
DeltaT_grad_plus[i] = (1/dx^2) * sqrt(
    max(max(delta_x_plus,0)^2, min(delta_x_minus,0)^2) +
    max(max(delta_y_plus,0)^2, min(delta_y_minus,0)^2) +
    max(max(delta_z_plus,0)^2, min(delta_z_minus,0)^2))
```

The maximum decrease is

```text
DeltaT_grad_minus[i] = (1/dx^2) * sqrt(
    max(min(delta_x_plus,0)^2, max(delta_x_minus,0)^2) +
    max(min(delta_y_plus,0)^2, max(delta_y_minus,0)^2) +
    max(min(delta_z_plus,0)^2, max(delta_z_minus,0)^2))
```

Define the limiter weight

`w[i] = (rho[i]-0.5)^3 * (1 - min(1, max_neighbor_abs_difference/tau))`,

where `tau = 0.4` and the maximum is over face-adjacent cells. Then

```text
if w[i] >= 0:
    delta_rho[i] = w[i] * DeltaT_grad_plus[i]
else:
    delta_rho[i] = w[i] * DeltaT_grad_minus[i]
```

The cube changes sign at 0.5 and drives values away from the interface midpoint: low densities lower, high densities rise. The neighbor-difference limiter shuts sharpening off when an adjacent jump already reaches `tau`, preventing over-sharpening and grid aliasing.

### 7.2 Why MMTD07's global correction is not used

MMTD07 sums the mass change from all cells and redistributes the opposite change according to an interface-area measure. It conserves mass globally but can move mass across the entire domain or connected component. CM12 demonstrates a small airborne droplet losing its density to a distant pool. Connected-component restriction [KLL+07] still allows harmful movement within one connected thin feature.

### 7.3 CM12 limiter before local return

CM12 modifies each proposed increment:

```text
LimitSharpeningIncrement(rho, delta, epsilon=1e-5):
    for each cell i:
        if rho[i] + delta[i] < 0 or rho[i] < epsilon:
            delta[i] = -rho[i]
        else if rho[i] > 0.5:
            delta[i] = 0
        else:
            keep delta[i]
```

Then it applies `rho += delta` and returns `-delta` locally.

Consequences:

- Cells with `rho > 0.5` are not directly modified by the deletion phase.
- Mass moves from the air side toward the liquid side only.
- Small positive tails are zeroed to reduce future work.
- The first branch must take precedence over the `rho > 0.5` branch exactly as printed, although a well-behaved increment should not make a high-density cell negative.

### 7.4 TraceAlongField

For every cell, CM12 follows `grad rho` from the cell center toward the 0.5 isocontour, using multiple forward Euler substeps. Stop when one of these occurs:

- The path reaches or crosses `rho = 0.5`.
- Path length reaches `D*dx`.
- The path crosses a solid boundary.

The paper uses `D` from 1.1 to 3.1 and `D = 2.1` in the general results. Increasing `D` visually resembles stronger surface tension because removed density can be deposited farther toward the main interface.

Robust reconstruction:

```text
TraceAlongField(x0, rho, D, dx):
    x = x0
    travelled = 0
    prev = sample(rho,x)
    while travelled < D*dx:
        g = sample_gradient(rho,x)
        if norm(g) < grad_epsilon:
            return x, STALLED
        direction = g / norm(g)
        h = min(trace_step, D*dx-travelled)
        x_new = x + h*direction
        if segment_crosses_solid(x,x_new):
            return last_non_solid_point, HIT_SOLID
        current = sample(rho,x_new)
        if (prev-0.5)*(current-0.5) <= 0:
            return linearly_interpolated_crossing(x,x_new), REACHED_INTERFACE
        x = x_new; prev = current; travelled += h
    return x, MAX_DISTANCE
```

CM12 does not specify substep size, gradient stencil, or stalled-gradient fallback. Use a substep no larger than roughly `0.25*dx` for reliable crossing detection, and test convergence under step refinement. A zero gradient in a low-density island is common; depositing at the current point preserves mass but may not sharpen, while deleting it would violate conservation.

### 7.5 ScatterValue and local mass return

```text
ReturnRemovedMassLocally(rho_after_delta, delta, D, solids):
    for each cell i:
        amount = -delta[i]
        if amount == 0: continue
        p, status = TraceAlongField(center(i), rho_before_or_frozen, D*dx)
        weights = trilinear_weights(p)
        zero weights whose nodes lie in solid
        if sum(weights) > 0:
            normalize weights
            atomic_scatter(rho_after_delta, amount*weights)
        else:
            rho_after_delta[i] += amount  # conservative fallback
```

The trace field should be frozen for the entire sharpening pass. Reading a density field while concurrent scatters modify it makes paths order-dependent and nondeterministic.

Correctness proof: each cell first changes mass by `delta[i]`, then scatters exactly `-delta[i]`. If normalized target weights sum to one, the net global change is zero cell by cell. Locality is bounded by `D*dx` unless the fallback or an erroneous trace violates the bound.

### 7.6 Solid-aware sharpening fluxes

For partial solids, coordinate mass changes use open face area. Along positive x:

`delta_x_plus[i] = -(rho[i]*Vf[i+x/2] - rho[i-x]*Vf[i-x/2]) * dx * DeltaT`.

The negative-x expression similarly uses the forward neighbor and the two face fractions. This prevents fictitious sharpening flux through a blocked portion of a face.

### 7.7 Sharpening edge cases

- A perfectly constant field has zero gradient and must remain unchanged.
- A one-cell droplet may have no reachable 0.5 crossing; use a conservative local fallback.
- Two nearby interfaces can cause the gradient to point to the wrong sheet. The distance cap limits damage but does not eliminate it.
- At a saddle or noisy gradient, forward Euler can oscillate; normalize the gradient and cap step size.
- Deposited density may exceed one. CM12 allows this temporarily and repairs it gradually through pressure divergence.
- Density above 0.5 is deliberately not sharpened, so a broad region just above 0.5 can theoretically double visible volume at fixed mass.
- Do not trace across periodic seams or open boundaries without explicit topology-aware handling.
- Atomics can produce tiny negative or over-one values due to ordering; repair only within a stated tolerance and include repair magnitude in the mass ledger.

### 7.8 Sharpening validation

1. One-dimensional smooth step: interface width should decrease, mass should remain constant.
2. Two separated humps: no mass transfers between them when farther apart than `D*dx`.
3. Small droplet above a pool: droplet mass should remain local, reproducing CM12 Figure 3's qualitative result.
4. Solid wall: no deposited density inside `V=0`; visible weights renormalize.
5. Random field: sum before and after differs only by reduction roundoff.
6. `D` sweep 1.1, 2.1, 3.1: quantify interface width, centroid drift, and maximum transport distance.
7. Trace-step refinement: results should converge as substep length is halved.

## 8. Handling moving and cut solid boundaries

### 8.1 Geometry fractions

`V[i]` is the non-solid volume fraction, and `Vf` values are non-solid face-area fractions. They must be generated consistently from the same solid geometry and time sample. A cell can contain both air and liquid; `V` describes only solid exclusion, not liquid occupancy.

Required geometric identities and checks:

- `0 <= V,Vf <= 1`.
- Fully solid cells have `V=0` and ordinarily all incident interior face fractions zero.
- A fully open cell has `V=1`.
- Face fractions shared by two cells are single stored values, not independently rasterized values.
- Fast moving solids require swept collision tests for characteristics; end-position voxelization alone can permit tunneling.

### 8.2 Excess density in a partial-solid cell

The invalid state is `rho[i] > V[i]`. For `V < 1`:

```text
EjectExcessDensityFromPartialSolids(rho, V, phi_s, S=1):
    for each cell i with V[i] < 1 and rho[i] > V[i]:
        excess = rho[i] - V[i]
        x = trace from center(i) along grad(phi_s), away from solid,
            for at most S*dx
        scatter excess to nearby non-solid grid points using normalized
            trilinear weights
        rho[i] -= excess
```

CM12 uses `S = 1`. The method guarantees `rho=0` inside a completely solid source cell when a valid target exists, and keeps `rho <= V` in most partial cells near solids. The qualification “most” matters: simultaneous scatters from many cells can overfill a target, and a target stencil may be blocked.

### 8.3 Moving-solid edge cases

- If a newly covered cell contains density, ejection must occur before it can be used as a valid liquid cell.
- If every scatter target is solid, keep the excess in a recovery buffer and retry after motion, or conservatively return it to the source while excluding that source from pressure classification. Never discard it silently.
- Use the solid signed-distance gradient pointing toward increasing distance; a reversed sign pushes mass deeper into the solid.
- At a non-differentiable medial location, the gradient may vanish. Use the closest-surface normal from the geometry query as a fallback.
- One-way coupling is assumed. The fluid does not update the solid's motion in CM12.
- Mass ejection and collision-aware advection must use compatible solid time intervals; otherwise mass may be moved through a wall that the characteristic considers closed.

## 9. Velocity advection and external forces: a genuine specification gap

CM12 lists “velocity advection and external force addition” but does not state the velocity advection discretization. It says the overall structure follows MMTD07, while its conservative-advection section explicitly discusses advection of `rho`, not staggered momentum. CM11b uses modified MacCormack for velocity, but CM12 cites CM11b only for velocity extrapolation. LAF11 contains conservative momentum advection, but CM12 says its density method is derived from LAF11 and does not say velocity uses that method.

Therefore no single velocity-advection algorithm can honestly be attributed to CM12 from the supplied text. An implementation must choose and declare one of these profiles:

| Profile | Velocity algorithm | Benefit | Risk relative to CM12 |
|---|---|---|---|
| Minimal first-order | Backward semi-Lagrangian component advection on MAC faces | Stable and simple | Dissipative; not momentum conserving |
| CM11b-like | Modified MacCormack with extrema fallback to first-order semi-Lagrangian | Less numerical diffusion | Not explicitly selected by CM12; limiter details matter |
| LAF11-like | Conservative momentum advection with collision-aware traces | Better momentum preservation at large CFL | More expensive; free-surface donor rules are nontrivial |

Whichever profile is chosen:

1. Trace each face component on its own staggered lattice or use a carefully defined vector interpolation.
2. Stop traces at solids and impose solid-relative boundary behavior through the pressure step, not by importing arbitrary solid momentum during advection.
3. Integrate gravity and other external acceleration explicitly, normally `u_star += dt*f`.
4. Record momentum change caused by external forces separately from numerical momentum error.
5. Verify a uniform velocity is a fixed point, solid walls are not crossed, and the pressure projection removes divergence without introducing a net internal-force momentum change in a closed domain.

This reference intentionally does not fill the gap by pretending CM11b's MacCormack scheme is part of CM12.

## 10. Effective liquid fraction for pressure

### 10.1 Why raw rho is insufficient in cut cells

Suppose a cell has non-solid fraction `V < 0.5` but all available non-solid volume is filled with liquid. Its surface density is naturally near `V`, hence below 0.5. Classifying it using raw `rho > 0.5` would incorrectly mark it as air.

CM12 defines

```text
rho_eff[i] = 0                  if V[i] == 0
rho_eff[i] = rho[i] / V[i]     otherwise
```

Interpretation: `rho_eff` estimates liquid occupancy relative to available non-solid volume. Use a robust test `V <= V_epsilon` rather than an exact floating-point comparison, but preserve mass and diagnostics separately; `rho_eff` is a pressure-classification field, not a replacement for conserved `rho`.

CM12 extrapolates `rho_eff` from cells with `V>0` into adjacent `V=0` cells so those solid-side cells can participate as pressure unknowns/stencil support. It then defines the approximate liquid signed distance

`phi_liquid[i] = -(rho_eff[i]-0.5)*dx`.

This is exact neither as a signed distance nor as an interface reconstruction. It is a one-cell pressure boundary proxy. Clamp or redistance it only if doing so preserves the 0.5 zero crossing used to construct coefficients.

### 10.2 Edge cases

- Dividing by tiny positive `V` amplifies noise. Treat geometry fractions below a documented threshold as solid and account for any density there through ejection.
- `rho_eff > 1` is allowed and drives the artificial divergence repair.
- `rho_eff < 0` is invalid.
- Extrapolation into solids must be limited to the stencil width; broad extrapolation can create fictitious liquid regions.
- A fully solid island not adjacent to liquid should not become a pressure unknown merely because a generic extrapolator reached it.

## 11. Variational cut-cell pressure projection

### 11.1 Variational meaning from BBB07

Pressure projection finds the pressure that minimizes post-projection kinetic energy subject to incompressibility. Discretizing the global energy with fractional fluid/solid volumes produces a symmetric positive semidefinite system in the unconstrained case and automatically accounts for sub-grid solid geometry. This is more accurate than voxelizing every cut cell as wholly solid or wholly fluid.

Let `u_star` be the advected, force-updated velocity. Abstractly, solve

`L(p) = D(u_star)`

and apply the compatible pressure gradient to velocity. `D` and the gradient must be negative adjoints under the chosen volume/mass weights; otherwise the energy argument and symmetry can fail.

### 11.2 CM11a discrete operator

On a regular staggered grid, each pressure equation has six neighbor contributions. Along +x, CM11a uses a coefficient proportional to the open fraction `V[i+1/2]` multiplying `(p[i]-p_plus)`. Other directions are analogous.

For an air neighbor, the ghost-fluid pressure is chosen so pressure reaches zero at the actual free surface, not the air cell center. In the notation printed by CM11a:

```text
p_plus = p[i] * phi_air / phi[i]     when the neighbor is air
p_plus = p[i+x]                      otherwise
```

Equivalent implementations commonly express this as a theta-scaled diagonal coefficient. The essential condition is linear interpolation of pressure to `p=0` at `phi=0`, with theta clamped away from zero to avoid an unbounded coefficient when the interface nearly coincides with the liquid cell center.

The right-hand side uses volume-weighted face flux plus solid-velocity corrections. Along x it contains the difference of open face fluxes and terms that incorporate solid motion as cell and face fractions differ. The y and z terms are analogous. Reuse CM11a's discretization as a matched operator pair; combining a different divergence with its Laplacian breaks consistency.

### 11.3 Ghost-fluid free-surface condition

ENGF03's operative idea is a Dirichlet air pressure imposed at the subcell interface. With zero gauge air pressure, interpolate between liquid pressure and a ghost pressure so the interpolant is zero where `phi` crosses zero. This avoids the first-order geometric error of imposing zero at the center of the neighboring air cell.

Edge cases:

- If both cells are liquid, use the ordinary pressure difference.
- If both are air, there is no liquid pressure equation for that face.
- If `phi` values do not bracket zero, do not invent a crossing.
- Clamp the interface fraction `theta` to a small positive minimum for conditioning, and report how often clamping occurs.
- Surface tension is not part of CM12; with surface tension the Dirichlet value would be a pressure jump, not zero.

### 11.4 Separating solid boundary complementarity

For a liquid-solid interface, BBB07/CM11a require

`0 <= p  perpendicular  (u-u_s) dot n >= 0`.

This means:

- Contact: `p > 0` and relative normal velocity is zero.
- Separation: `p = 0` and liquid normal velocity is at least the solid normal velocity; the wall cannot pull liquid toward itself.

The variational discretization already supplies the KKT structure, so CM11a enforces `p >= 0` at constrained solid-side nodes. Without this projection, negative pressure can glue liquid to a receding or curved wall.

CM11a discusses a more exact edge-based projection that adjusts all pressures around a non-grid-aligned interface by a minimum-norm correction. It finds a simpler node-based clamp visually indistinguishable and uses that in the multigrid solver. CM12 therefore inherits the node-based approximation, not the full edge-level solve.

## 12. Projected multigrid pressure solver

### 12.1 Hierarchy construction

For finest level `M`, each coarser level collapses `2x2x2` fine cells. Downsample non-solid fractions by an 8-to-1 average, including face-centered half-index quantities with border replication where needed.

Downsample `phi` as follows:

1. If all eight values have the same sign, use their average.
2. If the level is coarser than the `C` finest bubble-preserving levels, use their average.
3. Otherwise use the average of positive values.

CM11a uses `C=2`. This retains small air bubbles where they materially affect fine-grid pressure but lets them disappear on coarse levels where they can distort the broad pressure profile.

At every level:

1. Extrapolate `phi` into solid cells one cell away from liquid.
2. Build matrix `A_m` using the variational cut-cell and ghost-fluid coefficients.
3. Define the pressure lower bound `p_min = 0` in constrained solid cells and `-infinity` elsewhere.

### 12.2 Projected red-black Gauss-Seidel smoother

For each red or black pressure node, perform the ordinary Gauss-Seidel update from the row equation, then project:

`p[i] = max(p_unconstrained[i], p_min[i])`.

Red and black passes allow parallel updates because same-color nodes do not directly depend on each other on a seven-point stencil. The projection makes this PRBGS rather than ordinary RBGS.

Checks after every smoothing sweep:

- No constrained pressure is negative beyond tolerance.
- The complementarity residual is tracked separately from the linear residual.
- Matrix diagonal is positive for every active unknown.
- Isolated or singular all-Neumann components have a gauge condition or are removed.

### 12.3 Restriction and prolongation

CM11a uses trilinear interpolation for restriction and prolongation. At domain boundaries, clamp/replicate according to the hierarchy rule. The tall-cell paper instead ignores out-of-grid values and renormalizes; do not mix these rules without testing.

When carrying a fine-level separating constraint to a coarser correction equation, use

```text
DownsampleSubtract(p_min_fine, p_fine)[coarse] =
    max over the 8 children (p_min_fine[child] - p_fine[child])
```

Only the `S` finest levels receive these constraints; CM11a uses `S=3`. On coarser levels, set the correction lower bound to negative infinity because the fine geometric contact constraint has no meaningful coarse analogue.

### 12.4 V-cycle

```text
V_Cycle(m):
    if m == 1:
        solve A[1] p[1] = b[1] to high precision
        return

    repeat num_pre_sweeps:
        PRBGS_smooth(p[m], b[m], A[m], p_min[m])

    r[m] = b[m] - A[m] p[m]
    b[m-1] = Restrict(r[m])
    p[m-1] = 0

    if m > M-S:
        p_min[m-1] = DownsampleSubtract(p_min[m], p[m])
    else:
        p_min[m-1] = -infinity

    V_Cycle(m-1)
    p[m] += Prolong(p[m-1])

    repeat num_post_sweeps:
        PRBGS_smooth(p[m], b[m], A[m], p_min[m])
```

The residual line in the printed pseudocode contains a minor level-index typo (`b_l` rather than `b_m`); the intended residual is the current level's `b[m]-A[m]p[m]`.

### 12.5 Full multigrid cycle

```text
Full_Cycle():
    save = p[M]
    compute p_min[M]
    p_min[M] -= p[M]            # correction-space bound
    r[M] = b[M] - A[M]p[M]

    for m=M-1 down to 1:
        r[m] = Restrict(r[m+1])
        if m >= M-S:
            p_min[m] = DownsampleSubtract(p_min[m+1], 0)
        else:
            p_min[m] = -infinity

    b[1] = r[1]
    solve A[1]p[1]=b[1]

    for m=2 to M:
        p[m] = Prolong(p[m-1])
        b[m] = r[m]
        V_Cycle(m)

    p[M] = save + p[M]
```

CM11a's performance tests use three full cycles followed by four V-cycles, with four pre- and four post-sweeps. The tall-cell paper reports that one full multigrid cycle plus two V-cycles was visually sufficient in its different solver. CM12 does not publish its exact cycle counts or residual tolerance; this is a tuning gap.

### 12.6 Top-level solve

At the coarsest level solve to high precision. On a GPU, CM11b suggests clamping hierarchy depth where the whole top level fits in shared memory and running many Gauss-Seidel iterations in one kernel. A modern implementation may use a small direct solve or CPU solve if transfer overhead is acceptable.

### 12.7 Why full cycles and boundary preservation matter

The tall-cell dependency identifies three convergence-critical features that remain relevant conceptually:

1. Use full cycles to remove low-frequency error early.
2. Preserve air bubbles on the finest hierarchy levels.
3. Apply ghost-fluid and solid-fraction geometry on all relevant levels.

Omitting any can cause stagnation or divergence because coarse grids otherwise solve a different boundary-value problem.

### 12.8 Pressure update

After solving, apply the pressure gradient using the same cut-cell weights used to build `A`:

`u_next = u_star - (dt/d) grad p`.

At a separating wall, the corrected normal velocity must not penetrate the solid and must be allowed to move away. At a free surface, pressure reaches zero at the reconstructed interface.

### 12.9 Pressure solver edge cases

- No liquid unknowns: skip the solve and keep valid boundary velocities.
- Entirely enclosed liquid with pure Neumann conditions: remove the pressure nullspace by fixing one gauge value or subtracting the component mean.
- Disconnected liquid components: each all-Neumann component needs its own gauge handling.
- Tiny cut fractions: coefficients can become ill-conditioned; fraction thresholds must be paired with conservative geometry repair.
- A diagonal of zero indicates an isolated unknown; remove or constrain it.
- Coarse-level topology can change. Preserve fine bubbles only for `C` levels as specified.
- Projecting after prolongation alone is insufficient; constraints must be enforced during smoothing and transferred into correction space.
- A small linear residual does not imply complementarity satisfaction. Report both.

### 12.10 Pressure validation

1. Hydrostatic tank, including a 45-degree wall: pressure should be linear with depth; velocity remains near zero.
2. Receding curved wall: liquid peels off without negative pressure or sticky filaments.
3. Approaching wall: no penetration and nonnegative contact pressure.
4. Manufactured Poisson solution on cut geometry: verify convergence under grid refinement.
5. Divergence reduction: report `L2`, `Linf`, and volume-weighted flux imbalance before and after projection.
6. Multigrid convergence: residual reduction per cycle; compare with a direct or PCG/LCP reference on small grids.
7. Bubble preservation: a small air pocket survives on the finest `C=2` levels without corrupting coarse pressure.
8. Complementarity: measure `min(p)`, minimum separation velocity where `p=0`, and product residual `p * relative_normal_velocity`.

## 13. Excess-density divergence repair

When `rho_eff > 1`, CM12 adds

`source[i] = min(lambda*(rho_eff[i]-1), eta) / dx`

to the divergence, using `lambda=0.5` and `eta=1`. The pressure response pushes excess density away over subsequent advection steps. MMTD07 used `lambda=1` and no cap; CM12 reports instability when fast liquid accumulates at a solid.

Sign convention warning: “add to divergence” depends on whether the linear system uses `b = div(u_star)`, `b = -div(u_star)`, or includes `dt/d`. Verify with a one-cell test that `rho_eff > 1` creates an expanding corrected velocity, not compression.

This is gradual feedback, not an exact volume constraint. It does not alter conserved `rho` directly. The cap improves stability but lengthens recovery for severe over-density.

Tests:

- One overfull isolated region: maximum `rho_eff` should decay over frames without mass loss.
- Very large `rho_eff`: divergence source saturates at `eta/dx`.
- `rho_eff <= 1`: zero source.
- Fast jet into a wall: no pressure blow-up and mass remains constant.

## 14. Density post-processing for sub-grid rendering detail

This algorithm is for rendering only. Never feed its output back into advection, sharpening, or pressure.

### 14.1 Detail discriminator

Start from

`detail_gamma[i] = 2 * min(rho[i], 0.5)`.

Thus values at or above 0.5 map to one, and small positive densities map into `(0,1)`. Apply a Gaussian blur to `detail_gamma`. CM12 uses `sigma = 2*dx` in its demonstrated example.

Reasoning:

- On the air side of a large liquid body, nearby interior values of one blur outward and raise gamma.
- In an isolated sub-grid sheet or splash where all nearby values are small, blurred gamma remains small.

### 14.2 Density amplification

CM12 defines

`rho_render[i] = rho[i] / min(max(detail_gamma_blur[i], theta), 1)`

with `theta = 0.01` in its example. Extract the 0.5 isosurface of `rho_render`.

The denominator is in `[theta,1]`, so this only increases density. The `theta` floor prevents division by zero and caps amplification at `1/theta = 100`.

### 14.3 Gaussian blur implementation

Use a separable Gaussian for efficiency:

1. Choose a truncation radius, commonly `ceil(3*sigma/dx)`.
2. Build normalized one-dimensional weights.
3. Convolve x, then y, then z into separate buffers.
4. Apply explicit boundary conditions: reflect, clamp, periodic, or zero. For closed containers, do not blur detail support through solids; mask and renormalize the kernel on the visible side.

The paper specifies sigma but not truncation, boundary behavior, or solid masking. These choices can materially change which features appear.

### 14.4 Limitations and edge cases

- A tiny numerical-density speck can be amplified into visible geometry. Threshold input noise or require connected support, but document the deviation.
- Amplification thickens features and can create apparent volume not present in the simulation.
- CM12 suggests future thinning of surfaces created from original `rho<0.5`; it does not provide that algorithm.
- Two nearby thin sheets may merge after blur.
- Near a solid, unmasked blur can make detail appear through the wall.
- `theta <= 0` is invalid; `theta > 1` disables the intended bound.
- Use a copy so rendering does not change the mass ledger.

### 14.5 Validation

1. Large smooth interface: `rho_render` should remain close to `rho` on its air side.
2. Isolated thin sheet: a coherent 0.5 rendering surface should appear.
3. Pure zero field: remains zero despite division floor.
4. Solid separator: no detail crosses the solid.
5. Parameter sweep of sigma and theta: report rendered volume and connected-component count.

## 15. Marching cubes surface extraction and volume measurement

CM12 applies marching cubes [LC87] to the 0.5 isosurface of `rho` or `rho_render`.

Implementation requirements:

1. Evaluate each grid cube's eight scalar values relative to 0.5.
2. Use the case table to create edge intersections and triangles.
3. Linearly interpolate each edge intersection.
4. Share edge vertices or weld them deterministically to avoid cracks.
5. Resolve ambiguous cases consistently; the original marching-cubes table can produce topology ambiguities.
6. Orient triangles consistently before computing signed volume.

CM12 measures visible volume from the enclosed triangle mesh. For an oriented watertight mesh, sum signed tetrahedron volumes, e.g. `dot(a,cross(b,c))/6`, and take the consistent signed total. If the mesh is open, nonmanifold, or inconsistently oriented, this volume is unreliable and the failure must be reported.

Critical limitation: nonzero density below 0.5 contributes to conserved mass but produces no surface. A liquid layer thinner than one cell can visually disappear while mass remains exactly present. High surface-area-to-volume configurations likewise show significant apparent volume loss.

Tests:

- Sphere at multiple radii/resolutions: surface and volume convergence.
- Plane: no cracks, predictable triangle count.
- Ambiguous checkerboard cases: deterministic topology.
- Thin sheet below 0.5: expected empty visible surface but nonzero mass, validating metric separation.
- Closed-mesh check before volume integration.

## 16. Correctness properties: what is proved, heuristic, or not guaranteed

### 16.1 Exact or algebraically enforced properties

**Advection mass conservation.** In a closed domain, each donor's final transport weights sum to one. Therefore the sum of `rho` is unchanged except for floating-point reduction error.

**Sharpening mass conservation.** Every local deletion/addition `delta` is paired with a scatter of `-delta` whose normalized weights sum to one. Net mass change is zero if every scatter has a conservative fallback.

**Gamma-diffusion conservation.** Each pairwise `rho` transfer is equal and opposite; gamma averaging preserves the pair sum.

**Nonnegative sharpening state.** Equation 17 replaces a would-be negative result with zero and sets `delta=-rho`. This does not prove all later atomic sums remain nonnegative, so post-kernel tolerance checks are still required.

**Separating pressure feasibility.** PRBGS projects constrained pressures to `p>=0` at every smoothing update. This enforces the node-level approximation of complementarity.

### 16.2 Properties controlled heuristically

**Locality.** Sharpening mass return is bounded by `D*dx` along the chosen gradient trace, but direction accuracy and fallback behavior are heuristic.

**Destination fill.** Cumulative gamma balancing and 1-7 diffusion iterations keep row sums near one; they do not make the operator exactly doubly stochastic.

**Visible volume.** Sharpening, effective-fill pressure classification, and excess-density divergence keep the 0.5 volume near its target in many cases. No exact volume theorem is provided.

**Solid cleanup.** Ejection removes `rho-V` from a source cut cell but simultaneous target deposits can create new overfull cells. The pressure feedback repairs remaining excess over time.

**Multigrid convergence.** The reported solver converges well when full cycles, bubble preservation, and geometric coefficients are used. A universal convergence proof for every changing cut-cell complementarity problem is not given.

### 16.3 Explicit non-guarantees

- Visible volume can temporarily or significantly decrease while global mass is exact.
- A broad region with `rho` just above 0.5 can theoretically expand visible volume by almost a factor of two without changing mass.
- Density post-processing creates apparent geometry and is not conservative.
- Velocity/momentum conservation is not specified by CM12.
- Large-CFL stability does not guarantee trajectory or interface accuracy.
- Local conservation is meant in the bounded sharpening-transport sense, not a finite-volume flux balance for every physical control volume at every substep.

## 17. Mass, volume, divergence, and error ledgers

Maintain separate metrics:

```text
surface_mass = sum_i rho[i] * dx^3
available_liquid_fraction_mass = sum_i min(rho[i], V[i]) * dx^3
overfull_mass = sum_i max(rho[i]-V[i], 0) * dx^3
visible_volume = volume(marching_cubes(rho, iso=0.5))
render_volume = volume(marching_cubes(rho_render, iso=0.5))
div_L2 = sqrt(weighted_mean(div(u)^2))
div_Linf = max(abs(div(u)))
```

For open domains, use the LGF11 ledger:

`conservation_error(t) = current_mass - initial_mass - cumulative_inflow + cumulative_outflow - explicit_sources + explicit_sinks`.

Do not compare `surface_mass` numerically to `visible_volume` as if they had identical discrete meaning. In a perfect binary field they coincide up to units, but diffuse sub-grid density intentionally separates them.

Use double precision for diagnostic reductions and a stable parallel reduction. Suggested acceptance criteria should scale with operation count and storage precision; do not hard-code a universal epsilon. A practical regression threshold for a closed single-precision GPU run might be on the order of tens to hundreds of machine epsilons per frame times total mass, verified empirically against a double-precision reference.

## 18. End-to-end validation suite

### 18.1 Unit tests

| Component | Test | Pass condition |
|---|---|---|
| Trilinear interpolation | Random points in a cell | Nonnegative weights, sum one, exact constants and linear fields |
| Solid masking | Partially blocked stencil | Blocked weights zero, visible weights sum one |
| Characteristic collision | Segment through thin moving wall | Trace stops before wall; no tunneling |
| FIM local update | Known one-, two-, three-axis cases | Satisfies Godunov residual and upwind causality |
| Conservative donor | Random beta above/below one | Final emitted coefficient sum one |
| Gamma pair diffusion | Random positive pair | Pair rho and gamma sums unchanged; no negative rho |
| Sharpen limiter | Values around 0, epsilon, 0.5 | Exact branch behavior and no negative result |
| TraceAlongField | Analytic linear rho | Reaches known 0.5 point within integration error |
| ScatterValue | Solid-masked target | Deposited amount equals input amount |
| Effective fill | V in {0,tiny,partial,1} | No division error; documented classification |
| PRBGS | Small constrained matrix | Pressure lower bound and residual both satisfied |
| Post-process | rho=0 and rho>=0.5 | Zero remains zero; high density unchanged |

### 18.2 Conservation tests

1. Closed periodic translation for thousands of steps.
2. Reversible vortex: run forward then reverse velocity; record mass exactly and shape error separately.
3. Obstacle circulation with high CFL.
4. Moving container with no sources.
5. Repeated sharpening without advection: mass constant; interface reaches a bounded width rather than diverging.
6. Empty-to-filled source test: every inserted liquid ball is recorded as an explicit source and persists in the ledger.

### 18.3 Pressure and boundary tests

1. Static hydrostatic column in axis-aligned and rotated containers.
2. Dam break in a sphere.
3. Fast jet into a wall, exercising excess-density cap.
4. Solid object moving rapidly across a tank, exercising swept traces and one-way coupling.
5. Receding solid surface, comparing sticky versus separating constraints.
6. Partially filled cut cell with `V<0.5` and `rho=V`: must be treated as full liquid by `rho_eff`.

### 18.4 Surface and volume tests

1. Ball falling onto a plane and spreading below one-cell thickness: expect mass conservation but visible-volume loss.
2. Crown splash and thin sheet: compare raw and post-processed surfaces.
3. High surface-area foam-like geometry: quantify mass/volume divergence.
4. Small droplets rejoining the main body: their stored density must restore the global visible level.

### 18.5 Reference comparisons

On small grids, compare against:

- Explicit sparse-matrix construction of the CM12 advection operator.
- LAF11 five-scatter method.
- LGF11 baseline.
- A direct constrained pressure solve or trusted QP/LCP solve.
- Full-band FIM velocity extension instead of two-zone extrapolation.
- Double-precision CPU kernels with deterministic summation.

The comparison should include fields, invariants, residuals, and topology—not only rendered images.

## 19. GPU implementation notes

### 19.1 Memory layout

Use structure-of-arrays storage. Keep cell-centered fields contiguous and each face component in its native staggered allocation. Separate `advect_gamma` from `detail_gamma`. Double-buffer any field that a synchronous update reads and writes.

### 19.2 Scatter implementation choices

- Floating-point atomics are simplest but nondeterministic in last bits.
- Sort-by-destination plus segmented reduction is deterministic for a fixed sort and can improve accuracy, at higher memory cost.
- A prefix-scan can compact active donors or FIM tiles.
- Fuse forward `rho` and gamma scatter only if both maintain identical visibility and normalization decisions.

### 19.3 Active regions

CM12 is purely Eulerian but still benefits from active masks:

- FIM active tiles in the accurate velocity band.
- Density donors with nonzero `rho` or relevant gamma.
- Sharpening cells near the diffuse interface.
- Pressure unknowns in liquid and required solid-adjacent stencil cells.

Active-mask construction itself must be conservative: pruning a small positive donor without accounting for its mass changes the result.

### 19.4 Numerical precision

- Store simulation fields in float if performance requires it.
- Accumulate global diagnostics in double.
- Consider double or compensated accumulation for beta/gamma on very large grids.
- Never use infinity in arithmetic paths that can multiply it by zero; FIM may use a large sentinel with explicit validity bits.
- Track NaN/Inf counters per kernel and fail fast in debug builds.

### 19.5 Determinism and reproducibility

Record grid, `dt`, `dx`, all thresholds, trace integrator, blur kernel, multigrid cycle counts, smoothing counts, residual tolerances, fraction thresholds, atomic/reduction mode, and GPU architecture. The paper's headline parameter defaults are:

| Parameter | Published value/range |
|---|---|
| Simulation `dt` | `1/30 s` |
| Grid spacing `dx` | `0.05 m` in reported examples |
| Gravity | `10 m/s^2` |
| Sharpen fictitious `DeltaT` | `3*dt` |
| Sharpen jump limit `tau` | `0.4` |
| Tiny density `epsilon` | `1e-5` |
| Local return distance `D` | 1.1 to 3.1; typically 2.1 |
| Solid ejection distance `S` | 1 cell |
| Excess divergence `lambda` | 0.5 |
| Excess divergence cap `eta` | 1 |
| Gamma diffusion iterations | 1 to 7 |
| Accurate velocity band | 2 cells in CM11b |
| Multigrid bubble levels `C` | 2 in CM11a |
| Constraint levels `S` | 3 in CM11a |
| Detail blur sigma | `2*dx` in demonstrated example |
| Detail denominator floor `theta` | 0.01 in demonstrated example |

## 20. Algorithm inventory and provenance

| # | Algorithm or primitive | Provenance | Status in CM12 |
|---:|---|---|---|
| 1 | MAC staggered-grid storage | HW65 | Directly used |
| 2 | Four-stage frame split | CM12/MMTD07 | Directly used |
| 3 | Backward characteristic tracing | semi-Lagrangian tradition; LGF11/LAF11 | Directly used |
| 4 | Forward characteristic remainder scatter | LGF11 | Directly used |
| 5 | Trilinear gather/scatter | LGF11/LAF11/CM12 | Directly used |
| 6 | Donor column normalization | LGF11 | Directly used |
| 7 | Cumulative gamma balancing | LAF11 | Directly used |
| 8 | Dimension-split gamma diffusion | LAF11 | Directly used |
| 9 | Three-scatter matrix-free conservative advection | CM12 | Novel direct contribution |
| 10 | Normal-extension PDE for air velocity | CM11b/EMF02 lineage | Directly used |
| 11 | Fast Iterative Method active-list eikonal solve | JRW07 | Direct transitive dependency |
| 12 | GPU tile FIM and reduction | JRW07 | Applicable implementation detail |
| 13 | Hierarchical coarse/fine velocity extrapolation | CM11b | Directly used |
| 14 | Godunov-style normal-flow sharpening increment | MMTD07 | Directly used |
| 15 | Sharpen jump limiter | MMTD07 | Directly used |
| 16 | Local increment limiter | CM12 | Novel direct contribution |
| 17 | Gradient-field tracing with forward Euler | CM12 | Novel direct contribution |
| 18 | Solid-aware trilinear mass return | CM12 | Novel direct contribution |
| 19 | Face-fraction sharpening flux | CM12 | Novel direct contribution |
| 20 | Excess-density ejection from solids | CM12 | Novel direct contribution |
| 21 | Effective occupancy `rho/V` | CM12 | Novel direct contribution |
| 22 | Approximate density-derived liquid phi | CM12 | Directly used |
| 23 | Ghost-fluid free-surface pressure | ENGF03/CM11a | Direct transitive dependency |
| 24 | Variational cut-cell pressure projection | BBB07 | Direct transitive dependency |
| 25 | Separating-wall complementarity | BBB07/CM11a | Direct transitive dependency |
| 26 | Projected red-black Gauss-Seidel | CM11a | Directly used |
| 27 | Bubble-aware multigrid coarsening | CM11a | Directly used |
| 28 | Constraint-aware V-cycle | CM11a | Directly used |
| 29 | Constraint-aware full cycle | CM11a | Directly used |
| 30 | Capped excess-density divergence feedback | CM12/MMTD07 | Directly used |
| 31 | Velocity advection | Unspecified by CM12 | Specification gap |
| 32 | Forward-Euler external force integration | Standard; implied | Directly used |
| 33 | Rendering detail discriminator and amplification | CM12 | Novel direct contribution |
| 34 | Separable Gaussian blur | Standard; implied | Directly used for optional rendering |
| 35 | Marching cubes | LC87 | Directly used for rendering/evaluation |
| 36 | Oriented mesh volume integration | Standard; implied | Directly used for evaluation, formula unspecified |
| 37 | Global mass integration | CM12 | Directly used for evaluation |

## 21. Algorithms in dependencies that are explicitly excluded

To prevent dependency creep, the following algorithms appear in cited papers but are not part of CM12's published execution path:

- Tall-cell compression, tall-cell remeshing, level-set reinitialization, tall-cell Laplacian, tall-cell multigrid, rigid-body two-way approximation, particle thickening, and spray/foam generation from CM11b.
- General travel-time/geoscience uses of FIM from JRW07.
- Conservative momentum, vorticity-confinement correction, and energy correction from LAF11, unless selected to fill CM12's velocity-advection gap.
- Compressible-flow extensions and quadratic interpolation from LGF11.
- Mean-curvature flow, general foliation processing, WENO-5 density fluxes, and narrow-band reinjection variants from MMTD07.
- Two-way fluid-rigid-body coupling and PATH/QP solve from BBB07.
- Particle level set correction and surface-tension tests from ENGF03.
- PLS comparison implementation, anti-diffusion future work, and connected-component global sharpening correction mentioned by CM12.

## 22. Recommended implementation acceptance gate

Do not call an implementation conformant until all of these are true:

1. The algorithm profile documents the velocity-advection choice and every specification gap.
2. Closed-domain advection and sharpening conserve mass within a predeclared floating-point tolerance.
3. Every failed or fully masked scatter uses a conservative fallback and increments diagnostics.
4. No characteristic crosses a solid in swept tests.
5. Effective occupancy handles `V=0` and tiny `V` without division faults.
6. Pressure satisfies both divergence and complementarity tolerances.
7. Multigrid demonstrates mesh-independent or acceptably bounded convergence on representative cut-cell scenes.
8. Hydrostatic and separation tests pass on rotated geometry.
9. Raw mass, raw 0.5 volume, and post-processed render volume are reported separately.
10. The optional post-processing field is never fed back into simulation.
11. GPU and deterministic CPU reference results agree within declared field and invariant tolerances.
12. All parameter values and algorithm variants are serialized with the simulation.

## 23. Source notes and primary links

The local source files used as the starting corpus are:

- `docs/papers/massConservingLiquids.txt` and PDF.
- `docs/papers/tallCells.txt` and PDF.
- `docs/papers/A_fast_eikonal_equation_solver_for_parallel_system.txt`.
- `docs/papers/A_Multigrid_Fluid_Pressure_Solver_Handling_Separat.txt`.

The added dependency corpus is:

- `docs/papers/LGF11_Unconditionally_Stable_Fully_Conservative_Semi_Lagrangian.pdf` and `.txt`.
- `docs/papers/LAF11_Mass_and_Momentum_Conservation_for_Fluid_Simulation.pdf` and `.txt`.
- `docs/papers/MMTD07_Variational_Eulerian_Geometry_Processing.pdf` and `.txt`.
- `docs/papers/BBB07_Fast_Variational_Solid_Fluid_Coupling.pdf` and `.txt`.
- `docs/papers/ENGF03_PLS_Second_Order_Pressure_Boundary.pdf` and `.txt`.

Primary copies consulted for missing operational dependencies:

- LGF11: [author-hosted PDF](https://www.ulfhedinn.net/papers/stanford2010-01.pdf).
- LAF11: [Stanford/PhysBAM author PDF](https://physbam.stanford.edu/papers/stanford2011-03.pdf).
- MMTD07: [Caltech Applied Geometry Lab PDF](https://geometry.caltech.edu/pubs/MMTD07.pdf).
- BBB07: [UBC author PDF](https://www.cs.ubc.ca/~rbridson/docs/batty-siggraph2007-variationalcoupling.pdf).
- ENGF03: [Stanford/PhysBAM author PDF](https://physbam.stanford.edu/papers/stanford2003-03.pdf).

Bibliographic keys follow CM12 where possible. OCR damaged several author accents and equation glyphs in the TXT files; equations in this reference were cross-checked against the supplied PDFs or primary-author PDFs. Where a dependency states only a high-level adaptation—most notably FIM-based velocity extension—the reconstruction is labeled rather than presented as verbatim source pseudocode.

## 24. Final completeness conclusion

The executable CM12 stack is now covered end to end: grid state, time-step order, near and far velocity extrapolation, all three conservative-advection generations, gamma diffusion, sharpening and local return, cut-solid density repair, effective liquid classification, variational/ghost-fluid/separating-boundary pressure projection, projected multigrid, excess-density feedback, sub-grid detail enhancement, marching cubes, and validation.

The only material algorithmic hole left by the paper itself is the exact velocity-advection scheme. The five nonstandard missing dependencies have now been added locally. The collection still lacks the original HW65 and LC87 papers; they are standard but remain direct provenance dependencies.
