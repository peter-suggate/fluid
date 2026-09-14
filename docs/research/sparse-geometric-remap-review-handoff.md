# Sparse geometric remap: mathematical and algorithmic review handoff

## Summary

We have built a sparse, adaptive geometric method for transporting liquid volume. It traces shared cell boundaries backward through a divergence-free velocity field, corrects their swept areas to match pressure-derived fluxes, and gathers liquid from the previous frame's PLIC geometry. Conservation and geometry checks reject invalid steps without falling back to baseline transport.

The remaining challenge is accuracy and speed across sustained impacts and topology changes. The checks are strong fail-closed diagnostics, but they are not yet a complete proof of receiver coverage, global injectivity, or accumulated error. Review the actual implementation and current artifact manifests referenced here. Do not use an older design or checkpoint document as a description of current behavior.

## Current scope

The committing method is opt-in through `CellwiseRemap`; native `WorldOptions` still defaults to baseline transport. The reviewed scope is 2D, static full-capacity fluid cells against static walls, with zero source rates and no moving capacity. A committing remap rejects non-2D graphs, active material sources, and moving solid capacity.

Both native regressions use the paper timestep `Δt = 1/30 s`, 256 primary pressure iterations, relative tolerance `1e-6`, and requested `traceSegments = 1`, `edgeSamples = 1`, `BandProjection` closure.

- `cm12-figure-7` is the 2D centre slice of a radius-1 m liquid sphere centred at `(3.2, 4.5) m` in a sealed 6.4 m tank, with `h = 0.05 m` and gravity `-10 m/s²`.
- `coarse-first-pool-impact-half` is a 3.2 by 2.4 m open-top tank, one-third filled, with a radius-0.5 m liquid sphere centred at `(0, 1.825) m`, `h = 0.05 m`, and gravity `-9.80665 m/s²`.

The sparse solver uses strict 2:1-graded cell widths 1, 2, 4, and 8 in fine-cell units. A width-4 cell in this runtime is a `4 × 4` aggregate, not a `4³` cell. This handoff makes no validated 3D geometric-remap claim.

For geometric remap only, sizing is invariant under uniform translation. Strain, resolved velocity variation, curvature, and thin features can raise the rung, while absolute translation still expands swept support and activates receiver pages. Baseline and probe modes retain their earlier sizing and support policy. There is no coarse-cell quota or dense-grid fallback.

## Mathematical construction

Let `q_f` be the signed physical volume rate on subface `f`, `s_if ∈ {-1,+1}` its incidence sign for cell `i`, `C_i` the capacity, and `D_i` the discrete divergence:

```text
D_i = Σ_f s_if q_f
Δt |D_i| / C_i ≤ 2 ε32
```

The primary pressure projection owns pressure-cell and wall fluxes. A private f64 cleanup may remove only residuals justified by f32 evaluation of the predicted-velocity and pressure-correction operands. The strict final normalized-divergence target above does not change. A transient static-wall contact solve may promote predicted swept support for the primary solve only; physical pressure membership is restored before projected-support transfer so the impulse is not replayed.

The compatible 2D rates are integrated into a scalar streamfunction with this orientation:

```text
q_vertical   = ψ_top  − ψ_bottom
q_horizontal = ψ_left − ψ_right
u = (∂y ψ, −∂x ψ)
```

The global scalar interpolant is C1, so its curl velocity is C0 across represented cell and sparse-support boundaries. Pressure-owned and `ClosedWorld` rates are hard constraints. Harmonic continuation fills remaining scalar support without feeding remote private rates back into the accepted receiver-band problem. Forward-error accounting follows streamfunction construction paths and endpoint subtraction; it must not become a fixed tolerance.

Shared physical face points are traced backward with RK4. Effective temporal segmentation rises from the requested lower bound using measured Courant and velocity-gradient limits. Every committing unit edge has an actually traced interior point. The swept strip contains the source edge, its backward-traced image, and both endpoint pathlines. Its oriented area is

```text
I = ½ ∫ (x dy − y dx)
```

so endpoint-path area is part of the face target.

The traced endpoints remain fixed. Interior points on each shared edge chain are corrected by

```text
x*_j = x_j + δ 4 t_j (1 − t_j) n
```

where `t_j` is the actual, possibly irregular, source-edge parameter and `n` is the shared correction normal. Swept area is affine in `δ`. Neighbours consume the same canonical chain in reverse, including across coarse/fine seams. Adaptive dyadic refinement inserts canonical shared points when traced or corrected geometry has a material-relevant crossing; exhaustion fails closed.

For receiver cell `i`, backward tracing builds preimage polygon `P_i`, with area contract

```text
A(P_i) = C_i − Δt D_i
```

Previous-frame donor liquid `L_j` is the donor cell clipped by its PLIC plane. One gather computes

```text
V_i^(n+1) = Σ_j |P_i ∩ L_j|
```

before material mutation. Total and per-donor gathered volume, density bounds, area identity, continuity, material-relevant folds, and support use all pass before the single commit. Rejection keeps material unchanged and publishes a diagnostic receipt.

## Robustness questions

1. **Pressure cleanup.** The first half-pool frame failed because the bound used only a small post-cancellation rate. It now includes the large predicted-velocity and pressure-correction operands. A direct test accepts and cleans a `2e-5` residual produced by `±50` cancellation operands and rejects an injected `+0.1`. Verify that the bound cannot mask a materially unconverged pressure solve.
2. **Streamfunction arithmetic.** A later exact face exposed endpoint cancellation accumulated along the streamfunction forest. Verify that carried path error and unit-subface assembly cover every arithmetic route without becoming an arbitrary tolerance.
3. **Sparse coverage.** A resolution-1 partial cell spans both sides of its brick, but the old low-side `if` / high-side `else if` emitted only one halo direction. A missing `+y` page caused a donor gap although represented receiver polygons were locally valid. Both-side cells now emit every touched direction, and geometric remap consumes this halo during projected-support and missing-page allocation. Seek counterexamples to support completeness under the continuous velocity field.
4. **Global geometry.** Local simple-polygon checks previously missed small proper crossings and overlapping receivers. Review robust orientation signs, convex-hull-versus-PLIC material relevance, shared-chain identity, and whether accepted polygons form a partition.
5. **Transactions.** Check topology-transfer conservation, current rather than stale pressure membership, one-shot contact pressure, and persistence of transport faults through later resolution planning.

Open proof obligations include global injectivity of an accepted map, gap-free and overlap-free closure at hanging interfaces and walls, error bounds derived from the actual f32/f64 operation graph, guaranteed adaptive termination with either a certificate or a fault, and Galilean-invariant sizing that retains coarse bulk cells without under-resolving impact strain.

## Verified checkpoint and limits

The exact half-pool regression passes 10 frames. Every frame commits without a fault; liquid measure changes from `1337` to `1336.9999823626822`, relative drift `-1.3191711e-8`, and topology generation reaches 14.

The exact Figure 7 regression passes 30 strict frames. Liquid measure ends at `1252.0000058637852`, relative drift `4.6835345e-9`, with worst one-step relative drift `1.68343e-9`. At least eight width-4 wet liquid cells remain in every frame, minimum width-4 liquid measure is `80.3186`, and the frame-20 pre-contact covariance aspect ratio is `1.00463`.

The resolution suite passes 13 tests. Pressure-cleanup and streamfunction reconciliation tests pass both roundoff and large-error lanes. The unchanged production World golden passes; no golden was updated. These results do **not** establish a current 90-frame Figure 7 result. Sustained post-impact behavior beyond frame 30 remains unverified.

The built production manifest has source snapshot
`44e311bcf284b9528fc8415ef2573741cdb18bf3a7eaf7285865ad2734dc5174`.
Its Wasm hashes are:

- scalar: `fb3b310a009296ceb407030b564ce082116b03ac84cbc75dc1eea757b32c0535`;
- SIMD: `5dacc9a2250d06f4fbea45ecff9f2756b1eb1ac789c21b647415ec1e9119504e`;
- threaded: `880de63a2f410063a90017d4f5378e5ffd22e1a3e4d16251cdd2238e4ca9d7a1`.

The manifest identifies the built and served checkpoint, not later working-tree edits. Compare it before reproducing results.

## Performance state

Performance optimization is paused behind correctness. In the current Figure 7 30-frame receipt, the requested trace and edge values remain 1, while maxima reach 102 effective RK segments, 512 effective edge samples, 4,808,280 RK evaluations in one frame, 12,762 traces, 17,015 chain points, 7,241 inserted edge points, nine refinement passes, and 43 points on one subface. Maximum post-closure Courant is 11.255. A wrapper-level native run took about 11.27 seconds for 30 frames on the development host, but that includes Node/process startup and is not a UI frame-rate measurement.

A prior profile attributed roughly 61% of runtime to harmonic continuation, but it used a different experimental source and binary. Treat it only as a hypothesis for a fresh profile. A separate `c4fc` redundancy experiment reported large early work reductions but was reverted after a donor-conservation rejection. Neither is current performance evidence. Profile the manifest checkpoint again and retain every continuity, fold, support, and per-donor gate.

## Current implementation and native validation

Primary implementation:

- [world transaction and mode routing](../../rust/crates/fluid-core/src/world.rs)
- [pressure cleanup and receiver-band projection](../../rust/crates/fluid-core/src/band_projection.rs)
- [streamfunction and compatible velocity](../../rust/crates/fluid-core/src/numerics.rs)
- [shared-edge tracing, correction, geometry, and gather](../../rust/crates/fluid-core/src/adaptive_remap.rs)
- [sparse support and resolution policy](../../rust/crates/fluid-core/src/resolution.rs)
- [focused geometric-remap tests](../../rust/crates/fluid-core/tests/cellwise_remap_lab.rs)

Exact native harnesses:

- [Figure 7](../../tools/verify-cm12-figure-7-native.ts)
- [coarse-first half-pool](../../tools/verify-coarse-first-pool-impact-half-native.ts)

Built artifact manifests:

- [scalar](../../public/wasm/fluid-wasm/scalar/build-info.json)
- [SIMD](../../public/wasm/fluid-wasm/simd/build-info.json)
- [threaded](../../public/wasm/fluid-wasm/threaded/build-info.json)

Run without UI or Dawn:

```bash
cargo build --manifest-path rust/Cargo.toml --release -p fluid-core --example verify_world

npx tsx tools/verify-coarse-first-pool-impact-half-native.ts \
  --frames=10 \
  --binary=/Users/petersuggate/code/me/fluid/rust/target/release/examples/verify_world

npx tsx tools/verify-cm12-figure-7-native.ts \
  --frames=30 \
  --binary=/Users/petersuggate/code/me/fluid/rust/target/release/examples/verify_world

cargo test --manifest-path rust/Cargo.toml -p fluid-core resolution::tests::
cargo test --manifest-path rust/Cargo.toml -p fluid-core --test cellwise_remap_lab \
  pressure_rate_cleanup_uses_projection_operands_without_admitting_physical_residuals
cargo test --manifest-path rust/Cargo.toml -p fluid-core --test world_golden
```

Use the golden only as an unchanged compatibility check; do not regenerate or rebless it. Preserve sparse 2:1 topology and the one-gather geometric path rather than routing failures through baseline or a dense-grid substitute.

## Requested review output

Return a prioritized list of concrete bugs, unproved assumptions, missing validation, and performance opportunities. For each item:

1. cite the current source location;
2. state the invariant or mathematical claim at risk;
3. give a counterexample or minimal native experiment where possible;
4. distinguish a correctness defect from a conservative fail-closed limit;
5. for performance work, identify the protected certificate and report work counters and wall time on the same source manifest.

Prioritize mass loss, overlap, unbounded density, stale pressure ownership, and non-injective maps. Then cover missing proofs and tests, followed by optimizations that preserve the certificates.

## Review 2026-09-14

Source: commit 7f6a2cdc plus the uncommitted docs. Reviewed by reading `adaptive_remap.rs`, `band_projection.rs`, `numerics.rs`, `world.rs`, `tests/cellwise_remap_lab.rs`, one fresh native profile of the 30-frame Figure 7 lane (`sample`, 8,996 samples, native binary on stdin JSON, no Node), and two small native experiments for the geometry findings. Baseline lane run on the same source for comparison. No source or golden touched.

### Verdict

The large step is real: one gather per frame at post-closure Courant 11.3 with zero material substeps and drift 4.7e-9 over 30 frames. The fast part is not on track, and the reasons are structural rather than tuning:

| lane, cm12-figure-7, 30 frames | wall (native, startup incl.) | microsteps | cells | drift |
|---|---|---|---|---|
| CellwiseRemap | 11.27 s | 0 | ~1,400 | 4.7e-9 |
| baseline substepped transport | 2.87 s | 334 total (1 rising to 51) | ~2,500 | 1.4e-7 |

Wall-time split of the remap lane (one 11.27 s run):

| bucket | share | note |
|---|---|---|
| harmonic continuation `numerics.rs:446-571` (2 calls/frame) | 46.2 % | ~14,200 of 16,641 lattice vertices are dry air; `supportExtrapolationSamples = 0` every frame, so nothing ever samples the field it produces |
| tracing `trace_lattice_points` | 27.4 % | of which point location + solid clipping 20.3 %, velocity sampling 5.7 % |
| streamfunction extension CG + assembly | 5.2 % | |
| topology transition + resolution | 5.3 % | |
| polygon assembly + fold checks | 3.3 % | |
| pressure PCG (2 solves/frame) | 1.2 % | |
| gather / clipping | 0.8 % | |
| f64 cleanup | 0.4 % | |

Two facts decide the performance question:

1. **Half the frame is a solve over empty air that is never consumed.** The harmonic fill runs over every non-finite vertex of the full domain lattice, twice per frame, to 1e-11, whether Courant is 0.2 or 11. It is fixed overhead unrelated to how much liquid exists.
2. **RK segmentation is one global scalar, and it chases the tracing field's own discontinuity.** `adaptive_remap.rs:545-570`: `segments = max(ceil(4·C), ceil(8·dt·max|∇v|))`, applied to all ~12k traces. Receipts satisfy `rkEvaluations = traces × segments × 4` exactly. The 102 on frame 25 is the strain term (max|∇v| = 380 at Courant 11: velocity changes by its whole magnitude across one cell). The gradient maximum on frames 25, 28, 30 sits at the free surface where represented support ends and harmonic continuation begins (cells 782, 877, 1267; none at a 2:1 seam, only frame 28 also wall-adjacent). Travel-only segmentation would cut RK work 23 % over 30 frames; the real fix is the field, not the segment count.

Work classification: fixed and independent of Courant are the two harmonic solves, the 4×-over-request base edge sampling on every subface in the graph (traces ≈ 3.5 × subfaces ≈ 10 × receivers, dry ones included, `:569,:609`), the diagnostic 5th velocity sample per RK segment (`:1163-1180`, 25 % tax on tracing), and the 36-sample-per-unit-cell gradient census (`:965-995`). Adaptive and unbounded are `trace_segments` (1 to 102 here, hard error at 256) and edge refinement (fired on frame 28 only: 9 passes, 4 to 512 samples, clips 7k to 38.7k).

The design intent "one trace, one correction, one gather, work independent of Courant" does not hold: 39.5k RK evaluations at C = 0.22 vs 4.81M at C = 11.3 on a 2× cell-count change. What is inherent is that an injective discrete flow map of a piecewise-RT0 field needs tracing resolution proportional to travel; that is cheap only if a velocity evaluation is cheap, and today it costs ~110 ns because point location dominates.

### Prioritized findings

Severity order: mass/overlap/injectivity first, then certificate validity, then work.

**P1. Chain correction δ is unbounded (`adaptive_remap.rs:1503-1527`). Correctness.** The singular guard is `|coefficient| ≤ 1e-12` but the coefficient is area-dimensioned (~2L/3 healthy). It vanishes when the traced chain backtracks along its own chord, which does not need a self-crossing. Native experiment (unit face, 4 intervals, target area 0.02, chain with third point at `b`): `b−b* = 1e-2` gives δ = 10 cells with `chain_simple = true`; `1e-6` gives δ = 1e5; `1e-8` gives δ = 1e7. `max_correction_delta_over_h` (`:218`) is recorded and gated nowhere (only printed by `examples/verify_world.rs:510`). Fix: gate on relative coefficient `|coef| < c·L²` and `|δ| ≤ ½h`; route failure to a receipt fault.

**P2. Crossing predicates fail open (`:2938-2968`, `:2287`, `:2317-2328`). Certificate weakness.** `segments_cross` returns "no crossing" when any of the four orientations is inside its own error bound. `point_in_polygon` and `segments_intersect_inclusive` compare an uncompensated area-dimensioned orient against absolute 1e-12; f64 roundoff of orient at coordinate ~128 is ~3.6e-12, so the collinearity test is below its noise floor. Undecidable must escalate (refine or exact predicate), not resolve as safe.

**P3. The live BandProjection closure never runs `band_projection.rs`'s projection. Validation aimed at dead code.** `adaptive_remap.rs:457` and `:520` call `streamfunction_extension_rates_2d`; `project_receiver_band_rates_2d` (`band_projection.rs:320`) is reached only from `tests/cellwise_remap_lab.rs:671,704,724`. Consequences on the live path: no enclosed-component existence check (infeasible circulation is least-squares absorbed, not rejected); `max_courant_after_closure` is measured (`:2501`) but only consumed for segment count, never as a gate, so Courant 11.3 is an unbounded post-closure amplification nothing rejects; non-pressure non-closed rates are replaced wholesale including remote faces (`numerics.rs:1630-1656`).

**P4. Cleanup bound scales with operand magnitude (`band_projection.rs:147-182`). Correctness.** On the shipped fixture the bound is ~7.2e-3 in rate, normalized 2.4e-4, 1000× the 2ε32 target; in a 100 cell/s impact it reaches 3200×. Counterexample: in `tests/cellwise_remap_lab.rs:186` replace `2.0e-5` with `5.0e-3`; the cleanup accepts and removes it. `PressureReceipt.converged` is stored at `world.rs:732,845` and read nowhere, so an unconverged PCG proceeds into a bound proportional to ‖q‖. Also the bound is rebuilt against restored physical membership (`numerics.rs:2791-2821`, `world.rs:773-782`) while the projection ran on promoted swept-wall membership (`world.rs:745`), so the operand graph differs from the one evaluated on every face the contact solve touched. Residue is vented onto one face (`band_projection.rs:262-282`) whose far side is not in `active` and is not checked at `:286-295`.

**P5. No partition oracle; donor conservation is not tight (`gather_material:2757-2783`). Missing validation.** Per-donor `Σ_i |P_i ∩ L_j| = |L_j|` is blind on empty donors, cancels an overlap of area a against a gap of area a in the same donor, and uses `|signed_area|` per piece. The area identity and balance cannot help: correction forces `A(P_i)` by construction and shared chains telescope `Σ A(P_i)` exactly regardless of crossings (native experiment: sum = 4.000000000000 with a genuine T×U crossing). Cheapest fix: a coverage-count assertion in the lab (every sample point in exactly one accepted polygon). Partial reassurance: 8M random 4-chain corner configurations found no crossing between corner-only-sharing chains with all four polygons simple, so the per-polygon test appears to cover the case empirically; unproved.

**P6. Rejection is not atomic. Correctness (state).** `adaptive_remap.rs:949-953` writes density/gamma then calls `commit(dt, fields)?`. Seven paths return `Err` instead of receipt+fault (`:507, :559, :1095, :1347, :1577, :1742, :2731`); `world.rs:918/943` propagate with `?`, leaving the receipt `None` (cleared at `:890`), pressure and velocity advanced, frame not advanced, no diagnostic. On a cleanup rejection (`:404-418`) the frame continues with `microsteps = 0`: gravity and projection are committed to `face_velocity`, time advances (`world.rs:1024-1025`), no dt backoff, so repeated rejections accumulate velocity against frozen material. `world.rs:665` clears the fault every advance and `plan_resolution` (`:996`) runs before restore.

**P7. Streamfunction field has artificial discontinuities exactly where the strain term fires. Correctness of the tracing field, root cause of the segment explosion.** Constraint-connected components are seeded independently at ψ = 0 (`numerics.rs:967-971`, `:1472-1476`), so two sparse islands get arbitrary relative offsets and the harmonic fill turns the offset into a spurious velocity with no bound. The fill imposes homogeneous Neumann at the domain boundary (`:466-474`), which pins tangential velocity and leaves normal velocity free at a wall: the opposite of no-penetration. `active_lattice.fill(true)` at `:1062` erases the support-fallback diagnostic (`:820-832`). Tolerances at `:515,:561,:1570,:1611` are absolute (scale floored at 1), not relative. The endpoint derivatives (`:1129-1150`) are unconstrained and can overshoot the face mean velocity by ≥1.5×. Face flux consistency holds in the integrated sense only. The 2:1 seam assembly and C1 bicubic patch are sound.

**P8. Solve inventory (answer to the "how many global solves" question).** Per frame: primary PCG (`world.rs:763`, cap 256, `converged` ignored), a second full PCG after in-frame topology transition (`:861`), streamfunction extension CG (`numerics.rs:1572-1615`, cap 8192) once per stabilization pass with the pass loop `for pass in 1..=cells+1` (`adaptive_remap.rs:394`, worst case N+1 global solves) plus once globalized (`:520`), and the harmonic fill twice (cap 4096). Each extension pass clones the whole `PotentialUnion` per pressure-owned face (`numerics.rs:1437`), O(faces × vertices). The embedding path (`world.rs:722-736`) never applies the swept-wall contact pressure.

**P9. Refinement doubling and per-pass cost (`:1697-1711`, `:632-712`, `:1649-1673`). Work boundedness.** `corrected_only_area_intervals` marks every interval of every incident face of a corrected-only folded cell, so those faces double each pass (4 to 512 in 9 passes). Each pass reruns `build_geometry` and two full `receiver_mask` sweeps over all cells, and `crossing_chain_intervals` is O(B²) in boundary segments (~4M predicate tests per refine cell per pass at 512). Termination is guaranteed (32 passes, 4096 intervals/subface) and exhaustion rejects because `refine_cells` and the gate at `:864` share a predicate.

**P10. Receipt counters misreport (`:713-731`, `:618-628`).** `edge_samples` is overwritten with max ⌈1/shortest chord⌉; `adaptive_edge_points_inserted` is seeded from the pre-doubling baseline so band faces count as inserted; the quoted 7,241 is inflated. Receipts carry no `maxCourantCell`.

**P11. Small one-signed drifts (`gather_material:2706`, `:2720`).** Donors with `0 < volume ≤ 1e-12` are discarded; near-full donors round up to capacity. Both sit at the per-donor bound, invisible to the gate. Probe mode breaks out of the refinement loop (`:682`), so the seam test at `cellwise_remap_lab.rs:833` and every `assert_geometry_certificate` lane exercise neither band doubling nor refinement.

### Recommended order

1. Restrict the streamfunction and harmonic domain to represented support plus a backward-travel halo, or drop the fill: it is never sampled. Protected certificate: `streamfunction_max_integrated_flux_residual` and the per-face flux check; counters: harmonic unknowns, wall per frame. Expected: about half the frame.
2. Fix P7 (component seeding, wall BC, restore the support-fallback diagnostic), then remove or localize the strain term. Protected: fold count on Figure 7 frames 25 and 28; counters: `effectiveTraceSegments`, `rkEvaluations`, `tracingMaxVelocityGradient` and its point.
3. Trace only subfaces touching the receiver band and drop the diagnostic fifth sample; then make point location a per-cell table so an evaluation is a table lookup plus a bicubic patch. Protected: gather drift; counters: traces, RK evals, ns per eval.
4. Certificates before any 90-frame claim: P1 δ gate, P2 fail-closed predicates, P5 coverage oracle, P4 tighten or re-test the cleanup bound and wire `converged`, P3 either route the live closure through `project_receiver_band_rates_2d` or port its existence and amplification gates to the streamfunction path, P6 atomic rejection.
5. Only then the 90-frame Figure 7 run and the half-pool sustained run.

## Performance plan 2026-09-14

Goal restated: one gather per frame at any Courant, with per-frame work that is O(receiver band) times a small constant, the same shape as adaptive-mass (trace, scatter, gather, pressure). Lane: cm12-figure-7 30 frames and coarse-first half-pool 10 frames, native binary, same source manifest for every arm. Protected on every step: final drift, `pre_correction_receiver_band_folds`, `corrected_receiver_band_folds`, gather donor residual. Nothing below touches the certificate bugs listed above; they are sequenced after the work shape is right.

Where the 0.35 s/frame goes today and what the target shape is:

| stage | today | cause | target |
|---|---|---|---|
| harmonic fill | 0.174 s | full-domain solve over dry air, never sampled | 0 |
| tracing | 0.103 s | 12k traces × global `max(4C, 8·dt·max∇v)` segments × 5 samples, 110 ns/sample from point location | band faces only, per-point segments, ~20 ns/sample |
| streamfunction extension | 0.020 s | CG plus O(faces × vertices) union clones | O(band) |
| geometry + gather | 0.015 s | fine | unchanged |
| pressure | 0.005 s | fine | unchanged |

### Steps, each measured before the next starts

**S0. Stage timers in the receipt.** Wall per stage (field build, trace, geometry, gather, pressure) published with the counters, so every arm below reports its own split instead of a sampler. Half a day. No behaviour change.

**S1. Delete the harmonic fill.** `fill_streamfunction_holes_harmonic` (`numerics.rs:446`, called at `:1061` and `:1625`) and the `active_lattice.fill(true)` at `:1062`. ψ exists only on vertices with an incident rate-bearing subface, which is every vertex of a represented cell. Outside support a sample returns the fallback flag; a trace that leaves support is a receipt fault (`supportExtrapolationSamples` is already 0 on all 30 frames, so this fails nothing today). Expected: −46 % wall. Go/no-go: drift and fold counts unchanged on both lanes.

**S2. Shrink the trace set to what the gather consumes.** Trace only subfaces with at least one receiver-band cell (`adaptive_remap.rs:600-615`); remove the ×2 sample doubling at `:608-612` (2 samples per unit edge is the floor, adaptive refinement handles the rest); remove the diagnostic fifth sample per RK segment (`:1163-1180`); replace the 36-samples-per-cell gradient census (`:965-995`) with the gradient at the RK stage points already being evaluated. Expected: traces 12k → ~4-5k, samples per segment 5 → 4. Go/no-go: same certificates; `receiver_band` count unchanged.

**S3. Per-point travel-adaptive tracing; drop the strain term.** Replace the global `courant_trace_segments` (`:545-570`) with per-point step control: each RK4 step advances at most a quarter cell of that point's own travel, so a point in still water takes one step and a point in the jet takes 4C steps. Work becomes Σ_points local Courant instead of N × max Courant. Prerequisite check after S1: `tracingMaxVelocityGradient` on frames 25, 28, 30 must have dropped from 380/253/96 to the liquid's own strain; if it has not, the streamfunction still has a seam inside support and that is fixed first (component seeding `numerics.rs:967-971`, wall condition `:466-474`). Then remove the strain term. Expected on frame 25: RK evaluations 4.8M → under 0.5M. Go/no-go: fold counts on frames 25 and 28 do not rise.

**S4. Cheap velocity evaluation.** Today a sample walks `projected_stage → clip_segment → inside_solid → owner_at`. Replace with a per-fine-cell owner image (one u32 per unit cell, built once per topology, the same structure the advance lab already uses for point location) and precomputed bicubic patch coefficients per unit cell (16 f64) so an evaluation is one lookup plus one polynomial. Solid clipping is consulted only when the owner changes between stages. Expected: 110 ns → ~20 ns per sample. Go/no-go: bit-identical traces are not required; drift and folds are.

**S5. Extension solve O(band).** The stabilization loop at `adaptive_remap.rs:394` may run N+1 global CGs, and each pass clones the `PotentialUnion` per pressure-owned face (`numerics.rs:1437`). Make the union-find in-place with path compression, restrict the CG to band vertices with support vertices as Dirichlet data, and cap the stabilization loop at a small constant with a fault beyond it. Expected: 0.020 s → a few ms.

**S6. Measure against the goal.** Same two lanes, three arms: baseline substepped, remap before S1, remap after S5. Report wall per frame excluding startup, RK evaluations, traces, and the certificates. Target for Figure 7 frames 20-30, where baseline pays 40-51 microsteps: remap under baseline. Target for the pre-impact frames: remap within 1.5× of baseline (baseline pays one microstep there and the remap's fixed costs are the whole story).

Only after S6 is green: the 90-frame Figure 7 run and the certificate fixes above.

### Expected budget after S1-S5 (Figure 7 frame 25, today 0.35 s)

| stage | estimate |
|---|---|
| pressure (2 solves) | 5 ms |
| band projection + streamfunction on ~1,300 band cells | 5-10 ms |
| tracing: ~5k points, mean ~10 segments at impact, 4 samples, 20 ns | 4-8 ms |
| geometry, folds, gather | 15 ms |
| total | 30-40 ms |

Baseline on the same frame is roughly 70 ms and rising with its microstep count; adaptive-mass-shaped work is the tracing and gather lines. If S1-S3 alone do not bring frame 25 under 0.1 s, stop and reassess before S4-S5, because that would mean the work is somewhere the profile did not show.

### What carries to 3D

S1 (no global fill), S2 (band-only traces), S3 (per-point tracing) and S4 (owner image plus per-cell patch) are all local per-point or per-brick work and map directly onto the sparse CM12 lattice. S5 is 2D-specific; the 3D analogue is the band projection port already planned in the lab plan.
