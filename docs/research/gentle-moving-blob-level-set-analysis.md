# Gentle moving blob: level-set-volume analysis

## Decision

The current level-set-volume transport is a **no-go for sustained gentle translation**. It conserves the authoritative liquid measure to `1.95e-8` relative drift over 6 s, but the moving blob develops an internal hole and loses 56.0% of its UI-visible RDF area. Its authoritative centroid lags the exact translation by 91.2 mm, while the published surface centroid lags by 157.7 mm. The stationary level-set-volume arm and moving cellwise baseline remain stable, so the scene authoring, pressure solve, and RDF publication do not explain the failure on their own.

The strongest direct explanation for the hole is a mismatch between conserved volume and drawable geometry. At the last moving frame, 30.8% of the conserved measure is stored above cell capacity, 18.7% more is excluded by the phi/sub-cell PLIC eligibility rule, and RDF reconstruction under-represents the remaining PLIC geometry by another 5.6%. This is an area-equivalent visibility budget, not deleted mass.

[Final contour comparison (PNG)](../../artifacts/level-set-volume/gentle-moving-blob-analysis.png) · [scalable version (SVG)](../../artifacts/level-set-volume/gentle-moving-blob-analysis.svg)

## Method

`tools/analyze-gentle-moving-blob.ts` advances each arm twice: the release native verifier supplies dynamics and receipt diagnostics, while the current SIMD Wasm build supplies the exact snapshot and RDF segments consumed by the UI. Before combining a frame, the analyzer requires native and Wasm topology generations to match exactly and liquid measure to agree within `1e-5`. All 724 paired snapshots (four arms, frames 0 through 180) passed those checks, and no arm reported a simulation failure.

Each arm uses 180 frames at `dt=1/30 s`, one external step per frame, 256 pressure iterations, and a 50 mm finest cell. The moving speed is 0.08 m/s, giving an exact 0.48 m displacement and a trace Courant number of 0.053333 finest cells per frame. Runs were sequential on an otherwise uncontrolled desktop load; timings are recorded in the artifact but are not used for the numerical conclusion.

The complete per-frame data and initial/final segment arrays are in `artifacts/level-set-volume/gentle-moving-blob-analysis.json`. The Rust source fingerprint in the Wasm build manifest was `4dbed318237c813a678ca8ae7426d735d63241be1150428b9f84cf8651868559`; the SIMD Wasm binary fingerprint was `919001b8877b92aada74f444b180e117139ed62db95d19fd661705d906a8b032`. These identify the source manifest and built Wasm used with the current working-tree scene correction, rather than a source tree reproducible from one Git commit.

## Initial representation

The authored object is a three-dimensional sphere of radius 0.300 m centered at `(-0.25, 0.6, 0)`. The runtime two-dimensional authority selects source z cell 8, whose physical center is `z=+0.025 m`, although `SliceFrame.center_z` is published as zero. The exact sphere section at that cell center therefore has radius 0.2989565 m and area 0.2807798 m². Initial VOF values come from eight voxel samples at offsets `±0.4h`, rather than an analytic disk fraction.

| Initial representation | Area (m²) | Difference from exact selected section | Other error |
|---|---:|---:|---:|
| Exact selected sphere section | 0.280780 | — | radius 0.298957 m |
| Eight-sample VOF seed | 0.277500 | -1.17% | centroid exact by symmetry |
| Published RDF | 0.282715 | +0.69% | 5.84 mm radial RMS; circularity 0.9884 |

This initial 5.84 mm RDF error is reported separately from the 144.3 mm final moving error. The depth correction to 0.8 m removes z-boundary clipping; it does not make the selected voxel-center slice identical to the authored z=0 center plane.

## Six-second comparison

| Arm | Relative mass drift | Density centroid error | RDF centroid error | RDF area (m²) | Circularity | Radial RMS | Density axis ratio | Excess / authoritative area | Topology changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSV moving, natural adaptive | `1.95e-8` | 91.2 mm | 157.7 mm | 0.124494 | 0.3512 | 144.3 mm | 1.203 | 30.8% | 84 |
| LSV stationary | 0 | 0 | 1.17 mm | 0.277863 | 0.9961 | 3.63 mm | 1.000 | 0 | 180 |
| Baseline moving | `1.01e-7` | 0.368 mm | 0.476 mm | 0.276050 | 0.9927 | 5.03 mm | 1.014 | 0 | 17 |
| LSV moving, refinement requested | `2.18e-8` | 28.8 mm | 89.1 mm | 0.109884 | 0.6341 | 116.5 mm | 1.539 | 8.2% | 58 |

The fourth arm is not a fixed uniform-grid control. The refinement request retained width-2 cells and eight wet mixed-resolution seams at the last frame, so it only measures the effect of requesting more fine coverage. It reduces centroid lag and overcapacity, but visible area becomes slightly worse and density elongation grows. A clean frozen-uniform isolation was not available without changing production numerics and was not pursued.

The moving LSV degradation begins well before the large late topology churn:

| Frame (time) | Density centroid error | RDF area (m²) | Circularity | Radial RMS | Excess fine area | Topology generation |
|---|---:|---:|---:|---:|---:|---:|
| 0 (0 s) | 0 | 0.282715 | 0.9884 | 5.84 mm | 0 | 1 |
| 30 (1 s) | 9.77 mm | 0.227335 | 0.6912 | 79.5 mm | 1.139 | 8 |
| 60 (2 s) | 14.9 mm | 0.194270 | 0.4358 | 119.1 mm | 5.078 | 8 |
| 90 (3 s) | 24.7 mm | 0.159464 | 0.5866 | 95.2 mm | 9.822 | 9 |
| 120 (4 s) | 39.8 mm | 0.144259 | 0.3214 | 132.9 mm | 13.818 | 41 |
| 150 (5 s) | 61.6 mm | 0.138015 | 0.3330 | 130.7 mm | 22.736 | 101 |
| 180 (6 s) | 91.2 mm | 0.124494 | 0.3512 | 144.3 mm | 34.153 | 161 |

The ordering rules out excess and repeated topology changes as causes of the first observed area shrink. RDF area falls from 0.282715 to 0.271028 m² on frame 1 while reported excess is still zero. Width-2 cells first appear on frame 2, and the first nonzero excess appears on frame 3. From frames 20 through 60, topology generation stays fixed at 8 while excess grows from 0.486 to 5.078 fine-area units and RDF area continues falling from 0.246697 to 0.194270 m². Motion-dependent resampling therefore starts the measured shrinkage, while excess and later topology churn amplify or redistribute it. The saved aggregate metrics do not locate the exact frame or cell where the visually recognizable hole first opens.

## Why the internal hole appears

At frame 180, all values below use finest-cell area units so they can be compared directly:

| Stage | Area-equivalent measure | Increment hidden from the UI | Share of authoritative measure |
|---|---:|---:|---:|
| Conserved authoritative `V` | 111.0000 | — | 100.0% |
| Clamp `V` to cell capacity | 76.8475 | 34.1525 | 30.8% |
| Phi-eligible PLIC geometry | 56.0573 | 20.7902 | 18.7% |
| Published RDF geometry | 49.7975 | 6.2598 | 5.6% |
| Total absent from RDF | — | 61.2025 | 55.1% |

The first loss of visibility is overcapacity: mass conservation permits some cells to hold more `V` than their geometric capacity, but surface construction can draw at most a full cell. The phase fields can also disagree in the opposite direction. Semi-Lagrangian scalar interpolation diffuses an initially full interior cell to `0 < V/C < 1`; the plane fitter treats every such fraction as a real interface and chooses its offset from volume, even when advected phi still places the cell deeply inside liquid. A phi-derived normal cannot make that volume-imposed interior plane part of the physical outer contour. These artificial PLIC segments can corrupt the rebuilt RDF and therefore the phi source for the next trace. The phi eligibility rule then creates the more direct hole path: a positive, full, or overfull `V` cell whose phi lies beyond the half-width cutoff is assigned zero geometry before the RDF rebuild. A cell can therefore be liquid in authoritative `V` and air in the surface passed forward, reinforcing the mismatch. The RDF fit finally represents less area than the eligible PLICs. Its frame-180 area is 49.7975 fine units, or 0.124494 m², matching the 55.1% invisible budget.

This accounting describes representation loss; it does not mean the solver deleted 55.1% of liquid. Its authoritative total remains 111.000002 fine units.

The evidence is strongest for the final hole mechanism and weaker for its first spatial trigger. The code supports interpolation-created interior PLICs as an initiating mechanism that needs no missing donor, row residual, pressure error, or topology transition, but this run does not prove the identity of the first bad segment. The code and final area budget do show exactly how positive authoritative volume becomes invisible. The aggregate per-frame receipt does not identify which cell first opens the hole, and the stored initial/final contours cannot recover that event after the fact.

## Factor isolation

Repeated moving resampling is the necessary trigger in these arms. The stationary LSV case preserves density exactly and converges to a 3.63 mm RDF radial error despite rebuilding topology generations each frame. The moving Baseline arm also preserves translation and surface shape on an adaptive grid. Motion through the LSV gather/reconciliation path is therefore the differentiator.

The transport operator explains why exact mass is insufficient. Write `Tᵢⱼ` for the coupling from donor cell `j` to receiver cell `i`, and `C` for cell capacity. Exact donor sums, `Σᵢ Tᵢⱼ = Cⱼ`, preserve the total transported volume. Preserving a constant fill field additionally requires receiver sums, `Σⱼ Tᵢⱼ = Cᵢ`. Correct rigid translation also requires the appropriate first moment, for example `Σᵢ xᵢ Tᵢⱼ = Cⱼ(xⱼ + uΔt)` for a constant translation under a donor-local formulation. The current donor normalization establishes the first property. It does not establish the receiver or first-moment properties, and using the same RK2 landing trace for `V` and phi does not supply them. This is consistent with exact total mass alongside spreading, centroid lag, excess, and incompatible phase geometry.

For reference, repeated one-dimensional linear interpolation at Courant 0.053333 predicts an added x variance of 0.02272 m² after 180 frames, comparable to the seeded disk variance of about 0.02247 m². The refinement-request arm develops x/y covariance elongation consistent with this diffusion signature: its final physical variances are approximately 0.0480 and 0.0228 m². The natural-adaptive arm adds less x variance but accumulates much more overcapacity and centroid lag, so interpolation diffusion alone does not explain the hole; capacity balancing, phi eligibility, and RDF reconstruction determine how that spread remains visible.

Pressure and velocity error are not the primary cause. Through frame 120 the moving LSV liquid mean speed remains 0.079999995 m/s with negligible spread, while density centroid error has already reached 39.8 mm and RDF area has fallen to 0.1443 m². At frame 180 its mean is still 0.0799630 m/s with 0.00113 m/s standard deviation. The baseline carries the same velocity essentially exactly and remains round.

For the natural moving arm, topology generation and overcapacity have Pearson correlations of 0.632 and 0.764 with radial error; row residual has correlation 0.210. These are descriptive trends confounded by time and cannot establish that topology changes cause the error. The refinement-request contrast supports the narrower conclusion that refinement policy changes the balance between lag, elongation, excess, and visible-area loss.

No frame reports an invalid phi sample or a zero-weight donor, so the observed run does not support NaN phi propagation or an uncovered donor stencil as the cause. The receipt's global maximum normalized row residual is 0.04855386, showing that at least one receiver marginal remains inexact even though donor normalization preserves total measure. The receipt does not record the residual's cell location, so this maximum cannot establish that the defective row lies in the hole or blob core.

## Design recommendation and alternatives

The recommended next design step is to remove the contradictory mass and phase authority inside the existing reconcile stage, then redesign transport and reconstruction together so capacity and first moment are deliberate constraints. Every positive authoritative volume must have a compatible geometric interpretation, and any excess treatment must preserve the intended moment behavior. This addresses the measured failure as one coherent change rather than tuning its symptoms. The known-good Baseline result should remain the reference and is also a practical hybrid transport option while that design is developed.

One implementation family would retain conservative cell volume as interface authority and use phi for normals, with a capacity- and moment-consistent transport that keeps every authoritative cell geometrically representable. Another would complete the approximate conservative semi-Lagrangian coupling with explicit receiver-capacity and moment correction plus a defined excess policy. A third is an independently advected phi-first level set with global or regional mass correction; it removes the present full-volume/air-geometry contradiction but gives weaker local conservation guarantees.

More normalization passes could reduce the measured row residual, cubic phi interpolation could reduce one interpolation error, and draining excess could reduce one part of the final area budget. None alone enforces both receiver capacity and first moment or resolves the conflicting `V`/phi geometry authority, so they are not proposed as standalone fixes. These are design conclusions only; this analysis changes no production numerics.

## Limits and validation

Contour metrics use the actual UI RDF segments. Circularity uses RDF-reported represented area and segment perimeter; centroid, radial error, and boundary covariance use segment-length-weighted midpoints. Unsupported RDF vertices make the coarse raster symmetric-difference proxy unresolved in some frames, so it is retained in the artifact but not used as primary evidence. Absent contours return null rather than a zero metric.

`tests/gentle-moving-blob-metrics.test.ts` passes three focused tests covering translated-circle invariance, a known ellipse, and absent-contour handling. ESLint passes for the analyzer, metric utility, and test. Full TypeScript checking still reports pre-existing errors elsewhere in the working tree; it reports none in these new paths. No browser or Dawn gate was run for this read-only analysis.
