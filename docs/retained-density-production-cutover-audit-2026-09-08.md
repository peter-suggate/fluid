# Current-field transport: production dependency audit

This is an implementation boundary audit, not a production acceptance. The full-fine prescribed-flow capture fails before meshing. See [the measured diagnosis](retained-imposed-flow-diagnosis-2026-09-08.md) and [the transport authority contract](retained-density-motion-authority-design-2026-09-08.md). Changing publication tie handling or smoothing triangles cannot supply missing transported density information.

## The current update order

The resident's `encode` path in [`webgpu-sparse-cm12-resident.ts`](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts) currently executes these operations:

| Stage | What it reads or changes | Consequence for replacement |
| --- | --- | --- |
| Transport velocity extension | Creates the effective velocity used by the accepted transport stencil. | Freeze this source velocity or compile the departure map here. |
| Face preparation | Prepares staggered face support. | Keep the distinction between face velocity, collocated velocity and effective extended velocity explicit. |
| Conservative transport | `traceGammaAndBeta`, `scatterDensityDeficit`, then `gatherConservativeDensity` update native density, gamma and momentum. | These mean interpolation weights are not integrals of the spatial density over departure volumes. |
| Tracer advection | Runs after gather. | The comment describing the effective velocity as untouched no longer matches gather's write; do not use that comment as a freshness contract. |
| Gamma diffusion and surface sharpening | Change destination native density/gamma through their own receipts. | Retaining these density writes would reintroduce a second, independently evolved mass authority. |
| Rigid displacement and capacity redistribution | Move native amounts through bounded routes. | These operations currently have no matching transport of the retained spatial shape. They need explicit geometric/measure operations. |
| Retained support advance | Fits `a*q_seed+b` to the final native means. | Replace this operation, rather than adding shape advection followed by the same fitting. |
| Integral compilation and retained bank commit | Compares candidate integrals with destination native means; publishes retained generation. | Reverse the dependency: candidate field integrals should author the destination native means. Validate completeness and conservation before committing. |
| Final scalar masks, scalar publication, forces and pressure | Consume the destination scalar state. | Publish one accepted field generation and its matching native restriction before these consumers run. |

The exact write is in `gatherConservativeDensity` in [`webgpu-sparse-cm12-resident.wgsl.ts`](../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts): after calculating `momentumNext/nextDensity`, it calls `cm12PublishTransferredEffectiveVelocity`. Dry outputs can reset that effective velocity to zero. The full-fine diagnostic independently confirms the reset occurs after the native transport consumed the source velocity. Thus calling the existing characteristic helper at the end of sharpening would sample a different velocity generation. Copying only density coefficients does not resolve this dependency.

The packed coarse gather must follow the same replacement contract. Updating only the ordinary packet kernel would leave two different evolution methods selected by native rung.

## One accepted transport transaction

1. Take the accepted field and an immutable source velocity generation. Compile or evaluate one departure map, including its derivatives and admitted domain. A query's origin must not choose a different map on the two sides of a shared geometric support face.
2. Determine destination support coverage from the forward reach of every potentially wet source. Independently validate every destination's departure donors. A valid destination pullback alone does not prove that omitted destinations carried no liquid.
3. Transport shared smooth degrees of freedom and intentional sharp branches into a candidate bank. Derive integral constraints from that same source field and map. Reject unsupported departures, missing donors, range failures and unresolved representation errors; no rescaling of an initial primitive or fitting to old CM12 targets.
4. Integrate the candidate field over each current open geometric support. Restrict those measures into all allocated native ownership, using compiled topology. This includes allocated frontier pages that have not yet entered the compact accepted manifest, while excluding retired or free slots.
5. Validate finite values, density range, support coverage, integral accuracy, shared traces, generation stamps and global/local conservation. Commit the candidate field and matching native density image together. The failure path must leave the accepted field, its integral image and published generation mutually consistent.
6. Derive activity, pressure membership, presentation and native receipts from this committed pair. Zero-time native refinement is restriction only; it must not retrace, refit or evolve the field.

The isolated quadratic prototype tests part of steps 1–5 with a prescribed volume-preserving affine map. It does not yet implement this complete transaction in the resident or support arbitrary fluid deformation.

## Coupled quantities and operations cannot be left implicit

**Momentum.** The present conservative gather transports `rho*u` with the same native interpolation weights as density. Replacing density but retaining that momentum numerator and dividing by the new mean would change velocity even in cases where shape transport is exact. The replacement must either integrate momentum with the same transported measure or establish and validate a separate staggered velocity transport contract. Uniform velocity must remain uniform, and coarse/fine comparisons must include momentum, kinetic energy and symmetry rather than only scalar mass.

**Gamma and sharpening.** CM12 gamma corrects its particular discrete transport matrix; it is not an independent material density or a Jacobian certificate for a newly chosen characteristic map. Its diffusion and sharpening operations cannot be applied as unexplained native corrections to a new spatial authority. Initially validate prescribed transport with both disabled, as the diagnostic does. For production, either implement an explicit conservative operation on the current field or remove an obsolete operation after its numerical purpose has been addressed and validated. The original paper's density post-processing is not part of this proposed correction.

**Numerical incompressibility.** A divergence-reduced velocity does not establish that an RK2 departure map has exactly unit determinant. Even the analytic divergence-free planar saddle `u=(a*x,-a*y,0)` has an explicit-midpoint map determinant `1+(a*dt)^4/4`, whereas its exact flow has determinant one. A general scalar pullback and its new-domain integral can therefore change mass. The map must have a validated conservative construction, or its Jacobian must enter measure transport consistently. The affine probe rejects nonunit determinants; silently extending that acceptance to general projected velocities would be wrong.

**Solids.** Clipping a trajectory against a wall can make the map noninvertible. Moving/static closure must have an explicit conservative displacement rule before geometric clipping; deleting covered density is not a shape-preserving transport operation. The current editor now prepares solid edits and rejects wet-overlap insertion atomically. That contract must survive the new field representation, and historical direct `applySceneUniforms` replays need to be updated honestly.

**Injection and removal.** Current injection writes native mass and momentum and then invokes the retained mean-fitting path. The new operation must instead author a current spatial density/measure with defined overlap, capacity and velocity semantics. Allocation, support stamps, solid masks and both accepted banks must remain consistent on failure. Union of implicit primitives alone is not an additive mass rule when an injected shape overlaps existing liquid.

## Next evidence required for cutover

- Exact prescribed affine plane/sphere transport through formerly dry supports, actual GPU point/gradient queries and per-support integrals, with missing-donor and missing-destination rejection.
- Shared-trace general-field transport, including a smooth nonquadratic profile and an intentional sharp branch. Establish convergence and positivity admission independently; exact global-quadratic motion is insufficient.
- A divergence-free non-affine map and the linear saddle above, measuring Jacobian, local/global mass and reversible deformation error. Include a genuine deformation, not just translation of an analytic seed.
- The same prescribed motion on all-fine and mixed native rungs, followed by coupled quarter/half pool impact and the unchanged core scene regression gate.
- Live insertion, moving solids, domain frontier growth, rejected transactions and zero-time topology changes using the same accepted field/measure transaction.

GPU execution and bounded work are part of these gates. CPU implementations remain independent small-fixture oracles; production tracing, reconstruction, integration and restriction must not materialize evolving per-cell JavaScript geometry or require per-step field readback.
