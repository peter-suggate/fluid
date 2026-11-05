# Cloth mechanics implementation and validation notes

The implementation is a physical implicit energy solve, with a Float64 reference and an optional graph-colored WebGPU solver. It does not certify the paper's scene behavior.

**2026-11-06 full material audit:** All historical source trajectory receipts in this document are superseded pending recertification with independent source biradius, per-mode viscosity, four-component DER and finite supports. Plastic thickness is A+B, not the historical 3.2 mm estimate. See `wetcloth-mechanics-source-audit-2026-11-06.md`.

**Earlier attachment correction:** historical source-step and 0.5-second receipts below used hard source pins. They are superseded for fixtures with fixed soft vertices and do not validate the corrected finite-attachment source model. Corrected CPU simple-cloth500 and full towel first-step reruns passed below. Corrected GPU oracle and production source trajectories are pending; generic hard-pin checks remain valid.

## Constitutive model

- Shell membrane energy uses Green strain and plane stress, with stiffness multiplied by the source solid-fraction exponent (2).
- Shell bending uses source dihedral models: theta, `2 tan(theta/2)`, or `sin(theta/2)`. The source default is theta; both large dam-break drag comparison inputs select the tangent model. Bending solid fraction is averaged at the two shared-edge endpoints, matching `ShellBendingForce.cpp`.
- Rods have actual extension, material-frame discrete curvature, and independent twist degrees of freedom. Reference directors undergo parallel transport. No spring-network or prescribed-animation substitute is used.
- Shell membrane, hinge, and rod extension derivatives are analytic. Local DER bending derivatives use numerical differentiation. Energy line searches and the requested Newton correction determine convergence; an unaccepted, nonzero correction cannot be mistaken for convergence.
- The GPU solves displacement increments. Adding tiny timesteps to world-coordinate float32 positions would lose physically meaningful motion; increments are accumulated onto the accepted Float64 state after readback.
- Contact uses swept conservative advancement and refitted exact AABB broadphases for vertex/face and edge/edge self-contact. Rigid geometry comes from the shared shape authority. Dynamic rigid impulses account for inverse mass and inverse inertia and are returned to the resident rigid system.

## Source settings which must remain explicit

The pinned source is `raymondyfei/libwetcloth` commit `b43993d775a0f9e3a9e32611f26936f2528fac37`.

1. `solveSolid` defaults to true. Ring tests and capillary suction set it to false: they study transport on a stationary mesh. Their mesh must not fall under gravity or drag.
2. `useVaryingFraction` defaults to false and is false in all released water cloth/yarn inputs. In `TwoDScene.cpp` at lines 5700, 5785 and 5885, both material volume and fraction updates are conditional. Source-compatible runtime therefore keeps rest control volumes and rest fractions; geometric volume variation is an explicit separate option.
3. `applyPorePressureSolid` defaults to false (`TwoDSceneXMLParser.cpp:1887`) and no released water XML enables it. `LinearizedImplicitEuler.cpp:582–592` always adds Eulerian pore pressure to liquid, and conditionally adds it to solid. This is not a saturation-driven rest-length shrink operation. `Pressure.cpp:26–139` evaluates a grid pressure gradient multiplied by phase volume, timestep, saturation, solid weight, and an orientation coefficient. A standalone shrink force would not reproduce this source path. The optional solid force has not been added to mechanics; Eulerian liquid pore forcing belongs to the fluid plugin.

Paper Eq. 8/12 describes an interphase pressure-gradient term; the optional source discretization should be audited with its liquid and air/surface-stress contributions before enabling a new corrected mode.

## Measurements and checks

- All 32 released water configurations containing cloth or rods construct with their complete imported connectivity and materials. This includes the 159,201-vertex shell and 79,204-vertex yarn. Initial rest energies are numerically zero (largest approximately `1.1e-25 J` for stiff plastic).
- Full 47,072-vertex tighten-towel source: first dry reference step converged in 8 sweeps to a maximum Newton correction of `9.97e-9 m`. Reference mechanics took approximately 20.2 seconds; initial complete self-contact broadphase/narrowphase took 0.94 seconds. These are CPU oracle timings, not a real-time or reproduction claim.
- Exact simple-cloth source: 500 steps at its `0.001 s` timestep completed through fixed supports and self-contact without nonlinear failure. Maximum 37 sweeps; maximum accepted convergence measure `9.996e-9 m`; 3,094 cumulative self-contact resolutions. This source has no solid fields and uses free space.
- Actual Metal/Dawn shell and four-node bending/twist rod comparisons passed against the Float64 reference, including released strain and millimetre hinges at metre-scale coordinates. Exact full towel GPU first step passed in 8 sweeps, residual `6.2625e-9 m`, exclusive mechanics timing 277.7 ms, initial self-contact pass 1114.7 ms. The initial failure was underflow in the squared dihedral Hessian denominator; normal-polynomial rescaling preserves the derivative and avoids it.
- Unit checks cover dry/wet inertial mass, rigid invariance, strain/bending energy, source scripts and rollback, impulse identity/once-only consumption, true between-frame contact crossings, source rounded solid SDFs, SI rigid impulse transfer, analytic energy derivatives, BVH refit completeness, and stationary source mechanics.

## Outstanding gates

Full-duration coupled source scenes, quantitative saturation/drag/buoyancy comparisons, general grazing yarn cuts, and final visual comparisons to the paper remain unverified. GPU contact acceleration and faster global nonlinear solving may be required for practical full-duration runs. Do not describe imported source meshes or an isolated passing dry step as reproduced paper results.


## Contact and cut follow-up

Through-section box cuts split yarn while retaining original rest geometry, material frames, source clock, water density and surviving rod IDs. Exact material-frame end planes clip analytic capsules in picking, rendering and contact. Oblique ends extend the analytic cylinder before clipping, preserving its outside surface. Grazing section cuts still fail explicitly; they are not replaced with coarse voxels or round caps.

Whole finite rod segments now collide against rigid/source solids, including contacts between distant endpoints. Convex support/GJK and conservative sampled-field bounds use the same clipped volume. Rod material-point Jacobians distribute a force couple to endpoints for transverse moment and apply axial torque to independent twist. Tests cover momentum conservation, friction-induced spin, between-frame sphere crossing, flat-end openings and multipoint plane contact. Vertex/face and edge contacts include Coulomb friction. Contact constraints iterate until displacement and timestep-scaled velocity changes are below `1e-8 m`; unresolved constraints fail and trigger runtime rollback.

GPU candidates that reach a finite float32 constitutive precision floor receive Float64 iterative refinement of the same energy, with the original previous-step strain and inertial target. Reports distinguish GPU and refinement iteration counts. This does not alter timestep, material stiffness or convergence tolerance. The exact tiny-source GPU host completed 500 steps at its original `0.001 s` timestep with refinement. A separate CPU rerun using iterative contacts completed 500 steps in 18.9 seconds, at most 37 mechanical sweeps and four contact sweeps. The complete source first-step mechanics sweep is pending. These dry subsystem checks do not validate water dynamics or paper reproduction.


## Source capillary cohesion

`cohesion.ts` implements the released source's soft-element bridge search and force law. It uses rest-angle Gauss weights for shells, midpoint weights for rods, the actual source bucket neighborhood, two/four normal cones, native liquid signed-distance line samples, and the source 0.4–0.6 bridge occupancy gate. Reciprocal pairs halve their orientation weight. The resulting constant-magnitude attraction is an implicit `k * distance` energy with analytic gradient/Hessian. GPU mechanics uses its candidate as the initial guess for the same full Float64 energy whenever wet bridges exist. Real native phi is mandatory; invalid or unavailable samples fail explicitly. Fixed/rigid surface endpoint mechanics now uses the same native-phi-gated force law with exact energy derivatives and paired linear/angular impulses; its quadrature/admission integration remains a fidelity gate. No unsupported direct bound-to-bound proximity transfer has been added: source intersection records are consumed by cohesion, while liquid transfer goes through the existing free-fluid capture/release path.

## Large source upload and camera validation

The exact 159,201-node splash-cloth-large source completed its first Metal GPU mechanics step at the authored `0.0002 s` timestep: eight sweeps, maximum update `7.038438720741169e-9 m`, one contact sweep, no Float64 refinement. The preceding Dawn native abort came from a large Metal buffer upload; bounded 1 MiB uploads and eight-byte buffer allocation alignment fixed it without altering the mesh or numerical method. The complete source sweep remains pending.

The shared view ABI now appends the camera right vector (108 floats / 432 bytes), retaining every previous lane. Production cloth, water, rigid, SVO, grid, particle, and diagnostic camera paths read it. Actual Metal shader compilation and rolled/polar camera rays passed against the shared CPU picking basis. The full-image rolled-cloth raster/pick comparison also passed on actual Metal, including analytic yarn and shell depth.

Accepted solid cohesion reactions are evaluated at the implicit minimizer before contact projection. Dynamic body IDs share the existing reciprocal Box3D impulse route; fixed samples retain explicit support-reaction records. Focused tests verify energy derivatives and total linear/angular momentum. Source `solveSolid=false` omits these unused forces and reactions. Actual GPU shell/DER, precision-floor refinement, and soft cohesion refinement tests passed (three checks).

## Thick shell contact and exact clamp visibility

The first-step sweep passed eleven original sources, then the 40,000-node plastic sheet exposed a false self-contact pathology: its 3.2 mm thickness spans several 1.44 mm mesh edges, so excluding just one triangle ring made a flat continuous sheet collide with itself. The owned runaway diagnostic was stopped; no timestep, thickness, or mesh was reduced. Self-contact now excludes the connected rest-surface patch inside each feature's physical thickness support. Spatially close disconnected layers remain collision candidates. A full exact plastic rest-state contact diagnostic completed with zero contacts and bit-identical positions; focused tests retain between-frame collisions with a disconnected layer. Subsequent source runs use bounded per-source subprocesses and report timeouts as failures.

Rounded clamps use a distinct display tag (10, preserving physical tag 5 and existing robot display tag 5) and the shared local cut atlas. Finite analytic face, cylinder, sphere, and cut-plane enumeration replaces iterative marching in primary/water visibility. The GPU fluid-only depth test passed faces, curved rims, openings, rotated local cuts, undo, and a near-tangent ray. A primary-raster comparison initially exposed the generic CPU field-program picker's broad early-hit tolerance (23 microns outside the surface); shared rounded-plate picking now uses the same finite analytic patches. Both actual primary and fluid-only full-image parity checks passed after this correction.

## Finite source supports

The released `TwoDScene.cpp:630–681` creates quadratic attachment forces for soft fixed bits. It does not eliminate these DOFs: `applyScript` updates their rest targets, and `SceneStepper::advectScene` advances their physical positions. The unused `zeroFixedDoFs` helper is not an effective source constraint. The source material's `attachMultiplier` (default 0.001) now selects finite springs; its absence retains explicit generic kinematic pins.

Translation stiffness is the Gauss-volume-weighted attachment Young modulus times the cube root of the rest particle volume. Angular stiffness uses the analogous shear modulus and the source cross-section polar moment divided by rest particle length. Source outgoing-segment twist uses its start particle's dry angular inertia. The final rod tip's extra scalar is uncoupled and does not pin the preceding segment. The source quaternion projection returns an unsigned AngleAxis angle; the target update retains that behavior.

CPU/GPU energy, impulse acceptance, contact inverse masses, manifold transport, water exchange, and fluid coupling use the effective kinematic mask. Source-script targets remain separate from physical positions, are rolled back on failure, and survive local edits and save/load. CPU tests verify exact implicit spring response, fluid/contact momentum at source fixed nodes, the twist convention, and generic hard-pin behavior. The mixed-source radius audit found no mismatch: fuzzy cloth and towel use the same shell/rod radius with no particle overrides. Rod angular inertia excludes shell mass. Initial source dry mass uses the material rest fraction even when pore fraction is authored per vertex, matching the parser. The nonfluid predictor stages targets once before fluid projection and exposes gravity plus the same elastic/attachment/cohesion force derivatives without advancing the material state; final mechanics consumes only returned fluid impulses and integrates nonfluid forces once.

Corrected finite-support CPU receipts: exact simple cloth completed 500 authored 0.001-second steps (maximum 36 mechanics sweeps, four contact sweeps, correction 9.996368685574684e-9 m); exact full towel completed its first 0.0002-second step (eight mechanics sweeps, one contact sweep, correction 8.826089400518805e-9 m). Receipts are `artifacts/wetcloth/simple-cloth-finite-cpu500.json` and `towel-finite-cpu1.json`. These are dry mechanics checks, not coupled reproduction certificates.
