# Released wet-cloth mechanics audit — 2026-11-06

Reference: `raymondyfei/libwetcloth` commit `b43993d775a0f9e3a9e32611f26936f2528fac37`. This audit separates matching material laws from numerical adaptations. It does not certify reproduced paper trajectories. Historical source trajectory receipts predating the radius-B, damping, finite-support and DER corrections are superseded, including the plastic-sheet 3.2 mm thickness assumption.

## Material and force paths

| Source behavior | Current implementation and evidence |
| --- | --- |
| Independent radius and biradius defaults; shell thickness is their sum | Explicit `rodRadiusB_m`; circular fallback for generic authored materials. Source importer keeps independently defaulted B. Rod area is πab. Shared roots inherit the same material radii in both released mixed meshes. |
| Plane-stress Green strain; triangle-average ψ raised to λ | Same membrane energy and analytic derivatives; released λ defaults to 2. Source shell thickness now A+B. |
| Hinge theta, tangent-half or sine-half measure; edge-endpoint ψ average | Same three measures and source area/edge-length weighting. |
| Four rod curvature components, two from each adjacent material frame | Four-component quadratic energy; each side contributes one quarter of the local inverse-Voronoi-length bending quadratic. This replaces squaring an averaged curvature. |
| Elliptical rod bending matrix and twist modulus | B00=Eπa³b/4, B11=Eπab³/4; twist uses source Gπab(a²+b²)/4. No circular-radius or Saint-Venant torsion substitution. |
| Rod ψ sampling | Stretch uses segment-start particle; bending/twist use the central particle. Generic authored rods retain their documented average policy. |
| Material damping switches and coefficients | Five explicit relaxation times. Enabled shell/rod bending and extension use 3η/E; twist uses η/G. `accumulateWithViscous` disables all modes; `accumulateViscousOnlyForBendingModes` disables membrane/stretch. |
| Viscous reference state | Membrane Green-strain difference, hinge measure difference, rod start-step curvature/twist, and rod extension relative to start-step length. Rod viscous extension denominator is the start-step length, independently of elastic rest length. |
| Reference frame and twist history | Source `findNormal` seed and spatial parallel transport persisted in decoded topology. Accepted time transport retained. Geometric reference twist is unwrapped relative to accepted history and saved through edit/rebase/publication. |
| Soft fixed bits | Finite material attachment springs, including twist's outgoing-particle convention; generic authored pins remain kinematic. Rest targets advance once before fluid prediction and are reused by mechanics. Terminal tip bit2 does not attach the preceding segment. |
| Particle mass and inertia | Source dry mass uses material rest fraction even for nonuniform pore fraction. Rod angular inertia excludes preexisting shell mass and uses quarter rod mass times a²+b². Bound water contributes translational inertia; source angular inertia remains dry. |
| Stationary solid and fixed material fractions | `solveSolid=false` holds mechanics and coupling stationary. Released `useVaryingFraction=false` preserves material control volume and fraction. |
| Optional solid pore pressure | Released default is false and no released water input enables it; no arbitrary saturation shrink force is substituted. |

Primary code: [material parser](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/App/TwoDSceneXMLParser.cpp), [viscous coefficients](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/ElasticParameters.h), [section stiffness](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/DER/Dependencies/ElasticityParameters.h), [DER curvature](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/DER/Dependencies/Kappas.cpp), [viscous references](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/DER/Forces/ViscousOrNotViscous.h).

## Deliberate numerical and geometry differences

The released solver predicts both phase velocities, projects mixture pressure, solves a grid-based linearized elastic system, accepts phase velocities, transfers APIC velocities, and advects. The implementation here retains the required native narrowband FLIP fluid owner, predicts the same nonfluid material forces for coupling, and solves the Lagrangian material energy nonlinearly after fluid impulses. This is an adaptation of the material model, not an identical discrete integrator. Cloth-enabled native fluid uses an integer velocity clock, skipping the initial fluid-only half-kick so paired phase velocities share the same time. This first-order partitioned scheme requires timestep-convergence validation. See [source step order](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/WetClothCore.cpp).

The source's effective constructor overrides the approximate/Tournier Hessian flags; the reference here uses actual energy derivatives and safeguarded local Newton iteration, with Float64 refinement of GPU candidates. This changes the numerical solve, not the intended static material force. Source DER bending's diagnostic energy omits ψ although its force includes ψ; our energy includes frozen ψ so its derivative matches the effective force.

Source `frictionAngle` and `collisionMultiplier` parameterize Gauss normal/shear elastoplastic contact stress. The current physical CCD/constraint solver with Coulomb friction does not reproduce that MPM constitutive contact operator. It enforces nonpenetration with paired material/Box3D impulses, and this difference remains a source-behavior validation gate. Legacy `friction_angle` is ignored by the original parser and must not be treated as an active source coefficient.

The released preview draws yarn as fixed-width GL lines; it supplies no volumetric end-cap specification. Shared rendering, picking, editing and collision use an explicitly authored segment swept by the material-frame ellipsoid `(a,b,sqrt(ab))`, with exact cut planes. The geometric-mean axial cap is an adaptation, circular when a=b. Broadphase may use a bounding radius; narrowphase uses the actual ellipse. No maximum-radius capsule replaces the visible or collision surface.

## Validation and admission

Focused CPU checks cover separate DER components under opposite material twists, ellipse bending axes, source ψ quadrature, start-length viscous extension against an analytic implicit solution, unwrapped full-turn history, attachment/free-fall predictors, local edit history, and actual ellipse plane/sphere contact. Actual Metal GPU oracles passed finite attachments and the four-component elliptical DER against Float64 (17 checks in the latest run). Full source runs must be repeated only after the shared ellipse geometry admission checks and this material correction settle.

Outstanding behavioral gates remain full-duration coupled trajectories, pressure/drag timing, source Gauss contact-law differences, and quantitative/visual paper comparisons. A passing unit test or isolated dry source step is not paper reproduction.
