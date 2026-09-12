# Shared RDF surface reconstruction in 2D and 3D

Advance Slice transports the same piecewise-linear VOF authority as Sparse
CM12. A smoother view can be reconstructed from the accepted volume fractions
and PLIC normals, but smoothness, continuity, and exact local volume are
different guarantees. No surveyed method provides all three on arbitrary 3D
adaptive and cut-cell grids.

## Candidate methods

| Method | Inputs and construction | Conservation | Continuity and curvature | Generality and cost | Fit here |
| --- | --- | --- | --- | --- | --- |
| Shared RDF isocontour | Each accepted PLIC supplies an interface centre and normal. Scheufler and Roenby reconstruct a distance value from point-neighbour planes using `d_ij = n_j dot (x_i - xS_j)` and the orientation weight `abs(n_j dot delta)^2 / abs(delta)^2`, then interpolate those values to shared vertices. | The volume-correct PLIC planes remain the VOF authority. One shared zero isovalue does not generally reproduce every cell fraction; the paper explicitly contrasts continuous global isosurfaces with volume-matching per-cell isovalues. | A single scalar value at each shared vertex gives a watertight C0 contour or surface. The reconstructed distance field improves normal and position accuracy, but the paper does not claim a C1 extracted mesh. | Demonstrated in 2D and 3D on Cartesian, triangular, tetrahedral, and general polyhedral grids. The reported plicRDF reconstruction cost is about 2.4–7.3 times Youngs. One-plane-per-cell ambiguity and immersed cut geometry still need explicit fallbacks. | Best reversible presentation preview. Keep PLIC transport unchanged and publish the signed, mean, and maximum surface-implied volume error. |
| Curved DPIR | Dynamic programming first finds a globally continuous edge-crossing chain. A per-cell correction adds a rational quadratic Bézier segment chosen to recover the target fraction; the improved form includes the supplied normal in its objective. | Exact area is recovered in each handled mixed cell. | Published guarantee is continuity (C0). The sources do not establish C1 joins. Curved segments improve circular reconstruction and the normal-aware objective suppresses oscillation. | Two-dimensional. The dynamic-programming step costs `O(N L^2)` for `N` interface edges and `L` edge samples. The published method cannot handle a cell edge crossed twice, including unresolved filaments. Extensions cover distorted/unstructured grids and triple points within stated restrictions. | Strong 2D choice when exact visible cell area matters, but it is not a direct 3D route and is a larger topology solve than an RDF preview. |
| PPIC / PLVIRA / PMOF | Fits a parabola in each mixed cell. Its position satisfies the cell fraction; orientation and curvature minimize neighbouring-volume or moment objectives. | Cellwise fraction is exact for each fitted parabola. | Gains one reconstruction order over PLIC and produces convergent curvature in the reported 2D dynamic tests. Independent cell parabolas have no published cross-cell C0 or C1 guarantee. | Two-dimensional in the cited formulation. Curved geometric flux intersection replaces the current swept-plane calculation. | Relevant to a future higher-order transport experiment, not sufficient by itself for a seamless view. |
| Iterative piecewise paraboloids | Fits a paraboloid by iterative least squares to initial PLIC patches, translates it to conserve cell volume, and intersects curved regions through local refinement and polyhedral approximation. | Cellwise volume positioning and conservative unsplit VOF advection are part of the method. | Provides higher-order interface representation and curvature behavior. It remains piecewise; the source does not claim watertight C0 or C1 joins between cell paraboloids. | Three-dimensional and demonstrated on arbitrary convex and non-convex cells. Its curved intersection and iterative fit are substantially more involved than PLIC. | Best surveyed route to genuinely curved 3D VOF transport, but it does not alone meet the smooth shared-surface requirement. |
| CLSVOF | Advects both VOF and a level set. PLIC reconstructed from the coupled fields corrects the level set, which is reset to signed distance near the interface. | VOF supplies the conservative mass authority; the zero level set is not an exact local-volume representation. | A smooth implicit field supplies normals and curvature and handles merging and breakup. | Demonstrated in 3D and with adaptive meshes, but adds another advected field, redistancing, and coupling stages. | Established but invasive. It would change Sparse CM12's numerical authority rather than only its publication. |
| Campbell tensor-product B-spline reconstruction | Normals divide connected interface cells into single-valued orientation patches. Cumulative volume-fraction column integrals are fitted with B-splines and differentiated to recover a smooth surface. | The constraints preserve column integrals, not each mixed-cell volume independently. | Arbitrarily high order within each spline patch; the paper reports fourth- through tenth-order convergence. It does not establish C1 continuity across separately oriented patches. | Three-dimensional Cartesian grids. The method identifies unreconstructable cells when the interface is multi-valued in every available orientation. It does not cover Sparse CM12 cut cells or nonconforming AMR. | Valuable evidence that smooth high-order 3D reconstruction from fractions is possible, but its grid and topology assumptions exclude it as a direct sparse proxy replacement. |

## Numerical boundary

The implemented default is a **shared RDF isocontour** over the accepted
VOF/PLIC state. It uses the established distance reconstruction in both
dimensions and leaves transport and pressure unchanged. The legacy PLIC view
remains selectable. Advance Lab reports the area implied by the shared contour
against accepted VOF amount. Partial-capacity cells, unresolved multi-interface
cells, and missing sparse support produce explicit counters rather than silently
inventing a smooth surface.

Sparse CM12's 3D publication already extracted a watertight shared scalar
lattice: its regression suite covers spheres, tori, ellipsoids, disconnected
components, and adaptive ratios with no interior open or nonmanifold edges. Its
former scalar was formed directly from extended PLIC signed-distance supports
with trilinear blending. RDF therefore targets shape and volume error rather
than a 3D crack repair. The production path now follows the paper's two
interpolation levels. It first reconstructs an orientation-weighted RDF value
at every accepted cell centre from the complete point-neighbour PLIC stencil.
It then fits one free affine least-squares value at each shared topology vertex
and trilinearly evaluates those vertex values at the page compiler's canonical
sample coordinate. It does not create a second mass field or alter the PLIC
cache used by transport.

## Implemented 2D reconstruction and measurements

Advance Lab now defaults to `shared RDF` and retains `legacy transport PLIC`
for comparison. The RDF view uses the accepted, volume-correct PLIC interface point
and normal in every mixed cell, evaluates the Scheufler--Roenby orientation
weighted reconstructed distance for every contributing plane, including the
cell's own plane, and least-squares interpolates one scalar to each shared
lattice vertex. Marching that single scalar makes the displayed
contour watertight (C0). Transport, pressure, adaptivity, and stored liquid
amount continue to use the production PLIC state.

Sparse AMR needs one adaptation beyond the paper's conforming polyhedral
vertices. At real coarse corners and fine-side T-junctions, all incident cells
share one least-squares value. At dense display samples inside a coarse cell or
an unsplit coarse face, the cell's affine PLIC distance is prolongated instead
of holding the coarse centre RDF constant. A constant prolongation displaced a
flat B8 pool by almost one fine cell. Exact full/empty, face-aligned jumps are
published at zero directly because they contain no mixed cell from which to
construct a PLIC interface point. Both rules derive solely from the accepted
fractions and PLIC planes.

| Accepted production slice | Transport PLIC shared-face mismatches | Shared RDF mismatches | Shared RDF area error |
| --- | ---: | ---: | ---: |
| Half-size sphere and pool, frame 0 | 44 / 58 joins; mean gap 0.244, max 0.581 fine cells | 0 | -0.882 fine cells squared (-0.066% of all liquid in the slice) |
| Same scene, frame 1 after advance | 64 / 82 joins; mean gap 0.153, max 0.464 fine cells | 0 | -1.772 fine cells squared (-0.133% of all liquid in the slice); the authored frame-0 circle is not used as a motion reference |
| CM12 Figure 3, four disconnected drops and pool | 84 / 131 joins; mean gap 0.223, max 0.791 fine cells | 0 | -5.073 fine cells squared (-0.212% scene total) |

The initial sphere's shared-RDF radial RMS is 0.072 fine cells, compared with
0.149 for the disconnected PLIC endpoints. These are measurements of the
sampled VOF field, not a claim that an under-resolved sphere is mathematically
perfect. The preview is C0, not C1, and it does not preserve every cell's
fraction. The Figure 3 receipt remains just outside the unchanged 0.2% test cap
at 0.212%; the assertion is intentionally left failing rather than changing the
bound or tuning the established formula. It reports signed/mean/maximum
implied-area error, ambiguous cells, unresolved cells, and partial cut cells.

The canvas consumes that one shared scalar unchanged for every finite,
full-capacity interface cell. An earlier display-only guard replaced 30 of the
68 supported sphere cells with owner-local PLIC and changed shared vertex signs
per owner; this produced 72 unmatched endpoints and a 0.581-fine-cell maximum
gap even though the underlying RDF contour was continuous. The guard is gone.
The integrated frame-0 and frame-1 receipts now report respectively 0 of 68 and
0 of 92 supported cells falling back, zero owner conflicts, and zero dangling
endpoints or gaps. Even the smallest resolved frame-1 minority fraction
(`2.8849e-5`) remains RDF. PLIC is used only for partial-capacity solid-boundary
cells or a non-finite RDF sample, which is reported as an explicit
reconstruction failure.

Current partial-capacity cells use the
visible PLIC fallback because scalar capacity alone does not define the open
cell polygon required for a correct RDF interface point. A mixed cell that
contains more than one interface is likewise outside the one-plane plicRDF
model.

The 2D reconstruction itself takes about 10.1 ms for the native 64 x 48 sphere
slice and 29.5 ms for the 128 x 128 Figure 3 slice (20-run Node means on the
development machine). It is computed only when that view is selected and is
cached by the accepted frame/topology state in the UI.

## Implemented 3D reconstruction and validation

Sparse Geometric now defaults its production water publication to `Shared RDF`.
`Legacy PLIC field` remains a runtime method option. Each valid reconstructed
PLIC contributes its plane/box intersection area centroid and normal to the
accepted point-neighbour stencil. The weight is the Scheufler--Roenby orientation
weight with exponent two for every contributing plane, including the cell's own
plane. The former self-weight of one was removed: Equation 10 has no self-cell
exception. A first GPU pass publishes immutable cell-centre RDF values; its
separate gradient pass remains available for cache validation. Presentation
then evaluates Equation 11 once at each canonical topology vertex from the
connected incident cell-centre values. Each P8 page cooperatively caches its
9-by-9-by-9 vertex values (about 7.1 KiB of workgroup storage) and reuses eight
values for every trilinear sample. The FPP slot `s` represents the physical
fine-lattice point `s + 0.5`; evaluating the vertex field at that coordinate is
required to avoid a half-cell mesh displacement. Initial publication, paused
edits, final scalar publication, and an accepted adaptive generation all refresh
the cache before a presentation page reads it. Legacy mode skips the RDF work.
On a uniform lattice, point neighbours are the 26 cells sharing a vertex. At a
2:1 join, the canonical integer vertex lookup includes every fine owner incident
on the coarse corner, edge, or face and deduplicates stable cell ids. A local
eight-octant connectivity closure prevents diagonal RDF support from crossing a
solid separation. A paused runtime switch between RDF and PLIC explicitly
republishes unchanged accepted scalar pages, so the selected representation does
not wait for a later physics or topology change.

An actual Metal differential seeded six signed/oblique plane cases plus a
diagonal-only point-neighbour case through the shipping Sparse Geometric solver.
Both density banks remained bit-identical. The final direct probe and shipping
curved run both recorded an unchanged source fingerprint (`716fa1d...` for the
RDF helper and `62acd51...` for resident publication).

The unchanged shipping FPP and mesh consumer were also run on the same accepted
sphere and torus residents in RDF and PLIC modes. All four meshes were closed
and manifold:

| Accepted resident | Published view | Mesh volume error vs accepted VOF | RMS / maximum surface error |
| --- | --- | ---: | ---: |
| Sphere | Shared RDF | 2.899% | 4.827 / 10.661 mm |
| Sphere | Legacy PLIC | 0.055% | 4.057 / 9.756 mm |
| Torus | Shared RDF | 0.258% | 5.936 / 23.459 mm |
| Torus | Legacy PLIC | 0.586% | 5.151 / 14.549 mm |

These measurements use the literal Equation 10 self weight and the topology-
vertex Equation 11 interpolation. They show the intended continuity and shape
improvement over the earlier owner-affine RDF trace, while also showing that RDF
does not dominate PLIC on every error measure.

On the matched mini32 B8/P8 benchmark (8 warmups and 24 hardware-timestamped
frames), presentation publication measured 5.898 ms median / 6.160 ms p95 for
PLIC and 7.340 / 7.602 ms for RDF. RDF adds 1.442 ms to that stage (24.45%) and
0.983 ms to the whole advance median (166.134 to 167.117 ms, 0.592%). Terminal
physics work was identical. This pair isolates the incremental RDF cost on the
source immediately before the later swept air-support optimization; it is not a
whole-frame timing claim for that subsequent integrated source. Raw receipts are under
`artifacts/sparse-cm12-rdf-presentation-performance-final/`.

The RDF selector disables the otherwise independent coarse column-height
shortcut. This keeps `Shared RDF` from silently becoming a height/RDF hybrid on
calm pages; the legacy PLIC option retains its column-height policy. A cell with
partial solid capacity keeps the fill-derived fallback and a zero RDF gradient,
because scalar capacity does not define the clipped polyhedron needed to locate
an RDF interface centre.

The published 3D mesh remains C0 rather than C1. One global isovalue also does
not reproduce every mixed cell's volume. These are properties of the published
RDF representation, while accepted VOF volume and PLIC transport remain the
numerical authority. If a boundary vertex fit is rank deficient, publication
uses the mean RDF value rather than inventing a normal extrapolation. Immersed
partial-capacity cells stay on the explicit fill/PLIC fallback because scalar
capacity alone does not define the clipped open polyhedron. The connectivity
closure follows accepted full-capacity owners; a zero-volume internal baffle
whose adjacent cell quadratures both remain exactly full would require an
additional row-aperture connectivity test.

## Primary sources

- Scheufler and Roenby, *Accurate and efficient surface reconstruction from volume fraction data on general meshes*, JCP 383 (2019): <https://arxiv.org/abs/1801.05382>
- Dumas et al., *A new volume-preserving and continuous interface reconstruction method for 2D multi-material flow* (2017): <https://doi.org/10.1002/fld.4372>
- Chollet et al., *Curved Interface Reconstruction for 2D Compressible Multi-material Flows* (2020): <https://doi.org/10.1051/proc/202067011>
- Liu et al., *An improved continuity-preserving interface reconstruction method for multi-material flow* (2021): <https://doi.org/10.1016/j.compfluid.2021.104960>
- Remmerswaal and Veldman, *Parabolic interface reconstruction for 2D volume of fluid methods* (2022): <https://arxiv.org/abs/2111.09627>
- López, *Unsplit geometric volume-of-fluid method with iterative piecewise-paraboloid interface reconstruction on arbitrary three-dimensional grids* (2026): <https://doi.org/10.1016/j.jcp.2026.114714>
- Sussman and Puckett, *A Coupled Level Set and Volume-of-Fluid Method for Computing 3D and Axisymmetric Incompressible Two-Phase Flows* (2000): <https://doi.org/10.1006/jcph.2000.6537>
- Campbell, *An arbitrarily high-order three-dimensional Cartesian-grid method for reconstructing interfaces from volume fraction fields* (2021): <https://doi.org/10.1016/j.jcp.2020.109727>
- López et al., *A new volume of fluid method in three dimensions—Part II: Piecewise-planar interface reconstruction with cubic-Bézier fit* (2008): <https://doi.org/10.1002/fld.1775>
