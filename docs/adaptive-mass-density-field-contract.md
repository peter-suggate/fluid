# Density averages, implicit geometry, and refinement

This specifies the representation contract before choosing an implementation.
It incorporates the user's sharp-feature insight: the useful analogy is that
an adaptive implicit representation can retain planes, corners and edges inside
large cells. It is not a proposal to implement dual contouring.

## What a cell currently tells us

The resident CM12 scalar is a cell-volume density average. Its contribution
to the conserved amount is `rho_i * cellVolume(i)`. Solid occupancy is separate;
the effective fill used by several consumers divides by the open fraction.
CM12 tracking density can exceed one. Neither velocity nor gamma supplies the
missing liquid-interface location inside a mixed cell.

The average alone does not specify a continuous density function, an interface
normal, the position of a plane, a corner, or the number of liquid components.
It is not a point sample of an SDF, nor automatically a point sample of a smooth
density function at the cell centre.

For a fully open parent of volume V split into eight equal-volume children:

```
rho_parent = (rho_0 + ... + rho_7) / 8
```

That is one constraint on eight unknown child averages. It is sufficient for
mass conservation and insufficient for geometry. General clipped cells use
their actual volumes; the underlying identity is `sum(child mass) = parent mass`.

## The subtle distinction when copying the average

If the declared field is piecewise constant inside each accepted cell, copying
the parent average to every child DOES preserve that field exactly. It also
preserves mass. In that interpretation a partially filled cell contains diffuse
material throughout its entire volume, with no retained subcell interface.

But our presentation interprets means through additional interpolation and
reconstruction. Copying a parent average to eight new sample positions does
not generally reproduce the old interpolant. Changing the neighbours, support
width or reconstruction family can change it again. The numerical remap can
therefore be conservative while its displayed contour moves.

Consequently the desired contract is not merely "conservative prolongation".
It is preservation of the complete represented field, under an explicitly
declared relationship between that field and the averages.

## Define the field first; derive averages from it

Let `q(x,t)` be the persistent dimensionless tracking-density field in physical
coordinates, and let `O(t)` be the open (non-solid) domain. For a physics cell C:

```
M(C)       = integral over C intersect O of q(x,t) dx
rho(C)     = M(C) / volume(C)
rho_eff(C) = M(C) / volume(C intersect O)
```

The last expression applies only to nonzero open volume. This convention
matches the current full-cell-volume mass accounting; it does not claim that
existing grid means already determine such a field uniquely.

For continuous q at the interface, define the surface by `q = 0.5`. More
generally define it as the boundary of `{x in O : q(x,t) > 0.5}`, separating
the free surface from solid contacts. This also accommodates a sharp indicator
field, whose jump need not take the value 0.5 anywhere.

Given a retained field, splitting C does not change q. It changes only the
integration domains:

```
rho(child_j) = integral over child_j intersect O of q(x,t) dx
              / volume(child_j)
```

Exact integration then conserves parent mass because the children partition
the parent. The surface and normals remain unchanged because q remains
unchanged. Merely resampling q at child centres is not the same operation.

This is independence from the PHYSICS partition, not an absence of discretization
error. A finite representation of q has an accuracy limit and its own support
structure. Changing that structure must reproduce q exactly or obey a separately
measured approximation budget. Coarsening physics does not authorize discarding
interface information.

## A concrete split

Take an open unit cube containing liquid in its lower 30% in y. For this
example only, q is the sharp 0/1 occupancy of that geometry.

| Group | Number of children | Correct child average |
| --- | ---: | ---: |
| Lower half of cube | 4 | 0.6 |
| Upper half of cube | 4 | 0 |

The parent average is 0.3 before and after splitting. Its surface remains
`y = 0.3`. Copying 0.3 into all children gives the same total mass but a different
material distribution. It preserves the piecewise-constant interpretation,
not the sharp lower-fill interpretation.

Even the correct child means must not simply become density point samples:
linear interpolation of 0.6 at y=0.25 and 0 at y=0.75 puts its 0.5 crossing at
y=1/3, not y=0.3. The geometric interpretation still matters after an exact
volume remap.

## Sharp features are compatible with implicit fields

An implicit representation need not be globally smooth. A signed function
constructed from intersecting planar constraints, for example, can have a
sharp corner in its zero set. Requiring a smooth blend of neighbouring average
densities can round away that corner even when it could be represented compactly.

A plane through a cell needs orientation and position. With a known normal
and a single planar cut of known open geometry, the cell amount determines the
plane offset. An edge or corner generally needs multiple local constraints;
opposing sheets and disconnected components need additional structure. These
can be fitted from neighbouring information where sufficient, or retained
explicitly as part of the representation. They cannot be recovered uniquely
from the parent mean alone.

This is the relevant adaptive-SDF/dual-contouring analogy: subcell geometry can
contain more information than corner signs or one average, so increasing the
physical cell width does not necessarily require rounding every feature.
The analogy does not make density fractions interchangeable with SDF samples,
and changing only mesh extraction cannot satisfy the density-field contract.

The proposed field may therefore be piecewise smooth, with retained feature
constraints. Requiring a globally C1/C2 scalar everywhere would unnecessarily
exclude useful sharp-interface representations. On smooth portions, accurate
normals and curvature still matter. At a crease, preserving the intended pair
of normals matters more than forcing a single smooth normal across it.

## What the experiment must now prove

The density-field contract is the primary comparison criterion. A representation
of geometry disconnected from the conserved density is insufficient. A local
moment plane remains a possible component, not automatically a complete global
implicit field.

1. Construct a field from declared initial data; record what geometric
   information it retains and its initial error. Retain no callable authored
   shape in the evolving candidate.
2. Integrate that same field into different physical partitions, including
   partial region changes. Check local amounts and unchanged interface queries.
3. Refine the field's own representation separately. Verify exact preservation
   for its supported family, including sharp planes/edges/corners. Explicitly
   reject or retain detail for an unrepresentable merge.
4. Compare smooth bodies, creased bodies, thin sheets and disconnected material
   under the same field-query and volume-integration contracts. Measure all
   interface crossings and independent normals on each side of creases.
5. Evolve density with prescribed motion. Check conservative transport and
   geometry together; a successful repartition does not validate advection.

For a CM12-compatible diffuse field, `integral q` and volume enclosed by its
0.5 surface remain distinct quantities. Grid independence does not by itself
remove that paper limitation. A sharp indicator makes them equal in open
space, but changes the scalar model and requires suitable conservative
transport. This choice must be explicit in the eventual implementation.

No new representation or solver change is selected by this definition. It
replaces the premature choice of a smooth continuous field as the presumed
answer and sharpens the standalone prototype's acceptance criteria.
