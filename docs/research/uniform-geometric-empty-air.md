# Empty-air sparsity for Uniform Geometric

Investigation: 2026-09-19, current working tree, including existing uncommitted
uniform solver changes. No simulation implementation changed.

## Conclusion

Yes: keep the same finest-resolution Cartesian lattice and put its fields in
fixed-size resident tiles. Empty distant air need not own fine simulation
storage or receive fine-grid dispatches. This preserves uniform geometry and
avoids coarse/fine interfaces. It does not require adopting adaptive-volume's
topology machinery.

The first useful rung is **sparse work on dense storage**, with the existing
dense solver as the reference. Memory savings follow in a separate rung.
Skipping only cells with V=0 is incorrect for this implementation. The support
of transport, phi, velocity extension and pressure must be handled explicitly.

## What the current code pays for

Sources: `lib/methods/uniform/uniform-volume-method.ts`,
`webgpu-uniform-reference.ts`, `uniform-volume.wgsl.ts`,
`uniform-host-allocation.ts`, and the uniform pressure/extension modules.

- The method explicitly disables the existing active-region path. Its cell
  kernels dispatch the entire grid in 4³ workgroups; phi covers all vertices.
- Each receiver owns an 80-byte nine-donor stencil, including empty receivers.
  This arena is reused for sharpening, so those costs must not be added twice.
- Geometric coupling always builds edges, sums donors, adds fallback, and runs
  three row/sum/donor normalization rounds: twelve full cell dispatches, plus
  four full scratch clears. The gather is another dispatch.
- Sharpening, when enabled, runs eight sweeps of four full-grid passes.
- Dense texture copies, velocity fields, extension scratch/hierarchy,
  multigrid textures and surface publication add bounding-box costs.
- Current picker defaults are semi-Lagrangian velocity and liquid-capacity
  balancing **off**. The optional 64-round balancing investigation is a
  different configuration. The three geometric normalization rounds above
  still run with that option off.
- Construction allocates four velocity textures and two transport textures
  even for semi-Lagrangian. Removing unused allocations is a separate possible
  saving; it cannot remove the cubic growth with empty domain size.

CPU arithmetic from the allocation plans and constructors:

| Cubic lattice | Nine-donor arena | Approximate major payload |
| --- | ---: | ---: |
| 32³ | 2.5 MiB | 15.3 MiB |
| 64³ | 20 MiB | 115.6 MiB |
| 128³ | 160 MiB | 899.2 MiB |
| 256³ | 1,280 MiB | 7,093.2 MiB |

Major payload includes the host allocation plan, pressure hierarchy and backup,
extension hierarchy, two vertex phi fields, and stencil arena. It excludes
small control buffers, terrain, rigid resources, audit fields and driver
padding; these are estimates, not measured VRAM. Some sizes may be rejected
by the actual device's single-binding limits before total memory is exhausted.
The constructor checks the stencil buffer limit explicitly.

## Initial scene evidence

The new CPU probe evaluates the real initial vertex phi, includes every cell
with any corner phi <= band × maximum cell spacing, then counts occupied tiles.
It includes liquid interiors, not just the surface. These are **band sensitivity
measurements**, not certified residency sets. Independent V, solid clipping,
motion, sources, transport dependencies and velocity hierarchy support are not
included. In particular, no result establishes a safe fixed halo width.

| Scene | Lattice | 4³ tiles: +4 cells | 4³ tiles: +8 cells | 4³ tiles: +16 cells |
| --- | --- | ---: | ---: | ---: |
| mini32 dam | 32³ | 56.3% | 76.6% | 100% |
| mini64 dam | 64³ | 47.3% | 56.3% | 76.2% |
| large-power dam | 64×20×64 | 6.3% | 9.4% | 18.0% |

For the large-power dam's eight-cell band, 8³ tiles occupy 12.5% of its tile
directory; 16³ tiles occupy 18.8%. Fractions are tile counts, not exact cell
payload fractions: partial boundary tiles include padding. Alignment can also
make a larger tile size look unusually favorable in one snapshot.

This supports the requested focus on large empty scenes. Small dam scenes can
lose most savings to the support band. Large-power has substantial initial
headroom even under the wider illustrative band. No GPU speedup or runtime
peak occupancy has been measured by this probe.

Reproduce: `node --import tsx tools/census-uniform-geometric-air.ts`.
Raw results: `docs/research/uniform-geometric-empty-air-census.json`.

## What “empty” must mean

**Both V and phi matter.** V is independently conserved and can sit outside
phi's liquid region. Keep any positive V until it has actually transferred
away; never retire it by a cosmetic threshold. Conversely, negative/crossing
phi remains geometrically important even if V is zero.

**Transport has dependencies through air.** `uvNormalizeRows` sums every edge,
including those pointing to zero-V donors. `uvSumDonors` sums contributions
from every receiver. Removing an air donor changes a row scale, which changes
the weight of a liquid donor; donor normalization then changes its distribution
to other receivers. A zero-volume node is not a zero-influence node. Preserve
the backward dependency closure through all three rounds, including fallback
and complete donor sums. Because the schedule is finite, this can be a finite
closure rather than the entire connected graph; it must follow the actual
pass dependencies. A complete graph-component closure is safe but may be far
too broad. The optional liquid balancing schedule requires its own analysis.

**Destinations must be found before transport.** This is receiver-backtraced
transport. Enumerating only existing wet receivers misses newly wetted air.
Use a conservative displacement bound to find candidate receivers, then trace
their actual samples/edges and expand support. Forward-advection of wet-cell
centres alone is not proof that all inverse-map receivers were found.

**Phi needs more than a sign.** The shader redistances within a four-cell-sized
metric band, but its closest-point search travels up to four lattice cells
per axis and samples gradients/interpolation around those positions. Advection
adds the RK2 sampling footprint and closed-wall continuation. A four-cell halo
around wet cells is therefore not a proven bound. Missing far-air phi may
eventually use a positive truncated-distance convention, but arbitrary positive
constants can alter interpolation and future surface arrival. Establish and
validate that convention explicitly; never regenerate phi from V.

**Velocity extension is not just a local halo today.** Its narrow-band front
is followed by hierarchy restriction/prolongation filling farther air. Preserve
the required ancestors and interpolation siblings, or retain this stage dense
in the first experiment. Replacing its far-field behavior with zero would be a
numerical change. Long characteristics need midpoints, endpoint interpolation,
and traversed solid checks. MacCormack additionally needs reverse/predicted
support when selected.

**Pressure remains global through liquid.** Empty far-air pressure rows are
not independent fluid unknowns, but all liquid rows and their free-surface/
solid coefficients must remain. A tile seam must not become a wall. Preserve
global red/black colors, cross-tile reads between passes, and multigrid parent
support. Fine tiles can be sparse with a dense small coarse solve; independent
tile pressure solves would change the method.

**Sources and edits can create support anywhere.** Explicit drops, inflow and
moving/edited solids must activate their required region before kernels read
or write it. Retirement must respect both ping-pong states and queued uses.

## A GPU-friendly layout to investigate

### User direction: a scene-wide 4h control grid first

Keep pressure reduction as future work. Add one coarse control cell per 4³
fine cells: this maps directly onto the current workgroup footprint. Use it to
classify and schedule fine work, while the h grid remains the authority for
V, interface geometry and dynamics. At 256³ fine resolution the control grid
has 64³ entries; a 16-byte record costs 4 MiB. The first implementation can
use a few flags and min/max phi bounds rather than require a coarse simulation.

Do not average phi or V to decide that a block is empty. Record any positive V,
the minimum/maximum of all shared vertices touching the block, and relevant
source/solid activity. A negative corner or small detached feature must survive
coarsening. These flags seed separate dilated work masks using conservative
stage-specific displacement/stencil bounds; refine with actual dependencies
where useful. "No interface" is not "air": all-negative blocks contain liquid.

A coarse signed-distance interpolant may help later, but is not automatically
a safe distance bound. Advected phi is not guaranteed to remain an exact SDF.
Use occupancy plus conservative reach initially, or store a certified bound
on interpolation error before rejecting a sample based on coarse distance.
The near-surface path still uses fine phi for geometry and interpolation.

Rebuilding the map from all fine cells once per step still costs O(N), but can
pay by eliminating expensive work in many later passes. It is a useful first
experiment, not the final large-world algorithm. Later updates can read only
resident/carried/newly activated fine tiles, with source events and conservative
motion expansion preventing missed arrivals. No need to read the map back to
the CPU to dispatch work.

This provides three distinct potential wins: coarse rejection before expensive
fine sampling, whole-workgroup rejection/compacted dispatch, and eventually
fine payload residency. Only the third saves substantial fine storage.
The three geometric normalization rounds and global extension hierarchy still
need the dependency treatment above: the control grid alone cannot certify
their omission. Initial sparse-work experiments should leave those stages
dense until their closure is implemented. A later pressure hierarchy can share
the coarse coordinates, but needs pressure-specific coefficients, connectivity
and transfers; an occupancy record is not yet a coarse pressure operator.

Start with 8³ storage tiles and 4³ work tiles; compare 16³ storage after
measurement. One 8³ tile carries eight existing-size workgroups. Storage and
dispatch granularity need not match. Keep contiguous local arrays and resolve
tile addresses at workgroup or boundary granularity wherever possible.

For a bounded scene, use a direct logical-tile→physical-slot array, a pool of
stable slots, and compact stage worklists. At 256³ with 8³ tiles a uint directory
is only 128 KiB. This is a much simpler starting point than a per-cell hash or
adaptive tree. A larger-world two-level directory can follow if needed.

Use one authoritative owner for every shared vertex and MAC face, including
outer boundary faces. Cross-tile access must read that owner. Begin with direct
neighbor addressing; add cached halos only if profiling pays for their refresh.
A one-cell full apron costs 95% extra samples for an 8³ interior, 42% for 16³;
it is also insufficient by itself for long traces or redistancing.

GPU classification/compaction should produce stage-specific worklists and
dispatch counts. Compare compact indirect dispatch against direct tiled
dispatch with a uniform early exit. The existing balancing experiment favored
direct early exits locally, but did not benchmark sparse spatial worklists;
it does not settle this choice. Avoid CPU readback scheduling and per-tile
command submission.

Distinguish active slots, reserved pool capacity and actual allocated bytes.
Inactive slots inside a large preallocated pool do not release GPU memory.
To obtain memory savings, allocate a working-set-sized pool, with controlled
growth or multiple slabs. Capacity exhaustion must stop/retry before publishing
partial state or dropping liquid. Initial scene construction and presentation
also need sparse paths before claiming total large-domain scalability.

## The ladder

1. **Dynamic shadow census on the dense reference.** Record separate per-stage
   tile sets for V/phi, required transport edges, velocity sampling and pressure
   ancestors. Capture real trajectories, not only the initial geometry above.
   Log predicted support misses, tile churn and peak union bytes. Include the
   exact current defaults and optional MacCormack/balancing configurations.
2. **Skip work, retain dense fields.** Add an experimental tiled schedule for
   stages with proven support; initially keep coupled normalization and velocity
   hierarchy dense. Validate artificial seams against the same dense step.
   Explicitly initialize newly active outputs and clear stale ping-pong/scratch
   state; simply skipping writes leaves old liquid behind. Charge classification,
   compaction, copies and clears to total time. This rung saves work, not memory.
3. **Move payloads into resident tiles.** Preserve the same worklists and
   numerics. Tile the stencil arena and independent V/phi, then velocity and
   fine pressure storage. Retain the dense reference as a selectable comparison.
   Remove dense mirrors and domain-sized publication only after parity checks.

Rung 2 has its first shipped stage: conservative volume **sharpening** now runs
on a 4h work map by default in Uniform Geometric, dense schedule retained as a
live comparison (`docs/benchmarks/uniform-geometric-tile-work-2026-09-19.md`).
Its fixed-phi admission predicate is what made that classification provable
without a predictive halo; no other stage has that property yet. Velocity
extension and conservative transport stay research on this rung until their
dependency closures above are implemented, and rung 3 is untouched.

The decisive benchmark holds cell size, liquid geometry, forces and timestep
fixed while adding distant empty domain. Use distant/open boundaries so boundary
changes do not masquerade as a sparsity effect; compare dense and sparse at each
domain size. Track stage and full-step times, reserved/active bytes, topology
cost and simulated-time-matched conservation, phi surface, symmetry, pressure
residual and divergence. Include two separated drops, a travelling front,
cross-tile wall impact, live inflow/solid edits, and a mostly wet negative control.

Success means the dominant fine work and payload grow with occupied/support
tiles instead of empty domain volume, with no lost conservation or visible tile
seams. It does not mean every coarse or directory operation is constant cost.

## Verification and limits

Ran the CPU census successfully against three repository scenes. No Dawn or
browser was started, and no simulation source was changed; the large-change
Sparse CM12 regression gate was therefore not triggered. Existing uncommitted
work was preserved. This investigation establishes a concrete opportunity and
the dependency risks, not a proven sparse solver or a measured speedup.
