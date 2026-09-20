# Uniform Geometric: pages own the simulation domain

Design review, 2026-09-21. Supersedes the window-backed phi optimization plan.
This is the proposed architecture, not a claim that the current prototype implements it.

## Required contract

The accepted GPU page generation is the sole authority for simulation membership,
field addressing, pressure topology, and presentation. The liquid solve window is
retired. A bounding box may be derived for display or culling, but must not define
valid cells, allocations, physical boundaries, or solver dispatches.

There are no CPU page-demand receipts, allocations, or scheduling decisions between
stages of an advance. The host uploads external inputs and encodes a fixed GPU
pipeline; GPU metadata and indirect counts determine the work. Diagnostic readbacks
are asynchronous observations and cannot be required to finish the advance.

## What the current work does not satisfy

- Production scratch pages map a finite dense logical grid into transient physical
  slots. Their membership is rebuilt for each phase; they do not own persistent state.
- The window still defines `activeId`, vertex support, pressure origin/capacity, and
  the work of most stages. A page list underneath this remains a second authority.
- Dense phi, velocity, V, pressure and correction/reduction storage still scale with
  the authored lattice. Empty space between disconnected bodies is not free.
- The recent uncommitted GPU-only prototype removes page-demand waits by reserving
  the entire finite-domain scratch arena at construction. That demonstrates how to
  remove the waits, but is not an acceptable final sparse-memory architecture.
- Freezing far-air phi in the attempted sparse advection path failed a longer moved-
  inlet test. The separate census summary-index race was fixed in 6671ce77; that fix
  did not make sparse advection equivalent. A new field representation contract is
  necessary, not simply a larger guessed page halo.

## Authority and storage

Use a fixed-resolution signed page coordinate and a stable physical slot. Start by
measuring 32³ storage pages with 4³ compute tiles; keep the edge configurable in
research until the thin hose/pond memory measurements justify it. There is no array
indexed by the bounding box of logical page coordinates.

One accepted generation contains:

- Coordinate-to-slot directory, compact resident-slot list and cached adjacency.
- Persistent V, MAC velocity, phi, solid/open-fraction state and required history.
- Page role masks: liquid/interior, interface, source/frontier and stencil support.
- Derived pressure, transport, redistance, correction and presentation work lists.
- Counts, resource-budget/fault state and a generation identifier.

Separate work lists are legal derived views of this generation, not independently
allocated definitions of the domain. A pressure row must refer to resident cells;
a presentation primitive must refer to the same accepted generation.

Interior addressing is arithmetic. Neighbor caches handle normal seams; directory
lookups handle farther characteristic samples. Give shared MAC faces and phi
vertices a canonical owner. Ghost copies must have an explicit refresh order; two
pages may not independently evolve the same face or vertex.

## Residency and growth

Resident does not mean only phi-negative cells. The page domain is the union of
conservative liquid, the interface band, source footprints and all required support.
Keep deep liquid even when it is quiet: it still carries mass and pressure coupling.

At each advance, GPU kernels derive candidate residency from current residents and
explicit source/edit commands, allocate frontier neighbors from a free list, initialize
new slots, and construct adjacency and work lists. Do not scan a scene-sized page
catalogue to rediscover fluid. Remote drops seed their own page coordinates directly.

Support must account for each operator's actual reads and writes: characteristics,
redistance, velocity extension, sharpening, global surface correction and solids.
Use measured motion plus conservative force/source allowances. Validate support after
stages that can change the bound. Insufficient support must never be reinterpreted as
solid or silently omit transport. A GPU-only expansion/retry or an uncommitted failed
step is preferable to publishing a truncated solve.

Retirement requires zero retained mass and no interface, stencil, source, pressure,
or presentation dependency. Add hysteresis against allocation churn. Initialize all
newly visible field/history values; quarantine retired slots until every old consumer
is ordered before reuse. Stable slots alone are not a lifetime guarantee.

## Boundaries and phi

Physical boundaries come from authored terrain, bodies and explicit container walls.
The edge of allocated pages is not a wall. Outside validated support, absent pages
can represent ambient air; an absent page required by an operator is a support fault.

Replace the implicit globally evolving far-air phi contract with an explicit finite
interface band and well-defined exterior values. Preserve the transport/redistance
read band across page activation and retirement. This is a numerical change and must
be validated against physical/conservation metrics, not declared equivalent because
short trajectories happen to match bit-for-bit. All negative liquid and V remain
represented independently of the finite positive-air distance band.

## Pressure is part of the rewrite

Pressure uses one coupled sparse graph across all resident liquid cells. Page seams
are ordinary interior faces, never artificial zero-pressure boundaries. Real free
surfaces and solids retain their existing coefficient/boundary rules.

Build the multigrid hierarchy from active cells/pages in registered world coordinates.
Restriction, prolongation, residuals and coarse work lists must also be sparse; moving
only the fine stencil to pages while retaining a rectangular coarse hierarchy is not
completion. Coarsening must not introduce spurious connections between disconnected
liquid components. Preserve the original operator and pressure residual/convergence
requirements before optimizing topology maintenance.

## GPU allocation and publication

Reserve pools by a memory budget, independent of empty world extent. Shaders allocate
slots inside those pools; they cannot create WebGPU buffers. Infinite spatial addressing
does not imply infinite resident memory. Any optional host pool expansion happens
between advances, never as a prerequisite between stages of an advance.

Use candidate and accepted generations. Capacity, directory and support checks gate
all dependent work and final publication on the GPU. Overflow leaves the accepted
state intact and records a diagnostic; it must not partly advance time, consume a
source twice, leak mass, or expose half-initialized pages. This requires deliberate
state buffering and GPU commit guards, not just an overflow flag read after writes.

Simulation and presentation consume a consistent accepted generation. Diagnostics
report accepted/resident/support pages, new/retired pages, stage work counts, budget
usage and support faults. The pages overlay should show these roles, rather than only
scratch pages from the last phase.

## What to reuse from adaptive-mass

Reuse the design contracts in:

- `sparse-cm12-world-directory.ts`: signed GPU-mutable directory and bounded free slots.
- `sparse-cm12-topology-generation-store.ts`: accepted/staged/retired generation lifetime.
- `sparse-cm12-pressure-execution-image.ts`: pressure membership and indirect dispatches
  compiled from a single authority.

The generation-store implementation includes host-side preparation and allocation;
its invariants are useful, but it cannot be transplanted as the per-frame scheduler.
Likewise, do not import adaptive resolution, B8/P8-specific ABIs or an independent
presentation-domain authority into the fixed-resolution design.

The existing `uniform-page-layout.ts` is a useful host oracle for signed addressing,
stable slots and reuse tests. Its host planner is not the production authority. The
seven-point page stencil probe has artificial missing-neighbor Dirichlet conditions;
it is not a production pressure implementation.

## Implementation order and acceptance

1. Implement the GPU page-domain generation, fixed-budget pools, activation,
   initialization, retirement and publication guards. Prove signed coordinates,
   remote insertions, slot reuse and overflow without mid-frame host intervention.
2. Implement persistent field addressing and a globally coupled paged pressure
   operator/hierarchy on captured inputs. Include cross-page ponds, hydrostatics,
   disconnected components and page-order permutations.
3. Connect transport, explicit phi-band handling, extension, sources/solids, global
   corrections and presentation to the same authority. Retire window membership,
   pressure-lattice replanning, dense census and dense domain-sized field allocation.
4. Promote only after complete scene validation and matched performance gates.

The reference implementation can remain in a test-only oracle during the rewrite.
Production should not have a hybrid fallback where the old liquid window silently
becomes the authority again. A contiguous-page fast path for mini64 is appropriate,
but its membership and boundaries must still come from the page generation.

Required evidence:

- At fixed resident fluid/support, enlarging empty world extent or increasing the
  distance between disconnected ponds does not increase field allocation or fine
  work counts. Account separately for legitimate hierarchy/topology changes.
- No in-advance page map/readback wait, host decision or buffer resize.
- Seam-crossing hydrostatics, inflow, moved sources, drops, solids, emptying/refilling,
  retirement and capacity exhaustion preserve the stated numerical contracts.
- Global mass/correction reductions cover unique active owners in a stable order;
  no duplicate seam volume or dropped sleeping liquid.
- Mini64 retains at least 95% of frozen-baseline throughput at matched numerical
  settings. Measure full steps and growth-frame latency, not just a stencil kernel.
- Report GPU compute and host/queue gaps separately. A shorter kernel does not
  establish a faster frame, and reserved capacity is not active memory use.
