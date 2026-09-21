# Residency-owned domain, native numerical execution

## Invariants

1. A domain generation owns required cell pages. Required support is wet/source
   support expanded by a conservative velocity/acceleration reach over the
   allocation horizon, operator stencil reach, and physical-boundary support.
   Allocation latency is part of the horizon. A topology generation cannot be
   published until its resources and execution plans are ready.
2. Pressure coupling spans connected liquid, not just the velocity sweep. All
   liquid and the existing free-surface/boundary closure must remain in the
   solve. A page boundary must never become a physical wall.
3. Numerical workspaces are an execution layout for those cells. They need not
   reproduce the domain's allocation-page packing. In particular, multigrid
   pressure, RHS, topology, residual and minimum fields use native rectangular
   textures, including their existing halos. Ordinary dispatch extents are
   ceil(logical dimensions / workgroup edge) at every level.
4. Page placement and neighbor relationships are resolved when a generation is
   compiled. Repeated stencil taps must not perform catalogue lookups or atlas
   division/remainders. Existing operators, sweeps, tolerance, acceptance and
   recovery are retained.
5. Topology growth has a real allocation/compilation/remap cost, measured
   separately and included in end-to-end performance. No silent clipping while
   waiting for allocation, no full-world fallback disguised as sparse residency,
   and no hidden padding/copy work outside the reported cycles stage.

## Current implementation constraints

The current `initialUniformPageDomain` enumerates the whole authored domain.
Its catalogue is immutable; root field textures, transport scratch, root
shader constants and publication adapters are built against that catalogue.
The existing lagged active-window code is not a safe residency allocator: it
can deliberately under-cover a step and later fall back to the whole domain.
It must not be reused as an allocation safety proof.

The single-page native specialization proved the execution-layout separation.
The immediate multigrid implementation extends that separation to multi-page
root fields. Domain ownership and root operators remain unchanged in this
controlled change, so layout equivalence can be established before changing
residency. This alone does not implement demand-only domain allocation.

## Controlled implementation sequence

A. Retain the former tiled pressure layout as a QA oracle. Add a logical-dispatch
   atlas oracle to measure padded launches separately. Production multigrid uses
   native fields and direct dispatch with the existing residual-gated prefix.
   The already-dense pressure consumer binds the native result directly; the
   atlas-to-dense publication pass disappears.

B. Use the same root inputs, schedule and launch mode for all three layouts.
   Compare pressure and subsequent volume/velocity/phi, including non-divisible
   dimensions, boundary cases and rejected-cycle recovery. Record stage times,
   workspace bytes and dispatch counts. Numerical equivalence comes before
   timing interpretation.

C. Replace immutable all-resident root ownership with an allocation transaction:
   GPU demand evidence -> required support -> prepared generation -> migration ->
   acceptance -> retirement after in-flight consumers. Live liquid insertion,
   moved inlets and solid edits must participate before their first advance.
   Disconnected regions require multiple execution blocks; merging/splitting
   blocks cannot silently omit pressure coupling. Do not advertise this step as
   complete while root allocations still cover the authored world.

## Acceptance

- Same-input layout oracles agree numerically; the native cycles contain no
  atlas mapping, padded page launches or pressure-publication conversion.
- Long-dam cycles approach main's measured 6–8 ms range at matching configured
  work. Report total frame time as well; MiniDam32 must retain parity.
- Demand allocation additionally requires allocation-byte/page-count tests for
  an identical moving fluid body in differently sized empty worlds, measured
  growth across multiple boundaries, high-speed motion and source/solid edits.
- Run the canonical Sparse CM12 Dawn regression without weakened gates.

## Allocation transaction design (not yet connected to production)

Production fields must become owned by an accepted execution generation; the
current independently allocated readonly texture members prevent an atomic
resource swap. A generation must own the canonical V/phi/velocity fields,
operator scratch, pressure workspace, native dispatch plan and consumer views.
The world boundary description stays separate from the resident bounds.

The transaction has these states:

1. **Accepted:** the sole simulation authority. Existing reductions produce
   occupied/interface bounds and velocity reach; source commands carry their
   prospective footprint. The current `UniformPageSupport` prototype already
   closes forward destinations and backward reads, but its interleaved buffer
   is not the production authority and must not become a per-frame copy target.
2. **Required:** union retained mass/interface support, source support and read/
   destination closure. Pressure retains the complete connected liquid and
   separating-boundary continuation. Prediction uses the velocity field that
   actually transports volume, including its extension/interpolation reach.
   Current velocity alone is not a certificate for a later, force-modified
   characteristic. No retirement of mass-bearing or history-required cells.
3. **Prepared:** compile native execution blocks, allocate changed fields and
   build bindings outside the numerical advance. A rectangular required region
   gives one native block. Disconnected regions must not be replaced by their
   potentially enormous whole-world bounding box. Blocks that become coupled
   must merge or receive an explicit, verified boundary coupling implementation.
4. **Migrated:** copy retained canonical fields bit-for-bit once for this
   transaction; initialize newly admitted support to the correct ambient state.
   Rebuild derived scratch. Migration never mutates the accepted source.
5. **Validated:** verify all reads/destinations of the next step are covered.
   If evidence is late or a source invalidates it, prepare sufficient support
   before advancing; do not knowingly clip a step and repair it later.
6. **Published:** swap simulation bindings and presentation ownership together,
   then retire the old generation only after all queued consumers release it.
   Failed allocation or validation retains the old generation and its time.

The difficult distinction is between sparse residency and dense execution.
WebGPU textures do not provide application-controlled sparse virtual-memory
mapping. Arbitrary disconnected page packing cannot be made equivalent to a
single dense texture merely by precomputing a directory. Native interiors can
be free of paging arithmetic; boundaries between separately stored blocks need
precompiled addressing or explicit coupling, whose dispatch/transfer cost must
be measured. A rectangular long-dam region is the first integration target.

The initial whole-world catalogue must be removed only when these ownership and
transaction paths are in place. Changing `count` or a dispatch bound while
leaving readonly world-sized fields in place would be execution culling, not
reachability-driven allocation, and would not satisfy the allocation invariant.
