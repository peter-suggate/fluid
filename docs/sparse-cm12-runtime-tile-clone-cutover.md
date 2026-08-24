# Sparse CM12 runtime tile-clone cutover

Status: implementation contract, 2026-08-24.

## Decision

Sparse CM12 residency is a set of physical tiles, not a bounded logical world.
Construction, GPU memory, recurring work, and shader compilation must scale with
resident tiles and their changed seams. Empty coordinates allocate nothing.

World extent is therefore metadata, not a sizing input. Translating the same
resident tile set into a coordinate space billions of tiles across must leave
pool layout, construction work, and memory unchanged. Vast mostly-empty worlds
are the architectural acceptance case; the bounded Long Dam is only the first
dynamic-front regression.

The Long Dam is the first cutover lane. At generation zero it must contain only
tiles whose authored density is nonzero. It must not construct an immediate
receiver shell, a dormant receiver apron, a logical-domain owner plane, or
per-coordinate rung/seam variants. Before transport can enter an empty
coordinate, the GPU allocates, initializes, validates, and publishes a complete
tile there. The dam front must advance through those runtime clones without a
host rebuild or a hidden numerical wall.

This deliberately distinguishes three terms which the current implementation
conflates:

- **capacity** is uncommitted storage in a bounded physical page slab;
- **residency** is a coordinate-to-page mapping in the accepted generation;
- **activity** is the subset of resident pages scheduled by a frame.

Capacity is allowed to have headroom. Residency may not be prefilled to avoid
implementing allocation. Activity is not evidence of sparse allocation.
The first cutover uses a fixed 1,024-page slab budget: Long Dam seeds only 80
complete resident pages into it. This budget is deliberately unrelated to the
logical domain dimensions. Reaching it must saturate with a receipt; a later
cutover may chain another fixed slab or retire/reuse pages, never resize from
world volume.

## Current cutover status

- Generation-zero accepted activity and presentation residency are the 80
  authored wet Long Dam tiles. Dry coordinates have no presentation page.
- Swept receiver publication allocates pages on the GPU and the 96-step Long
  Dam gate advances through newly cloned B8 receivers.
- Presentation residency and page allocation are sparse, but bounded physics
  atlases temporarily retain the complete four-rung plus 2:1 seam catalogue.
  Receiver publication needs that catalogue to enter transport at B8; the
  geometry-only dynamic page cannot yet replace its rows and incidence without
  changing Long Dam's material flow.
- The remaining compatibility debt is below the presentation seam: construction still
  expands the physics atlas to 1,152 dormant keys, builds 102,792 accepted cells
  and 298,614 accepted rows, and uploads a logical-domain owner directory.
  Removing those structures requires the TCP1 coordinate directory and
  complete field/topology pages to become the physics address space, not just
  the residency/presentation address space.

## Measured reason for the cutover

The current Long Dam generation-zero path grows 80 authored fluid bricks to 116
immediate-support bricks and then to all 1,152 logical-domain bricks. It packs
the resulting 102,792 accepted cells and 298,614 accepted rows into a library
containing every rung and 2:1 seam variant:

| quantity | accepted generation | compatibility library |
| --- | ---: | ---: |
| cells | 102,792 | 673,920 |
| rows | 298,614 | 2,112,048 |
| packed topology | 35.8 MiB | 266.0 MiB |
| construction time | 0.57 s | 35.32 s |
| transient JS heap | 109 MiB | 2.79 GiB |

The compatibility library exists because the current dynamic page owns only
cell geometry. It cannot publish rows, incidence, field slots, pressure edges,
or neighbor services. Lowering the host-template threshold therefore makes the
Long Dam front stop; raising it restores motion by prebuilding its future.

The cutover removes that choice rather than tuning the threshold.

## Runtime object model

### Sparse coordinate directory

The directory maps a signed brick coordinate and span to a physical tile slot.
It is an open-addressed, generation-stamped GPU hash table sized from tile-pool
capacity. No key is linearized through domain dimensions and no empty coordinate
has a record.

The accepted and candidate directory slots are distinct generations. Readers
use only the accepted generation. A clone transaction may mutate the candidate
directory and neighbor links, but publication is one generation flip after all
receipts seal.

### Physical tile page

A physical B8 page is complete enough to execute every paper map. Its record
contains:

- signed world coordinate, span, rung, generation, and lifecycle flags;
- field-page and presentation-page slots;
- accepted cell count and the local cell-address rule;
- six direct neighbor slots and their accepted rungs;
- six boundary programs, covering wall, sparse-air, same-rung, and 2:1 seams;
- row/face ranges, incidence or implicit-neighbor service, and pressure-edge
  ranges;
- classification, dirty-tile, transfer, and publication masks;
- parent/child aggregate handles used by the sparse pressure hierarchy.

Uniform brick interiors are arithmetic. Only boundary programs and exceptional
rows are materialized. The first implementation may compile those six programs
at clone time. A later implementation may replace compilation with cloning from
a finite device-global topology catalogue; the page ABI does not change.

### Field pages

Fields are indexed by `(tile slot, local cell/face address)`, never by a global
template cell ID. Cell and face slabs are capacity-shaped and contain only
physical page slots. A rung uses a prefix of the canonical B8 local address
space or a rung-specific compact layout; either choice is acceptable while the
number of slots remains proportional to tile-pool capacity.

Velocity sampling uses a tile-local halo populated from the page's direct
neighbors. The finest-domain face-velocity raster is not part of this ABI.

### Sparse pressure hierarchy

Aggregate and hierarchy nodes are allocated from pools when a resident tile
needs their coordinate path. Empty groups have no node. A tile clone patches
only its own aggregate, the six incident aggregate edges, and its ancestor
chain. No hierarchy level allocates `ceil(domain / scale)^3` groups.

### Presentation

Presentation pages are allocated for resident surface/receiver closure, not for
logical-domain coordinates. A retired tile releases its presentation page after
the accepted generation no longer references it.

## Clone transaction

One GPU transaction performs the following steps:

1. Compact unique requested coordinates from swept-front, injection, and 2:1
   closure producers.
2. Probe the accepted directory. Existing coordinates become ordinary rung or
   activity work; absent coordinates request a page.
3. Preflight tile, field, boundary-program, pressure, and presentation capacity
   for the whole closure. On failure, publish no part of the transaction.
4. Pop physical slots and write candidate coordinate/rung descriptors.
5. Initialize fields by exact zero/identity construction for an empty receiver,
   restriction from children, or prolongation/copy from a donor/parent.
6. Compile the six boundary programs and patch both sides' candidate neighbor
   links. Interior topology is selected arithmetically.
7. Build tile-local row/incidence/pressure services and sparse aggregate
   ancestry. Validate conservation, 2:1 grading, address uniqueness, and
   generation coverage.
8. Insert candidate directory records and seal per-tile receipts.
9. Flip the accepted directory/page generation once. Physics never observes a
   geometry-only page or a half-patched seam.

Retirement is the inverse transaction. A page is returned to a free list only
after no accepted directory, neighbor, hierarchy, presentation, or in-flight
frame generation references it.

## Phased cutover

Each phase changes production ownership; shadow-only structures do not count as
a cutover.

### C0 — Gate and fixed ABI

- Add a construction receipt distinguishing capacity, resident pages, active
  pages, and authored-fluid pages.
- Add the Long Dam gate: generation-zero resident pages equal authored-fluid
  pages exactly; logical-domain allocations are zero.
- Add the world-extent-invariance gate: translate those same pages to very large
  signed coordinates and prove that pool layout and bytes are identical.
- Make shader layouts capacity-independent: offsets live in runtime headers so
  one device pipeline family is reusable across scenes.

Exit: the gate exists and fails on the current receiver-domain implementation.

### C1 — Complete B8 receiver clone (immediate production cutover)

- Introduce the sparse coordinate directory and complete B8 page/field slabs.
- Seed generation zero directly from the authored 80 Long Dam fluid bricks.
- Convert point ownership, conservative transport, velocity support, pressure,
  projection, activity, and presentation to consume page-local services.
- Allocate swept receivers through the clone transaction before any transport
  writes can target them.
- Remove Long Dam from `dormantReceiverDomain` and
  `packResidentTopologyTemplates`; do not retain a fallback selector.

Exit: the Long Dam regression reaches its existing front checkpoints, every new
protected receiver is B8 in its first accepted generation, initial residency is
exactly the authored-fluid set, host simulation-sized work remains zero, and
construction performs no all-rung packing.

### C2 — Adaptive rerung and retirement

- Move restriction/prolongation and 2:1 closure onto the same page transaction.
- Allocate and retire sparse pressure ancestors with page lifecycle.
- Recycle dry pages behind the advancing front after dependency-safe grace.

Exit: long-run Long Dam residency follows the moving working set rather than
monotonically approaching traversed-world volume.

### C3 — Predefined topology cloning

- Intern the finite set of rung-interior and neighbor-rung boundary programs at
  device creation.
- Replace clone-time seam compilation with descriptor copies and coordinate
  patching.
- Cache the resulting scene-invariant shader/pipeline family per device.

Exit: clone latency contains no topology compilation and scene size does not
change shader source or pipeline keys.

### C4 — Remove transitional global services

- Delete the logical owner directory, TEI spatial owner plane, BTI finest-tile
  owner plane, finest-domain face-velocity support, domain-shaped pressure
  hierarchy, and per-template global field IDs.
- Delete host mutable-template thresholds and receiver-apron capacity controls.

Exit: a source audit contains no allocation formula based on logical brick
volume or finest-domain volume in Sparse CM12 production code.

## Long Dam acceptance receipt

The production regression records at least:

- authored-fluid tile count;
- generation-zero resident and active tile keys;
- physical pool capacity and high-water mark;
- per-step allocated, cloned, retired, and accepted tile keys;
- first accepted generation and rung for each clone;
- front positions at the existing trace, surface, and liquid thresholds;
- topology/page/directory/pressure receipt faults;
- host construction time, transient heap, and allocated GPU bytes;
- proof that no host template variant packer ran.

Required assertions:

1. Generation-zero resident keys equal authored-fluid keys.
2. Pool layout and bytes depend on physical capacity only; translating the
   resident set into a vast coordinate space changes neither.
3. Every coordinate ahead of the dam is absent until a GPU producer requests
   it.
4. Every newly protected receiver is published at B8 before transport targets
   it.
5. The front is monotone and reaches the existing Long Dam checkpoint.
6. Capacity high-water follows the live working set; no allocation equals the
   1,152-brick logical domain merely because the domain is bounded.
7. There are no validation, capacity, generation, seam, conservation, pressure,
   or presentation faults.

## Non-goals for C1

- Unbounded coordinate precision and multi-chunk buffer growth may follow once
  the Long Dam owns the correct lifecycle.
- Clone-time seam compilation is acceptable until C3.
- Bit-identical brick ordering is not required. Conservation, front motion,
  pressure/divergence, symmetry where applicable, and deterministic replay are
  required.
- A smaller dormant apron, dense owner page, or accepted-only host superset is
  not an intermediate success; each preserves the wrong ownership model.
