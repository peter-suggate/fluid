# VRD1 reverse VEX dependency integration gap

Status: truthful standalone scaffold. An initial accepted-CSR compiler,
immutable lookup library, candidate actual-read logger, transactional CPU oracle,
fixtures, and Naga gate exist. Live GPU canonical replacement and paged reverse
adjacency mutation do not. The resident does not import or allocate VRD1;
production behavior and FPA1 authority are unchanged.

VRD1 maps an exactly changed accepted VEX stable cell to every preparation row
whose accepted prior execution actually read that value. It does not treat
physical incidence or a geometric speed radius as a substitute:
`prepareTransportFaceRow` reads
VEX through adaptive trilinear `sampleVelocity` calls at the initial and RK2
midpoint characteristic queries, with up to sixteen substeps. HTP1 incidence
and pressure/extrapolation edges are exact topology assets but are not exact
trace-read assets, so they are deliberately not reused for this mapping.

The causality proof is local. If only a VEX cell outside a row's accepted prior
read set changes, every read and branch that chose the trace/sample positions is
bit-identical, so the row cannot redirect or change. If an old donor changes,
the row is already scheduled; that execution may redirect A→B and publishes the
new B dependency atomically with its result. The next frame therefore schedules
the row for a B-only change. Unexecuted rows inherit their accepted journal.

## Construction and acceptance boundary

Construction allocates the bounded double-bank CSR/journal capacity and may seed
an initial complete journal. Thereafter the execution accessor records incident
validity reads, probe owner/span dependencies, every valid interpolation-corner
xyz read, and the final sample reads. Result generation and dependency
generation must match before candidate acceptance. The journal carries topology
generation/hash and a hash of RK2 direction, clipping, adaptive-owner
interpolation, corner count, substep limit, and compiler version.
VRD1 rejects unsupported policy fields, a policy-hash mismatch, missing or
duplicate stable rows, invalid stable cells, generation mismatch, and edge
capacity overflow before publishing a candidate CSR.

The builder deduplicates each forward row read set, reverses it, and stores rows
in ascending stable order. The immutable header records edge count/capacity,
maximum rows per cell, byte accounting, topology/policy hashes, and a content
hash over the canonical CSR. Runtime lookup is read-only and bounded.

## Live blockers

- The live VEX accessor is not yet instrumented to append the four read kinds.
  Candidate storage should be chunked/bounded rather than a fixed ~802-read slab
  per row: worst-case row terms plus 32 probe/interpolation queries make that
  slab several GiB at ocean row counts. Chunk allocation may be unordered, but
  canonical seal must sort/dedupe stable cells/rows before acceptance.
- Candidate dependency storage and the candidate face-result authority must be
  accepted or rejected as one generation pair. Overflow/uncovered read faults
  preserve both prior accepted banks; there is no abandon-stale fallback.
- The standalone CPU transaction proves edge replacement, including retiring
  A when a scheduled row redirects A→B and scheduling that row for a B-only
  change on the next frame. Equivalent GPU canonicalization/add-remove mutation
  is not implemented here.
- A donor-4³-tile reverse graph may be substantially smaller while remaining
  safe through explicit false-positive coarsening. Do not fix its per-row K,
  page capacity, or replace the exact cell graph until an observational ocean
  census reports unique donor cells and donor tiles per executed row. A viable
  GPU design groups add/remove mutations by donor tile so one workgroup owns a
  subscriber list, preflights all required pages, then commits or preserves the
  prior accepted bank in full.
- The live FPE1 begin gate must require
  `fpeaVexReverseProvenanceValid()` before accepting any VEX-family event. The
  lookup returns `INVALID` ranges on a provenance fault, but callers must fault
  rather than interpret that as an empty dependency.
- Topology or trace/CFL policy changes require rebuilding VRD1 and publishing its
  new immutable hashes before the corresponding FPE1/FPA1 generation begins.
- Construction memory must be budgeted from `offsetBytes + rowBytes`; a chosen
  spare `edgeCapacity` is explicit, and overflow is a hard construction error.

Static gate:

```bash
node --import tsx tools/check-sparse-cm12-vex-reverse-dependency.ts
```
