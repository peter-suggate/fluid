# Uniform Geometric fixed-resolution pages: first milestone

Follow-up: [production scratch-page integration](uniform-volume-pages-production-2026-09-21.md).
The scope below describes the earlier addressing experiment.

## Scope

Implemented an experimental host residency planner and a GPU seven-point
stencil probe for 16³ and 32³ pages. Production Uniform Geometric and Sparse
CM12 are unchanged. This is an addressing/performance experiment, not a paged
fluid solver or a certified MiniDam64 regression result.

The design carries over Sparse CM12's signed-coordinate directory, stable
physical leaf slots, and arithmetic interior / explicit seam distinction. It
reuses the shared signed spatial hash. It does not import the adaptive topology
catalogue or change its allocator. Planning is currently host-owned; GPU-owned
frontier allocation remains future work.

## Implemented contract

- Requested coordinates compile into a collision-resolving directory, compact
  active-slot list, and six cached face neighbors per physical page. No arrays
  scale with the bounding box between pages.
- Retained coordinates keep their physical slots through growth and reorder.
- Capacity failure leaves the previous plan untouched. Slots released by a
  transition cannot be reused within that same transition.
- New/released slot lists are explicit. The planner does not initialize GPU
  fields, submit commands, or fence lifetime. A caller must wait for every old
  consumer before reusing a hole from an earlier generation. This includes
  presentation users. These candidates are not production publication objects.
- GPU stencil dispatch uses 4³ workgroups within larger storage pages, with
  arithmetic local offsets and cached neighbor loads only at page seams.
- Missing pages mean zero Dirichlet values only in this test operator. This is
  not a fluid boundary rule and must not be inherited by production pressure.

The scalar stencil uses storage buffers in both arms; current production
Uniform Geometric also uses textures, so this does not measure the eventual
texture-to-buffer tradeoff. No halos are copied in the experiment.

## Measurements

Final capture: 40 measured samples after five warmups, each applying the same
operator to the same input 32 times in one compute pass. Dense / paged / paged /
dense order for each page size. Hardware timestamps exclude compilation and
readback, and are divided by dispatch count. Both arms represent the same 64³
scalar field at the same world origin. Physical page allocation order is
reversed to prevent reliance on spatial slot adjacency.

| Page edge | Dense GPU ms/application | Paged GPU ms/application | Time change |
| --- | ---: | ---: | ---: |
| 16 | 0.024211 | 0.025438 | +5.07% |
| 32 | 0.024254 | 0.023472 | −3.23% |

Entries average the two run medians, not the samples across runs. The original
20-sample pilot observed approximately +6.1% / −3.1%. Both captures have large
timing outliers; this is not a confidence interval or a portable speed claim.
The final raw samples, adapter identification and layout/shader source hashes
are in [uniform-pages-2026-09-21.json](uniform-pages-2026-09-21.json).

Both packed arms reserve 1 MiB per scalar field (input and output each have one
field). Planned directory + neighbor + active-list metadata is 3,840 bytes for
64 B16 pages or 480 bytes for eight B32 pages. The hash directory is host-side
in this probe; only neighbors and the active list are uploaded. Sparse fixtures
place a page one million page coordinates away without increasing capacity.

32³ is promising for local stencil execution. This does not settle resident
memory around thin streams, where larger pages can waste more space, or the
95% full-scene throughput requirement. Repeated same-input dispatches have a
cache-friendly working set; the actual pressure hierarchy and its dependencies
are not represented. The kernel uses one dispatch with seam branches, not a
separately dispatched interior/seam specialization.

## Verification

- `npm run test:uniform-pages`: four tests pass. Covers signed floor division,
  deliberate hash collisions, reciprocal neighbors, stable slots, all-or-nothing
  capacity failure, retirement/reuse, separated pages, and invalid inputs.
- `npm run benchmark:uniform-pages -- --samples=40 --out=docs/research/uniform-pages-2026-09-21.json`:
  all active outputs exactly match independently addressed CPU stencils. Checks
  2,392,064 cell outputs across full domains, disconnected signed pages, holes,
  and reused slots; outputs start as NaNs. No uncaptured GPU validation errors.
- Repository TypeScript check reports 15 errors in unrelated existing files;
  no errors in the new files. `git diff --check` passes.
- The canonical Sparse CM12 suite was not triggered: no production simulation,
  sparse-world topology, terrain, presentation or live-edit path was changed.

## Next milestone

1. Measure stage-support residency on actual hose/pond trajectories for B16/B32,
   including independent V, phi, characteristic samples, terrain and sources.
2. Integrate a fixed-resolution paged pressure operator with real coefficients,
   cross-page global coupling, multigrid transfer and residual checks. Compare
   identical captured inputs with current Uniform Geometric.
3. Add GPU directory allocation and field initialization under a publication
   transaction, with queue/presentation lifetime protection and capacity faults.
4. Benchmark complete MiniDam64 advances at fixed numerical settings against a
   frozen current baseline. Required throughput is at least 95% of baseline;
   report growth-frame latency separately. This milestone does not pass that gate.
