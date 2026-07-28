# Power-liquids mini dam: 5x GPU plan

Date: 2026-07-28. Target device: Apple M1 Max. Target lane:
`benchmark:power-dam-moving-interface`.

## Result to optimize

The current working tree measures **38.11 ms/advance** over 62 moving-interface
advances. The counter capture measures **36.75 ms GPU-busy** and **1.59 ms of
gaps** inside a 38.34 ms frame. The live fine band is **4087/4096 bricks
(99.8%)**, pressure converges in **4 of 10** encoded outer iterations, and the
frame encodes about **1654 dispatches in 168 compute passes**.

A strict 5x target from the current tree is **7.62 ms/advance**. Removing every
GPU gap would only reach 36.75 ms, so queue overlap, pass merging, and dispatch
deletion cannot deliver the target. The work inside the kernels must fall by
about 80%.

Approximate current GPU-busy ledger from the supplied/captured task spans:

| subsystem | current ms | target ms | required change |
|---|---:|---:|---:|
| pressure / MGPCG / Section 4.3 / A2 | 8.8 | 1.8 | materialized operator plus recycled solve |
| JFA-CPT, fine volume, fine publication | 6.6 | 1.4 | temporal closest-point repair |
| topology and structured publication | 6.2 | 0.9 | delta-only products and phase-packed work |
| fine advection | 3.9 | 0.8 | resident dense-address fast path |
| two Section 5 air-support builds | 6.1 | 1.2 | persistent identity graph plus wide value refresh |
| remaining work and unavoidable gaps | 5.2 | 1.5 | cached authorities and fewer global phases |
| **total** | **36.8** | **7.6** | **4.8x GPU-busy, 5x wall** |

This budget is deliberately hard. A design that only improves one row cannot
meet it.

## Architecture

### 1. Materialize the pressure graph once per accepted topology

The hot A2 and Section 4.3 kernels repeatedly derive the same row relationship
through `pageSlot`: brick record, two masks, rank, physical slot, owner, then
coefficient direction. Those dependent loads are repeated for every apply,
every smoother wave, and every CG iteration.

Publish a compact, generation-keyed operator image alongside accepted
topology:

- `direct[row][18] = {otherRow, coefficient}` in canonical channel order;
- `fineAdjoint[row] = {offset, count}` into a canonical ordered edge array;
- boundary diagonal and transition class per row;
- page/block row lists for the smoother.

Build only dirty rows. Unchanged rows retain their previous bank. A2 then does
one coalesced edge load per term and the existing ordered row fold; it performs
no page lookup, rank, owner, catalog-direction scan, or geometry reconstruction.
The same image must feed Section 4.3 so the publication cost is amortized over
all pressure work.

After materialization, add a **recycled pressure solve**. Carry the last two
accepted pressure/search subspace vectors by stable row identity, project the
new residual into that space, then run ordinary MGPCG. The safe first target is
four outer iterations to two without changing the stopping criterion. This is
a Gate-B numerical change; non-convergence remains fail-closed.

Do not revive the one-workgroup persistent solve. It reduced launches but
measured 15.1 ms slower because it used roughly one GPU core. The production
shape remains many row/page workgroups.

### 2. Turn cold JFA into temporal closest-point repair

The current redistancer deliberately discards the closest-point field, seeds
from sign changes, then runs seven global 27-tap floods. Persist instead:

- closest-point world coordinate;
- stable seed edge key and quantized edge fraction;
- seed generation/valid bit;
- per-page `newSeed`, `retiredSeed`, and `repairRequired` masks.

During A-to-B page publication, carry valid closest points by logical sample.
Validate each carried seed against the new endpoint signs. Scatter invalidation
from retired seed keys, insert new seeds, and compact only repair pages plus a
bounded halo. Run local strides `8,4,2,1,+1` on that set. Fall back to the full
seven-pass cold path if the repaired set exceeds an authored fraction or if
validation finds an unresolved sample.

The important cache is the closest-point result, not a large workgroup page
directory. A 7x7x7 shared directory prototype reduced residency and regressed
the lane from 38.11 to 40.35 ms.

### 3. Add a saturated-residency fine-grid specialization

At 99.8% brick occupancy, the sparse representation pays validation and
physical-page indirection without saving work. Keep sparse storage as the
authority, but publish a `denseResident` bit when every logical brick in the
required transport/redistance support is present and valid.

The dense kernels use logical sample index arithmetic and a direct
logical-brick-to-page table loaded once per brick. They retain the same f32
payloads, boundary rules, and evaluation order. Specialize:

- fine semi-Lagrangian departure sampling;
- JFA/repair neighbor samples;
- volume overlap reduction;
- fine-to-coarse restriction.

The sparse kernels remain the fallback. The cutover is GPU-authored from the
published worklist header, so a stale host decision cannot select an invalid
path. This specialization is for this mini dam-break case; large sparse scenes
continue to use SPGrid.

### 4. Separate stable graphs from per-frame values

Several low-occupancy tasks rebuild identities and relationships although only
values changed:

- Section 5 closest-face identities and adjacency;
- structured boundary row/family classes;
- row-to-catalog-slot/channel mappings;
- fine/coarse restriction ownership;
- topology owner-page and halo directories.

Key each graph by its actual topology authority, not the advance number. On an
unchanged authority, refresh only velocities, phi-derived apertures, and other
dynamic values. On a changed authority, rebuild dirty rows/pages and retain the
other bank's records. In particular, the two required Section 5 publications
need different velocity values but can share one stable identity graph.

### 5. Phase-pack independent narrow work

WebGPU exposes one queue and no portable cross-workgroup grid barrier. It cannot
overlap arbitrary existing passes into a 5x win. It can, however, stop launching
one tiny grid at a time.

Build a GPU-authored task list and use one wide dispatcher per dependency phase:

1. classify dirty topology rows, boundary rows, fine pages, and air identities;
2. scan/compact all products;
3. scatter/rebuild independent products;
4. validate and commit.

Each workgroup claims a typed task record and runs the relevant phase body. This
co-schedules many small worksets across the GPU without using a single
workgroup, while retaining real storage-to-indirect pass boundaries. It targets
the 0.1-2% occupancy publication spans, not the already 50% occupancy JFA
floods.

## Delivery order and gates

1. **Pressure graph differential harness.** Apply gathered and materialized
   operators to identical vectors; require bit-identical row results. Then
   measure the 62-step lane. Stop unless pressure falls below 4 ms.
2. **Closest-point carry and repair.** A/A determinism first, then Gate B over
   500 steps. Record cold-fallback frequency and repair-page fraction. Stop
   unless JFA/fine publication falls below 3 ms.
3. **Dense-resident fine specialization.** Differentially compare sparse and
   dense kernels on the same generation. Target fine advection below 1.2 ms.
4. **Stable Section 5 and boundary graphs.** Force topology rejection and
   verify fallback authority. Target both support builds below 2 ms combined.
5. **Phase packing.** Apply only after kernel work is reduced; it is the final
   occupancy multiplier, not the first optimization.
6. **Pressure recycling.** Enable only after the materialized operator is the
   default. Gate the full 500-step physical envelope and retain an immediate
   ordinary-MGPCG fallback.

Every throughput comparison must be interleaved A/B on an otherwise idle GPU.
Every restructuring change uses bit-exact Gate A. Closest-point repair and
recycled Krylov state use Gate B, with zero validation errors, zero topology
rollback, zero unaccepted restriction rows, and converged pressure on every
executed step.

## Experiments already rejected

- **Full 7x7x7 workgroup page/owner cache:** 38.11 -> 40.35 ms. Shared-memory
  residency loss exceeded lookup reuse.
- **Eight-entry per-invocation terminal-phi cache:** 38.11 -> 38.26 ms. The
  existing direct directory and hardware cache already cover it.
- **One 256-lane adjoint workgroup per transition row:** 38.11 -> 38.27 ms.
  The fine-adjoint span is limited by dependent address chains, not its
  child-by-direction loop.
- **One-workgroup persistent MGPCG:** 79.7 -> 94.8 ms in the earlier isolated
  comparison despite removing hundreds of launches. Parallelism matters more
  than dispatch count on this GPU.

These failures all point to the same conclusion: cache **published results and
stable relationships across frames**, not small lookup fragments inside one
invocation, and keep enough independent row/page workgroups to occupy the GPU.
