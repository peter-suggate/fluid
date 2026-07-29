# Power-liquids mini dam: 5x GPU plan

Date: 2026-07-28. Target device: Apple M1 Max. Target lane:
`benchmark:power-dam-moving-interface`.

## Result to optimize

The optimization baseline measured **38.11 ms/advance** over 62 moving-interface
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

### 1. Compiled pressure graph -- landed and measured

The hot A2 and Section 4.3 kernels repeatedly derive the same row relationship
through `pageSlot`: brick record, two masks, rank, physical slot, owner, then
coefficient direction. Those dependent loads are repeated for every apply,
every smoother wave, and every CG iteration.

The accepted-topology commit now publishes two index-only operator images:

- `direct[row][0..18]`: one row status plus eighteen destination-row/report
  codes in canonical channel order;
- `fineAdjoint[row][0..143]`: eight children by eighteen candidate directions,
  encoded as destination row plus coefficient channel, or the exact report
  code the old traversal would have produced.

Both images are rebuilt immediately after an accepted hierarchy commit and are
then reused by every A2 apply until the next accepted commit. A rejected
candidate cannot overwrite them. Coefficients remain in the accepted Section
6.3 bank: storing only addresses preserves the original coefficient loads, FMA
shape, staged-term indices, and ordered row fold. The production apply performs
no page lookup, brick-mask rank, physical-slot indirection, owner lookup, or
catalog-direction scan. The old chase remains behind measurement-only
environment flags so this cost can be priced in the same binary.

On the 4087/4096-brick moving-interface lane, interleaved uninstrumented A/B
runs measured **36.90 ms/advance with the images versus 37.40 ms with both
chases**, a repeatable **0.50 ms/advance (1.3%) whole-frame saving**. An
identically isolated xctrace comparison measured the repeated direct, adjoint,
and ordered-fold stages at **4.353 ms cached versus 9.831 ms chased**
(**-5.478 ms, -55.7%**); direct fell 72.0%, adjoint 59.5%, and the unchanged
fold was flat within noise. The isolated numbers intentionally amplify these
stages and must not be added to the whole-frame result. A WebGPU timestamp
capture put the complete pressure phase at **10.09 ms cached versus 11.85 ms
chased**. Therefore the page-slot traversal was the inner loop of operator
staging, but removing it is not most of the complete pressure solve or frame.

This closes the address-materialization item. Further pressure work must target
the remaining cached adjoint gather/staging, ordered folds and V-cycle work, or
reduce the number of applies with the recycled solve below.

After materialization, add a **recycled pressure solve**. Carry the last two
accepted pressure/search subspace vectors by stable row identity, project the
new residual into that space, then run ordinary MGPCG. The safe first target is
four outer iterations to two without changing the stopping criterion. This is
a Gate-B numerical change; non-convergence remains fail-closed.

Do not revive the one-workgroup persistent solve. It reduced launches but
measured 15.1 ms slower because it used roughly one GPU core. The production
shape remains many row/page workgroups.

### 2. Halve the symmetric Section 4.3 shell -- landed and measured

The paper's general `k≈8` boundary-smoothing choice was overly conservative for
this small domain. Keeping the preconditioner fixed and symmetric but reducing
both matching halves to `k=4` cuts merged-band A2 applies from 15 to 7 per
preconditioner application. It does not interlace row solvers or change A2;
ordinary MGPCG still owns the full domain and the same `1e-4` residual gate.

An even-depth sweep measured `k=8/6/4/2` at **37.92/37.00/36.29/37.56
ms/advance**. The corresponding terminal outer counts were `4/4/5/8`: `k=4`
is the knee, while `k=2` repays the saved shell work with extra Krylov steps.
Two more interleaved pairs measured **37.90/36.37** and **37.82/36.32 ms** for
`k=8/k=4`, a repeatable **1.5 ms/advance (4.0%) saving**. Dispatches fell from
1656 to 1304 per advance.

Timestamp isolation reduced the three repeated merged-band stages from **3.520
to 1.785 ms/advance (-49.3%)**; full-domain A2 stages remained flat. Matched
240-step xctrace controls measured **37.04 ms at k=8 versus 35.94 ms at k=4**.
The xctrace per-stage frame anchors differed, so only those clean controls—not
the traced per-stage totals—are used for the A/B claim.

The two-step Dawn smoke and Section 4.3 shader test pass. Across the 500-step
mini-dam smoke, every k=4 pressure solve converged in 3--8 iterations; k=8 used
3--8 as well. Both configurations trip the same pre-existing long-run physical
and raster gates, with comparable maximum pressure residuals, so that run does
not establish a shell-depth regression. Production now selects `k=4` only for
the measured two-level profile (`maximumLeafSize=2`, dimensions no larger than
16 cubed); deeper or larger profiles retain the paper-backed `k=8`. Strict even
depths 2--16 remain selectable through `FLUID_OCTREE_SECTION43_SHELL_DEPTH` for
validation and rollback.

### 3. Turn cold JFA into temporal closest-point repair

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

### 4. Add a saturated-residency fine-grid specialization

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

### 5. Separate stable graphs from per-frame values

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

### 6. Phase-pack independent narrow work

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

1. **Pressure graph differential harness -- complete.** Direct and fine-adjoint
   codecs cover every edge and failure code, and Dawn applies both addressings
   to identical vectors with bit-identical staged terms. The cache is retained,
   but the measured complete pressure phase remains above the 4 ms gate.
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

- **Per-row regular/adaptive A2 interlacing:** Two bit-identical variants were
  measured against the cached eighteen-lane operator. Serially staging the six
  Cartesian terms in one regular-row lane regressed the moving-interface lane
  from 37.87 to 39.05 ms and raised the isolated merged direct stage from 0.957
  to 2.669 ms. Fusing the complete seven-point regular row and bypassing its
  term arena/fold improved that prototype, but still regressed 37.89 to 38.69
  ms; its direct stage was 1.666 ms and the fold was effectively unchanged
  (1.253 versus 1.219 ms). The three-layer Section 4.3 shell on this 16-cubed
  case contains too little profitable regular interior, while one-row lanes and
  mixed-class lookup sacrifice the occupancy of the wide cached gather. Keep
  one global operator and the existing 18/144-lane staging on this lane.
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
