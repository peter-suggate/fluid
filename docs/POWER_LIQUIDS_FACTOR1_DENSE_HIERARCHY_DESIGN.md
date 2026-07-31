# Factor-1 dense-indexed pressure hierarchy

Status: concrete prototype design following
`POWER_LIQUIDS_COARSE_5X_WALLCLOCK_PLAN.md`. This is a factor-1 specialization;
the existing sparse factor-4/8 hierarchy remains the fallback and correctness
oracle.

Implemented result (2026-07-31): the transactional dense shadow and recurring
Stage C correction are production-wired behind
`FLUID_OCTREE_FACTOR1_DENSE_MG=0`. Two clean 500-step paired runs measured
21.534/21.354 ms per dense advance versus 23.308/23.274 ms for the sparse
fallback. Both arms had identical terminal counters and zero validation
errors. Dense correction removes 55 MGPCG records per advance (839 to 784),
for a paired mean wall saving of 1.847 ms (7.9%).

Subsequent factor-1 work kept degree-four Chebyshev smoothing as an explicit
experiment while restoring degree two by default for the established mini-dam
impulse response, and made the existing inline accepted-row A2 kernel the
ordinary-apply default. Inline A2
keeps the convergence gate, replaces its three staged body dispatches with one
dispatch, and leaves residual A2 staged. It removes another 24 dispatches per
mini advance. Clean A/B measurements were 20.538/21.212 ms inline versus
20.624/20.690/21.402 ms staged on the 500-step moving lane, and 31.183 ms
inline versus 31.329 ms staged on the 240-step hydrostatic lane; all reported
zero validation errors. `FLUID_OCTREE_FACTOR1_SERIAL_ACCURATE_A2=0` restores
the staged ordinary apply.

The same inline row kernel now defaults on for Section 6.3 merged-band A2.
It replaces each direct/adjoint/fold triple with one merged-workset dispatch.
A balanced 300-step `on/off/off/on` sequence measured 23.203/23.110 ms inline
versus 23.587/23.957 ms staged, a 0.615 ms mean saving (2.6%), while deleting
154 dispatches per advance. The 240-step hydrostatic pair measured
35.175 ms inline versus 36.063 ms staged with identical two-iteration
convergence and zero errors. Set
`FLUID_OCTREE_FACTOR1_INLINE_MERGED_BAND_A2=0` for the staged fallback.

An explicit compiled-image inline variant
(`FLUID_OCTREE_FACTOR1_COMPILED_INLINE_A2=1`) was also tested. It fits the
portable ABI (eight storage buffers for ordinary rows, nine for merged-band)
and removes topology/state searches, but streams the 19-word direct and
144-word adjoint images through each hot row. A balanced 300-step
`on/off/off/on` sequence measured 25.417/25.397 ms compiled versus
23.703/23.897 ms for inline chase, with identical outputs and command counts.
The 1.61 ms mean regression refutes the assumption that fewer dependent
lookups are necessarily more cache-efficient; the image working set costs
more here than the compact chase. The compiled arm therefore remains off.

Dense native correction publication is also folded into the final level-0
post-smooth/tail kernels. This preserves the stored Chebyshev value and reduces
the recurring dense correction from 24 to 23 dispatches (degree two: 16 to
15). Separately, topology ready-commit and owner-page publication now share
their semantically continuous passes, reducing the ready-flip prefix from five
compute passes to two while retaining its storage-to-indirect visibility
boundary. A second storage-only compaction removes another four passes on a
normal advancing step: advection staging to commit, topology transfer to
reconstruction, and march to reconstruction at both Section 5 call sites. A
balanced 300-step sequence measured 23.890/23.607 ms compact versus
23.700/24.117 ms with the legacy boundaries, a 0.160 ms mean saving, with
identical outputs/dispatches and exactly 136.02 versus 140.02 passes per
advance.

The host-side dead-tail predictor is implemented but remains explicit-on via
`FLUID_OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL=1`. With an immediately adjacent,
converged, unchanged-topology observation, a prior four-iteration solve plus
the two-iteration margin would encode six iterations. At the current graph
that removes four encoded outer iterations. The exact record count must be
remeasured under the corrected k=8 mini shell. Normal throughput deliberately
receives none of this saving: its snapshots are mapped only at
sampling/end points, not before the
next encode, so the adjacency guard restores the full ten-iteration envelope.
Adding a recurring map would introduce the CPU/GPU synchronization the plan
explicitly rules out. A real shipping dead-tail saving therefore needs a
GPU-side bundled/rescue graph rather than a host predictor.

Degree-two smoothing remains an A/B arm rather than the default. It reduced
the mini lane from 20.624 to 20.488 ms by deleting 88 dispatches, but raised
the final solve from three to four iterations. On the hydrostatic lane it
raised the solve from two to three iterations and regressed wall time from
31.329 to 32.038 ms. Likewise, combined reduction drains remain explicit-on:
they removed 20 MGPCG dispatches but regressed the mini wall by about
0.422 ms. These results supersede the old assumption that fewer encoded
records necessarily means a faster factor-1 solve: after live indirect shaping,
extra live arithmetic and barriers can dominate record savings.

## 1. Decision

Use a **dense coordinate address with sparse semantics**, not a dense solve.
For level `l`,

```text
d_l       = ceil(domainDimensions / 2^l)
levelBase = exclusive prefix sum(product(d_l))
slot(l,q) = levelBase[l] + q.x + d_l.x * (q.y + d_l.y * q.z)
```

The slot is therefore arithmetic and stable for the lifetime of the domain.
`flags[slot] == 0` means absent. An occupied slot has exactly one of
`ACTIVE`, `GHOST`, or `MG_ONLY`.

This x-fast slot is the **semantic serialization used by CPU/GPU differential
tests**, not an immutable physical GPU ABI. The first GPU prototype should use
x-fast structure-of-arrays with explicitly versioned channel bases and
strides, but may change physical layout without changing the coordinate-keyed
oracle. If vector channels share one buffer, pad each channel stride to 64
floats so every channel begins on a 256-byte boundary.

This deliberately removes from the recurring factor-1 correction:

- open-addressed keys and hash probes;
- brick mask/rank directories and ranked-slot vectors;
- physical-page records used only to turn a coordinate back into a slot;
- published 18-column stencil indices;
- transfer records and parent linked lists;
- level row maps (a row's native slot is derived from its geometry);
- capacity-sized candidate worklists.

The captured hierarchy census is `[1475, 214, 35, 8, 1]`: 1,733 of 4,681
slots (37.0%) are occupied. That originally suggested compact work selection,
but the Metal cache microbenchmark refuted it for the wide recurring kernels.
At workgroup size 64, a direct x-fast full-volume six-face apply measured
32.768 µs versus 40.960 µs through the sorted occupied worklist. Full-volume
prolongation also won its measured pair. The saved invocations do not repay
the index load and indirect address on this 16³ domain.

The production executor therefore uses direct x-fast full-volume WG64 scans
for level-0/1 smoothing and prolongation, with a `flags == 0` early exit.
Initialization, publication, and parent-owned restriction still use sorted
occupied worklists where they avoid unrelated row/vector work or define the
deterministic parent fold. `tools/benchmark-dense-hierarchy-cache.ts` retains
the exhaustive A/B harness and coordinate hashes.

Do not copy the existing 8x8x4 sparse-page swizzle into this ABI. It pads the
mini hierarchy to 5,376 cells (+14.85%), and levels 2–4 alone consume 768
physical cells for 73 logical cells. Its original advantage came from sparse
page/directory lookup, which this design removes. Consecutive x-fast lanes
still issue contiguous streams for ±Y and ±Z because every lane applies the
same fixed address offset. The Metal sweep found logical 8x8x4 shared staging
consistently slower: halo loading and barriers dominate this mini lane.
Follow-up Metal sweeps also refuted logical 4x4x4 transfer scheduling over
x-fast storage: at WG32, restriction measured 49.15 µs versus 15.02 µs for
the worklist, and prolongation measured 16.38 µs versus 7.08 µs. Coordinate
reconstruction and scanning the tile erase the locality benefit seen in a
physical 4³ swizzle, while the physical swizzle still hurts the stencil.

## 2. Mini-lane sizing

For a 16x16x16 domain and 4,096-row capacity:

| level | dimensions | dense slots | current hash capacity |
|---:|---:|---:|---:|
| 0 | 16x16x16 | 4,096 | 8,192 |
| 1 | 8x8x8 | 512 | 1,024 |
| 2 | 4x4x4 | 64 | 128 |
| 3 | 2x2x2 | 8 | 16 |
| 4 | 1x1x1 | 1 | 2 |
| total | | **4,681** | **9,362** |

The current plan allocates 2,704,388 bytes of topology and 973,648 bytes of
state for each accepted hierarchy. Candidate topology/state duplicates those
large buffers, so accepted plus candidate consumes 7,356,072 bytes before
ghost scratch.

The factor-1 rediscretized M1 stencil stores only six nonzero face
coefficients, and every present face coefficient is the uniform
`cellWidth * 2^l`. The twelve edge channels are always zero. Consequently the
dense representation needs no off-diagonal coefficients or column indices:

```text
acceptedFlags     u32[4681]   18,724 bytes
acceptedOwner     u32[4681]   18,724 bytes
acceptedDiagonal f32[4681]   18,724 bytes
candidateFlags/Owner/Diagonal             56,172 bytes
rhs/a/b           f32[3 * 4681]           56,172 bytes
spectralUpper     f32[5]                      20 bytes
```

The core accepted/candidate M1 image is therefore **168,536 bytes
(164.6 KiB)**, versus 7,356,072 bytes (7.02 MiB) for the present
accepted/candidate topology and state buffers: a **7,187,536-byte
(6.85 MiB, 97.7%) reduction**. The accurate A2 row and adjoint images remain
in the first prototype and are not included in either figure.

Two full-capacity occupied-index vectors plus per-level counts add only 37.5
KiB; one accepted vector contains about 6.8 KiB of live indices on the captured
mini hierarchy.

## 3. Mapping the accepted octree rows

The dense candidate builder consumes the same transactional sources as
`WebGPUOctreeSPGridVCycle`: captured row geometry, topology metrics, accepted
Section 6.3 coefficients, and the exact row delta.

For accepted row `r`:

1. Decode finest-cell origin `o` and `native = ctz(rowGeometry[r].size)`.
2. For every `l >= native`, address `q = min(o >> l, d_l - 1)`.
3. At `l == native`, publish `ACTIVE` and `owner = r + 1`.
4. At `l > native`, merge `MG_ONLY` unless a higher-priority class already
   occupies the coordinate.

Role priority stays `ACTIVE > GHOST > MG_ONLY`, exactly as `cMergeClass`.
Parallel publication uses a packed atomic role word or separate
`atomicMax(role)` and `atomicCompareExchange(owner, 0, r + 1)`. Two nonzero,
different owners for one coordinate report the existing candidate error. The
valid octree invariant means this is an error path, not a scheduling choice.

After active/MG publication, reuse the existing row-parallel ghost predicate:
for each nonzero Section 6.3 channel, form its transformed contact coordinate;
if the fine coordinate is absent and its direct-addressed parent is `ACTIVE`,
publish `GHOST` with that parent's owner. Repeated identical ghost proposals
are idempotent.

No row map is stored. Seeding and publication recover the native dense slot
from the row's immutable geometry. A ghost's owner remains available for
Section 4.2 propagation and accurate-image construction.

## 4. Matrix-free first-order operator

At an occupied dense slot `(l,q)`:

```text
y(q) = diagonal(q) * x(q)
for direction in [+x,-x,+y,-y,+z,-z], in the existing channel order:
    n = q + direction
    if n is in bounds and flags(slot(l,n)) != 0:
        y(q) -= (cellWidth * 2^l) * x(n)
```

This is the exact operator currently published by
`buildCandidateStencils`:

- only its first six channels can become nonzero;
- every such coefficient is the same per-level scalar;
- a coefficient is nonzero exactly when direct lookup finds an occupied
  neighbour;
- `diagonal` retains the accepted L2 free-surface reaction and the existing
  missing-neighbour/MG-only terms.

The channel loop order must remain `+x,-x,+y,-y,+z,-z`; this preserves the
per-row floating-point expression. The slot numbers change, but slot numbers
are not numerical operands.

`initializeDenseCorrection` can replace all of `clearCorrection`, the five
`zeroVectors` dispatches, and `seedRhs`:

- every dense invocation clears its three vectors;
- every active native slot loads its owner's input RHS;
- every active owner clears its output correction.

This reduces a five-level correction from 30 encoded dispatches to 24 before
any larger fusion:

```text
gate 1 + initialize 1
+ (pre 4 + restrict 1 + prolong 1 + post 4) * two wide levels
+ fused level-2-to-bottom tail 1
+ publish 1
= 24
```

The final post-smooth dispatch at each wide level and the tail can eventually
publish their native active owners directly, deleting the separate publish
dispatch and reaching 23. That fusion is optional because it is less isolated.

## 5. Aggregate transfer without records

For every occupied fine coordinate, factor-1 transfer is

```text
parent(l,q) = slot(l + 1, q >> 1)
P[fine,parent] = 1
```

Prolongation is a single parent load:

```text
a_l(q) += a_(l+1)(q >> 1)
```

Restriction is parent-owned. One parent invocation visits its in-bounds
children in the fixed order

```text
(0,0,0), (1,0,0), (0,1,0), (1,1,0),
(0,0,1), (1,0,1), (0,1,1), (1,1,1)
```

and includes only occupied children. It adds each child's residual to the
already seeded native parent RHS. No atomic, transfer count, weight load,
head/tail load, or pointer chase remains.

This order is deterministic but is not promised bit-identical to the old hash
slot/record order. It represents the identical matrix `P`; using the same
unit mapping in both directions guarantees `R = P^T`. The hydrostatic
bit-stability gate must still pass because its represented residual is exact.
Transient trajectories use the existing quality tolerances, not an artificial
slot-order byte comparison.

## 6. Accurate A2 integration

Do not rewrite the numerical A2 expression in the first prototype. Keep its
19-word direct row image and 144-word adjoint row image and only change how
their epoch builders resolve a coordinate:

```text
old: page neighbour -> brick occupancy -> rank -> sparse slot -> owner
new: bounds check -> dense slot arithmetic -> flags/owner
```

The compiled image format, apply kernels, coefficient loads, term staging, and
fold order remain unchanged. This isolates the hierarchy-layout change from
the second-order numerical operator. Once differential tests show the two
image builders produce identical row/code images, the sparse page, directory,
and column-index storage can be retired.

This is also why a stand-alone new solver class is preferable to adding more
branches to the generic WGSL. Suggested integration:

- add `WebGPUOctreeFactorOneDenseVCycle` in a new file;
- implement `OctreeFirstOrderSPDVCycle` and expose the same accurate-operator
  interface;
- construct it in `webgpu-octree.ts` only when global `fineFactor === 1`;
- retain `FLUID_OCTREE_FACTOR1_DENSE_MG=0` as the A/B fallback;
- leave `WebGPUOctreeSPGridVCycle` unchanged for factor 4/8.

The implemented Stage B/C seam keeps sparse setup and accurate A2
authoritative while publishing a separate, versioned dense arena. The dense
correction replaces only `encodeCorrectionGate`/`encodeCorrectionBody` when
`FLUID_OCTREE_FACTOR1_DENSE_MG` is enabled. This preserves the exact sparse
fallback without encoding both recurring graphs.

## 7. Setup graph

Keep the existing capture and unchanged-input authority. Replace the 17 sparse
candidate phases and five broad publication phases with:

1. clear dirty dense candidate ranges;
2. publish active/MG roles directly from rows;
3. publish direct-addressed ghosts;
4. publish aggregate parent closure for levels 0->1, 1->2, 2->3, 3->4
   (one ordered dispatch boundary per dependency);
5. build diagonals over all dirty levels;
6. reduce/publish spectral bounds;
7. validate roles, owners, bottom count, positivity, and finite values;
8. copy or bank-swap the accepted dense metadata;
9. finalize lifecycle and fingerprint inputs;
10. compile the unchanged A2 direct/adjoint row images through dense lookup.

The conservative graph encodes about 20 setup dispatches versus the current
31. A bank swap can remove the metadata copy after validation, but should be a
second step because all consumers must select the same accepted bank.

Setup dispatch savings alone are not the thesis. The recurring benefit is
six fewer dispatches per V-cycle plus materially cheaper stencil and transfer
kernels. With the observed three-iteration solve, measure the number of
V-cycle applications from command accounting and multiply it by six to set the
command-count acceptance gate before implementation.

## 8. Correctness invariants

The prototype must assert:

1. Dense dimensions and bases cover each domain coordinate exactly once and
   never overlap between levels.
2. An occupied cell has exactly one role.
3. `ACTIVE` and `GHOST` have one valid owner; `MG_ONLY` has owner zero.
4. Every accepted row has exactly one native active cell, recovered
   arithmetically from geometry.
5. Every occupied non-bottom cell has an occupied `q >> 1` parent.
6. Every ghost's aggregate parent resolves to the same owner relation as the
   sparse oracle.
7. The bottom contains exactly one occupied cell.
8. Direct M1 diagonal and six neighbour-presence decisions equal the sparse
   oracle for every occupied coordinate.
9. Dense restriction and prolongation are exact transposes.
10. Pre/post Chebyshev schedules remain reverse-paired with the same
    transactional spectral bound.
11. A rejected candidate cannot change the accepted bank or its A2 images.
12. Factor 4/8 never select this executor.

The strongest differential artifact is a per-generation CPU/GPU dump keyed by
`(level,coordinate)`, comparing role, owner, diagonal, six neighbour-presence
bits, native row mapping, and aggregate parent. This avoids treating sparse
slot numbers as semantic.

## 9. Staged prototype

### Stage A: CPU oracle and sizing

Implement a pure dense hierarchy builder from the existing SPGrid oracle
inputs. Differential-test all invariants above, including odd domain
dimensions, adaptive active/ghost contacts, boundaries, and duplicate ghost
proposals. No production selection changes.

### Stage B: shadow GPU image

Allocate the 168,536-byte dense buffers beside the current hierarchy. Build them
from the same candidate epoch, read back only under diagnostics, and compare
coordinate-keyed images. The sparse solver remains authoritative.

### Stage C: recurring M1 cutover

Run dense correction while retaining the existing sparse setup and A2 images.
This prices the recurring matrix-free operator and recordless transfer with
the smallest correctness surface. Compare both the full-dense correctness arm
and the sorted occupied-index performance arm; the latter is the intended
shipping selector. Acceptance:

- validation/error flags unchanged;
- three terminal PCG iterations on the clean factor-1 mini lane;
- hydrostatic 500-step bit-stability;
- free-fall and volume gates pass;
- correction dispatches 30 -> 24;
- dense V-cycle GPU time improves enough to survive a 500-step wall A/B.

If Stage C does not improve GPU time, stop before rewriting setup: direct
indexing has then disproved the expected recurring gain.

### Stage D: dense setup and A2 image builder

Replace sparse candidate construction and resolve the unchanged A2 images from
dense metadata. Differential-test emitted A2 image words before deleting the
sparse buffers. Acceptance adds setup dispatches <= 20 and removal of at least
7 MiB of hierarchy allocation on mini.

### Stage E: physical-layout microbenchmark only if needed

If the x-fast occupied-worklist arm does not deliver the expected recurring
gain, freeze one captured topology and compare x-fast, 4x4x4 tiled, 8x8x4
tiled, and Morton storage under equal work. Timestamp repeated matrix-free
apply, parent-owned restriction, fine-owned prolongation, and seed/publish
separately; subtract an empty-dispatch batch and require coordinate-keyed
output hashes to match. Sweep 32/64/128/256 lanes and distinguish physical
layout from logical workgroup tiling. Do not select a swizzle from intuition
or full-solver noise.
