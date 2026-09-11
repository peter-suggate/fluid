# Prospective transport destinations: first measurement

The radius-0.1 Figure 7 capture supports reducing the destination **query set**, but does not yet establish a reduction in persistent brick allocation. A local velocity sweep with one source-cell-width of padding covered every measured mass receiver in four sampled steps. It selected 52.5–95.8% fewer resident cells than the accepted roster. These are cell-count reductions, not measured dispatch or runtime savings.

## Results

Production Sparse CM12, balanced, coarse first, B8/presentation B8, scene timestep 1/30 s, pressure tolerance 0.194 to match the existing small-fluid workload. The only scene edit is the radius-0.1 sphere at (0, 4.5, 0), plus extended run duration. Existing transport QA captures are enabled; allocation, topology policy, and solver kernels are unchanged.

| Step | Time (s) | Accepted cells | Selected resident cells | Reduction | Actual receiver union | Missed receivers | Forward-only receivers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.033 | 18,432 | 768 | 95.8% | 64 | 0 | 0 |
| 8 | 0.267 | 4,352 | 1,232 | 71.7% | 164 | 0 | 0 |
| 24 | 0.800 | 14,624 | 6,944 | 52.5% | 632 | 0 | 48 |
| 32 | 1.067 | 14,624 | 6,560 | 55.1% | 898 | 0 | 0 |

The receiver union includes every backward stencil touching a nonzero-density donor with positive weight, every receiver with nonzero fixed-point forward density return, and every positive gathered transport density. No visual density threshold is used. The step-24 forward return reaches 124 receivers, including **48 absent from the backward set**.

A half-source-width margin misses four backward/gathered receivers at step 32. One and two source widths miss none in these captures. One width is an empirical candidate, not a general characteristic-support proof.

## Candidate construction

For each nonzero source cell, construct the axis-aligned union of its initial box and its box translated by `dt * sourceVelocity`. Expand each side by one source-cell width along that axis. Enumerate the overlapping logical 4-cell bins, including coordinates not presently resident, and deduplicate them. A resident receiver is selected when its center lies in one of these bins.

Selection uses only the pre-step source snapshot. Receiver receipts are used afterward to judge coverage. Candidate bins are coordinate records; they are not allocated native simulation packets. Source cell centers and widths are read from the accepted topology, with GPU-grown cell geometry reconstructed from the production world-directory coordinate and page-local index. The capture asserts matching source/transport topology generations and that all referenced donors belong to that source roster.

For comparison, symmetric boxes based on the global speed bound and largest accepted cell width were tested with one and two widths of padding. They cover all receivers, but select **every resident cell** at steps 24 and 32. A bound dominated by unrelated coarse air cells loses the useful spatial selectivity.

## Allocation implications

| Step | Resident logical 8-cell bins | Bins containing required receivers | Local candidates | Currently absent candidates |
|---:|---:|---:|---:|---:|
| 1 | 36 | 4 | 8 | 0 |
| 8 | 12 | 12 | 12 | 0 |
| 24 | 36 | 36 | 55 | 19 |
| 32 | 36 | 32 | 55 | 19 |

These are logical coordinate bins derived from cell centers, not physical allocator-page counts. At step 24, required receivers already occupy all 36 represented bins. Thus a much smaller cell query set does **not** imply fewer whole bricks can be retained. Also, allocating every candidate would add 19 currently absent bins in the later captures. The candidate stage must remain lightweight, with allocation deferred until after actual transport demand is resolved.

## What this does and does not establish

- Dawn completed 32 steps, with captures at 1, 8, 24, and 32, no reported simulation/validation faults, and valid source-generation matching.
- One-source-width local candidates contain every captured mass receiver; half-width candidates fail at step 32.
- The selector runs on the CPU. Timings in the JSON cover box construction only, excluding readback and receipt matching; they do not predict GPU performance.
- The probe does not trace from absent cells or remove any velocity support. It cannot yet detect new demand that a virtual trace outside the existing transport roster would discover.
- Local velocity does not bound general RK2 travel through a deforming velocity field. More scenes and a conservative field bound are required before changing production behavior.
- Mass-receiver coverage does not establish that dry-cell gamma evolution, velocity extension, pressure boundaries, or presentation support can be discarded.

The next experiment is a GPU coordinate-only trace list evaluated against the unchanged source velocity field, including the absent candidate bins. Measure its construction and tracing cost, compare every overlapping trace with the resident path, and separately count nonzero destination demand before considering page removal.

## Reproduce

Close fluid browser tabs and ensure no other Dawn run holds the repository GPU lease.

```bash
node --import tsx tools/probe-cm12-prospective-destinations-dawn.ts
node --import tsx tools/analyze-cm12-prospective-destinations.ts
```

The probe defaults to Metal and local `node_modules/webgpu`. Per-frame source snapshots and receiver sets are written to `artifacts/cm12-prospective-destinations/`. The compact result is retained at `benchmarks/results/cm12-prospective-destinations-2026-09-11.json`.

Repository typecheck still fails on existing unrelated files; neither new measurement tool appears in its errors. This change adds measurement tools only, so no simulation post-refactor gate is required.
