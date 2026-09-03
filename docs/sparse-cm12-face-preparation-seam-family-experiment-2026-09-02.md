# Sparse CM12 face-preparation seam-family experiment

## Question

Ocean-seiche B8/P8 with a full-domain minimum-cell-size 8 region spends about
3.7 ms in `face-preparation`, of which about 3.0 ms is the oriented face-row
preparation substage. BFA1 stores physical-boundary and positive sparse-air
addresses in one stream. Both specialized kernels dispatch the complete stream
and reject the other family before entering the RK2 trace.

The experiment compiled the two families into separate, packet-aligned ranges
and dispatched each specialized kernel over only its range. This was a general
immutable-address transformation: it used no scene or rung condition and did
not change accepted-row authority, row order within a family, or numerical code.

## Work reduction

The ocean construction contained:

| family | addresses | compact packets |
|---|---:|---:|
| physical boundary | 9,840 | 154 |
| positive sparse air | 1,000 | 16 |
| total | 10,840 | 170 |

The production control dispatched both consumers over 170 mixed packets, or
340 packet workgroups in total. The split dispatched 170 workgroups in total,
a 50% reduction in launched seam groups. The 10,840 useful row traces were
unchanged.

## Result

The experiment was rejected and removed from production.

| measurement | production control | split families | change |
|---|---:|---:|---:|
| complete face preparation, median | 3.7356 ms | 3.9977 ms | +7.0% |
| complete face preparation, p95 | 4.5220 ms | 4.7186 ms | +4.3% |
| dirty face-row preparation, median | 2.9491 ms | 3.1457 ms | +6.7% |
| dirty face-row preparation, p95 | 3.5389 ms | 3.8011 ms | +7.4% |

The terminal census, pressure residuals, and convergence receipts were exactly
equal, and both runs reported zero validation errors. The split therefore
isolated an execution-shape regression rather than a physics difference.

The likely explanation is occupancy and latency hiding. The split removes idle
lanes but packs the register-heavy characteristic traces into fuller groups.
This agrees with the earlier failed seam-kernel fusion and row-center live-range
experiments: once a lane enters `prepareTransportFaceRow`, launch count is not
the limiting resource. Reducing inexpensive rejected lanes does not reduce the
number of expensive traces.

Future face work should instead remove or cheapen complete traces:

1. Cache one effective velocity/flag record per accepted coarse cell while
   rasterizing dense face support. A B1 leaf currently re-evaluates the same
   cell record for as many as 512 finest-lattice writes. Keep the scale-1 path
   unchanged to protect fine scenes.
2. Test a direct compact accepted-row preparation oracle against BFA1, then use
   a GPU-authored cost selector only if both coarse and fine-biased scenes show
   a stable crossover.
3. Longer term, publish a conservative generation-stamped characteristic
   receipt whose dependency closure includes dynamic and sparse-air lifecycle
   rows, so unchanged rows avoid the RK2 trace entirely.

## Receipts

- Control: `artifacts/ocean-seiche-min8-dynamic-coarse-transport-production-20260902.json`
- Split experiment: `artifacts/ocean-seiche-min8-face-seam-family-split-20260902.json`
- Command: `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-sparse-cm12-stage-cost.ts --scene=ocean-seiche --brick-fine=8 --presentation-page=8 --minimum-cell-size=8 --region-scope=domain --warmup=8 --frames=24 --capture-gap-ms=110 --final-qa=0 --quiet=1 --out=artifacts/ocean-seiche-min8-face-seam-family-split-20260902.json`
