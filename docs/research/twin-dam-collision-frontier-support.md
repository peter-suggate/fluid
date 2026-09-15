# Twin dam collision: retained dilute donor support

## Reproduction

On the working-tree transport ordering, twin-dam-collision stopped with:

```
GEOMETRIC_VOLUME_TRANSPORT
kernel=addWholeFrameUncoveredDonorFallbacks
frame=15; generation=17; owner=14747
operands=21,1,0.6612697839736938,1
```

The donor retained 0.000235296 finest-cell volumes. Its positive-y physical face had no accepted opposing cell. The guard correctly rejected outward flow to an absent in-domain receiver rather than treating allocation failure as physical outflow.

## Cause and fix

`measureBrickActivity` already visits the local face/edge/corner support stencil when a cell is an interface, thin fluid, or has any positive density. However, inside that loop only interface/thin-fluid cells wrote a support bit. The positive-density branch did no work. Such a donor depended solely on the measured velocity sweep. Face preparation on the next frame can change its outward face velocity, leaving that prior sweep insufficient.

Record the same adjacent stencil in the exact swept-demand word for every retained nonzero donor. The interface-support word alone is insufficient because its publication is gated on the coarser occupied classifier. Allocation, activation and retirement already consume the union of these words, so all now agree that the neighbour must remain available.

This is one mask update inside the existing census loop: no new dispatch, runtime topology repair or weakened transport guard. It can retain additional adjacent pages while they support nonzero mass. Existing explicitly recorded residue deletion still removes eligible whole dilute deep-air pages; this fix does not silently discard mass or change its threshold.

## Verification

```sh
WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx \
  tools/probe-figure7-deformation-dawn.ts \
  --scene=twin-dam-collision --steps=90 --verify-health --failure-debug \
  --output=artifacts/level-set-volume/twin-dam-support-3s.json
```

Dawn/Metal: all 91 checkpoints through frame 90 (3 s) passed health checks, with no rejected topology, activity faults, transport faults, coupling overflow or GPU validation errors. Maximum resident pages: 28. No physical outflow was recorded. Initial volume was 2048 finest-cell volumes; final volume 2047.953638 plus recorded residue deletion 0.049025 leaves approximately 0.002663 finest-cell volumes of numerical accounting drift (1.30 ppm), not an exact-conservation claim.

Artifacts: `twin-dam-failure.json` reproduces the failure; `twin-dam-support-fix.json` passes 30 frames; `twin-dam-support-3s.json` passes 90. These used the shared working tree, including the ongoing frame-order refactor. The full regression suite was not run, per the user's earlier instruction.
