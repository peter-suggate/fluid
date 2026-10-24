import assert from "node:assert/strict";
import test from "node:test";
import { ellipsoidBounds, subcellEllipsoidPatch } from "../lib/core/solid-patch/ellipsoid-bounds";
import { validateSolidPatch } from "../lib/core/solid-patch/compose";
import { ellipsoidSolidPatchKind } from "../lib/core/solid-patch/kinds/ellipsoid";
import { createSolidPatchRowBuffer } from "../lib/core/solid-patch/kind";
import { inscribedQuadricSignedDistance_cells, coverageForSignedDistance } from "../lib/core/solid-patch/inscribed-quadric";
import { solidVoxelPatchesKey } from "../lib/core/solid-world";
import { mirrorPatch } from "../lib/core/voxel-editor/geometry";
import { packSvoSculptCuts, sampleSvoSculptCuts } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";

test("subcell ellipsoids share CPU coverage and GPU cut bounds across serialization", () => {
  const patch = { ...subcellEllipsoidPatch([1.125, 2.25, -3.375], [6.125, 7.25, 1.625], "clear"), affectsScenery: true };
  validateSolidPatch(patch);
  const roundtrip = JSON.parse(JSON.stringify(patch));
  assert.equal(solidVoxelPatchesKey([patch]), solidVoxelPatchesKey([roundtrip]));
  const packed = packSvoSculptCuts([roundtrip], { origin_m: [0, 0, 0], cellSize_m: [1, 1, 1] });
  const raster = ellipsoidSolidPatchKind.rasterizer(roundtrip), row = createSolidPatchRowBuffer(8);
  for (let y = 2; y < 8; y++) for (let z = -4; z < 2; z++) {
    const result = raster.row(y, z, 1, 7, row);
    for (let x = 1; x < 7; x++) {
      const p = [x + .5, y + .5, z + .5] as const;
      const exact = Math.hypot(p[0] - 3.625, p[1] - 4.75, p[2] + .875) - 2.5;
      assert.ok(Math.abs(inscribedQuadricSignedDistance_cells(roundtrip, undefined, p) - exact) < 1e-12);
      assert.ok(Math.abs(sampleSvoSculptCuts(packed, p).distance - exact) < 1e-12);
      assert.equal(result === "full" ? 255 : result === "empty" ? 0 : row.coverage[x - 1], coverageForSignedDistance(exact));
    }
  }
  const shifted = subcellEllipsoidPatch([1.25, 2.25, -3.375], [6.25, 7.25, 1.625], "clear");
  assert.deepEqual(shifted.minimum, patch.minimum);
  assert.notEqual(solidVoxelPatchesKey([shifted]), solidVoxelPatchesKey([patch]), "subcell motion invalidates cached geometry");
});

test("subcell ellipsoid bounds survive mirroring and rescaling without becoming their integer box", () => {
  const patch = subcellEllipsoidPatch([1.125, 2.25, -3.375], [6.125, 7.25, 1.625], "clear");
  const reflected = mirrorPatch(patch, 0, 16);
  assert.deepEqual(ellipsoidBounds(reflected), { minimum: [9.875, 2.25, -3.375], maximumExclusive: [14.875, 7.25, 1.625] });
  assert.deepEqual(ellipsoidBounds(mirrorPatch(reflected, 0, 16)), ellipsoidBounds(patch));
  const resized = ellipsoidSolidPatchKind.resized!(patch, [2, 4, -8], [14, 16, 4]);
  assert.deepEqual(ellipsoidBounds(resized), { minimum: [2.25, 4.5, -6.75], maximumExclusive: [12.25, 14.5, 3.25] });
  assert.throws(() => validateSolidPatch({ ...patch, payload: "[0,0,0,6,0,0]" }), /subcell bounds/);
  assert.throws(() => validateSolidPatch({ ...patch, payload: "[0,0,0,-1,0,0]" }), /subcell bounds/);
});
