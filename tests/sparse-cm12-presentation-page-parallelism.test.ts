import assert from "node:assert/strict";
import test from "node:test";
import {
  createSparseCM12FramePlanPresentationLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import {
  createSparseCM12FramePlanPresentationWGSL,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation.wgsl";

for (const resolution of [4, 8, 16] as const) {
  test(`FPP1 P${resolution} distributes page samples across the workgroup`, () => {
    const layout = createSparseCM12FramePlanPresentationLayout({
      pageCapacity: 4,
      brickFineResolution: resolution,
      pageResolution: resolution,
    });
    const wgsl = createSparseCM12FramePlanPresentationWGSL({ layout });
    assert.match(wgsl,
      /for\(var linear=lane;linear<cm12FppTiles\s*\*64u;linear\+=64u\)/);
    assert.match(wgsl, /let tile=linear\/64u;\s*let sample=linear-tile\*64u;/);
    assert.match(wgsl,
      /cm12PresentationExactSample\(brick,page,tile,sample,/);
    assert.match(wgsl,
      /cm12PresentationStoreCandidate\(page,cm12FppLocalIndex\(tile,sample\),/);
    assert.match(wgsl,
      /cm12PresentationStoreAccepted\(page,localIndex,\s*cm12PresentationLoadCandidate/);

    const covered = new Set<number>();
    for (let lane = 0; lane < 64; lane += 1) {
      for (let linear = lane; linear < layout.tilesPerPage * 64; linear += 64) {
        covered.add(linear);
      }
    }
    assert.equal(covered.size, resolution ** 3);
    assert.deepEqual([...covered].sort((a, b) => a - b),
      Array.from({ length: resolution ** 3 }, (_, index) => index));
  });
}
