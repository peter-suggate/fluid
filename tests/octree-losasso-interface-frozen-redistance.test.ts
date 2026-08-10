import assert from "node:assert/strict";
import test from "node:test";

import {
  octreeLosassoAdaptivePhiRedistanceFinishWGSL,
  octreeLosassoAdaptivePhiRedistanceInitializeWGSL,
  octreeLosassoAdaptivePhiRedistanceAtoBWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectAWGSL,
  octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL,
  octreeLosassoAdaptivePhiWorklistReachWGSL,
} from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";

const mixed = (values: readonly number[]) => Math.min(...values) <= 0
  && Math.max(...values) >= 0;

test("every corner of a mixed leaf is a D4-invariant accepted interface node", () => {
  const phi = [-0.25, 0.75, 0.75, 1.75, 0.75, 1.75, 1.75, 2.75] as const;
  assert.equal(mixed(phi), true,
    "diagonal negative support must still identify the incident cube as mixed");
  const before = new Float32Array(phi);
  const after = new Float32Array(before);
  // Interface-frozen finish is identity on every mixed-leaf corner. This also
  // covers +0/-0 bit preservation because production returns before signing.
  assert.deepEqual(new Uint32Array(after.buffer), new Uint32Array(before.buffer));

  const permutations = [
    [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
  ] as const;
  const corner = (index: number) => [index & 1, (index >> 1) & 1, (index >> 2) & 1] as const;
  for (const permutation of permutations) {
    const permuted = new Array<number>(8);
    for (let index = 0; index < 8; index += 1) {
      const p = corner(index);
      const q = permutation.map((axis) => p[axis]);
      permuted[q[0]! | (q[1]! << 1) | (q[2]! << 2)] = phi[index]!;
    }
    assert.equal(mixed(permuted), true, `axis permutation ${permutation.join("")}`);
  }
  assert.equal(mixed([1, 2, 2, 3, 2, 3, 3, 4]), false,
    "an all-air incident leaf remains available for Eikonal propagation");
});

test("sparse redistance freezes mixed-leaf corners and propagates only outward", () => {
  assert.match(octreeLosassoAdaptivePhiWorklistReachWGSL,
    /incidentLeaves\[8u\*i\+incident\][\s\S]*minimumPhi<=0\.&&maximumPhi>=0\.[\s\S]*bandMask\[i\]=select\(0u,1u,scheduled\)\|select\(0u,2u,frozen\)/,
    "the compact scheduler must accept compiled incident mixed-leaf corners without dense lookup");
  assert.match(octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL,
    /atomicOr\(&bandMask\[master\],marked\)/,
    "a constrained mixed corner must freeze and schedule every independent master");
  assert.match(octreeLosassoAdaptivePhiRedistanceAtoBWGSL,
    /bitcast<u32>\(prior\)&FROZEN[\s\S]*distanceOutput\[i\]=prior;return/,
    "sweeps must copy accepted interface values rather than lower them");
  assert.match(octreeLosassoAdaptivePhiRedistanceAtoBWGSL,
    /abs\(distanceInput\[negative\]\)[\s\S]*abs\(distanceInput\[positive\]\)/,
    "outward Eikonal propagation must consume the accepted distance magnitude");
  assert.match(octreeLosassoAdaptivePhiRedistanceProjectAWGSL,
    /bitcast<u32>\(distance\[i\]\)&FROZEN[\s\S]*return;[\s\S]*let d0=abs\(distance\[c\.masters\.x\]\);let d1=abs/,
    "hanging-distance closure must retain frozen mixed corners and not interpret markers as signs");
  assert.match(octreeLosassoAdaptivePhiRedistanceFinishWGSL,
    /bitcast<u32>\(encoded\)&FROZEN[\s\S]*atomicAdd\(&redistanceReceipt\[1\],1u\);return/,
    "finish must retain transported interface phi bit-for-bit");

  const storageBindings = new Set([...octreeLosassoAdaptivePhiRedistanceInitializeWGSL.matchAll(
    /@group\(0\)@binding\((\d+)\)var<storage/g,
  )].map((match) => Number(match[1])));
  assert.equal(storageBindings.size, 10, "the one graph scan stays at Dawn's storage-buffer limit");
});
