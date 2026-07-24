import assert from "node:assert/strict";
import test from "node:test";

import { octreeFaceBandWGSL } from "../lib/webgpu-octree-face-closest-point";

const compact = (source: string): string => source.replace(/\s+/g, "");

function wgslFunction(name: string): string {
  const shader = compact(octreeFaceBandWGSL);
  const start = shader.indexOf(`fn${name}(`);
  assert.notEqual(start, -1, `missing WGSL function ${name}`);
  const open = shader.indexOf("{", start);
  assert.notEqual(open, -1, `missing WGSL body for ${name}`);
  let depth = 0;
  for (let cursor = open; cursor < shader.length; cursor += 1) {
    if (shader[cursor] === "{") depth += 1;
    else if (shader[cursor] === "}" && --depth === 0) {
      return shader.slice(start, cursor + 1);
    }
  }
  assert.fail(`unterminated WGSL function ${name}`);
}

test("closest-point CPT rejection is fail-closed and never scans the row arena", () => {
  const closest = wgslFunction("closestPointLiquidVector");
  assert.match(closest,
    /owner=ownerAt\(vec3u\(floor\(ownerPoint\)\)\).*band=rowOfIdentity\(cell\(owner\.origin\),owner\.size\).*sampled=resolveWetFaceVectorAtPoint\(band,point\)/s,
    "the physical query is resolved through an exact owner identity");
  assert.match(closest,
    /if\(!velocityValid\(sampled\.value\)\)\{sampled\.value\.w=-f32\(band\+1u\);\}returnsampled/,
    "a rejected local simplex remains attributable to its exact attempted row");
  assert.doesNotMatch(closest, /for\(|while\(|loop\{|rowCount|rowCapacity/,
    "the top-level CPT query cannot repair a miss with a row-arena scan");

  const resolver = wgslFunction("resolveWetFaceVectorAtPoint");
  assert.match(resolver,
    /direct=wetFaceVectorAtPoint\(initialAnchor,pointGrid\).*localFanGap=direct\.reason==CPT_NO_SIMPLEX\|\|direct\.reason==CPT_MISSING_VERTEX.*dx=-2i;dx<=2i.*candidate=rowOfIdentity\(cell\(origin\),size\).*sampled=wetFaceVectorAtPoint\(candidate,pointGrid\)/s,
    "a local fan or seeded-vertex miss retries only geometrically indexed local identities with the same physical point");
  assert.doesNotMatch(resolver, /rowCount|rowCapacity|arrayLength\(&rowDirectory\)/,
    "the bounded local retry cannot scan the row arena or directory");

  const identity = wgslFunction("rowOfIdentity");
  assert.match(identity,
    /slot=rowIdentitySlot\(cellKey,size\).*encoded=rowDirectory\[slot\].*candidate=rows\[row\].*candidate\.cell==cellKey&&candidate\.size==size/s,
    "an exact candidate identity uses one collision-free direct-table load");
  assert.doesNotMatch(identity, /for\(|while\(|rowIdentityLess/,
    "identity lookup never degenerates into a directory or capacity walk");

  const wet = wgslFunction("wetFaceVectorAtPoint");
  assert.match(wet, /for\(varcorner=0u;corner<8u;corner\+=1u\)/,
    "uniform interpolation examines the fixed eight-corner stencil only");
  assert.match(wet, /for\(varlocal=0u;local<header\.count;local\+=1u\)/,
    "adaptive interpolation examines only the selected catalog entry");
  assert.doesNotMatch(wet, /rowCount|rowCapacity|rowDirectory|rowOfIdentity/,
    "a containing-simplex miss remains local and fail-closed");

  const seeded = wgslFunction("seededIncidentVector");
  assert.match(seeded,
    /count=min\(incidence\[rowIndex\],p\.axisStride\).*for\(varlocal=0u;local<count;local\+=1u\)/s,
    "wet velocity carriers are limited by the row's fixed incidence capacity");

  const carrier = wgslFunction("closestSeededFaceCarrier");
  assert.match(carrier,
    /ROW_SUPPORT3_ENDPOINT.*count=min\(incidence\[rowIndex\],p\.axisStride\).*incidence\[p\.rowCapacity\+rowIndex\*p\.axisStride\+local\].*face\.flags&\(LIVE\|SEED\|PHI_VALID\|FACE_VELOCITY_VALID\).*distanceSquared=dot\(delta,delta\).*distanceSquared==best\.distanceSquared&&faceIndex<best\.face/s,
    "the empty-stencil carrier is the deterministic closest immutable seeded face in one fixed incidence star");
  assert.doesNotMatch(carrier, /rowCount|rowDirectory|while\(|loop\{|powerRowVelocities/,
    "the carrier cannot scan rows, consume power-cell velocity, or observe a concurrently extended dry face");

  assert.match(resolver,
    /carrierEligible=localFanGap.*bestCarrier=closestSeededFaceCarrier\(initialAnchor,pointGrid\).*carrierEligible&&bestRow==INVALID.*closestSeededFaceCarrier\(candidate,pointGrid\).*if\(bestRow!=INVALID\)\{returnbest;\}if\(bestCarrier\.face!=INVALID\)\{returnLiquidInterpolation\(bestCarrier\.value,0u\);\}returndirect/s,
    "only an exhausted local-fan gap may copy the closest seed, and every valid bounded fan wins first");
});
