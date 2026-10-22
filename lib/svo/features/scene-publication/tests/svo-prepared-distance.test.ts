import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { registerMeshAsset } from "../../../../core/imported-assets/asset-library";
import { decodeMeshVolume } from "../../../../core/imported-assets/mesh-volume";
import { prepareSvoPrimitiveDistance, sampleSvoPrimitive, type SvoPrimitiveDescriptor } from "../../../contracts/svo-primitive-abi";
import type { SvoFieldProgram } from "../svo-field-program";

const identity = { primitiveId: 1, materialId: 2, ownerId: 3,
  center_m: { x: .3, y: -.2, z: .1 }, orientation: { w: 2, x: .2, y: -.1, z: .3 } };
const program: SvoFieldProgram = { ops: [
  { op: "box", out: 0, point: 0, parameters: { a: .4, b: .3, c: .2 } },
], result: 0 };
const descriptors: SvoPrimitiveDescriptor[] = [
  { ...identity, kind: "sphere", radius_m: .3 },
  { ...identity, kind: "box", halfExtents_m: { x: .4, y: .3, z: .2 } },
  { ...identity, kind: "capsule", radius_m: .2, segmentHalfLength_m: .3 },
  { ...identity, kind: "cylinder", radius_m: .2, halfHeight_m: .3 },
  { ...identity, kind: "rounded-cylinder", radius_m: .3, halfHeight_m: .4, edgeRadius_m: .05 },
  { ...identity, kind: "cup", radius_m: .3, halfHeight_m: .4, wallThickness_m: .04 },
  { ...identity, kind: "ellipsoid", radii_m: { x: .4, y: .3, z: .2 } },
  { ...identity, kind: "torus", majorRadius_m: .3, minorRadius_m: .08 },
  { ...identity, kind: "cone", baseRadius_m: .3, halfHeight_m: .4, topRadius_m: .1 },
  { ...identity, kind: "round-cone", baseRadius_m: .3, halfHeight_m: .4, topRadius_m: .1 },
  { ...identity, kind: "smooth-union-cluster", clusterReference: 0,
    lobeRadii_m: { x: .065, y: .04, z: .056 }, packing: {
      field: "noise-foliage", seed: 4258, smoothRadius_m: 0, clusterPeriod_m: .038,
      detailPeriod_m: .028, threshold: .5, clusterWeight: .32, detailWeight: .68, interiorBias: .06,
    } },
  { ...identity, kind: "field-program", envelopeRadii_m: { x: .4, y: .3, z: .2 }, fieldProgramReference: 0, program },
];
const points = [{ ...identity.center_m }, ...Array.from({ length: 257 }, (_, i) => ({
  x: identity.center_m.x + ((i * 73 % 257) / 256 - .5),
  y: identity.center_m.y + ((i * 137 % 257) / 256 - .5),
  z: identity.center_m.z + ((i * 193 % 257) / 256 - .5),
}))];

test("prepared solid distances exactly match all primitive samplers under rotation", () => {
  for (const descriptor of descriptors) {
    const distance = prepareSvoPrimitiveDistance(descriptor);
    for (const point of points) assert.equal(distance(point), sampleSvoPrimitive(descriptor, point).signedDistance_m,
      `${descriptor.kind} at ${JSON.stringify(point)}`);
    assert.throws(() => distance({ x: NaN, y: 0, z: 0 }), /must be finite/);
  }
});

test("prepared field and cluster distances resolve arena references once", () => {
  for (const descriptor of descriptors) {
    if (descriptor.kind !== "field-program" && descriptor.kind !== "smooth-union-cluster") continue;
    let resolves = 0;
    const input = { ...descriptor, program: undefined, packing: undefined };
    const distance = prepareSvoPrimitiveDistance(input,
      () => { resolves++; return descriptor.kind === "smooth-union-cluster" ? descriptor.packing : undefined; },
      () => { resolves++; return descriptor.kind === "field-program" ? descriptor.program : undefined; });
    for (const point of points) assert.equal(distance(point), sampleSvoPrimitive(descriptor, point).signedDistance_m);
    assert.equal(resolves, 1);
    assert.throws(() => prepareSvoPrimitiveDistance(input), /requires its authored/);
  }
});

test("prepared imported Pool Party distances retain the sampled field and Lipschitz bound", () => {
  const asset = "teak-lounger";
  registerMeshAsset(asset, decodeMeshVolume(gunzipSync(readFileSync(
    new URL("../../../../../public/assets/pool-party/teak-lounger.fvx.gz", import.meta.url)))));
  const descriptor: SvoPrimitiveDescriptor = { ...identity, kind: "field-program",
    fieldProgramReference: 0, envelopeRadii_m: { x: 1, y: 1, z: 1 },
    program: { ops: [], result: 0, sampledVolume: { asset } } };
  const distance = prepareSvoPrimitiveDistance(descriptor);
  for (const point of points) assert.equal(distance(point), sampleSvoPrimitive(descriptor, point).signedDistance_m);
});
