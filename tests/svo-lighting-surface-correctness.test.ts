import assert from "node:assert/strict";
import test from "node:test";

import { cloneScene, defaultScene } from "../lib/model";
import {
  SVO_PRIMITIVE_FEATURES,
  intersectSvoPrimitive,
  type SvoEllipsoidPrimitive,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveRayHit,
  type SvoSpherePrimitive,
} from "../lib/svo-primitive-abi";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import {
  evaluateUnifiedLighting,
  type LinearRgb,
  type UnifiedLightingMaterial,
} from "../lib/webgpu-lighting";

const reflectiveMaterial: UnifiedLightingMaterial = {
  baseColorLinear: [0.85, 0.55, 0.2],
  emissiveLinear: [0, 0, 0],
  roughness: 0.35,
  metallic: 1,
  specularF0Linear: [0.04, 0.04, 0.04],
  specularWeight: 1,
  ambientDiffuse: 0,
  rimColorLinear: [0, 0, 0],
  rimWeight: 0,
};

const diffuseMaterial: UnifiedLightingMaterial = {
  ...reflectiveMaterial,
  baseColorLinear: [0.7, 0.7, 0.7],
  roughness: 0.65,
  metallic: 0,
};

function expectHit(hit: SvoPrimitiveRayHit | null): SvoPrimitiveRayHit {
  assert.ok(hit);
  return hit;
}

function rgb(hit: SvoPrimitiveRayHit, towardViewer: LinearRgb, towardLight: LinearRgb, lightColorLinear: LinearRgb) {
  const normal: LinearRgb = [hit.normal.x, hit.normal.y, hit.normal.z];
  return evaluateUnifiedLighting(reflectiveMaterial, {
    shadingNormal: normal,
    geometricNormal: normal,
    towardViewer,
    towardLight,
    lightColorLinear,
  });
}

const luminance = (color: LinearRgb) => 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
const dot = (left: LinearRgb, right: LinearRgb) => left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

test("smooth sphere and ellipsoid normals move a highlight continuously", () => {
  const shapes: Array<SvoSpherePrimitive | SvoEllipsoidPrimitive> = [
    {
      kind: "sphere", primitiveId: 1, materialId: 1, ownerId: 1,
      center_m: { x: 0, y: 0, z: 0 }, radius_m: 1,
    },
    {
      kind: "ellipsoid", primitiveId: 2, materialId: 1, ownerId: 2,
      center_m: { x: 0, y: 0, z: 0 }, radii_m: { x: 1.4, y: 1, z: 0.75 },
    },
  ];

  for (const shape of shapes) {
    const xRadius = shape.kind === "sphere" ? shape.radius_m : shape.radii_m.x;
    const hits = Array.from({ length: 41 }, (_, index) => {
      const x = -0.72 * xRadius + index * 0.036 * xRadius;
      return expectHit(intersectSvoPrimitive(shape, {
        origin_m: { x, y: 0, z: 3 }, direction: { x: 0, y: 0, z: -1 },
      }));
    });

    assert.ok(hits.every((hit) => hit.normalPolicy === "smooth"));
    assert.ok(hits.every((hit) => hit.featureId === SVO_PRIMITIVE_FEATURES.smooth));
    for (let index = 1; index < hits.length; index += 1) {
      const previous: LinearRgb = [hits[index - 1].normal.x, hits[index - 1].normal.y, hits[index - 1].normal.z];
      const current: LinearRgb = [hits[index].normal.x, hits[index].normal.y, hits[index].normal.z];
      assert.ok(dot(previous, current) > 0.995, `${shape.kind} adjacent analytic normals must not quantize`);
    }

    const highlightPositions = [-0.65, 0, 0.65].map((lightX) => {
      const inverseLength = 1 / Math.hypot(lightX, 1);
      const towardLight: LinearRgb = [lightX * inverseLength, 0, inverseLength];
      const samples = hits.map((hit) => ({
        x: hit.position_m.x,
        response: luminance(rgb(hit, [0, 0, 1], towardLight, [5, 5, 5])),
      }));
      assert.ok(samples.every(({ response }) => Number.isFinite(response) && response >= 0));
      return samples.reduce((peak, sample) => sample.response > peak.response ? sample : peak).x;
    });
    assert.ok(highlightPositions[0] < highlightPositions[1]);
    assert.ok(highlightPositions[1] < highlightPositions[2]);
    assert.ok(Math.abs(highlightPositions[1]) < 1e-10, `${shape.kind} centered light must peak at its centered normal`);
  }
});

test("cube faces stay planar and the edge response is a hard discontinuity", () => {
  const cube: SvoFinitePrimitiveDescriptor = {
    kind: "box", primitiveId: 3, materialId: 1, ownerId: 3,
    center_m: { x: 0, y: 0, z: 0 }, halfExtents_m: { x: 1, y: 1, z: 1 },
  };
  const xFace = [-0.8, 0, 0.8].map((z) => expectHit(intersectSvoPrimitive(cube, {
    origin_m: { x: 3, y: 0, z }, direction: { x: -1, y: 0, z: 0 },
  })));
  for (const hit of xFace) {
    assert.equal(hit.normalPolicy, "hard-feature");
    assert.equal(hit.featureId, SVO_PRIMITIVE_FEATURES.boxFaceX);
    assert.deepEqual(hit.normal, { x: 1, y: 0, z: 0 });
  }

  const eitherSideOfEdge = [
    expectHit(intersectSvoPrimitive(cube, {
      origin_m: { x: 3, y: 0, z: 0.999 }, direction: { x: -1, y: 0, z: 0 },
    })),
    expectHit(intersectSvoPrimitive(cube, {
      origin_m: { x: 0.999, y: 0, z: 3 }, direction: { x: 0, y: 0, z: -1 },
    })),
  ];
  assert.deepEqual(eitherSideOfEdge.map((hit) => hit.featureId), [
    SVO_PRIMITIVE_FEATURES.boxFaceX,
    SVO_PRIMITIVE_FEATURES.boxFaceZ,
  ]);
  assert.deepEqual(eitherSideOfEdge.map((hit) => hit.normal), [
    { x: 1, y: 0, z: 0 },
    { x: 0, y: 0, z: 1 },
  ]);
  assert.ok(eitherSideOfEdge.every((hit) => {
    const nonzeroAxes = [hit.normal.x, hit.normal.y, hit.normal.z].filter((component) => component !== 0);
    return nonzeroAxes.length === 1 && Math.abs(nonzeroAxes[0]) === 1;
  }), "a cube edge must never synthesize a diagonal/interpolated normal");

  const towardViewer: LinearRgb = [Math.SQRT1_2, 0, Math.SQRT1_2];
  const litX = luminance(rgb(eitherSideOfEdge[0], towardViewer, [1, 0, 0], [3, 3, 3]));
  const darkZ = luminance(rgb(eitherSideOfEdge[1], towardViewer, [1, 0, 0], [3, 3, 3]));
  assert.ok(litX > 0.1);
  assert.equal(darkZ, 0, "the adjacent face remains geometrically back-lit instead of smoothing across the edge");
});

test("every authored closed room-shell primitive blocks an outside direct light", () => {
  for (const environmentId of ["conservatory", "night-lab", "research-station"] as const) {
    const publication = buildSvoScenePrimitives(cloneScene(defaultScene), { environmentId });
    const shell = publication.metadata.filter((entry) => entry.shell && !entry.openShell);
    const floor = shell.find((entry) => entry.key.endsWith("/floor"));
    const ceiling = shell.find((entry) => entry.key.endsWith("/ceiling"));
    assert.ok(floor && ceiling);
    const floorPrimitive = publication.descriptors[floor.primitiveIndex];
    const ceilingPrimitive = publication.descriptors[ceiling.primitiveIndex];
    assert.equal(floorPrimitive.kind, "box");
    assert.equal(ceilingPrimitive.kind, "box");
    if (floorPrimitive.kind !== "box" || ceilingPrimitive.kind !== "box") continue;
    const roomCenter = {
      x: 0,
      y: 0.5 * (floorPrimitive.center_m.y + ceilingPrimitive.center_m.y),
      z: 0,
    };

    for (const target of shell) {
      const descriptor = publication.descriptors[target.primitiveIndex];
      assert.notEqual(descriptor.kind, "terrain-heightfield");
      if (descriptor.kind === "terrain-heightfield") continue;
      const delta = {
        x: descriptor.center_m.x - roomCenter.x,
        y: descriptor.center_m.y - roomCenter.y,
        z: descriptor.center_m.z - roomCenter.z,
      };
      const centerDistance = Math.hypot(delta.x, delta.y, delta.z);
      const towardLight: LinearRgb = [delta.x / centerDistance, delta.y / centerDistance, delta.z / centerDistance];
      const blocker = intersectSvoPrimitive(descriptor, {
        origin_m: roomCenter,
        direction: { x: towardLight[0], y: towardLight[1], z: towardLight[2] },
        tMin_m: 1e-4,
        tMax_m: centerDistance + 1,
      });
      assert.ok(blocker && blocker.t_m < centerDistance + 1, `${target.key} must block the light before it reaches the room`);

      const unoccluded = evaluateUnifiedLighting(diffuseMaterial, {
        shadingNormal: towardLight,
        geometricNormal: towardLight,
        towardViewer: towardLight,
        towardLight,
        lightColorLinear: [4, 4, 4],
      });
      const blocked = evaluateUnifiedLighting(diffuseMaterial, {
        shadingNormal: towardLight,
        geometricNormal: towardLight,
        towardViewer: towardLight,
        towardLight,
        lightColorLinear: [0, 0, 0],
      });
      assert.ok(luminance(unoccluded) > 0.1);
      assert.deepEqual(blocked, [0, 0, 0], `${target.key} visibility zero must remove all direct light`);
    }
  }
});
