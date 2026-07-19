import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalSvoThickGlassVolume,
  evaluateSvoThickGlassInterface,
  intersectSvoThickGlassVolume,
  packSvoThickGlassVolumes,
  querySvoThickGlassVolume,
  SVO_THICK_GLASS_RECORD_STRIDE_BYTES,
  svoThickGlassBounds,
  svoThickGlassMediumHandoff,
  svoThickGlassWGSL,
  unpackSvoThickGlassVolumes,
  type SvoThickGlassVolume,
} from "../lib/svo-thick-glass";

const sphere: SvoThickGlassVolume = {
  glassId: 41,
  materialId: 7,
  ownerId: 13,
  revision: 9,
  shape: "sphere",
  center_m: [0, 0, 0],
  radii_m: [1, 1, 1],
  indexOfRefraction: 1.5,
  absorption_mInv: [0.8, 0.2, 0.1],
  surfaceEpsilon_m: 1e-5,
  maximumOpticalPath_m: 4,
};

function close(actual: number, expected: number, epsilon = 1e-6): void {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} ~= ${expected}`);
}

function close3(actual: readonly number[], expected: readonly number[], epsilon = 1e-6): void {
  actual.forEach((value, axis) => close(value, expected[axis], epsilon));
}

test("80-byte thick-glass ABI preserves stable geometry, owner, material, ID, and revision", () => {
  assert.equal(SVO_THICK_GLASS_RECORD_STRIDE_BYTES, 80);
  const packed = packSvoThickGlassVolumes([sphere]);
  assert.equal(packed.byteLength, 80);
  const restored = unpackSvoThickGlassVolumes(packed)[0];
  assert.equal(restored.glassId, 41);
  assert.equal(restored.materialId, 7);
  assert.equal(restored.ownerId, 13);
  assert.equal(restored.revision, 9);
  assert.equal(restored.shape, "sphere");
  close3(restored.radii_m, [1, 1, 1]);
  assert.throws(() => canonicalSvoThickGlassVolume({ ...sphere, glassId: 0 }), /zero is reserved/);
  assert.throws(() => canonicalSvoThickGlassVolume({ ...sphere, shape: "sphere", radii_m: [1, 1.1, 1] }), /equal radii/);
});

test("sphere reports exact outside entry/exit and inside exit with outward normals", () => {
  const outside = intersectSvoThickGlassVolume(sphere, { origin_m: [-3, 0, 0], direction: [2, 0, 0], tMax_m: 6 });
  assert.ok(outside?.entry);
  close(outside!.entry!.t_m, 2);
  close(outside!.exit.t_m, 4);
  close3(outside!.entry!.geometricNormal, [-1, 0, 0]);
  close3(outside!.exit.geometricNormal, [1, 0, 0]);
  assert.equal(outside!.entry!.frontFacing, true);
  assert.equal(outside!.exit.frontFacing, false);
  assert.equal(outside!.insideAtStart, false);
  close(outside!.opticalPath_m, 2);

  const inside = intersectSvoThickGlassVolume(sphere, { origin_m: [0, 0, 0], direction: [1, 0, 0], tMax_m: 2 });
  assert.ok(inside);
  assert.equal(inside!.entry, undefined);
  assert.equal(inside!.insideAtStart, true);
  close(inside!.first.t_m, 1);
  close3(inside!.first.geometricNormal, [1, 0, 0]);
  close(inside!.opticalPath_m, 1);
});

test("tangent and rotated ellipsoid roots/normals remain exact and bounded", () => {
  const tangent = intersectSvoThickGlassVolume(sphere, { origin_m: [-2, 1, 0], direction: [1, 0, 0], tMax_m: 4 });
  assert.ok(tangent);
  assert.equal(tangent!.tangent, true);
  close(tangent!.first.t_m, 2);
  close3(tangent!.first.geometricNormal, [0, 1, 0]);
  assert.equal(tangent!.opticalPath_m, 0);

  const rotated: SvoThickGlassVolume = {
    ...sphere,
    glassId: 42,
    shape: "ellipsoid",
    radii_m: [2, 1, 1],
    orientation: { w: Math.SQRT1_2, x: 0, y: 0, z: Math.SQRT1_2 },
  };
  const hit = intersectSvoThickGlassVolume(rotated, { origin_m: [0, 3, 0], direction: [0, -1, 0], tMax_m: 6 });
  assert.ok(hit?.entry);
  close(hit!.entry!.t_m, 1);
  close3(hit!.entry!.position_m, [0, 2, 0]);
  close3(hit!.entry!.geometricNormal, [0, 1, 0]);
  const bounds = svoThickGlassBounds(rotated);
  close3(bounds.minimum, [-1, -2, -1]);
  close3(bounds.maximum, [1, 2, 1]);
});

test("medium handoff exposes Beer absorption, Fresnel refraction, and inside TIR without thin-wall semantics", () => {
  const outside = intersectSvoThickGlassVolume(sphere, { origin_m: [-3, 0, 0], direction: [1, 0, 0], tMax_m: 6 })!;
  const handoff = svoThickGlassMediumHandoff(sphere, outside, [1, 0, 0], "air", "glass");
  assert.equal(handoff.boundary.medium, "glass");
  assert.equal(handoff.boundary.thinWall, false);
  assert.equal(handoff.boundary.boundaryId, 41);
  close(handoff.optics.fresnel, 0.04);
  assert.equal(handoff.optics.totalInternalReflection, false);
  close3(handoff.optics.refractedDirection!, [1, 0, 0]);
  close(handoff.absorptionTint[0], Math.exp(-1.6));

  const inside = intersectSvoThickGlassVolume(sphere, { origin_m: [0, 0, 0.9], direction: [1, 0, 0], tMax_m: 2 })!;
  const tir = evaluateSvoThickGlassInterface(sphere, inside.first, [1, 0, 0], "glass", "air");
  assert.equal(tir.totalInternalReflection, true);
  assert.equal(tir.fresnel, 1);
  assert.equal(tir.refractedDirection, undefined);
  close(Math.hypot(...tir.reflectedDirection), 1);
});

test("query adapter fails malformed and stale publications closed", () => {
  assert.equal(querySvoThickGlassVolume(sphere, { origin_m: [-3, 0, 0], direction: [1, 0, 0], tMax_m: 6 }, 9).status, "hit");
  assert.deepEqual(
    querySvoThickGlassVolume(sphere, { origin_m: [-3, 0, 0], direction: [1, 0, 0], tMax_m: 6 }, 10),
    { status: "stale", expectedRevision: 10, actualRevision: 9 },
  );
  assert.equal(querySvoThickGlassVolume({ ...sphere, radii_m: [1, 0, 1] }, {
    origin_m: [-3, 0, 0], direction: [1, 0, 0], tMax_m: 6,
  }, 9).status, "invalid");
  assert.equal(querySvoThickGlassVolume(sphere, {
    origin_m: [-3, 0, 0], direction: [0, 0, 0], tMax_m: 6,
  }, 9).status, "invalid");
  assert.equal(querySvoThickGlassVolume(sphere, {
    origin_m: [-3, 3, 0], direction: [1, 0, 0], tMax_m: 6,
  }, 9).status, "miss");
});

test("WGSL mirror is binding-free, bounded, and exposes status/identity/optical handoff", () => {
  assert.match(svoThickGlassWGSL, /struct SvoThickGlassRecord/);
  assert.match(svoThickGlassWGSL, /fn svoThickGlassIntersect/);
  assert.match(svoThickGlassWGSL, /SVO_THICK_GLASS_STALE/);
  assert.match(svoThickGlassWGSL, /fn svoThickGlassInterface/);
  assert.match(svoThickGlassWGSL, /totalInternalReflection/);
  assert.match(svoThickGlassWGSL, /absorptionTint/);
  assert.match(svoThickGlassWGSL, /fn svoThickGlassMaterialId/);
  assert.match(svoThickGlassWGSL, /fn svoThickGlassOwnerId/);
  assert.doesNotMatch(svoThickGlassWGSL, /@group|@binding|loop\s*\{|while\s*\(/);
});
