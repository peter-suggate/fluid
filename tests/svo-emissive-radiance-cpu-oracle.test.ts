import assert from "node:assert/strict";
import test from "node:test";
import type { SceneDescription } from "../lib/model";
import type { SparseSceneDomainPlan } from "../lib/sparse-scene-domain";
import { buildSvoEmissiveRadianceCpuOraclePublication } from "../lib/svo-emissive-radiance-cpu-oracle";
import { buildSvoNodeMipCpuOraclePublication } from "../lib/svo-node-mip-cpu-oracle";
import { unpackSvoRadianceRgb9e5 } from "../lib/svo-tetrahedral-radiance";
import type { EnvironmentProxyPrimitive } from "../lib/voxel-environments";

function scene(): SceneDescription {
  return {
    schemaVersion: "1.0.0", sceneId: "emissive-radiance-cpu-oracle-test", environment: "garden", randomSeed: 1, duration_s: 1,
    container: { width_m: 16, height_m: 8, depth_m: 16, fillFraction: 0, top: "open", fluidWallMode: "free-slip" },
    voxelDomain: { finestCellSize_m: 1, brickSize_cells: 8 },
    fluid: { density_kg_m3: 1_000, dynamicViscosity_Pa_s: .001, surfaceTension_N_m: .07, gravity_m_s2: { x: 0, y: -9.81, z: 0 }, initialCondition: "tank-fill" },
    nominalResolution: { length_m: 1 },
    numerics: { fixedDt_s: .01, maxDt_s: .01, pressureRelativeTolerance: 1e-5, pressureMaxIterations: 10 },
    rigidBodies: [],
  };
}

function domain(): SparseSceneDomainPlan {
  const point = (x: number, y: number, z: number) => ({ x, y, z });
  return {
    brickSize: 8, cellSize_m: [1, 1, 1], worldOrigin_m: point(0, 0, 0),
    solverGridOriginCells: [0, 0, 0], solverDimensionsCells: [16, 8, 16], sceneDimensionsCells: [16, 16, 16],
    brickDimensions: [2, 2, 2], solverBounds_m: { min: point(0, 0, 0), max: point(16, 8, 16) },
    worldBounds_m: { min: point(0, 0, 0), max: point(16, 16, 16) },
    solverBrickCoordinates: [], environmentBrickCoordinates: [], proxyBrickCoordinates: [], coordinates: [],
  };
}

function box(
  key: string,
  center: readonly [number, number, number],
  half: readonly [number, number, number],
  emission = 0,
  tags: readonly string[] = [],
  color: readonly [number, number, number] = [1, .5, .25],
): EnvironmentProxyPrimitive {
  return {
    kind: "box", key, ownerIndex: 0, group: "fixture", tags,
    center_m: { x: center[0], y: center[1], z: center[2] },
    halfSize_m: { x: half[0], y: half[1], z: half[2] },
    material: { colorLinear: color, emission, roughness: .3 },
    aabb_m: {
      min: { x: center[0] - half[0], y: center[1] - half[1], z: center[2] - half[2] },
      max: { x: center[0] + half[0], y: center[1] + half[1], z: center[2] + half[2] },
    },
  };
}

function word(page: ReturnType<typeof buildSvoEmissiveRadianceCpuOraclePublication>["interiors"][number], direction: number, x: number, y: number, z: number): number {
  return page.packedInterleaved[((z * 8 + y) * 8 + x) * 4 + direction];
}

test("emissive CPU oracle mirrors every opacity key and explicitly certifies black pages", () => {
  const primitives = [
    box("emitter", [1, 1, 1], [.4, .4, .4], 2, ["light", "emits-positive-x"]),
    box("ordinary", [9, 1, 1], [.4, .4, .4]),
  ];
  const opacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), primitives, { generation: 7, levelCount: 1, samplesPerAxis: 2 });
  const radiance = buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), primitives, { samplesPerAxis: 2 });
  assert.equal(radiance.plan, opacity.plan);
  assert.deepEqual(radiance.interiors.map(({ key }) => key), opacity.interiors.map(({ key }) => key));
  assert.equal(radiance.interiors.length, 2);
  assert.equal(radiance.emissiveProxyCount, 1);
  assert.equal(radiance.blackPageCount, 1);
  const black = radiance.interiors.find(({ key }) => key.coordinate[0] === 1)!;
  assert.equal(black.certifiedBlack, true);
  assert.ok(black.packedInterleaved.every((value) => value === 0));
  assert.equal(black.packedInterleaved.length, 8 ** 3 * 4);
  assert.equal(radiance.packedInteriorBytes, radiance.interiors.length * 8 ** 3 * 16);
});

test("authored direction tags produce coverage-premultiplied one-sided exitance", () => {
  const emitter = box("panel", [1, 1, 1], [.45, .45, .45], 2, ["light", "emits-positive-x"]);
  const opacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), [emitter], { generation: 8, levelCount: 1, samplesPerAxis: 2 });
  const radiance = buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), [emitter], { samplesPerAxis: 2 });
  const page = radiance.interiors[0];
  assert.equal(page.certifiedBlack, false);
  // Tetrahedral directions 0/1 have +X; 2/3 have -X and must see the back face as black.
  const positive = unpackSvoRadianceRgb9e5(word(page, 0, 1, 1, 1));
  assert.ok(positive[0] > 0 && positive[1] > 0 && positive[2] > 0);
  assert.equal(word(page, 1, 1, 1, 1), word(page, 0, 1, 1, 1));
  assert.equal(word(page, 2, 1, 1, 1), 0);
  assert.equal(word(page, 3, 1, 1, 1), 0);
  assert.ok(positive[0] <= 2, "surface sampling must retain coverage premultiplication");
});

test("parent texels are the eight-child mean with absent virtual children contributing black", () => {
  const emitter = box("small-emitter", [.5, .5, .5], [.45, .45, .45], 1, ["light", "emits-positive-y"], [1, 1, 1]);
  const opacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), [emitter], { generation: 9, levelCount: 2, samplesPerAxis: 2 });
  const radiance = buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), [emitter], { samplesPerAxis: 2 });
  const base = radiance.interiors.find(({ key }) => key.level === 0)!;
  const parent = radiance.interiors.find(({ key }) => key.level === 1)!;
  for (let direction = 0; direction < 4; direction += 1) {
    const expected = [0, 0, 0];
    for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
      const decoded = unpackSvoRadianceRgb9e5(word(base, direction, x, y, z));
      for (let channel = 0; channel < 3; channel += 1) expected[channel] += decoded[channel] / 8;
    }
    const actual = unpackSvoRadianceRgb9e5(word(parent, direction, 0, 0, 0));
    for (let channel = 0; channel < 3; channel += 1) assert.ok(Math.abs(actual[channel] - expected[channel]) < .003);
  }
});

test("conflicting one-sided tags are rejected before oracle construction", () => {
  const emitter = box("broken", [1, 1, 1], [.4, .4, .4], 1, ["emits-positive-x", "emits-negative-z"]);
  const opacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), [emitter], { generation: 10, levelCount: 1 });
  assert.throws(() => buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), [emitter]), /conflicting emission directions/);
});

test("primary directional light injects Lambertian bounce only on light-facing geometry", () => {
  const reflector = box("reflector", [4, 4, 4], [1, 2, 2], 0, [], [1, .5, .25]);
  const opacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), [reflector], { generation: 11, levelCount: 1, samplesPerAxis: 2 });
  const radiance = buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), [reflector], {
    samplesPerAxis: 2,
    primaryDirectionalLight: {
      towardLightDirection: [1, 0, 0], colorLinear: [1, 1, 1], intensity: Math.PI,
    },
  });
  const page = radiance.interiors[0];
  const facing = unpackSvoRadianceRgb9e5(word(page, 0, 4, 3, 3));
  assert.ok(facing[0] > 0 && facing[1] > 0 && facing[2] > 0);
  assert.ok(Math.abs(facing[0] - .5) < .01, "pi incident intensity cancels the Lambertian 1/pi at half sample coverage");
  assert.ok(Math.abs(facing[1] / facing[0] - .5) < .02);
  assert.ok(Math.abs(facing[2] / facing[0] - .25) < .02);
  assert.equal(word(page, 0, 2, 3, 3), 0, "the -X face is back-facing to a +X toward-light direction");
  assert.ok(radiance.directLightSampleCount > 0);
  assert.equal(radiance.shadowedDirectLightSampleCount, 0);
  assert.equal(radiance.emissiveProxyCount, 0, "direct bounce does not require authored emission");
});

test("another environment proxy conservatively shadows primary-light bounce", () => {
  const reflector = box("reflector", [4, 4, 4], [1, 2, 2], 0, [], [1, 1, 1]);
  const blocker = box("blocker", [6, 4, 4], [.25, 2, 2]);
  const light = { towardLightDirection: [1, 0, 0] as const, colorLinear: [1, 1, 1] as const, intensity: Math.PI };
  const clearOpacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), [reflector], { generation: 12, levelCount: 1 });
  const clear = buildSvoEmissiveRadianceCpuOraclePublication(clearOpacity, domain(), [reflector], { primaryDirectionalLight: light });
  assert.notEqual(word(clear.interiors[0], 0, 4, 3, 3), 0);

  const blockedOpacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), [reflector, blocker], { generation: 13, levelCount: 1 });
  const blocked = buildSvoEmissiveRadianceCpuOraclePublication(blockedOpacity, domain(), [reflector, blocker], { primaryDirectionalLight: light });
  const page = blocked.interiors.find(({ key }) => key.coordinate.every((value) => value === 0))!;
  assert.equal(word(page, 0, 4, 3, 3), 0);
  assert.ok(blocked.shadowedDirectLightSampleCount > 0);
});

test("omitting the primary light retains emissive-only surface sampling", () => {
  const ordinary = box("ordinary", [4, 4, 4], [1, 1, 1]);
  const opacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), [ordinary], { generation: 14, levelCount: 1 });
  const radiance = buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), [ordinary]);
  assert.equal(radiance.interiors[0].certifiedBlack, true);
  assert.equal(radiance.directLightSampleCount, 0);
  assert.equal(radiance.shadowedDirectLightSampleCount, 0);
});

test("tagged authored point lights inject their dominant local bounce", () => {
  const reflector = box("reflector", [4, 4, 4], [1, 2, 2], 0, [], [1, 1, 1]);
  const lamp = box("warm-lamp", [6.5, 4, 4], [.2, .2, .2], 11, ["light", "point-light"], [1, .4, .1]);
  const primitives = [reflector, lamp];
  const opacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), primitives, { generation: 15, levelCount: 1 });
  const emissionOnly = buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), primitives);
  const weakDirectional = buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), primitives, {
    primaryDirectionalLight: { towardLightDirection: [1, 0, 0], colorLinear: [1, 1, 1], intensity: .09 },
  });
  const localBounce = buildSvoEmissiveRadianceCpuOraclePublication(opacity, domain(), primitives, { injectAuthoredProxyLights: true });
  const emissionPage = emissionOnly.interiors.find(({ key }) => key.coordinate.every((value) => value === 0))!;
  const weakPage = weakDirectional.interiors.find(({ key }) => key.coordinate.every((value) => value === 0))!;
  const bouncePage = localBounce.interiors.find(({ key }) => key.coordinate.every((value) => value === 0))!;
  assert.equal(word(emissionPage, 0, 4, 3, 3), 0, "the receiver has no authored emission of its own");
  const bounced = unpackSvoRadianceRgb9e5(word(bouncePage, 0, 4, 3, 3));
  const weak = unpackSvoRadianceRgb9e5(word(weakPage, 0, 4, 3, 3));
  assert.ok(bounced[0] > .1, "the local intensity-11 fixture must produce visible first-bounce energy");
  assert.ok(bounced[0] > 10 * weak[0], "the dominant local lamp must materially exceed the garden's 0.09 directional key");
  assert.ok(bounced[0] > bounced[1] && bounced[1] > bounced[2], "the bounce retains the fixture's warm spectrum");
  assert.ok(localBounce.localLightSampleCount > 0);
  assert.equal(localBounce.shadowedLocalLightSampleCount, 0);
  assert.equal(localBounce.directLightSampleCount, 0);

  const blocker = box("blocker", [5.75, 4, 4], [.2, 2, 2]);
  const blockedPrimitives = [reflector, blocker, lamp];
  const blockedOpacity = buildSvoNodeMipCpuOraclePublication(scene(), domain(), blockedPrimitives, { generation: 16, levelCount: 1 });
  const blocked = buildSvoEmissiveRadianceCpuOraclePublication(blockedOpacity, domain(), blockedPrimitives, { injectAuthoredProxyLights: true });
  const blockedPage = blocked.interiors.find(({ key }) => key.coordinate.every((value) => value === 0))!;
  assert.equal(word(blockedPage, 0, 4, 3, 3), 0, "another proxy blocks the finite receiver-to-lamp segment");
  assert.ok(blocked.shadowedLocalLightSampleCount > 0);
});
