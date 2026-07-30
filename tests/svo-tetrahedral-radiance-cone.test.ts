import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  SVO_TETRAHEDRAL_RADIANCE_CONE_MAXIMUM_STEPS,
  integrateSvoTetrahedralRadianceCone,
  svoDiffuseHemisphereCones,
  svoTetrahedralRadianceConeWGSL,
} from "../lib/svo-tetrahedral-radiance-cone";
import type { SvoTetrahedralRadiance } from "../lib/svo-tetrahedral-radiance";

const close = (actual: number, expected: number, tolerance = 1e-9) => assert.ok(
  Math.abs(actual - expected) <= tolerance,
  `expected ${actual} to be within ${tolerance} of ${expected}`,
);
const black = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] as SvoTetrahedralRadiance;

test("radiance cone composites coverage-unpremultiplied light front to back", () => {
  const halfRed = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]] as SvoTetrahedralRadiance;
  const result = integrateSvoTetrahedralRadianceCone({
    origin_m: [0, 0, 0], direction: [1, 0, 0], aperture_radians: 0,
    minimumVoxelWidth_m: 1, maximumDistance_m: 2.5, opacityCutoff: 1,
  }, () => ({ coverage: 128, radiance: halfRed }));
  const alpha = 128 / 255;
  close(result.radiance[0], (alpha + (1 - alpha) * alpha) / alpha, 1e-12);
  close(result.radiance[1], 0);
  close(result.opacity, 1 - (1 - alpha) ** 2, 1e-12);
  assert.equal(result.termination, "distance");
  assert.equal(result.valid, true);
  assert.equal(result.coneTaps, 2);
  assert.equal(result.opacityTaps, 2);
  assert.equal(result.radianceSamples, 2);
  assert.equal(result.radianceTextureTaps, 8);
});

test("directional reconstruction samples outgoing radiance back toward the receiver", () => {
  const towardNegativeX = [[0, 0, 0], [0, 0, 0], [2, 1, .5], [2, 1, .5]] as SvoTetrahedralRadiance;
  let outgoing: readonly number[] | undefined;
  const result = integrateSvoTetrahedralRadianceCone({
    origin_m: [0, 0, 0], direction: [1, 0, 0], aperture_radians: 0,
    minimumVoxelWidth_m: 1, maximumDistance_m: 1,
  }, (query) => {
    outgoing = query.outgoingDirection;
    return { coverage: 255, radiance: towardNegativeX };
  });
  close(outgoing![0], -1); close(outgoing![1], 0); close(outgoing![2], 0);
  assert.ok(result.radiance[0] > 0);
});

test("missing radiance is black but still contributes authoritative opacity", () => {
  const result = integrateSvoTetrahedralRadianceCone({
    origin_m: [0, 0, 0], direction: [0, 0, 1], aperture_radians: .8,
    minimumVoxelWidth_m: .5, maximumDistance_m: 4,
  }, () => ({ coverage: 192 }));
  assert.deepEqual(result.radiance, [0, 0, 0]);
  assert.equal(result.valid, true);
  assert.equal(result.missingRadianceSamples, result.coneTaps);
  assert.equal(result.radianceTextureTaps, 0);
  assert.equal(result.termination, "opacity");
});

test("missing opacity fails closed by default and can remain diagnostic-only", () => {
  const closed = integrateSvoTetrahedralRadianceCone({
    origin_m: [0, 0, 0], direction: [0, 1, 0], aperture_radians: 1,
    minimumVoxelWidth_m: 1, maximumDistance_m: 10,
  }, () => undefined);
  assert.equal(closed.valid, false);
  assert.equal(closed.transmittance, 0);
  assert.equal(closed.opacity, 1);
  assert.equal(closed.termination, "invalid-opacity");
  assert.equal(closed.coneTaps, 1);
  assert.equal(closed.missingOpacitySamples, 1);

  const diagnostic = integrateSvoTetrahedralRadianceCone({
    origin_m: [0, 0, 0], direction: [0, 1, 0], aperture_radians: 0,
    minimumVoxelWidth_m: 1, maximumDistance_m: 2.5, failClosedOpacity: false,
  }, () => undefined);
  assert.equal(diagnostic.valid, false);
  assert.equal(diagnostic.transmittance, 1);
  assert.equal(diagnostic.termination, "distance");
  assert.equal(diagnostic.coneTaps, 2);
});

test("the coverage guard prevents empty and nearly empty radiance amplification", () => {
  const bright = [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]] as SvoTetrahedralRadiance;
  const result = integrateSvoTetrahedralRadianceCone({
    origin_m: [0, 0, 0], direction: [0, 1, 0], aperture_radians: 0,
    minimumVoxelWidth_m: 1, maximumDistance_m: 1,
  }, () => ({ coverage: 1, radiance: bright }));
  assert.deepEqual(result.radiance, [0, 0, 0]);
  assert.equal(result.unpremultiplyRejectedSamples, 1);
  assert.equal(result.radianceSamples, 1);
});

test("cone queries preserve node-mip LOD growth and obey the hard step bound", () => {
  const lods: number[] = [], diameters: number[] = [];
  const result = integrateSvoTetrahedralRadianceCone({
    origin_m: [0, 0, 0], direction: [1, 0, 0], aperture_radians: 1.2,
    minimumVoxelWidth_m: .1, maximumDistance_m: 1e9,
    maximumSteps: 8,
  }, ({ lod, diameter_m }) => {
    lods.push(lod); diameters.push(diameter_m);
    return { coverage: 0, radiance: black };
  });
  assert.equal(result.termination, "step-limit");
  assert.equal(result.coneTaps, 8);
  assert.ok(lods.at(-1)! > lods[0]);
  assert.ok(diameters.at(-1)! > diameters[0]);
  assert.throws(() => integrateSvoTetrahedralRadianceCone({
    origin_m: [0, 0, 0], direction: [1, 0, 0], aperture_radians: 1,
    minimumVoxelWidth_m: 1, maximumDistance_m: 1,
    maximumSteps: SVO_TETRAHEDRAL_RADIANCE_CONE_MAXIMUM_STEPS + 1,
  }, () => ({ coverage: 0 })), /maximum steps/);
});

test("three- and four-cone hemisphere patterns are normalized and moment matched", () => {
  for (const count of [3, 4] as const) {
    const samples = svoDiffuseHemisphereCones([0, 1, 0], count, .37);
    assert.equal(samples.length, count);
    close(samples.reduce((sum, sample) => sum + sample.weight, 0), 1);
    samples.forEach(({ direction }) => {
      close(Math.hypot(...direction), 1);
      assert.ok(direction[1] > 0);
    });
    close(samples.reduce((sum, sample) => sum + sample.weight * sample.direction[1], 0), 2 / 3);
    close(samples.reduce((sum, sample) => sum + sample.weight * sample.direction[0], 0), 0);
    close(samples.reduce((sum, sample) => sum + sample.weight * sample.direction[2], 0), 0);
  }
});

test("cone WGSL is binding-free and exposes its sampling hook, diagnostics, and hemisphere helpers", () => {
  assert.doesNotMatch(svoTetrahedralRadianceConeWGSL, /@group|@binding/);
  assert.match(svoTetrahedralRadianceConeWGSL, /fn svoTetraRadianceConeTrace/);
  assert.match(svoTetrahedralRadianceConeWGSL, /svoTetraRadianceConeLoad\(query\)/);
  assert.match(svoTetrahedralRadianceConeWGSL, /svoNodeMipCoverageOpacity\(coverage,step\/voxelWidth\)/);
  assert.match(svoTetrahedralRadianceConeWGSL, /svoNodeMipLod\(diameter,config\.minimumVoxelWidth_m\)/);
  assert.match(svoTetrahedralRadianceConeWGSL, /radianceSamples\*4u/);
  assert.match(svoTetrahedralRadianceConeWGSL, /fn svoTetraRadianceHemisphereDirection/);
  assert.match(svoTetrahedralRadianceConeWGSL, /fn svoTetraRadianceHemisphereWeight/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("binding-free tetrahedral cone gather WGSL compiles on the selected WebGPU adapter", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU radiance-cone checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    const code = `${svoTetrahedralRadianceConeWGSL}
fn svoTetraRadianceConeLoad(query:SvoTetraRadianceConeQuery)->SvoTetraRadianceConeSourceSample{
  let light=max(query.outgoingDirection.y,0.0);
  let value=SvoTetraRadiance(vec3f(light),vec3f(light),vec3f(light),vec3f(light));
  return SvoTetraRadianceConeSourceSample(.5,value,1u,1u);
}
@compute @workgroup_size(1) fn validate(){
  let config=SvoTetraRadianceConeConfig(vec3f(0.0),vec3f(0.0,1.0,0.0),1.0,.1,10.0,64u,.995,.003921568627451,1u);
  let result=svoTetraRadianceConeTrace(config);_=result.radiance;
  _=svoTetraRadianceHemisphereDirection(vec3f(0.0,1.0,0.0),2u,4u,.3);
  _=svoTetraRadianceHemisphereWeight(2u,4u);
}`;
    const shaderModule = device.createShaderModule({ label: "Tetrahedral radiance cone validation", code });
    const info = await shaderModule.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    await device.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint: "validate" } });
  } finally {
    device.destroy();
  }
});
