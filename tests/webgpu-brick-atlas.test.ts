import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/model";
import { octreeMethod } from "../lib/methods/octree";
import {
  BRICK_ATLAS_AIR_PHI,
  WebGPUFluidBrickAtlas,
  brickAtlasLifecycleShader,
  brickAtlasMirrorShader,
  brickAtlasToDenseShader,
  brickAtlasValidateShader,
  fluidBrickAtlasSamplingWGSL,
  fluidBrickAtlasAllocatedBytes,
  planFluidBrickAtlas,
} from "../lib/webgpu-brick-atlas";
import {
  FLUID_BRICK_ACTIVATED,
  FLUID_BRICK_RESIDENT,
  GPUFluidBrickResidency,
  fluidBrickResidencyShader,
} from "../lib/webgpu-fluid-brick-residency";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { legacyUniformComputeShader, retiredBulkFluxScaleClearShader, retiredBulkTransportClearShader, retiredBulkVelocityClearShader } from "../lib/webgpu-eulerian";

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("atlas mirror shader, layout, and dense-import bind group publish the same bindings", () => {
  const source = readFileSync(new URL("../lib/webgpu-brick-atlas.ts", import.meta.url), "utf8");
  const bindingNumbers = (text: string) => [...text.matchAll(/binding(?:\(|:\s*)(\d+)/g)]
    .map((match) => Number(match[1]));
  const mirrorShaderBindings = bindingNumbers(brickAtlasMirrorShader.slice(0, brickAtlasMirrorShader.indexOf("const INVALID")));
  const mirrorLayout = source.match(/this\.mirrorLayout = device\.createBindGroupLayout\([\s\S]*?\n\s*\]\s*\}\);/)?.[0];
  const denseImportGroup = source.match(/const mirrorGroup = this\.device\.createBindGroup\([\s\S]*?\n\s*\]\s*\}\);/)?.[0];
  assert.ok(mirrorLayout, "mirror layout declaration should remain discoverable");
  assert.ok(denseImportGroup, "dense-import bind group declaration should remain discoverable");
  assert.deepEqual(bindingNumbers(mirrorLayout), mirrorShaderBindings);
  assert.deepEqual(bindingNumbers(denseImportGroup), mirrorShaderBindings);
  assert.deepEqual(mirrorShaderBindings, [0, 1, 2, 3, 4, 5, 6, 7]);
});

test("uniform kernels bind the optional bulk worklist declared by every entry point", () => {
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  assert.match(legacyUniformComputeShader, /@group\(0\) @binding\(26\) var<storage,read> bulkWorklist/);
  assert.match(uniform, /binding: 26[^\n]*read-only-storage/,
    "the explicit pipeline layout must include the shader's sparse scheduling arena");
  assert.match(uniform, /binding: 26, resource: \{ buffer: sparse\?\.bulkWorklist \?\? this\.bulkAtlasFallbackWorklist \}/,
    "all bind groups must supply either the live GPU worklist or a never-read fallback");
  assert.match(uniform, /binding: 24, resource: \{ buffer: \(atlas \?\? sparse\)\?\.params \?\? this\.bulkAtlasFallbackParams \}/,
    "sparse-only reverse and occupancy groups must decode the live worklist with live atlas dimensions");
  assert.doesNotMatch(legacyUniformComputeShader, /struct ScheduledCell \{[^}]*\bactive\s*:/,
    "WGSL reserved keywords cannot be used as structure members");
});

test("octree atlas ownership defaults to mirror but compact authority keeps only bulk residency", () => {
  const parameter = octreeMethod.params.find((candidate) => candidate.key === "brickAtlas");
  assert.ok(parameter && parameter.kind === "select");
  assert.equal(parameter.default, "mirror");
  assert.deepEqual(parameter.options.map((option) => option.value), ["mirror", "authoritative", "off"]);
  assert.equal(octreeMethod.presetFor?.("balanced").brickAtlas, "mirror");

  const method = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const sparseWorld = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(method, /brickAtlas: brickAtlasMode\(values\.brickAtlas\)/);
  assert.match(uniform, /brickAtlas: options\.octree\.brickAtlas \?\? "mirror"/);
  assert.match(projection, /brickAtlas: faceTransportEnabled && !compactAtlasDiagnostic \? "off" : options\.brickAtlas \?\? "mirror"/);
  assert.match(projection, /bulkResidencyOnly: faceTransportEnabled/);
  assert.match(sparseWorld, /get bulkResidencyWorklist/);
  assert.match(sparseWorld, /mode: brickAtlasMode/);
  assert.match(sparseWorld, /get atlasSamplingSource/);
  assert.match(projection, /get fluidBrickAtlasSamplingSource/);
  assert.match(uniform, /get fluidBrickAtlasSamplingSource/);
  assert.match(smoke, /FLUID_BRICK_ATLAS_MODE/);
  assert.match(method, /brickSparseVelocityAdvection: values\.brickSparseAdvection !== "off"/);
  assert.match(method, /brickSparseTransportPreparation: values\.brickSparseTransport !== "off"/);
  assert.match(method, /brickSparseOccupancyFluxPreparation: values\.brickSparseOccupancyFlux !== "off"/);
  assert.match(method, /brickSparseExtrapolation: values\.brickSparseExtrapolation !== "off"/);
  assert.match(smoke, /FLUID_BRICK_SPARSE_ADVECTION/);
  assert.match(smoke, /FLUID_BRICK_SPARSE_TRANSPORT/);
  assert.match(smoke, /FLUID_BRICK_SPARSE_OCCUPANCY_FLUX/);
  assert.match(smoke, /FLUID_BRICK_SPARSE_EXTRAPOLATION/);
  const transport = octreeMethod.params.find((candidate) => candidate.key === "brickSparseTransport");
  assert.ok(transport && transport.kind === "select");
  assert.equal(transport.default, "off", "the no-win full-footprint ocean A/B stays opt-in");
  assert.equal(octreeMethod.presetFor?.("balanced").brickSparseTransport, "off");
  const occupancyFlux = octreeMethod.params.find((candidate) => candidate.key === "brickSparseOccupancyFlux");
  assert.ok(occupancyFlux && occupancyFlux.kind === "select");
  assert.equal(occupancyFlux.default, "off", "atomic column reduction remains a measured opt-in");
  assert.equal(octreeMethod.presetFor?.("balanced").brickSparseOccupancyFlux, "off");
  const extrapolation = octreeMethod.params.find((candidate) => candidate.key === "brickSparseExtrapolation");
  assert.ok(extrapolation && extrapolation.kind === "select");
  assert.equal(extrapolation.default, "off", "sparse extrapolation remains opt-in until the widened-ocean A/B wins");
  assert.equal(octreeMethod.presetFor?.("balanced").brickSparseExtrapolation, "off");
  assert.match(smoke, /\["off", "mirror", "authoritative"\]/);
  assert.doesNotMatch(sparseWorld, /encodeAtlasToDense\(/,
    "authoritative remains infrastructure-only until consumers are ported");
});

test("compact authority removes the exact full-capacity target atlas payload", () => {
  const plan = planFluidBrickAtlas([320, 96, 80], { brickSize: 8, maxTextureDimension3D: 2048 });
  assert.equal(plan.capacity, 4_800);
  assert.deepEqual(plan.atlasDimensions, [170, 170, 170]);
  assert.equal(plan.allocatedTextureBytes, 98_260_000);
  assert.equal(fluidBrickAtlasAllocatedBytes(plan), 98_317_792);
});

test("velocity predictor samples bulk atlas pages while reverse transport stays dense", () => {
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  assert.match(legacyUniformComputeShader, /@group\(0\) @binding\(22\) var<storage,read> bulkAtlasPageTable/);
  assert.match(legacyUniformComputeShader, /fn bulkAtlasSampleVelocity/);
  assert.match(legacyUniformComputeShader, /let slotBrick=bulkAtlasSlot\(q\);if\(slotBrick\.x!=BULK_ATLAS_INVALID\)/);
  assert.match(uniform, /this\.advectGroup =[\s\S]*surfaceAuthority, true/,
    "the predictor consumes the previous projected bulk atlas");
  assert.match(uniform, /this\.reverseGroup =[\s\S]*this\.transportB, surfaceAuthority, false, "velocity"\) : this\.advectGroup/,
    "the reverse pass must consume its newly predicted dense transport field");
  assert.match(uniform, /const correctionSurfaceAuthority = this\.adaptiveProjection\?\.levelSetTexture \?\? this\.volumeB[\s\S]*this\.correctGroup =[\s\S]*this\.transportA, correctionSurfaceAuthority, true/,
    "the correction trace uses the original atlas authority without aliasing its uniform-path output");
  assert.match(uniform, /mode === "mirror"/,
    "authoritative infrastructure stays disabled until solver kernels write its pages");
});

test("velocity scheduling refreshes per substep and clears the retired stream before transport", () => {
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  assert.match(legacyUniformComputeShader, /workgroupLinear=wid\.x\+wid\.y\*bulkWorklist\[12\]/,
    "cell64 kernels linearize the GPU-authored two-dimensional dispatch");
  assert.match(legacyUniformComputeShader, /let entry=16u\+activeIndex\*2u/);
  assert.match(retiredBulkVelocityClearShader, /stream=gid\.x\+gid\.y\*worklist\[5\]\*256u/,
    "retirement uses the producer's distinct 256-thread dispatch");
  assert.match(retiredBulkVelocityClearShader, /16u\+atlasParams\.brickDims\.w\*2u\+retiredIndex\*2u/);
  assert.match(uniform, /for \(let substep[\s\S]*encodeBulkRefresh[\s\S]*Clear retired sparse compatibility payloads[\s\S]*dispatchTransport\(prep/,
    "the current dt refresh and retirement clear precede transport on every substep");
  assert.match(uniform, /\[this\.velocityA, this\.velocityB, this\.velocityC, this\.velocityD\]\.map/,
    "every dense velocity ping-pong is cleared exactly when its brick retires");
  assert.match(uniform, /if \(sparseBulkTargets \|\| sparseExtrapolationTargets\)/,
    "sparse extrapolation must clear retired velocity bricks even when velocity advection uses the dense A\/B path");
  assert.match(uniform, /dispatchWorkgroupsIndirect\(source\.bulkWorklist, FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES\)/);
  assert.match(uniform, /dispatchWorkgroupsIndirect\(bulkSource\.bulkWorklist, FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES\)/);
  assert.match(uniform, /new Uint32Array\(\[1, 1, 0, 0\]\)/,
    "atlas sampling and sparse target mapping have independent control lanes");
  assert.match(uniform, /new Uint32Array\(\[0, 1, 0, 0\]\)/,
    "reverse transport can schedule sparsely without sampling the original atlas");
});

test("transport preparation uses the cell64 worklist while preserving and retiring its padded shell", () => {
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  assert.match(legacyUniformComputeShader, /fn buildTransport\([^)]*global_invocation_id[^)]*\)[^{]*workgroup_id[^{]*local_invocation_index/);
  assert.match(legacyUniformComputeShader, /if\(bulkAtlasControl\.y!=0u\)[\s\S]*scheduledVelocityCell[\s\S]*padded=id\+vec3i\(1\)/);
  assert.match(legacyUniformComputeShader, /else\{[\s\S]*textureStore\(transportOut,padded,vec4f\(0\.0\)\)/,
    "the dense A/B path must continue writing the complete padded zero shell");
  assert.match(retiredBulkTransportClearShader, /texture_storage_3d<rgba16float,write>/);
  assert.match(retiredBulkTransportClearShader, /stream=gid\.x\+gid\.y\*worklist\[5\]\*256u/);
  assert.match(retiredBulkTransportClearShader, /textureStore\(transportOut,vec3i\(cell\)\+vec3i\(1\),vec4f\(0\.0\)\)/,
    "retirement clears payload coordinates without ever overwriting the permanent shell");
  assert.match(uniform, /dispatchTransport[\s\S]*FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES/);
  assert.match(uniform, /retiredTransportClearGroups = \[\.\.\.new Set\(\[this\.transportA, this\.transportB\]\)\]/);
  assert.match(uniform, /sparseTransportTargets[\s\S]*retiredTransportClearPipeline[\s\S]*FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES/);
});

test("occupancy and flux preparation preserve dense mirrors over the resident cell64 worklist", () => {
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  assert.match(legacyUniformComputeShader, /@binding\(27\) var<storage,read_write> occupancyColumns: array<atomic<u32>>/);
  assert.match(legacyUniformComputeShader, /fn buildSparseOccupancy\([^)]*global_invocation_id[^)]*\)[^{]*workgroup_id[^{]*local_invocation_index/);
  assert.match(legacyUniformComputeShader, /atomicMax\(&occupancyColumns\[column\],u32\(id\.y\+1\)\)/,
    "all active y-bricks race safely into one exact column maximum");
  assert.match(legacyUniformComputeShader, /fn resolveSparseOccupancy[\s\S]*atomicLoad\(&occupancyColumns\[column\]\)\)-1\.0/,
    "zeroed columns publish the historical -1 empty sentinel");
  assert.match(legacyUniformComputeShader, /fn buildFluxScales\([^)]*global_invocation_id[^)]*\)[^{]*workgroup_id[^{]*local_invocation_index/);
  assert.match(legacyUniformComputeShader, /if\(bulkAtlasControl\.y!=0u\)[\s\S]*scheduledVelocityCell/);
  assert.match(retiredBulkFluxScaleClearShader, /texture_storage_3d<rg32float,write>/);
  assert.match(retiredBulkFluxScaleClearShader, /textureStore\(fluxScalesOut,cell,vec4f\(0\.0,1\.0,0\.0,0\.0\)\)/,
    "retired cells restore invalid-neighbor donor/receiver semantics");
  assert.match(uniform, /encoder\.clearBuffer\(this\.occupancyColumns\)/,
    "retired and emptied columns cannot retain stale maxima");
  assert.match(uniform, /buildSparseOccupancyPipeline[\s\S]*FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES[\s\S]*resolveSparseOccupancyPipeline/);
  assert.match(uniform, /buildFluxScalesPipeline[\s\S]*fluxScaleGroup[\s\S]*FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES/);
  assert.match(uniform, /retiredFluxScaleClearPipeline[\s\S]*FLUID_BRICK_RETIRED_DISPATCH_OFFSET_BYTES/);
});

test("atlas planning packs apron tiles near-cubically inside the texture limit", () => {
  const plan = planFluidBrickAtlas([61, 46, 41], { brickSize: 8 });
  assert.deepEqual(plan.brickDimensions, [8, 6, 6]);
  assert.equal(plan.logicalBrickCount, 288);
  assert.equal(plan.tileSize, 10);
  assert.equal(plan.capacity, 288, "mirror mode backs the whole logical lattice by default");
  assert.deepEqual(plan.tileGridDimensions, [7, 7, 6]);
  assert.deepEqual(plan.atlasDimensions, [70, 70, 60]);
  assert.equal(plan.degraded, false);
  assert.equal(plan.bytesPerTile, 1000 * 20);
  assert.equal(plan.allocatedTextureBytes, 70 * 70 * 60 * 20);
  const fractional = planFluidBrickAtlas([61, 46, 41], { brickSize: 8, maximumResidentFraction: 0.75 });
  assert.equal(fractional.capacity, 216);
});

test("atlas capacity degrades against maxTextureDimension3D instead of failing", () => {
  const plan = planFluidBrickAtlas([128, 128, 128], { brickSize: 8, maximumResidentFraction: 1, maxTextureDimension3D: 40 });
  assert.equal(plan.logicalBrickCount, 4096);
  assert.deepEqual(plan.tileGridDimensions, [4, 4, 4], "only four 10-texel tiles fit per 40-texel axis");
  assert.equal(plan.capacity, 64);
  assert.equal(plan.degraded, true);
  plan.atlasDimensions.forEach((extent) => assert.ok(extent <= 40));
});

test("atlas capacity honors the hard tile ceiling and fraction", () => {
  const capped = planFluidBrickAtlas([128, 128, 128], { brickSize: 8, maximumTiles: 50 });
  assert.equal(capped.capacity, 50);
  assert.equal(capped.degraded, false);
  const fraction = planFluidBrickAtlas([64, 64, 64], { brickSize: 8, maximumResidentFraction: 0.25 });
  assert.equal(fraction.capacity, Math.ceil(512 * 0.25));
  const floor = planFluidBrickAtlas([8, 8, 8], { brickSize: 8, maximumResidentFraction: 0.0001 });
  assert.equal(floor.capacity, 1, "at least one physical tile always exists");
  assert.throws(() => planFluidBrickAtlas([0, 8, 8]), /positive integer/);
});

test("atlas shaders follow the pooled sparse-page template", () => {
  assert.match(brickAtlasLifecycleShader, /atomicExchange\(&pageTable\[brickIndex\], INVALID\)/);
  assert.match(brickAtlasLifecycleShader, /atomicStore\(&control\[2\], 1u\)/, "pool exhaustion raises the overflow flag");
  assert.match(brickAtlasLifecycleShader, /atomicSub\(&control\[0\], 1u\)/);
  assert.match(brickAtlasLifecycleShader, /let x = min\(blocks, 65535u\)/, "indirect work tiles into two dimensions");
  assert.doesNotMatch(brickAtlasLifecycleShader, /states\[pageIndex\] = state & ~RESIDENT/, "atlas overflow must not clear residency it does not own");
  assert.match(brickAtlasMirrorShader, /vec3i\(local\) - vec3i\(1\)/, "tiles mirror a one-voxel apron around the payload");
  assert.match(brickAtlasMirrorShader, /fn initializeActivated/, "authoritative imports only newly allocated pages");
  assert.match(brickAtlasMirrorShader, /residencyStates\[brickIndex\] & ACTIVATED/, "authoritative import is activation-gated");
  assert.match(brickAtlasMirrorShader, /clamp\(cell, vec3i\(0\), vec3i\(atlasParams\.dims\.xyz\) - vec3i\(1\)\)/, "apron reads clamp to the dense domain like the dense sampler");
  assert.match(brickAtlasToDenseShader, /fn mirrorAtlasToDense/);
  assert.match(brickAtlasToDenseShader, /textureStore\(denseLevelSet, cell, vec4f\(AIR_PHI/, "missing pages publish air");
  assert.match(brickAtlasToDenseShader, /textureStore\(denseVelocity, cell, vec4f\(0\.0\)\)/, "missing pages publish zero velocity");
  for (const filterable of [true, false]) {
    const sampling = fluidBrickAtlasSamplingWGSL(filterable);
    assert.match(sampling, /fn brickAtlasSamplePhi\(position: vec3f\) -> f32/);
    assert.match(sampling, /fn brickAtlasSampleVelocity\(position: vec3f\) -> vec3f/);
    assert.match(sampling, /brickAtlasDensePhi/, "missing slots fall back to the dense field");
    const validate = brickAtlasValidateShader(filterable);
    assert.match(validate, /fn compareAtlasToDense/);
    assert.match(validate, /atomicMax\(&stats\[index\], bitcast<u32>/);
  }
  for (const filterable of [true, false]) {
    const authoritative = fluidBrickAtlasSamplingWGSL(filterable, "authoritative");
    assert.match(authoritative, new RegExp(`BRICK_ATLAS_AIR_PHI: f32 = ${BRICK_ATLAS_AIR_PHI}\\.0`));
    assert.match(authoritative, /return BRICK_ATLAS_AIR_PHI/);
    assert.match(authoritative, /return vec3f\(0\.0\)/);
    assert.doesNotMatch(authoritative, /texture(?:Load|SampleLevel)\(dense(?:LevelSet|Velocity)/,
      "authoritative sampling has no hidden dense fallback dependency");
  }
  assert.match(fluidBrickAtlasSamplingWGSL(true), /textureSampleLevel\(brickAtlasPhi, brickAtlasSampler/, "filterable devices use one hardware trilinear fetch");
});

test("residency pre-activation entry points exist and widen support by swept velocity", () => {
  assert.match(fluidBrickResidencyShader, /fn classifySwept/);
  assert.match(fluidBrickResidencyShader, /fn expandDownstream/);
  assert.match(fluidBrickResidencyShader, /fn emitWorklist/);
  assert.match(fluidBrickResidencyShader, /params\.settings\.z \* 1\.5/, "swept support = |v| dt with a 1.5 safety factor");
  assert.match(fluidBrickResidencyShader, /max\(params\.settings\.x, sweptSupport\)/);
  assert.match(fluidBrickResidencyShader, /minimumPhi <= 0\.0 && maximumPhi >= 0\.0/, "legacy classify keeps its exact band semantics");
});

async function createDevice() {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredFeatures: optionalFluidDeviceFeatures(adapter.features),
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  return { device, validationErrors };
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number) {
  const readback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return words;
}

async function readTextureFloats(
  device: GPUDevice,
  texture: GPUTexture,
  dimensions: readonly [number, number, number],
  components: 1 | 2 | 4,
) {
  const bytesPerRow = Math.ceil(dimensions[0] * components * 4 / 256) * 256;
  const readback = device.createBuffer({
    size: bytesPerRow * dimensions[1] * dimensions[2],
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer: readback, bytesPerRow, rowsPerImage: dimensions[1] },
    [...dimensions],
  );
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const raw = readback.getMappedRange();
  const output = new Float32Array(dimensions[0] * dimensions[1] * dimensions[2] * components);
  const source = new DataView(raw);
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    const sourceOffset = (z * dimensions[1] + y) * bytesPerRow + x * components * 4;
    const targetOffset = (x + dimensions[0] * (y + dimensions[1] * z)) * components;
    for (let component = 0; component < components; component += 1) {
      output[targetOffset + component] = source.getFloat32(sourceOffset + component * 4, true);
    }
  }
  readback.unmap();
  readback.destroy();
  return output;
}

test("GPU authoritative atlas preserves resident payloads and clears missing dense pages to air", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU atlas checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const dimensions = [16, 4, 4] as const;
    const residency = new GPUFluidBrickResidency(device, dimensions, [0.1, 0.1, 0.1], {
      brickSize: 8,
      haloCells: 1,
      retireAfterFrames: 0,
    });
    // Only the first of two logical bricks is resident and newly activated.
    device.queue.writeBuffer(residency.stateBuffer, 0, new Uint32Array([FLUID_BRICK_RESIDENT | FLUID_BRICK_ACTIVATED, 0]));
    const atlas = new WebGPUFluidBrickAtlas(device, dimensions, residency, {
      brickSize: 8,
      mode: "authoritative",
      validate: false,
    });
    assert.equal(atlas.mode, "authoritative");
    const source = atlas.getSamplingSource();
    assert.equal(source.mode, "authoritative");
    assert.equal(source.pageTable, atlas.pageTable);
    assert.equal(source.phi, atlas.phiView);
    assert.equal(source.velocity, atlas.velocityView);

    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const levelSet = device.createTexture({ size: [...dimensions], dimension: "3d", format: "r32float", usage });
    const velocity = device.createTexture({ size: [...dimensions], dimension: "3d", format: "rgba32float", usage });
    const cells = dimensions[0] * dimensions[1] * dimensions[2];
    const upload = (residentPhi: number, missingPhi: number, residentVelocity: number, missingVelocity: number) => {
      const phi = new Float32Array(cells);
      const vel = new Float32Array(cells * 4);
      for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
        const index = x + dimensions[0] * (y + dimensions[1] * z);
        phi[index] = x < 8 ? residentPhi : missingPhi;
        vel[index * 4] = x < 8 ? residentVelocity : missingVelocity;
      }
      device.queue.writeTexture({ texture: levelSet }, phi, { bytesPerRow: dimensions[0] * 4, rowsPerImage: dimensions[1] }, [...dimensions]);
      device.queue.writeTexture({ texture: velocity }, vel, { bytesPerRow: dimensions[0] * 16, rowsPerImage: dimensions[1] }, [...dimensions]);
    };
    upload(-2, -9, 3, 7);
    let encoder = device.createCommandEncoder();
    atlas.encode(encoder, levelSet, velocity);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Existing authoritative pages must not be re-imported on later frames.
    device.queue.writeBuffer(residency.stateBuffer, 0, new Uint32Array([FLUID_BRICK_RESIDENT, 0]));
    upload(-22, -99, 13, 17);
    encoder = device.createCommandEncoder();
    atlas.encode(encoder, levelSet, velocity);
    atlas.encodeAtlasToDense(encoder, levelSet, velocity);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const phi = await readTextureFloats(device, levelSet, dimensions, 1);
    const vel = await readTextureFloats(device, velocity, dimensions, 4);
    const resident = 3 + dimensions[0] * (2 + dimensions[1]);
    const missing = 11 + dimensions[0] * (2 + dimensions[1]);
    assert.equal(phi[resident], -2, "existing resident page remains authoritative");
    assert.equal(vel[resident * 4], 3);
    assert.equal(phi[missing], BRICK_ATLAS_AIR_PHI, "nonresident dense cells cannot retain stale liquid phi");
    assert.equal(vel[missing * 4], 0, "nonresident dense cells cannot retain stale velocity");
    const stats = await atlas.readStats();
    assert.equal(stats.residentTiles, 1, "atlas reports its own bulk-residency page count");
    assert.equal(stats.overflow, 0);
    levelSet.destroy();
    velocity.destroy();
    atlas.destroy();
    residency.destroy();
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});

test("GPU pre-activation schedules downstream and swept bricks before phi arrives", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU atlas checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const dims = [24, 8, 8] as const;
    const cell = [0.1, 0.1, 0.1] as const;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
    const levelSet = device.createTexture({ size: [...dims], dimension: "3d", format: "r32float", usage });
    const velocity = device.createTexture({ size: [...dims], dimension: "3d", format: "rgba32float", usage });
    // Brick 0 (x 0..7) straddles phi=0; bricks 1 and 2 sit far outside the
    // two-cell halo band (minimum phi 0.45 m and 1.25 m against 0.2 m).
    const phi = new Float32Array(dims[0] * dims[1] * dims[2]);
    for (let z = 0; z < dims[2]; z += 1) for (let y = 0; y < dims[1]; y += 1) for (let x = 0; x < dims[0]; x += 1) {
      phi[x + dims[0] * (y + dims[1] * z)] = x < 4 ? -0.05 : (x - 4 + 0.5) * cell[0];
    }
    device.queue.writeTexture({ texture: levelSet }, phi, { bytesPerRow: dims[0] * 4, rowsPerImage: dims[1] }, [...dims]);
    const writeVelocity = (vx: number) => {
      const data = new Float32Array(dims[0] * dims[1] * dims[2] * 4);
      for (let i = 0; i < data.length; i += 4) data[i] = vx;
      device.queue.writeTexture({ texture: velocity }, data, { bytesPerRow: dims[0] * 16, rowsPerImage: dims[1] }, [...dims]);
    };
    const run = async (vx: number, preActivation: boolean, dt_s: number) => {
      const residency = new GPUFluidBrickResidency(device, dims, cell, { brickSize: 8, haloCells: 2, retireAfterFrames: 0 });
      writeVelocity(vx);
      const encoder = device.createCommandEncoder();
      residency.encode(encoder, levelSet, velocity, { dt_s, preActivation });
      device.queue.submit([encoder.finish()]);
      const states = await readBuffer(device, residency.stateBuffer, 3 * 4);
      residency.destroy();
      return states;
    };
    // Baseline: no pre-activation keeps the dry downstream bricks unscheduled.
    const baseline = await run(2, false, 0.05);
    assert.ok((baseline[0] & FLUID_BRICK_RESIDENT) !== 0, "the interface brick is resident");
    assert.equal(baseline[1] & FLUID_BRICK_RESIDENT, 0);
    assert.equal(baseline[2] & FLUID_BRICK_RESIDENT, 0);
    // Downstream expansion: swept support 2*0.05*1.5 = 0.15 m cannot reach
    // brick 1 (0.45 m), so residency must come from the core neighbor whose
    // face velocity points into it — before any of its cells are wet.
    const downstream = await run(2, true, 0.05);
    assert.ok((downstream[1] & FLUID_BRICK_RESIDENT) !== 0, "downstream neighbor of a core brick pre-activates");
    assert.ok((downstream[1] & FLUID_BRICK_ACTIVATED) !== 0);
    assert.equal(downstream[2] & FLUID_BRICK_RESIDENT, 0, "bricks beyond the immediate downstream neighbor stay unscheduled");
    // Swept support: upstream flow disables the neighbor rule, but a fast
    // front widens the band itself: 8*0.05*1.5 = 0.6 m > 0.45 m.
    const upstreamSlow = await run(-2, true, 0.05);
    assert.equal(upstreamSlow[1] & FLUID_BRICK_RESIDENT, 0, "upstream flow must not pre-activate the neighbor");
    const upstreamFast = await run(-8, true, 0.05);
    assert.ok((upstreamFast[1] & FLUID_BRICK_RESIDENT) !== 0, "swept support widens the residency band");
    levelSet.destroy();
    velocity.destroy();
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});

test("GPU residency derives brick worklists directly from compact surface candidates", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU atlas checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const residency = new GPUFluidBrickResidency(device, [16, 8, 8], [0.1, 0.1, 0.1], {
      brickSize: 8,
      haloCells: 2,
      retireAfterFrames: 0,
    });
    const storage = (label: string, data: Uint32Array<ArrayBuffer>, indirect = false) => {
      const buffer = device.createBuffer({
        label,
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
          | (indirect ? GPUBufferUsage.INDIRECT : 0),
      });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    // One 8-cell compact surface leaf and one CORE candidate cover only the
    // first of the two logical x bricks. Candidate-control words 1..3 are the
    // producer-authored indirect dispatch consumed by encodeSurfaceCandidates.
    const leaves = storage("Compact surface leaf", new Uint32Array([
      0, 8, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ]));
    const candidates = storage("Compact surface candidate", new Uint32Array([0, 2]));
    const candidateControl = storage("Compact surface candidate control", new Uint32Array([1, 1, 1, 1]), true);
    const encoder = device.createCommandEncoder();
    residency.encodeSurfaceCandidates(encoder, leaves, candidates, candidateControl);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const states = await readBuffer(device, residency.stateBuffer, 8);
    const stats = await residency.readStats();
    assert.ok((states[0] & FLUID_BRICK_RESIDENT) !== 0);
    assert.equal(states[1] & FLUID_BRICK_RESIDENT, 0);
    assert.equal(stats.resident, 1);
    assert.equal(stats.core, 1);

    // The bulk topology scheduler is a persistent union: a surface candidate
    // may add brick zero, but the bootstrap liquid in brick one must survive
    // both that merge and a later publication with no surface candidates.
    const bulkResidency = new GPUFluidBrickResidency(device, [16, 8, 8], [0.1, 0.1, 0.1], {
      brickSize: 8,
      haloCells: 2,
      retireAfterFrames: 0,
      includeLiquidInterior: true,
    });
    device.queue.writeBuffer(bulkResidency.stateBuffer, 4, new Uint32Array([FLUID_BRICK_RESIDENT]));
    let bulkEncoder = device.createCommandEncoder();
    bulkResidency.encodeSurfaceCandidates(bulkEncoder, leaves, candidates, candidateControl);
    device.queue.submit([bulkEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    let bulkStates = await readBuffer(device, bulkResidency.stateBuffer, 8);
    assert.ok((bulkStates[0] & FLUID_BRICK_RESIDENT) !== 0, "moving surface support adds a bulk brick");
    assert.ok((bulkStates[1] & FLUID_BRICK_RESIDENT) !== 0, "bootstrap deep-liquid brick remains resident");
    device.queue.writeBuffer(candidateControl, 0, new Uint32Array([0, 0, 1, 1]));
    bulkEncoder = device.createCommandEncoder();
    bulkResidency.encodeSurfaceCandidates(bulkEncoder, leaves, candidates, candidateControl);
    device.queue.submit([bulkEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    bulkStates = await readBuffer(device, bulkResidency.stateBuffer, 8);
    assert.ok((bulkStates[0] & FLUID_BRICK_RESIDENT) !== 0);
    assert.ok((bulkStates[1] & FLUID_BRICK_RESIDENT) !== 0);
    assert.equal((await bulkResidency.readStats()).resident, 2);
    bulkResidency.destroy();
    leaves.destroy();
    candidates.destroy();
    candidateControl.destroy();
    residency.destroy();
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});

test("GPU atlas mirrors dam-break fields exactly, including brick seams, and pre-activates ahead of the front", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU atlas checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const scene: SceneDescription = cloneScene(defaultScene);
    scene.sceneId = "test-brick-atlas-dam-break";
    scene.fluid.initialCondition = "dam-break";
    delete scene.fluid.inflow;
    scene.rigidBodies = [];
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
    const values = Object.fromEntries(octreeMethod.params.map((parameter) => [parameter.key, "default" in parameter ? parameter.default : 0])) as Record<string, string | number | boolean>;
    // This fixture verifies the legacy dense velocity mirror consumed by the
    // atlas, so opt out of compact face-velocity authority explicitly.
    values.faceVelocityTransport = "off";
    const previousDirectPagedPhi = process.env.FLUID_OCTREE_DIRECT_PAGED_PHI;
    process.env.FLUID_OCTREE_DIRECT_PAGED_PHI = "0";
    const solver = octreeMethod.createSolver!(device, scene, "balanced", values, undefined) as unknown as {
      advanceTo(time_s: number, bodies: never[]): boolean;
      readStats(): Promise<Record<string, number | undefined>>;
      destroy(): void;
    };
    if (previousDirectPagedPhi === undefined) delete process.env.FLUID_OCTREE_DIRECT_PAGED_PHI;
    else process.env.FLUID_OCTREE_DIRECT_PAGED_PHI = previousDirectPagedPhi;
    const internals = solver as unknown as {
      octreeProjection: {
        levelSetTexture: GPUTexture;
        sparseBrickWorld: { residency: GPUFluidBrickResidency; atlas?: { plan: { capacity: number } } };
      };
    };
    const residency = internals.octreeProjection.sparseBrickWorld.residency;
    const capacity = residency.capacity;
    const [bx, by] = residency.brickDimensions;
    const dims = [61, 46, 41] as const;
    const bytesPerRow = Math.ceil(dims[0] * 4 / 256) * 256;
    const phiReadback = device.createBuffer({ size: bytesPerRow * dims[1] * dims[2], usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readPhiPerBrick = async () => {
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture: internals.octreeProjection.levelSetTexture },
        { buffer: phiReadback, bytesPerRow, rowsPerImage: dims[1] },
        [...dims],
      );
      device.queue.submit([encoder.finish()]);
      await phiReadback.mapAsync(GPUMapMode.READ);
      const raw = new Uint8Array(phiReadback.getMappedRange().slice(0));
      phiReadback.unmap();
      const minimumPhi = new Float32Array(capacity).fill(Number.POSITIVE_INFINITY);
      const view = new DataView(raw.buffer);
      for (let z = 0; z < dims[2]; z += 1) for (let y = 0; y < dims[1]; y += 1) for (let x = 0; x < dims[0]; x += 1) {
        const value = view.getFloat32((z * dims[1] + y) * bytesPerRow + x * 4, true);
        const brick = Math.floor(x / 8) + bx * (Math.floor(y / 8) + by * Math.floor(z / 8));
        if (value < minimumPhi[brick]) minimumPhi[brick] = value;
      }
      return minimumPhi;
    };
    const nextTurn = () => new Promise<void>((resolve) => setImmediate(resolve));
    const dt = scene.numerics.fixedDt_s!;
    const firstResident = new Int32Array(capacity).fill(-1);
    const firstWet = new Int32Array(capacity).fill(-1);
    const steps = 50;
    let time = 0;
    for (let step = 0; step < steps; step += 1) {
      while (!solver.advanceTo(time + dt, [])) await nextTurn();
      time += dt;
      await device.queue.onSubmittedWorkDone();
      const states = await readBuffer(device, residency.stateBuffer, capacity * 4);
      const minimumPhi = await readPhiPerBrick();
      for (let brick = 0; brick < capacity; brick += 1) {
        if (firstResident[brick] < 0 && (states[brick] & FLUID_BRICK_RESIDENT) !== 0) firstResident[brick] = step;
        if (firstWet[brick] < 0 && minimumPhi[brick] < 0) firstWet[brick] = step;
      }
    }
    // Pre-activation invariant: no brick may get wet without having been
    // scheduled on an earlier step (bricks wet from the first sample carry the
    // initial condition and are excluded).
    let lateWetBricks = 0;
    for (let brick = 0; brick < capacity; brick += 1) {
      if (firstWet[brick] <= 0) continue;
      lateWetBricks += 1;
      assert.ok(firstResident[brick] >= 0 && firstResident[brick] < firstWet[brick],
        `brick ${brick} became wet at step ${firstWet[brick]} but resident only at ${firstResident[brick]}`);
    }
    assert.ok(firstWet.some((step) => step === 0), "the initial dam must populate wet atlas bricks");
    const stats = await solver.readStats();
    assert.ok((stats.fluidBrickAtlasResidentTiles ?? 0) > 0, "resident bricks hold atlas tiles");
    assert.equal(stats.fluidBrickAtlasOverflow, 0);
    assert.ok((stats.fluidBrickAtlasResidentTiles ?? 0) >= (stats.fluidBrickResidentCount ?? 0),
      "bulk atlas residency may include liquid-interior bricks beyond the narrow surface band");
    assert.ok((stats.fluidBrickAtlasCapacity ?? 0) <= internals.octreeProjection.sparseBrickWorld.atlas!.plan.capacity);
    // Mirror-mode round trip: FP32 trilinear through the atlas tile (payload +
    // apron) must reproduce dense sampling exactly, including across seams.
    assert.ok((stats.fluidBrickAtlasMaxPhiErrorManual ?? 1) <= 1e-5,
      `atlas phi round-trip error ${stats.fluidBrickAtlasMaxPhiErrorManual}`);
    assert.ok((stats.fluidBrickAtlasMaxVelocityErrorManual ?? 1) <= 1e-5,
      `atlas velocity round-trip error ${stats.fluidBrickAtlasMaxVelocityErrorManual}`);
    if (device.features.has("float32-filterable" as GPUFeatureName)) {
      // Hardware trilinear quantizes footprint weights; bound the error by a
      // small multiple of the per-cell field variation.
      assert.ok((stats.fluidBrickAtlasMaxPhiError ?? 1) <= 0.02,
        `hardware phi sampling error ${stats.fluidBrickAtlasMaxPhiError}`);
      assert.ok((stats.fluidBrickAtlasMaxVelocityError ?? 1) <= 0.05,
        `hardware velocity sampling error ${stats.fluidBrickAtlasMaxVelocityError}`);
    }
    phiReadback.destroy();
    solver.destroy();
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});

test("GPU sparse occupancy republishes the dense column maxima exactly", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU occupancy checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const scene: SceneDescription = cloneScene(defaultScene);
    scene.sceneId = "test-sparse-column-occupancy";
    scene.fluid.initialCondition = "dam-break";
    delete scene.fluid.inflow;
    scene.rigidBodies = [];
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
    const run = async (sparse: boolean) => {
      const values = Object.fromEntries(octreeMethod.params.map((parameter) => [parameter.key, "default" in parameter ? parameter.default : 0])) as Record<string, string | number | boolean>;
      values.brickSparseOccupancyFlux = sparse ? "on" : "off";
      // This is a compatibility-height A/B. Production direct-page authority
      // intentionally allocates no duplicate sparse-world/atlas object.
      const previousDirectPagedPhi = process.env.FLUID_OCTREE_DIRECT_PAGED_PHI;
      process.env.FLUID_OCTREE_DIRECT_PAGED_PHI = "0";
      const solver = octreeMethod.createSolver!(device, scene, "balanced", values, undefined) as unknown as {
        info: { nx: number; nz: number };
        heightB: GPUTexture;
        octreeProjection: { sparseBrickWorld: { bulkResidencyWorklist: GPUBuffer } };
        advanceTo(time_s: number, bodies: never[]): boolean;
        destroy(): void;
      };
      if (previousDirectPagedPhi === undefined) delete process.env.FLUID_OCTREE_DIRECT_PAGED_PHI;
      else process.env.FLUID_OCTREE_DIRECT_PAGED_PHI = previousDirectPagedPhi;
      assert.equal(solver.advanceTo(0.004, []), true);
      await device.queue.onSubmittedWorkDone();
      const header = await readBuffer(device, solver.octreeProjection.sparseBrickWorld.bulkResidencyWorklist, 80);
      assert.ok(header[0] > 0 && header[12] > 0 && header[16] < 1_000_000, "bulk cell64 worklist is populated before occupancy");
      const heights = await readTextureFloats(device, solver.heightB, [solver.info.nx, solver.info.nz, 1], 2);
      solver.destroy();
      return heights;
    };
    const dense = await run(false);
    const sparse = await run(true);
    assert.deepEqual(sparse, dense, "resident atomics and area resolve preserve every compatibility height texel");
    assert.deepEqual(validationErrors, []);
  } finally {
    device.destroy();
  }
});
