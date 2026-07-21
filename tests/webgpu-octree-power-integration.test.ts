import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { octreePowerVolumeShader, resolveOctreePowerProjectionPolicy } from "../lib/webgpu-octree";
import { WebGPUOctreePowerDescriptor } from "../lib/webgpu-octree-power-descriptor";
import { WebGPUOctreePowerFaces } from "../lib/webgpu-octree-power-faces";
import { WebGPUOctreePowerOperator } from "../lib/webgpu-octree-power-operator";
import { WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";

test("power projection policy keeps authority fail-closed while enabling the mirror", () => {
  assert.deepEqual(resolveOctreePowerProjectionPolicy(undefined, [1, 1, 1], false, 0), {
    requested: "off", mirrorEnabled: false, authoritative: false,
  });
  assert.deepEqual(resolveOctreePowerProjectionPolicy("mirror", [1, 1, 1], false, 0), {
    requested: "mirror", mirrorEnabled: true, authoritative: false,
  });
  const authority = resolveOctreePowerProjectionPolicy("authoritative", [1, 1.1, 1], true, 1);
  assert.equal(authority.mirrorEnabled, true); assert.equal(authority.authoritative, false);
  assert.match(authority.fallbackReason!, /isotropic/); assert.match(authority.fallbackReason!, /terrain/);
  assert.match(authority.fallbackReason!, /rigid/); assert.match(authority.fallbackReason!, /transferred compact face velocity/);
  assert.deepEqual(resolveOctreePowerProjectionPolicy("authoritative", [1, 1, 1], false, 0, true), {
    requested: "authoritative", mirrorEnabled: true, authoritative: true,
  });
  assert.deepEqual(resolveOctreePowerProjectionPolicy("authoritative", [1, 1, 1], false, 1, true, true), {
    requested: "authoritative", mirrorEnabled: true, authoritative: true,
  });
  assert.match(resolveOctreePowerProjectionPolicy("authoritative", [1, 1, 1], true, 0, true, true)
    .fallbackReason!, /cell-vertex solid SDF/, "terrain remains fail-closed until its embedded-boundary geometry is supported");
  assert.match(resolveOctreePowerProjectionPolicy("authoritative", [1, 1, 1], true, 0, true, true)
    .fallbackReason!, /canonical compact-face rollback seed/,
  "terrain must not become authoritative from face-aperture quadrature alone");
  assert.deepEqual(resolveOctreePowerProjectionPolicy(
    "authoritative", [1, 1, 1], true, 0, true, true, false, true, true,
  ), { requested: "authoritative", mirrorEnabled: true, authoritative: true },
  "terrain authority opens only when both vertex SDF and rollback seed inputs exist");
  const imported = resolveOctreePowerProjectionPolicy(
    "authoritative", [1, 1, 1], false, 0, true, true, true,
  );
  assert.equal(imported.authoritative, false);
  assert.match(imported.fallbackReason!, /imported\/seeded geometry/,
    "imported and explicitly seeded shapes retain the compatibility projection");
});

test("Dawn publishes physical power-cell volume from normalized topology metrics", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-integration checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals); const nativeGpu = dawn.create(["backend=metal"]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  const shader = device.createShaderModule({ code: octreePowerVolumeShader });
  assert.deepEqual((await shader.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint: "publishPowerVolumes" } });
  const upload = (data: ArrayBufferView, usage = GPUBufferUsage.STORAGE) => {
    const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange(), 0, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap(); return buffer;
  };
  const paramsData = new ArrayBuffer(16); new Float32Array(paramsData)[0] = 0.125; new Uint32Array(paramsData)[1] = 1;
  const params = upload(new Uint32Array(paramsData), GPUBufferUsage.UNIFORM);
  const metricWords = new Uint32Array(4); new Float32Array(metricWords.buffer)[2] = 0.75;
  const metrics = upload(metricWords); const headerWords = new Uint32Array(12); headerWords[3] = 2;
  const headers = upload(headerWords); const count = upload(new Uint32Array([1]));
  const volumes = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: metrics } },
    { binding: 2, resource: { buffer: headers } }, { binding: 3, resource: { buffer: count } },
    { binding: 4, resource: { buffer: volumes } },
  ] });
  const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(volumes, 0, readback, 0, 16); device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone(); await readback.mapAsync(GPUMapMode.READ);
  assert.ok(Math.abs(new Float32Array(readback.getMappedRange())[0] - 0.75) < 1e-6); readback.unmap();
  params.destroy(); metrics.destroy(); headers.destroy(); count.destroy(); volumes.destroy(); readback.destroy(); device.destroy();
});

test("Dawn compiles and submits the complete GPU-resident power mirror graph", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-integration checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals); const nativeGpu = dawn.create(["backend=metal"]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  const bytes = readFileSync(join(process.cwd(), "lib/generated/octree-power-catalog.bin"));
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const descriptor = new WebGPUOctreePowerDescriptor(device, 1);
  const topology = new WebGPUOctreePowerTopology(device, 1, catalog);
  const faces = new WebGPUOctreePowerFaces(device, 1, 30, topology.source, 60);
  const operator = new WebGPUOctreePowerOperator(device, 1, 30, 30, 30);
  const storage = (label: string, size: number) => device.createBuffer({
    label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const headers = storage("integration leaf headers", 48);
  const owners = storage("integration dense owners", 4);
  const volumes = storage("integration physical volumes", 4);
  const pressure = storage("integration pressure", 4);
  const entries = storage("integration leaf entries", 30 * 8);
  const encoder = device.createCommandEncoder();
  descriptor.encode(encoder, headers, owners, {
    dimensions: [2, 2, 2], maximumLeafSize: 2, rowCount: 0, generation: 1, ownerMode: "dense",
  });
  topology.encode(encoder, descriptor.descriptors, 0, [1, 1, 1]);
  faces.encode(encoder, headers, { dimensions: [2, 2, 2], rowCount: 0, physicalCellSize: 1, generation: 1 });
  operator.encodeAssemblyFromControl(encoder, faces.faces, faces.source, volumes, faces.control);
  operator.encodeLeafRowPublication(encoder, headers, entries);
  operator.encodeProjectionFromControl(encoder, faces.faces, faces.source, pressure, faces.control);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  descriptor.destroy(); topology.destroy(); faces.destroy(); operator.destroy();
  headers.destroy(); entries.destroy(); owners.destroy(); volumes.destroy(); pressure.destroy(); device.destroy();
});

test("production dam-break RasterWaterPipeline initializes exact layouts and rasterizes signed global fine A", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the production power regression",
  timeout: 60_000,
}, () => {
  const child = spawnSync(process.execPath, ["--import", "tsx", "tools/run-webgpu-smoke.ts"], {
    cwd: process.cwd(), encoding: "utf8", timeout: 25_000, killSignal: "SIGKILL", maxBuffer: 32 * 1024 * 1024,
    env: { ...process.env, FLUID_SCENE: "dam-break-ui", FLUID_METHOD: "octree",
      FLUID_TARGET_S: "0.004", FLUID_ORACLE_STEPS: "1", FLUID_SURFACE_COLUMNS: "2400",
      FLUID_PRESSURE_CYCLES: "400", FLUID_STABILITY_ENVELOPE: "1", FLUID_CPU_ORACLE: "0",
      FLUID_FIELD_STATS: "0", FLUID_DISABLE_TIMESTAMPS: "1", FLUID_OCTREE_FACE_TRANSPORT: "1",
      FLUID_OCTREE_POWER_PROJECTION: "authoritative", FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
      FLUID_GLOBAL_FINE_GENERATION_TRANSITION: "1" },
  });
  assert.equal(child.error, undefined, `production transition process failed: ${child.error?.message ?? "unknown"}`);
  assert.equal(child.status, 0, `production transition smoke failed:\n${child.stderr}\n${child.stdout.slice(-8_000)}`);
  const result = child.stdout.split("\n").flatMap((line) => {
    try { const value = JSON.parse(line) as Record<string, unknown>; return value.phase === "result" ? [value] : []; }
    catch { return []; }
  }).at(-1) as Record<string, any> | undefined;
  assert.ok(result, "production transition emitted no result record");

  const generation = result.finalGlobalFineGeneration;
  assert.equal(generation?.publicationValid, true, "A is not a published global fine generation");
  assert.ok(generation?.generation > 0 && generation?.activePages > 0,
    "A published no globally indexed fine pages");
  assert.equal(generation?.taggedMetadataPages, generation?.activePages,
    "A page table and physical metadata generation tags disagree");
  assert.equal(generation?.malformedActivePages, 0, "A contains malformed active fine pages");
  assert.ok(generation?.validSamples > 0 && generation?.finiteValidSamples === generation?.validSamples,
    "A contains no finite valid fine samples");
  assert.ok(generation?.negativeValidSamples > 0 && generation?.positiveValidSamples > 0,
    "A does not straddle the liquid interface");

  const descriptor = result.octreePowerTopologyDiagnostics?.descriptor;
  const topology = result.octreePowerTopologyDiagnostics?.topology;
  const faces = result.octreePowerFaceDiagnostics;
  assert.ok(result.pressureRequiredRows > 0 && result.pressureRequiredEntries > 0,
    "production pressure rows collapsed");
  assert.equal(descriptor?.validCount, descriptor?.rowCount, "power descriptor publication is partial");
  assert.equal(topology?.resolvedCount, descriptor?.rowCount, "power topology publication is partial");
  assert.equal(faces?.valid, true, "power-face generation is invalid");
  assert.ok(faces?.rowCount > 0 && faces?.faceCount > 0 && faces?.incidenceCount > 0,
    "power rows or faces collapsed to zero");

  const raster = result.finalGlobalFineRaster;
  const transition = raster?.globalFineAuthorityTransition;
  assert.equal(raster?.rendererValidationErrorCount, 0,
    "production RasterWaterPipeline initialization or fine raster encode raised a scoped validation error");
  assert.equal(raster?.rendererUncapturedErrorCount, 0,
    "production RasterWaterPipeline initialization or fine raster encode raised an uncaptured GPU error");
  assert.ok(raster?.frontInterfacePixels > 0 && raster?.backInterfacePixels > 0,
    "clean fine/coarse A rasterized no closed water interface");
  assert.equal(raster?.surfaceGeometrySource, "global-fine-coarse",
    "A raster did not use the clean current fine/coarse publication");
  assert.equal(raster?.globalFineCrossingPublished, true,
    "A raster did not publish a current global crossing");
  assert.equal(raster?.presentationFallbackActive, false,
    "A raster used presentation fallback geometry");
  assert.ok((raster?.globalFineAuthorityLatch ?? 0) > 0,
    "A raster did not latch global fine/coarse authority");
  assert.equal(transition?.cleanFineCoarseRequired, true,
    "A raster did not require the clean compact-coarse member");
  assert.equal(transition?.validGeneration, generation?.generation,
    "renderer A generation differs from the published fine generation");
  assert.equal(transition?.unpublishedGeneration, generation?.generation + 1,
    "renderer B was not the immediate unpublished generation");
  assert.equal(transition?.retainedGeometrySource, "retained-previous",
    "unpublished B did not select retained presentation geometry");
  assert.equal(transition?.retainedFrontInterfacePixels, raster?.frontInterfacePixels,
    "unpublished B did not retain A's front raster");
  assert.equal(transition?.retainedBackInterfacePixels, raster?.backInterfacePixels,
    "unpublished B did not retain A's back raster");
  assert.equal(transition?.retainedFrontInterfaceHash, raster?.frontInterfaceHash,
    "unpublished B changed A's front vertex/raster content");
  assert.equal(transition?.retainedBackInterfaceHash, raster?.backInterfaceHash,
    "unpublished B changed A's back vertex/raster content");
  assert.deepEqual(transition?.retainedFrontInterfaceBounds_m, raster?.frontInterfaceBounds_m,
    "unpublished B changed A's extracted interface bounds");
  assert.ok(Number.isFinite(result.pressureRelativeResidual), "pressure residual is non-finite");
  assert.equal(result.nonFiniteCount, 0, "production publication contains non-finite values");
});
