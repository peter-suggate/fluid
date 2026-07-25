import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import { fineLevelSetJFACPTWGSL,
  WebGPUFineLevelSetRedistance } from "../lib/webgpu-octree-fine-levelset-redistance";
import { fineLevelSetSummaryWGSL,
  WebGPUFineLevelSetSummaries } from "../lib/webgpu-octree-fine-levelset-summary";
import { makeFineLevelSetTopologyWGSL, WebGPUFineLevelSetLeafSeeds,
  WebGPUFineLevelSetTopology } from "../lib/webgpu-octree-fine-levelset-topology";
import { fineLevelSetFusedTransportPublicationWGSL,
  WebGPUFineLevelSetTransport } from "../lib/webgpu-octree-fine-levelset-transport";
import { makeFineLevelSetProductionFusedTransportWGSL } from
  "../lib/webgpu-octree-fine-levelset-fused-transport";
import { fineToCoarseLevelSetWGSL,
  WebGPUFineToCoarseLevelSet } from "../lib/webgpu-octree-fine-to-coarse-levelset";
import { fineLevelSetVolumeCorrectionWGSL,
  WebGPUFineLevelSetVolumeCorrection } from "../lib/webgpu-octree-fine-levelset-volume";
import { WebGPUOctreeFaceClosestPointExtension } from
  "../lib/webgpu-octree-face-closest-point";
import type { OctreePowerFaceSource } from "../lib/webgpu-octree-power-faces";
import type { OctreePowerTopologySource } from "../lib/webgpu-octree-power-topology";
import { makePowerVelocityPrepassBuilderWGSL,
  WebGPUOctreePowerVelocityPrepass } from "../lib/webgpu-octree-power-velocity-prepass";

const MAX_PORTABLE_STORAGE_BINDINGS = 10;

function makeMinimalPowerPrepassSources(device: GPUDevice): {
  topology: OctreePowerTopologySource;
  faces: OctreePowerFaceSource;
  destroy(): void;
} {
  const buffers: GPUBuffer[] = [];
  const storage = (size: number) => {
    const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    buffers.push(buffer);
    return buffer;
  };
  const metrics = storage(16), topologyControl = storage(32);
  const catalogEntryHeaders = storage(16), catalogVolumes = storage(4), catalogFaces = storage(16);
  const catalogCoefficients = storage(19 * 4);
  const catalogTetrahedronHeaders = storage(12), catalogTetrahedra = storage(16);
  const catalogTetrahedronVertices = storage(16), catalogLookup = storage(4);
  const sameOrFinerDirect = storage(4), sameOrCoarserDirect = storage(4);
  const topology: OctreePowerTopologySource = {
    plan: { rowCapacity: 1, entryCount: 1, lookupCount: 1, metricBytes: 16,
      catalogBytes: 88, allocatedBytes: 136 },
    metrics, control: topologyControl, catalogEntryHeaders, catalogVolumes, catalogFaces, catalogCoefficients,
    catalogTetrahedronHeaders, catalogTetrahedra, catalogTetrahedronVertices,
    catalogLookup, sameOrFinerDirect, sameOrCoarserDirect,
  };
  const faces = storage(32), faceNormals = storage(16), faceCentroids = storage(16);
  const incidenceRows = storage(32), incidence = storage(16), faceControl = storage(64), rowDirectory = storage(16);
  const faceSource: OctreePowerFaceSource = {
    plan: { rowCapacity: 1, faceCapacity: 1, incidenceCapacity: 2, faceBytes: 32,
      normalBytes: 16, centroidBytes: 16, quadratureBytes: 80, incidenceBytes: 16, workspaceBytes: 32,
      boundaryQueryBytes: 32, liveFaceDispatchBytes: 12, workDispatchBytes: 48,
      rowDirectoryCapacity: 1, rowDirectoryBytes: 16, rowDiagnosticBytes: 16,
      deltaFaceBytes: 32, allocatedBytes: 208 },
    faces, faceNormals, faceCentroids, faceQuadrature: storage(80), incidenceRows,
    incidence, control: faceControl, rowDirectory, deltaStats: storage(16),
    liveFaceDispatch: storage(12), boundaryPhiQueries: storage(32),
  };
  return { topology, faces: faceSource, destroy: () => buffers.forEach(buffer => buffer.destroy()) };
}

function storageDeclarations(source: string): number {
  return source.match(/var<storage(?:,\s*\w+)?>/g)?.length ?? 0;
}

function reachableStorageBindings(source: string): ReadonlyMap<string, readonly number[]> {
  const bindings = new Map<string, number>();
  for (const match of source.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var<storage(?:,\s*\w+)?>\s*([A-Za-z_]\w*)/g,
  )) bindings.set(match[2], Number(match[1]));
  const bodies = new Map<string, string>();
  for (const match of source.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = source.indexOf("{", match.index); let depth = 0, close = -1;
    for (let at = open; at < source.length; at += 1) {
      if (source[at] === "{") depth += 1;
      if (source[at] === "}" && --depth === 0) { close = at; break; }
    }
    assert.notEqual(open, -1, `function ${match[1]} must have a body`);
    assert.notEqual(close, -1, `function ${match[1]} must close its body`);
    bodies.set(match[1], source.slice(open + 1, close));
  }
  const result = new Map<string, readonly number[]>();
  for (const entry of source.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z_]\w*)/g,
  )) {
    const pending = [entry[1]], visited = new Set<string>(), reached = new Set<number>();
    while (pending.length > 0) {
      const name = pending.pop()!;
      if (visited.has(name)) continue;
      visited.add(name);
      const body = bodies.get(name);
      assert.notEqual(body, undefined, `reachable function ${name} must exist`);
      for (const [global, binding] of bindings) {
        if (new RegExp(`\\b${global}\\b`).test(body!)) reached.add(binding);
      }
      for (const callee of bodies.keys()) {
        if (!visited.has(callee) && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) pending.push(callee);
      }
    }
    result.set(entry[1], [...reached].sort((a, b) => a - b));
  }
  return result;
}

test("global-fine modules document declarations separately from entry-point reachability", () => {
  const topology = makeFineLevelSetTopologyWGSL(`
@group(0) @binding(9) var<storage,read> coarsePhi:array<f32>;
fn sampleCoarseOctreePhi(position:vec3f)->f32{return coarsePhi[u32(position.x)*0u];}`);
  assert.deepEqual({
    topology: storageDeclarations(topology),
    velocityPrepass: storageDeclarations(makePowerVelocityPrepassBuilderWGSL()),
    transport: storageDeclarations(fineLevelSetFusedTransportPublicationWGSL),
    fusedTransport: storageDeclarations(makeFineLevelSetProductionFusedTransportWGSL()),
    redistance: storageDeclarations(fineLevelSetJFACPTWGSL),
    summary: storageDeclarations(fineLevelSetSummaryWGSL),
    restriction: storageDeclarations(fineToCoarseLevelSetWGSL),
    volume: storageDeclarations(fineLevelSetVolumeCorrectionWGSL),
  }, {
    // The topology module also owns the three-control final publication gate
    // and the prior worklist used to distinguish cold affine initialization;
    // each entry point remains below the portable ten-storage limit.
    // JFA-CPT declares its immutable page, closest-point work, phi, flag, and
    // transaction-control inputs. No individual entry point exceeds the
    // portable ten-storage limit.
    topology: 23, velocityPrepass: 10, transport: 11, fusedTransport: 10, redistance: 10,
    // Summary declares disjoint fine, compact-coarse, sorting, and indirect
    // arenas in one module. Its largest entry point reaches six storage
    // bindings; no entry point sees all module declarations at once.
    // Restriction also declares the fine topology transaction control. Its
    // prepare entry point reaches ten storage bindings exactly.
    summary: 16, restriction: 12, volume: 12,
  });
  // Restriction and volume deliberately share one module across several
  // smaller passes. Their declarations are not simultaneously reachable;
  // the constrained-device construction test below is the WebGPU validator's
  // authoritative proof for every pipeline entry point.
});

test("every global-fine compute entry point stays within ten reachable storage buffers", () => {
  const topology = makeFineLevelSetTopologyWGSL(`
@group(0) @binding(9) var<storage,read> coarsePhi:array<f32>;
fn sampleCoarseOctreePhi(position:vec3f)->f32{return coarsePhi[u32(position.x)*0u];}`);
  const modules = {
    topology,
    velocityPrepass: makePowerVelocityPrepassBuilderWGSL(),
    transport: fineLevelSetFusedTransportPublicationWGSL,
    fusedTransport: makeFineLevelSetProductionFusedTransportWGSL(),
    redistance: fineLevelSetJFACPTWGSL,
    summary: fineLevelSetSummaryWGSL,
    restriction: fineToCoarseLevelSetWGSL,
    volume: fineLevelSetVolumeCorrectionWGSL,
  };
  for (const [moduleName, source] of Object.entries(modules)) {
    for (const [entryPoint, bindings] of reachableStorageBindings(source)) {
      assert.ok(bindings.length <= MAX_PORTABLE_STORAGE_BINDINGS,
        `${moduleName}.${entryPoint} reaches ${bindings.length} storage buffers: ${bindings.join(", ")}`);
    }
  }
});

test("Dawn constructs every global-fine stage with maxStorageBuffersPerShaderStage=10", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the constrained construction gate",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= MAX_PORTABLE_STORAGE_BINDINGS);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBuffersPerShaderStage: MAX_PORTABLE_STORAGE_BINDINGS,
  } });
  assert.equal(device.limits.maxStorageBuffersPerShaderStage, MAX_PORTABLE_STORAGE_BINDINGS);
  const diagnosticStage = process.env.FLUID_FINE_PORTABILITY_STAGE;
  if (diagnosticStage === "device") return;
  device.pushErrorScope("validation");

  const finePlan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8 });
  const owner = new WebGPUFineLevelSetBricks(device, finePlan);
  const sourceA = owner.initializeEmptyGPUGeneration(1), sourceB = owner.prepareGPUGeneration(2);
  if (diagnosticStage === "bricks") { assert.equal(await device.popErrorScope(), null); return; }
  const coarsePhi = device.createBuffer({ size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const topology = new WebGPUFineLevelSetTopology(device, sourceA, sourceB, `
@group(0) @binding(9) var<storage,read> coarsePhi:array<f32>;
fn sampleCoarseOctreePhi(position:vec3f)->f32{return coarsePhi[u32(position.x)*0u];}`);
  const leafSeeds = new WebGPUFineLevelSetLeafSeeds(device, sourceB);
  if (diagnosticStage === "topology") { assert.equal(await device.popErrorScope(), null); return; }

  // This gate owns fine-stage pipeline portability, not the independently
  // tested power topology/face constructors. An earlier combined diagnostic
  // hit a Dawn/Metal SIGSEGV while creating those upstream dependencies, so
  // use the same ABI-shaped buffer sources that the prepass consumes and keep
  // that unrelated driver failure out of the max-binding verdict.
  const power = makeMinimalPowerPrepassSources(device);
  if (diagnosticStage === "prepass-builder") {
    device.createComputePipeline({ label: "Constrained direct power trajectory sampler", layout: "auto", compute: {
      module: device.createShaderModule({ code: makePowerVelocityPrepassBuilderWGSL() }),
      entryPoint: "sampleDirectPowerVelocity",
    } });
    assert.equal(await device.popErrorScope(), null); return;
  }
  const prepass = new WebGPUOctreePowerVelocityPrepass(device,
    finePlan.maximumResidentBricks * finePlan.samplesPerBrick, power.topology, power.faces);
  if (diagnosticStage === "prepass") { assert.equal(await device.popErrorScope(), null); return; }
  const faceExtension = new WebGPUOctreeFaceClosestPointExtension(
    device, sourceA, power.topology.plan.rowCapacity, 4, power.faces.plan.faceCapacity,
  );
  const ownerAuthority = device.createBuffer({
    label: "Constrained fused transport owner authority",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.createComputePipeline({ label: "Constrained fused production fine characteristic", layout: "auto",
    compute: { module: device.createShaderModule({ code: makeFineLevelSetProductionFusedTransportWGSL() }),
      entryPoint: "transportFineCharacteristic" } });
  if (diagnosticStage === "fused-transport") { assert.equal(await device.popErrorScope(), null); return; }
  const transport = new WebGPUFineLevelSetTransport(
    device, sourceA, prepass, faceExtension,
  );
  if (diagnosticStage === "transport") { assert.equal(await device.popErrorScope(), null); return; }
  const redistance = new WebGPUFineLevelSetRedistance(device, sourceA, topology);
  if (diagnosticStage === "redistance") { assert.equal(await device.popErrorScope(), null); return; }
  const summaries = new WebGPUFineLevelSetSummaries(device, finePlan);
  if (diagnosticStage === "summary") { assert.equal(await device.popErrorScope(), null); return; }
  const restriction = new WebGPUFineToCoarseLevelSet(device, 8,
    finePlan.maximumResidentBricks * finePlan.samplesPerBrick);
  if (diagnosticStage === "restriction") { assert.equal(await device.popErrorScope(), null); return; }

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const headers = device.createBuffer({ size: 8 * 48, usage: storage });
  const records = device.createBuffer({ size: 8 * 16, usage: storage });
  const physicalVolumes = device.createBuffer({ size: 8 * 4, usage: storage });
  const rowCount = device.createBuffer({ size: 4, usage: storage });
  const sampleDirectory = device.createBuffer({ size: 32 + 64 * 32, usage: storage });
  const publicationControl = device.createBuffer({ size: 64, usage: storage });
  const volume = new WebGPUFineLevelSetVolumeCorrection(device, sourceA, {
    headers, records, physicalVolumes, rowCount, sampleDirectory, publicationControl,
    dimensions: [2, 2, 2], physicalCellSize: 1, maximumLeafSize: 1, sampleRowCapacity: 64,
  });
  if (diagnosticStage === "volume") { assert.equal(await device.popErrorScope(), null); return; }

  const maximumBindingSize = Number(device.limits.maxStorageBufferBindingSize);
  const bindingSizes = {
    fineChannel: finePlan.payloadCapacityBytes / 4,
    fineMetadataGeneration: finePlan.metadataCapacityBytes / 2,
    fineWorklistGeneration: finePlan.worklistBytes / 2,
    topologySnapshot: finePlan.maximumResidentBricks * finePlan.samplesPerBrick * 4,
    velocityPrepassQueries: prepass.plan.queryBytes,
    velocityPrepassVertices: prepass.plan.vertexVelocityBytes,
    transportPositions: transport.plan.positionBytes,
    summaryDirectory: summaries.plan.directoryBytes,
    restrictionScratch: restriction.plan.aggregateScratchBytes,
    volumeScratch: volume.plan.reductionScratchBytes,
  };
  for (const [label, bytes] of Object.entries(bindingSizes)) {
    assert.ok(bytes <= maximumBindingSize, `${label} ${bytes} exceeds maxStorageBufferBindingSize ${maximumBindingSize}`);
  }

  const validation = await device.popErrorScope();
  assert.equal(validation, null, validation?.message);
  volume.destroy(); publicationControl.destroy(); sampleDirectory.destroy(); rowCount.destroy(); physicalVolumes.destroy(); records.destroy(); headers.destroy();
  restriction.destroy(); summaries.destroy(); redistance.destroy(); transport.destroy();
  ownerAuthority.destroy(); faceExtension.destroy(); prepass.destroy();
  power.destroy(); leafSeeds.destroy(); topology.destroy(); coarsePhi.destroy(); owner.destroy(); device.destroy();
});
