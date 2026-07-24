import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { WebGPUOctreeCoarseLevelSet } from "../lib/webgpu-octree-coarse-levelset";
import {
  OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES,
  OCTREE_POWER_COARSE_LEVELSET_ERROR,
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  WebGPUOctreePowerCoarseLevelSet,
  octreePowerCoarseLevelSetShader,
  planOctreePowerCoarseLevelSet,
  unpackOctreePowerCoarseLevelSetControl,
} from "../lib/webgpu-octree-power-coarse-levelset";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { OCTREE_POWER_TOPOLOGY_VALID, WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

test("WP8 planner is compact-row bounded and exposes independent coarse/fine diagnostics", () => {
  const plan = planOctreePowerCoarseLevelSet(32, 6);
  const source = readFileSync(new URL("../lib/webgpu-octree-power-coarse-levelset.ts", import.meta.url), "utf8");
  const encode = WebGPUOctreePowerCoarseLevelSet.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /^encode\(broker,input,options\)/);
  assert.doesNotMatch(encode, /newPassBroker/);
  assert.match(encode, /broker\.fence\("powercoarselevel-setschedulecomplete"\)/,
    "redistance passes require command-buffer ordering before a later caller resumes the broker");
  assert.match(encode, /broker\.copyBufferToBuffer/,
    "GPU-authored counts must use the broker's required copy boundary");
  assert.equal(plan.scratchBytes, 32 * 16 * 2);
  assert.equal(plan.selectorRowBytes, 32 * 256 * 4);
  assert.ok(plan.allocatedBytes < 128_000,
    "the aligned encoder-local parameter arena remains a small row-independent allocation");
  assert.match(octreePowerCoarseLevelSetShader, /redistancePowerCoarsePhi/);
  assert.match(octreePowerCoarseLevelSetShader, /validatePowerCoarseFineCorrection/);
  assert.match(octreePowerCoarseLevelSetShader, /nonobtuseAnchorSolidAngle|fn nonobtuse/);
  assert.match(octreePowerCoarseLevelSetShader, /NO_CAUSAL_SIMPLEX/);
  assert.equal(OCTREE_POWER_COARSE_LEVELSET_ERROR.noCausalSimplex, 512,
    "the WGSL no-causal-simplex failure must have a named host diagnostic bit");
  assert.doesNotMatch(octreePowerCoarseLevelSetShader, /nearestFallback|bestKnown|nearestKnown/,
    "Section 4.2 redistance must fail closed when no causal Delaunay simplex exists");
  assert.match(octreePowerCoarseLevelSetShader, /causalTetraCandidate/);
  assert.match(octreePowerCoarseLevelSetShader, /causalTriangleCandidate/);
  assert.match(octreePowerCoarseLevelSetShader, /causalEdgeCandidate/,
    "transition redistance must include lower-dimensional causal simplex updates");
  assert.match(octreePowerCoarseLevelSetShader,
    /fn causalAvailableSubsimplexCandidate[\s\S]*causalNeighborMagnitude\(selectorRow\(row,s\.x\),fromA\)[\s\S]*if\(nx\.y!=0\.0\)\{candidate=min\(candidate,causalEdgeCandidate[\s\S]*if\(nx\.y!=0\.0&&ny\.y!=0\.0\)\{candidate=min\(candidate,causalTriangleCandidate[\s\S]*nx\.y!=0\.0&&ny\.y!=0\.0&&nz\.y!=0\.0&&nonobtuse[\s\S]*causalTetraCandidate/,
    "Section 5 boundary rows must minimize over the available causal subsimplices of the local Delaunay tetrahedron");
  assert.match(octreePowerCoarseLevelSetShader,
    /var axes=vec3f\(1e30\)[\s\S]*inverseTransform\(v\.xyz,metric\.transformAndFlags&63u\)[\s\S]*if\(all\(axes<vec3f\(1e29\)\)\)\{magnitude=eikonal3\(axes,size\(row\)\);used=true;\}[\s\S]*if\(\(header\.flags&UNIFORM\)!=0u\)/,
    "the paper's regular-grid update is selected from the actual Cartesian cell-centre neighbors before wall-clipped pressure-catalog classification");
  assert.doesNotMatch(octreePowerCoarseLevelSetShader,
    /any\(rows==vec3u\(INVALID\)\)[\s\S]{0,120}\{continue;\}/,
    "a virtual vertex outside the physical domain must not discard its available boundary edge or triangle");
  assert.doesNotMatch(octreePowerCoarseLevelSetShader, /atomicAdd\(&control\.nearest/,
    "the boundary causal update is not a nearest-cell fallback");
  assert.match(octreePowerCoarseLevelSetShader,
    /source\.flags&\(PHI_VALID\|PHI_FINITE\)\)\!=\(PHI_VALID\|PHI_FINITE\)[\s\S]*fail\(row,INVALID_SOURCE\)/);
  assert.match(octreePowerCoarseLevelSetShader,
    /source\.flags&\(PHI_CORRECTED\|PHI_INTERFACE\)\)\!=0u/,
    "advected coarse interface rows remain fixed redistance seeds when no fine correction exists");
  assert.match(octreePowerCoarseLevelSetShader,
    /seedPhi[\s\S]*source\.phi<0\.0&&seedPhi>=0\.0[\s\S]*fixedSeed=true/,
    "coarse-only redistance must refresh fixed seeds from current sign-crossing neighbors");
  assert.match(octreePowerCoarseLevelSetShader,
    /shiftedMinimum=min\(value,source\.minimumPhi\+shift\)[\s\S]*shiftedMaximum=max\(value,source\.maximumPhi\+shift\)/,
    "advection must keep phi inside its shifted extrema despite one-ULP cancellation");
  assert.match(octreePowerCoarseLevelSetShader,
    /if\(params\.physical\.y>0\.0&&\(!finite\(velocity\.x\)[\s\S]*velocity\.w<=0\.0\)\)\{fail\(row,INVALID_VELOCITY\)/,
    "the dt=0 bootstrap must not depend on a Stage-A velocity fit");
  assert.match(octreePowerCoarseLevelSetShader,
    /var value=source\.phi;if\(params\.physical\.y>0\.0&&gradient\.w>0\.0\)\{value-=/,
    "dt=0 copies the seeded coarse phi exactly before redistance");
  assert.doesNotMatch(octreePowerCoarseLevelSetShader, /texture|readback/i);
  assert.match(source,
    /const params = this\.freeParameterArenas\.pop\(\) \?\? this\.device\.createBuffer/,
    "submitted parameter arenas must be recycled instead of allocated on every advance");
  assert.match(source,
    /retireSubmittedEncoder[\s\S]*this\.freeParameterArenas\.push\(arena\.params\)/);
  assert.doesNotMatch(source,
    /retireSubmittedEncoder[\s\S]{0,300}arena\.params\.destroy\(\)/,
    "retirement must return ordered queue storage to the pool rather than churn GPU allocations");
});

test("rejected fine correction preserves every byte of the prior coarse authority", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-power-coarse-levelset.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /guardRejectedFineCoarseEntryPoints|coarseFineTransactionEntryPoints|\.replace(?:All)?\(/,
    "transaction guards and scratch addressing must be authored directly in WGSL");
  assert.match(octreePowerCoarseLevelSetShader,
    /fn rejectedFine\(\)->bool\{return params\.hasFine!=0u&&\(arrayLength\(&fineControl\)<6u\|\|fineControl\[0\]==INVALID\|\|fineControl\[5\]!=VALID\);\}/);
  assert.match(octreePowerCoarseLevelSetShader,
    /fn preparePowerCoarsePhiSchedule[\s\S]{0,260}coarseScheduleEnabled=select\(0u,1u,!rejectedFine\(\)\)[\s\S]*workgroupUniformLoad\(&coarseScheduleEnabled\)[\s\S]*if\(enabled==0u\)\{if\(lid==0u\)\{control\.pad0=0u;\}return;\}/,
    "the first portable phase must broadcast rejected fine authority and disable every later transaction phase");
  assert.deepEqual([...octreePowerCoarseLevelSetShader.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z_]\w*)/g,
  )].map((match) => match[1]), [
    "preparePowerCoarsePhiSchedule",
    "buildPowerCoarseSelectorRows",
    "advectPowerCoarsePhiSchedule",
    "correctPowerCoarsePhiSchedule",
    "redistancePowerCoarsePhiPass",
    "publishPowerCoarsePhiSchedule",
    "commitPowerCoarsePhiSchedule",
  ], "portable ownership kernels replace the deleted eighteen-stage host schedule");
  assert.match(octreePowerCoarseLevelSetShader,
    /fn finalizePowerCoarsePhi[\s\S]{0,220}coarseFinalizeEnabled=select\(0u,1u,!rejectedFine\(\)\)[\s\S]*workgroupUniformLoad\(&coarseFinalizeEnabled\)/,
    "coarse finalization must broadcast rejected fine authority before its reduction barriers");
  assert.match(octreePowerCoarseLevelSetShader,
    /fn publishPowerCoarsePhiDeltaAndCommit[\s\S]{0,320}if\(lid==0u\)\{let rejected=rejectedFine\(\);let enabled=!rejected&&control\.valid==VALID&&candidateDirectory\.state==VALID;[\s\S]*workgroupUniformLoad\(&coarseDeltaCommitState\[0\]\)/,
    "delta publication must broadcast its fail-closed gate before touching the prior immutable directory");
  assert.doesNotMatch(octreePowerCoarseLevelSetShader,
    /fn publishPowerCoarsePhiDeltaAndCommit[\s\S]{0,260}if\([^)]*(?:rejectedFine|control\.valid|candidateDirectory\.state)[^)]*\)\{return;\}[\s\S]*workgroupBarrier/,
    "storage-derived authority must not let any lane bypass a later commit barrier");
  const encode = WebGPUOctreePowerCoarseLevelSet.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode,
    /constrowWorkgroups=Math\.max\(1,Math\.ceil\(maximumRows\/256\)\);constselectorWorkgroups=Math\.max\(1,Math\.ceil\(maximumRows\*this\.selectorCount\/64\)\);run\(this\.buildSelectorRowsPipeline,phaseBindings\[1\],selectorWorkgroups\);run\(this\.advectPipeline,phaseBindings\[2\],rowWorkgroups\);run\(this\.correctPipeline,phaseBindings\[3\],rowWorkgroups\);for\(constredistanceofthis\.redistancePipelines\)\{run\(redistance,phaseBindings\[4\],rowWorkgroups\)\}/,
    "each fixed redistance pass must occupy the full row-parallel device schedule");
  assert.equal((encode.match(/run\(this\.(?:prepare|advect|correct|publish|commit)Pipeline/g) ?? []).length, 5,
    "the non-iterative ownership phases remain explicit and fixed");
  assert.doesNotMatch(encode,
    /dispatch\("(?:migrate|prepare|clearStatus|advect|validateFine|applyFine|redistance|publish|finalize|publishDelta)"/,
    "the legacy string-addressed host stage machine must stay deleted");
  assert.match(octreePowerCoarseLevelSetShader,
    /migratePowerCoarsePhiSource[\s\S]*previousSampleAtCurrentLeaf\(header\)/,
    "recurring coarse phi must be sampled by spatial leaf identity before row-indexed advection");
  assert.match(octreePowerCoarseLevelSetShader,
    /output\.flags&MIGRATED_AIR[\s\S]*params\.hasFine!=0u[\s\S]*fail\(row,INVALID_SOURCE\)[\s\S]*output\.flags&=~MIGRATED_AIR/,
    "fine mode must reject an uncorrected migrated row while coarse-only mode publishes its explicit positive-air complement");
  assert.match(octreePowerCoarseLevelSetShader,
    /applyExactFineCorrection[\s\S]*output\.flags=.*PHI_CORRECTED[\s\S]*redistancePowerCoarsePhi[\s\S]*source\.flags&\(PHI_CORRECTED\|PHI_INTERFACE\)[\s\S]*return;/,
    "valid cell-center fine aggregates are fixed coarse seeds during the local Delaunay sweep");
  assert.match(octreePowerCoarseLevelSetShader,
    /let metric=metrics\[row\];if\(\(metric\.transformAndFlags&VALID\)==0u\)\{fail\(row,INVALID_ROW\);return;\}var fixedSeed=/,
    "a fine correction cannot bypass invalid topology validation");
  const scheduleEncode = WebGPUOctreePowerCoarseLevelSet.prototype.encode.toString();
  assert.match(octreePowerCoarseLevelSetShader,
    /validatePowerCoarseFineCorrection\(gid\.x\);applyExactFineCorrection\(gid\.x\);[\s\S]*override REDISTANCE_ITERATION:u32=0u;[\s\S]*redistancePowerCoarsePhi\(gid\.x,REDISTANCE_ITERATION\)/,
    "fine correction validation and seeding must precede the inter-dispatch Delaunay redistance passes");
  assert.match(scheduleEncode,
    /run\(this\.correctPipeline[\s\S]*for\s*\(const redistance of this\.redistancePipelines\)/,
    "the host must preserve correction-before-redistance command ordering");
  assert.match(scheduleEncode, /Power coarse level set [^"]*persistent schedule/,
    "command attribution must identify the single persistent coarse transaction");
});

test("coarse publication snapshots physical power-cell volume with phi authority", () => {
  assert.match(octreePowerCoarseLevelSetShader,
    /struct SampleEntry \{[^}]*row:u32, physicalVolume:f32 \}/);
  assert.match(octreePowerCoarseLevelSetShader,
    /let metric=metrics\[row\][\s\S]*physicalVolume=metric\.volume\*extent\*extent\*extent/,
    "the directory must remain a complete volume snapshot after pressure rows rebuild");
  assert.match(octreePowerCoarseLevelSetShader,
    /rowStatus\[row\]\.physicalVolume=physicalVolume[\s\S]*SampleEntry\([^)]*rowStatus\[row\]\.physicalVolume\)/);
});

test("every coarse schedule bind group equals transitive WGSL reachability", () => {
  const bindings = new Map<string, number>();
  for (const match of octreePowerCoarseLevelSetShader.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+([A-Za-z_]\w*)/g,
  )) bindings.set(match[2], Number(match[1]));
  const bodies = new Map<string, string>();
  for (const match of octreePowerCoarseLevelSetShader.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = octreePowerCoarseLevelSetShader.indexOf("{", match.index); let depth = 0, close = -1;
    for (let at = open; at < octreePowerCoarseLevelSetShader.length; at += 1) {
      if (octreePowerCoarseLevelSetShader[at] === "{") depth += 1;
      if (octreePowerCoarseLevelSetShader[at] === "}" && --depth === 0) { close = at; break; }
    }
    assert.ok(open >= 0 && close > open); bodies.set(match[1], octreePowerCoarseLevelSetShader.slice(open + 1, close));
  }
  const reachable = (entryPoint: string) => {
    const pending = [entryPoint], reached = new Set<string>(), result = new Set<number>();
    while (pending.length > 0) {
      const name = pending.pop()!; if (reached.has(name)) continue; reached.add(name);
      const body = bodies.get(name); assert.notEqual(body, undefined);
      for (const [global, binding] of bindings) if (new RegExp(`\\b${global}\\b`).test(body!)) result.add(binding);
      for (const callee of bodies.keys()) if (!reached.has(callee)
        && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) pending.push(callee);
    }
    return [...result].sort((a, b) => a - b);
  };

  const observed = new Map<string, number[]>(); let id = 0;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: () => ({ id: id += 1 }), createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({ entryPoint: compute.entryPoint,
      getBindGroupLayout: () => ({ entryPoint: compute.entryPoint }) }),
    createBindGroup: ({ layout, entries }: { layout: { entryPoint: string }; entries: { binding: number }[] }) => {
      observed.set(layout.entryPoint, entries.map(({ binding }) => binding).sort((a, b) => a - b)); return {};
    },
  } as unknown as GPUDevice;
  const buffer = {} as GPUBuffer;
  const coarse = { plan: { rowCapacity: 1 }, records: buffer } as unknown as WebGPUOctreeCoarseLevelSet;
  const topology = { metrics: buffer, catalogTetrahedronHeaders: buffer, catalogTetrahedra: buffer,
    catalogTetrahedronVertices: buffer };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const encoder = { beginComputePass: () => pass, copyBufferToBuffer() {} } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8 } });
  try {
    const schedule = new WebGPUOctreePowerCoarseLevelSet(device, coarse,
      topology as unknown as ConstructorParameters<typeof WebGPUOctreePowerCoarseLevelSet>[2], 1);
    schedule.encode(new PassBroker(encoder), { headers: buffer, cellVelocities: buffer, rowDirectory: buffer, rowCount: 1 },
      { dimensions: [1, 1, 1], physicalCellSize: 1, dt: 0, rowDirectoryCapacity: 1, generation: 1 });
    const secondEncoder = { beginComputePass: () => pass, copyBufferToBuffer() {} } as unknown as GPUCommandEncoder;
    assert.doesNotThrow(() => schedule.encode(new PassBroker(secondEncoder),
      { headers: buffer, cellVelocities: buffer, rowDirectory: buffer, rowCount: 1 },
      { dimensions: [1, 1, 1], physicalCellSize: 1, dt: 0, rowDirectoryCapacity: 1, generation: 2 }),
    "a second command buffer must be encodable before the first is submitted or retired");
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
  const entryPoints = [...octreePowerCoarseLevelSetShader.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z_]\w*)/g,
  )].map((match) => match[1]);
  assert.deepEqual([...observed.keys()].sort(), [...entryPoints].sort());
  for (const entryPoint of entryPoints) assert.deepEqual(observed.get(entryPoint), reachable(entryPoint), entryPoint);
});

test("Dawn runs live-row advection, redistance, and cell-center fine correction without readback dependencies", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WP8 GPU checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals); const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const compilation = await device.createShaderModule({ code: octreePowerCoarseLevelSetShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const rowCount = 8, topology = new WebGPUOctreePowerTopology(device, rowCount, catalog);
  const coarse = new WebGPUOctreeCoarseLevelSet(device, rowCount);
  const uniformEntry = catalog.sameOrFinerDirect[0x3ffff] & 0xffff;
  const metricData = new ArrayBuffer(rowCount * 16), metricWords = new Uint32Array(metricData), metricFloats = new Float32Array(metricData);
  for (let row = 0; row < rowCount; row += 1) { metricWords[row * 4] = uniformEntry;
    metricWords[row * 4 + 1] = OCTREE_POWER_TOPOLOGY_VALID; metricFloats[row * 4 + 2] = 1; }
  device.queue.writeBuffer(topology.metrics, 0, metricData);
  const records = new Map(Array.from({ length: rowCount }, (_, row) => { const x = row & 1, phi = x - 0.5;
    return [row, { phi, minimumPhi: phi, maximumPhi: phi,
      flags: OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite, generation: 0, fineSampleCount: 0 }] as const; }));
  coarse.upload(records);
  const headerData = new ArrayBuffer(rowCount * 48), headerWords = new Uint32Array(headerData);
  for (let row = 0; row < rowCount; row += 1) { headerWords[row * 12] = row; headerWords[row * 12 + 3] = 1; }
  const velocities = new Float32Array(rowCount * 4); for (let row = 0; row < rowCount; row += 1) velocities[row * 4 + 3] = 1;
  const rowDirectoryCapacity = rowCount;
  const rowDirectoryWords = new Uint32Array(rowDirectoryCapacity * 4);
  for (let row = 0; row < rowCount; row += 1) {
    rowDirectoryWords.set([row + 1, 1, row, row], row * 4);
  }
  const upload = (data: ArrayBufferView) => { const buffer = device.createBuffer({ size: Math.max(4, data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buffer.unmap(); return buffer; };
  const headers = upload(new Uint8Array(headerData)), velocityBuffer = upload(velocities);
  const rowDirectory = upload(rowDirectoryWords);
  const offsets = upload(new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]));
  const aggregateData = new ArrayBuffer(rowCount * 16), aggregateFloats = new Float32Array(aggregateData);
  const aggregateWords = new Uint32Array(aggregateData);
  aggregateFloats.set([-0.05, -0.05, 0.05]); aggregateWords[3] = 1;
  const contributions = upload(new Uint8Array(aggregateData));
  const schedule = new WebGPUOctreePowerCoarseLevelSet(device, coarse, topology.source, 2);
  const directoryOffset = OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + coarse.plan.recordBytes;
  const readback = device.createBuffer({ size: directoryOffset + schedule.plan.sampleDirectoryBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder(); schedule.encode(new PassBroker(encoder), { headers, cellVelocities: velocityBuffer, rowDirectory,
    rowCount, fineCorrection: { rowOffsets: offsets, contributions, contributionCount: rowCount,
      maximumContributionsPerRow: 1, aggregated: true } },
  { dimensions: [2, 2, 2], physicalCellSize: 1, dt: 0, rowDirectoryCapacity, generation: 41 });
  encoder.copyBufferToBuffer(schedule.control, 0, readback, 0, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
  encoder.copyBufferToBuffer(coarse.records, 0, readback, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, coarse.plan.recordBytes);
  encoder.copyBufferToBuffer(schedule.sampleDirectory, 0, readback, directoryOffset, schedule.plan.sampleDirectoryBytes);
  device.queue.submit([encoder.finish()]); schedule.retireSubmittedEncoder(encoder);
  await readback.mapAsync(GPUMapMode.READ); const result = readback.getMappedRange().slice(0); readback.unmap();
  const decoded = unpackOctreePowerCoarseLevelSetControl(new Uint32Array(result, 0, 16));
  assert.deepEqual(decoded, {
    flags: 0, firstErrorRow: 0xffff_ffff, rowCount, advectedRows: rowCount, uniformUpdates: 0,
    transitionUpdates: 0, redistancePasses: 2, correctedRows: 1, interfaceRows: 1,
    contributionCount: rowCount, generation: 41, valid: OCTREE_POWER_COARSE_LEVELSET_VALID,
  });
  assert.equal(decoded.uniformUpdates, 0,
    "the checkerboard fixture makes every row a sign-change seed, so the paper's redistance sweep must retain all rows");
  const output = new Float32Array(result, OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, rowCount * 4);
  assert.ok(Math.abs(output[0] + 0.05) < 1e-6); assert.ok(Math.abs(output[1] + 0.05) < 1e-6);
  assert.ok(Math.abs(output[2] - 0.05) < 1e-6);
  const directory = new Uint32Array(result, directoryOffset, schedule.plan.sampleDirectoryBytes / 4);
  assert.deepEqual(Array.from(directory.subarray(0, 7)),
    [OCTREE_POWER_COARSE_LEVELSET_VALID, 41, rowCount, 2, 2, 2, 2]);
  assert.equal(new Float32Array(result, directoryOffset + 28, 1)[0], 1);
  let indexedRows = 0;
  for (let slot = 0; slot < schedule.plan.rowCapacity; slot += 1) {
    const base = 8 + slot * 8;
    if (directory[base] === 0) continue;
    indexedRows += 1;
    assert.equal(directory[base + 5] & (OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite),
      OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite);
  }
  assert.equal(indexedRows, rowCount, "PUBLISHED implies every requested compact row has a directory key");

  // Bootstrap is a geometry/sign publication. A rank-deficient Stage-A row
  // has w=0, but at dt=0 its velocity is mathematically unused and must not
  // block the first coarse directory. Once time advances, the same input is
  // strictly invalid and publication remains fail-closed.
  device.queue.writeBuffer(velocityBuffer, 0, new Float32Array(rowCount * 4));
  const runVelocityGuard = async (dt: number, generation: number) => {
    const guardReadback = device.createBuffer({ size: OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const guardEncoder = device.createCommandEncoder();
    schedule.encode(new PassBroker(guardEncoder), { headers, cellVelocities: velocityBuffer, rowDirectory, rowCount },
      { dimensions: [2, 2, 2], physicalCellSize: 1, dt, rowDirectoryCapacity, generation });
    guardEncoder.copyBufferToBuffer(schedule.control, 0, guardReadback, 0,
      OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
    guardEncoder.copyBufferToBuffer(schedule.sampleDirectory, 0, guardReadback,
      OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, 4);
    device.queue.submit([guardEncoder.finish()]); schedule.retireSubmittedEncoder(guardEncoder);
    await guardReadback.mapAsync(GPUMapMode.READ);
    const bytes = guardReadback.getMappedRange().slice(0); guardReadback.unmap(); guardReadback.destroy();
    return { control: unpackOctreePowerCoarseLevelSetControl(new Uint32Array(bytes, 0, 16)),
      directoryState: new Uint32Array(bytes)[16] };
  };
  const bootstrapWithoutVelocity = await runVelocityGuard(0, 42);
  assert.equal(bootstrapWithoutVelocity.control.flags, 0);
  assert.equal(bootstrapWithoutVelocity.control.valid, OCTREE_POWER_COARSE_LEVELSET_VALID);
  assert.equal(bootstrapWithoutVelocity.directoryState, OCTREE_POWER_COARSE_LEVELSET_VALID);
  const advancedWithoutVelocity = await runVelocityGuard(0.01, 43);
  assert.equal(advancedWithoutVelocity.control.flags, OCTREE_POWER_COARSE_LEVELSET_ERROR.invalidVelocity);
  assert.equal(advancedWithoutVelocity.control.valid, 0);
  assert.equal(advancedWithoutVelocity.directoryState, 0);

  coarse.upload(new Map(Array.from({ length: rowCount }, (_, row) => [row, {
    phi: 0, minimumPhi: 0, maximumPhi: 0, flags: 0, generation: 0, fineSampleCount: 0,
  }] as const)));
  const invalidReadback = device.createBuffer({ size: OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES + 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const invalidEncoder = device.createCommandEncoder();
  schedule.encode(new PassBroker(invalidEncoder), { headers, cellVelocities: velocityBuffer, rowDirectory, rowCount },
    { dimensions: [2, 2, 2], physicalCellSize: 1, dt: 0, rowDirectoryCapacity, generation: 44 });
  invalidEncoder.copyBufferToBuffer(schedule.control, 0, invalidReadback, 0,
    OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES);
  invalidEncoder.copyBufferToBuffer(schedule.sampleDirectory, 0, invalidReadback,
    OCTREE_POWER_COARSE_LEVELSET_CONTROL_BYTES, 4);
  device.queue.submit([invalidEncoder.finish()]); schedule.retireSubmittedEncoder(invalidEncoder);
  await invalidReadback.mapAsync(GPUMapMode.READ);
  const invalidResult = invalidReadback.getMappedRange().slice(0); invalidReadback.unmap();
  const invalidControl = unpackOctreePowerCoarseLevelSetControl(new Uint32Array(invalidResult, 0, 16));
  assert.equal(invalidControl.flags, OCTREE_POWER_COARSE_LEVELSET_ERROR.invalidSource,
    "invalid source records fail closed instead of becoming valid zero phi");
  assert.ok(invalidControl.firstErrorRow < rowCount);
  assert.equal(invalidControl.valid, 0);
  assert.equal(new Uint32Array(invalidResult)[16], 0, "invalid source cannot publish an authoritative directory");
  schedule.destroy(); headers.destroy(); velocityBuffer.destroy(); rowDirectory.destroy(); offsets.destroy(); contributions.destroy();
  readback.destroy(); invalidReadback.destroy(); coarse.destroy(); topology.destroy(); device.destroy();
});
