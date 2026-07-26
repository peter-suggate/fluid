import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey, planFineLevelSetBricks } from
  "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks, type WebGPUFineLevelSetBrickSource } from
  "../lib/webgpu-octree-fine-levelset-bricks";
import { FINE_LEVELSET_SUMMARY_CENTER_COMPLETE, FINE_LEVELSET_SUMMARY_CONSUMERS,
  FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE, FINE_LEVELSET_SUMMARY_VALID,
  fineLevelSetSummaryDirectEntryBase,
  fineLevelSetSummaryWGSL, planFineLevelSetGPUSummaries,
  planFineLevelSetSummaryLeafLookup, WebGPUFineLevelSetSummaries } from
  "../lib/webgpu-octree-fine-levelset-summary";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { PassBroker } from "../lib/webgpu-pass-broker";

function reachableSummaryBindings(entryPoint: string): number[] {
  const source = fineLevelSetSummaryWGSL.replace(/\/\/[^\n\r]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "");
  const globals = new Map<string, number>();
  for (const match of source.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s*([A-Za-z_]\w*)/g,
  )) globals.set(match[2], Number(match[1]));
  const bodies = new Map<string, string>();
  for (const match of source.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = source.indexOf("{", match.index); let depth = 0; let close = -1;
    for (let at = open; at < source.length; at += 1) {
      if (source[at] === "{") depth += 1;
      if (source[at] === "}" && --depth === 0) { close = at; break; }
    }
    assert.ok(open >= 0 && close > open, `summary WGSL function ${match[1]} must be complete`);
    bodies.set(match[1], source.slice(open + 1, close));
  }
  const pending = [entryPoint]; const reached = new Set<string>(); const bindings = new Set<number>();
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (reached.has(name)) continue;
    reached.add(name);
    const body = bodies.get(name);
    assert.notEqual(body, undefined, `reachable summary WGSL function ${name} must exist`);
    const scopes: Set<string>[] = [new Set()]; let declaresLocal = false;
    for (const token of body!.matchAll(/[A-Za-z_]\w*|[{}]/g)) {
      const value = token[0];
      if (value === "{") { scopes.push(new Set()); declaresLocal = false; continue; }
      if (value === "}") { scopes.pop(); declaresLocal = false; continue; }
      if (value === "let" || value === "var" || value === "const") { declaresLocal = true; continue; }
      if (declaresLocal) { scopes.at(-1)!.add(value); declaresLocal = false; continue; }
      const binding = globals.get(value);
      if (binding !== undefined && !scopes.some((scope) => scope.has(value))) bindings.add(binding);
    }
    for (const callee of bodies.keys()) {
      if (!reached.has(callee) && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) pending.push(callee);
    }
  }
  return [...bindings].sort((left, right) => left - right);
}

function findSummaryEntry(words: Uint32Array, key: number): number | undefined {
  return fineLevelSetSummaryDirectEntryBase(words, key);
}

test("fine-summary publication mutates a direct rank directory without sort, merge, or search", () => {
  const encode = WebGPUFineLevelSetSummaries.prototype.encode.toString();
  assert.match(encode, /pageDelta[\s\S]*updateIndirectBuffer/);
  assert.match(encode, /removeFineSummaryPages[\s\S]*addFineSummaryPages[\s\S]*publishFineSummaryCoarseRows/);
  assert.match(encode, /recomputeFineSummaryBase[\s\S]*recomputeFineSummaryParents[\s\S]*publishFineSummaryDirect/);
  assert.doesNotMatch(encode, /sort|merge|recordScratch|carryRecords/);
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /sortFineSummary|mergeFineSummary|recordLowerBound|while\(low<high\)|while\(lo<hi\)/);
  assert.match(fineLevelSetSummaryWGSL,
    new RegExp(`fn topWord\\(key:u32\\)->u32\\{return 16u\\+key\\/${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u;\\}`
      + `[\\s\\S]*fn rankForKey\\(key:u32\\)[\\s\\S]*rankForKey\\(childKey\\)`));
  assert.match(encode,
    /removeFineSummaryPages[\s\S]*ensureFineSummaryDirectoryPages[\s\S]*ensureFineSummaryRanks[\s\S]*addFineSummaryPages/,
    "directory-page and rank ownership must be established at dispatch boundaries before mutation");
  assert.match(fineLevelSetSummaryWGSL,
    /prepareFineSummaryDirect[\s\S]*dirStore\(9u,0u\)[\s\S]*publishFineSummaryDirect[\s\S]*dirStore\(9u,select\(0u,PUBLISHED,error==0u\)\)/,
    "publication is invalidated before mutation and restored only after full validation");
  assert.deepEqual(FINE_LEVELSET_SUMMARY_CONSUMERS.map(({ classification }) => classification),
    ["simulation-critical", "simulation-critical", "diagnostics-only", "diagnostics-only"]);
});

test("fine-summary rank holes contribute neutral values without skipping workgroup barriers", () => {
  for (const entryPoint of ["recomputeFineSummaryBase", "recomputeFineSummaryParents"]) {
    const start = fineLevelSetSummaryWGSL.indexOf(`fn ${entryPoint}`);
    const firstBarrier = fineLevelSetSummaryWGSL.indexOf("workgroupBarrier()", start);
    assert.ok(start >= 0 && firstBarrier > start, `${entryPoint} must retain its reduction barrier`);
    const prefix = fineLevelSetSummaryWGSL.slice(start, firstBarrier);
    assert.doesNotMatch(prefix, /\breturn\s*;/,
      `${entryPoint} may not diverge on storage-derived rank state before a barrier`);
    assert.match(prefix, /rankInRange[\s\S]*keyPlusOne[\s\S]*enabled/,
      `${entryPoint} must predicate reads and writes while every lane remains live`);
  }
  assert.match(fineLevelSetSummaryWGSL,
    /var lo=3\.402823e38;var hi=-3\.402823e38;[\s\S]*var ma=3\.402823e38;var count=0u;var failure=0u/,
    "inactive base lanes must enter the reduction with min/max/count/error identities");
  assert.match(fineLevelSetSummaryWGSL,
    /var item=emptyEntry\(key\);var center=vec2u\(0u\);if\(enabled\)/,
    "inactive parent lanes must publish neutral child and center scratch");
});

test("fine-summary publication admits canonical empty safety-ring ranks but rejects partial evidence", () => {
  assert.match(fineLevelSetSummaryWGSL,
    /fn publishedEntryValid[\s\S]*let hasSamples=e\.samples!=0u;let hasBricks=e\.bricks!=0u;if\(hasSamples!=hasBricks\)\{return false;\}/,
    "sample and brick counts must either both be absent or both describe populated fine evidence");
  assert.match(fineLevelSetSummaryWGSL,
    /if\(hasSamples\)[\s\S]*e\.samples>e\.bricks\*p\.samplesPerBrick[\s\S]*finite\(lo\)&&finite\(hi\)&&finite\(ma\)&&lo<=hi&&ma>=0\.0/,
    "populated entries must retain bounded counts and finite ordered intervals");
  assert.match(fineLevelSetSummaryWGSL,
    /return e\.minimumPhi==INVALID&&e\.maximumPhi==0u&&e\.minimumAbsolutePhi==bitcast<u32>\(3\.402823e38\);/,
    "a rank with no fine or coarse samples is legal only in the canonical unavailable-evidence form");
  assert.match(fineLevelSetSummaryWGSL,
    /let bad=key>=p\.hierarchyKeyCapacity\|\|rankForKey\(key\)!=rank\|\|!publishedEntryValid\(value\)/,
    "publication must validate structure without treating a legal safety-ring rank as stale");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /let bad=[^;]*!present\(value\)/,
    "page allocation alone must not claim that the narrow-band summary is stale");
});

test("direct summary lookup is differential-exact against a key map and fails closed", () => {
  const fine = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [4, 2, 2],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 16 });
  const plan = planFineLevelSetGPUSummaries(fine, 8); const words = new Uint32Array(plan.directoryWords);
  words[0] = 0; words[2] = 3; words[3] = plan.entryCapacity;
  words[8] = 16 + plan.hierarchyTopLevelPages + plan.directoryPageCapacity * plan.directoryPageSize;
  words[9] = FINE_LEVELSET_SUMMARY_VALID; words[10] = plan.hierarchyKeyCapacity;
  words[14] = plan.directoryPageSize; words[15] = plan.hierarchyTopLevelPages;
  const keys = [0, Math.floor(plan.hierarchyKeyCapacity / 2), plan.hierarchyKeyCapacity - 1];
  const pageRanks = new Map<number, number>();
  keys.forEach((key, rank) => {
    const top = Math.floor(key / plan.directoryPageSize);
    let page = pageRanks.get(top);
    if (page === undefined) { page = pageRanks.size; pageRanks.set(top, page); words[16 + top] = page + 1; }
    const rankWord = 16 + plan.hierarchyTopLevelPages + page * plan.directoryPageSize
      + key % plan.directoryPageSize;
    words[rankWord] = rank + 1; words[words[8]! + rank * 8] = key;
  });
  for (let key = 0; key < plan.hierarchyKeyCapacity; key += 1) {
    const expected = keys.indexOf(key); const base = fineLevelSetSummaryDirectEntryBase(words, key);
    assert.equal(base, expected < 0 ? undefined : words[8]! + expected * 8);
  }
  words[9] = 0; assert.equal(fineLevelSetSummaryDirectEntryBase(words, keys[0]!), undefined);
  words[9] = FINE_LEVELSET_SUMMARY_VALID;
  const firstPage = words[16 + Math.floor(keys[0]! / plan.directoryPageSize)]! - 1;
  const firstRankWord = 16 + plan.hierarchyTopLevelPages + firstPage * plan.directoryPageSize
    + keys[0]! % plan.directoryPageSize;
  words[firstRankWord] = plan.entryCapacity + 1;
  assert.equal(fineLevelSetSummaryDirectEntryBase(words, keys[0]!), undefined);
});

test("every fine-summary pipeline binds exactly its transitively reachable resources", () => {
  const observed = new Map<string, number[]>();
  const buffer = { size: 4096, destroy() {} } as unknown as GPUBuffer;
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 },
    queue: { writeBuffer() {} },
    createBuffer: () => buffer,
    createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint,
      getBindGroupLayout() { return { entryPoint: this.entryPoint }; },
    }),
    createBindGroup: ({ layout, entries }: {
      layout: { entryPoint: string };
      entries: Iterable<GPUBindGroupEntry>;
    }) => {
      const bindings = Array.from(entries, ({ binding }) => binding).sort((left, right) => left - right);
      const prior = observed.get(layout.entryPoint);
      if (prior) assert.deepEqual(bindings, prior, `${layout.entryPoint} bind contract changed`);
      observed.set(layout.entryPoint, bindings);
      return {};
    },
  } as unknown as GPUDevice;
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [2, 1, 1],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 1 });
  const source = {
    plan, generation: 1, generationSlot: 0, params: buffer, metadata: buffer, worklist: buffer,
    flags: buffer, phi: buffer, workA: buffer, workB: buffer, rollbackPhi: buffer,
  } as WebGPUFineLevelSetBrickSource;
  const pass = {
    setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {}, end() {},
  };
  const encoder = { beginComputePass: () => pass, copyBufferToBuffer() {} } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    const summaries = new WebGPUFineLevelSetSummaries(device, plan, 1);
    summaries.encode(new PassBroker(encoder), source,
      { buffer, layout: { changedKeysOffsetWords: 16 } }, {
      directory: buffer, control: buffer, delta: buffer, deltaHeaderWords: 16, deltaRecordWords: 4,
    });
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }
  assert.equal(observed.size, 17,
    "the audit plan must exercise validation, direct mutation, recompute, and publication");
  for (const [entryPoint, bindings] of observed) {
    assert.deepEqual(bindings, reachableSummaryBindings(entryPoint),
      `${entryPoint} host bindings must equal transitive WGSL reachability`);
    assert.ok(bindings.filter((binding) => binding !== 0).length <= 10,
      `${entryPoint} must fit the hard ten-storage-buffer device limit`);
  }
  assert.equal(observed.get("removeFineSummaryPages")!.filter((binding) => binding !== 0).length, 9);
  assert.equal(observed.get("reclaimFineSummaryDirectoryPages")!.filter((binding) => binding !== 0).length, 6);
});

test("Dawn publishes factor-4/factor-8 sparse fine summaries across moving interface generations", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  for (const factor of [4, 8] as const) {
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [4, 2, 2],
      finestCellWidth: 1, fineFactor: factor, brickResolution: 4,
      maximumResidentBricks: factor === 4 ? 16 : 128 });
    const owner = new WebGPUFineLevelSetBricks(device, plan); const summaries = new WebGPUFineLevelSetSummaries(device, plan);
    const oracle = new FineLevelSetBrickOracle(plan);
    const pageDelta = device.createBuffer({ size: (16 + 2 * plan.maximumResidentBricks) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    let previousLogicalKeys: number[] = [];
    for (const [iteration, plane] of [1.25, 2.25].entries()) {
      const keys: number[] = []; const x = Math.floor(plane / (plan.brickResolution * plan.fineCellWidth));
      for (let z = 0; z < plan.brickDimensions[2]; z += 1) for (let y = 0; y < plan.brickDimensions[1]; y += 1) {
        keys.push(packFineLevelSetBrickKey(plan, [Math.min(plan.brickDimensions[0] - 1, x), y, z]));
      }
      oracle.publishInterfaceAndRing(keys, ([px]) => px - plane);
      const generation = oracle.exportGPUGeneration();
      const logicalKeys = Array.from(generation.worklistWords.slice(7, 7 + generation.activeCount),
        (id) => generation.metadataWords[id * 10 + 1]);
      const source = owner.uploadGeneration(generation);
      if (iteration > 0) {
        const changed = [...new Set([...previousLogicalKeys, ...logicalKeys])].sort((a, b) => a - b);
        const delta = new Uint32Array(16 + 2 * plan.maximumResidentBricks);
        delta[0] = changed.length; delta[1] = source.generation; delta[15] = 1;
        delta.set(changed, 16); device.queue.writeBuffer(pageDelta, 0, delta);
      }
      const encoder = device.createCommandEncoder(); const broker = new PassBroker(encoder);
      summaries.encode(broker, source,
        { buffer: pageDelta, layout: { changedKeysOffsetWords: 16 } },
        { directory: pageDelta, control: pageDelta, delta: pageDelta,
          deltaHeaderWords: 16, deltaRecordWords: 4 });
      device.queue.submit([broker.finish()]);
      previousLogicalKeys = logicalKeys;
    }
    await device.queue.onSubmittedWorkDone();
    const readback = device.createBuffer({ size: summaries.plan.directoryBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(summaries.directory, 0, readback, 0,
      summaries.plan.directoryBytes); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
    assert.equal(words[0], 0); assert.equal(words[1], 2); assert.ok(words[2] > 0); assert.equal(words[9], FINE_LEVELSET_SUMMARY_VALID);
    const topKey = summaries.plan.levelOffsets[summaries.plan.maximumLevel];
    const top = findSummaryEntry(words, topKey);
    assert.notEqual(top, undefined); assert.ok(words[top! + 4] > 0); assert.ok(words[top! + 5] > 0);
    summaries.destroy(); owner.destroy(); pageDelta.destroy(); readback.destroy();
  }
  device.destroy();
});

test("Dawn publishes exact factor-4/factor-8 fine cell-centre phase outside the narrow band", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  for (const factor of [4, 8] as const) {
    const residentBricks = factor === 4 ? 1 : 8;
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [1, 1, 1],
      finestCellWidth: 1, fineFactor: factor, brickResolution: 4, maximumResidentBricks: residentBricks });
    const owner = new WebGPUFineLevelSetBricks(device, plan);
    const summaries = new WebGPUFineLevelSetSummaries(device, plan);
    const oracle = new FineLevelSetBrickOracle(plan);
    const keys = Array.from({ length: plan.logicalBrickCount }, (_, key) => key);
    // The affine field's exact trilinear value at (0.5, 0.5, 0.5) is -0.25.
    oracle.publishInterfaceAndRing(keys, ([x, y, z]) => x + 2 * y + 4 * z - 3.75);
    const generation = oracle.exportGPUGeneration();
    // Redistance retains current finite signed phi outside the narrow
    // CPT/transport band but clears its VALID membership bit. Centre phase
    // must remain available to dynamic pressure topology even when every
    // sample in this minimal fixture is outside that band.
    generation.flags.fill(0);
    const source = owner.uploadGeneration(generation);
    const authority = device.createBuffer({ size: 80,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder(); const broker = new PassBroker(encoder);
    summaries.encode(broker, source, { buffer: authority, layout: { changedKeysOffsetWords: 16 } },
      { directory: authority, control: authority, delta: authority,
        deltaHeaderWords: 16, deltaRecordWords: 4 });
    device.queue.submit([broker.finish()]);
    await device.queue.onSubmittedWorkDone();
    const readback = device.createBuffer({ size: summaries.plan.directoryBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const copy = device.createCommandEncoder(); copy.copyBufferToBuffer(summaries.directory, 0, readback, 0,
      summaries.plan.directoryBytes); device.queue.submit([copy.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const bytes = readback.getMappedRange().slice(0); readback.unmap();
    const words = new Uint32Array(bytes); const floats = new Float32Array(bytes);
    const lookup = planFineLevelSetSummaryLeafLookup(plan.brickDimensions, plan.finestCellDimensions,
      [0, 0, 0], 1, plan.samplesPerBrick);
    const base = findSummaryEntry(words, lookup.key); assert.notEqual(base, undefined);
    assert.equal(words[base! + 4], 0);
    assert.equal(words[base! + 5], 0);
    assert.equal(words[base! + 6] & FINE_LEVELSET_SUMMARY_CENTER_COMPLETE,
      FINE_LEVELSET_SUMMARY_CENTER_COMPLETE);
    assert.ok(Math.abs(floats[base! + 7] - (-0.25)) < 1e-6,
      `factor ${factor} centre phi ${floats[base! + 7]}`);
    summaries.destroy(); owner.destroy(); authority.destroy(); readback.destroy();
  }
  device.destroy();
});
