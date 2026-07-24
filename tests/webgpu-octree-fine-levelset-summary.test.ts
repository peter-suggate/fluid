import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey, planFineLevelSetBricks } from
  "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks, type WebGPUFineLevelSetBrickSource } from
  "../lib/webgpu-octree-fine-levelset-bricks";
import { FINE_LEVELSET_SUMMARY_CENTER_COMPLETE, FINE_LEVELSET_SUMMARY_VALID,
  fineLevelSetSummaryWGSL, planFineLevelSetSummaryLeafLookup, WebGPUFineLevelSetSummaries } from
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
  for (let slot = 0; slot < words[2]; slot += 1) {
    const base = 16 + slot * 8;
    if (words[base] === key) return base;
  }
  return undefined;
}

test("fine-summary recurring publication is delta-only and caller-brokered", () => {
  const encode = WebGPUFineLevelSetSummaries.prototype.encode.toString();
  assert.match(encode, /pageDelta[\s\S]*updateIndirectBuffer/);
  assert.doesNotMatch(encode, /pageDelta\?\.|coarse\?\.|this\.fallback/,
    "the exact page delta and corrected-coarse authority are mandatory; no absent-authority switch remains");
  assert.match(encode, /dispatchWorkgroupsIndirect/);
  assert.doesNotMatch(encode, /new PassBroker|commandEncoder\(\)|\.fence\(/,
    "the summary stage must stay inside its caller's pass broker apart from the immutable indirect copy");
  assert.doesNotMatch(encode, /resetFineSummaryDeltaRecords/,
    "recurring updates must not clear the full fixed-capacity sort arena");
  assert.match(encode, /runIndirect\("recomputeFineSummaryBase"/);
  assert.match(encode, /runIndirect\("recomputeFineSummaryParents"/);
  assert.equal([...encode.matchAll(/\brun\("/g)].length, 2,
    "one persistent builder and one persistent publisher are the only direct recurring dispatches");
  const buildAt = encode.indexOf('run("prepareFineSummaryDelta"');
  const baseAt = encode.indexOf('runIndirect("recomputeFineSummaryBase"');
  const parentsAt = encode.indexOf('runIndirect("recomputeFineSummaryParents"');
  const publishAt = encode.indexOf('run("publishFineSummaryDelta"');
  assert.ok(buildAt >= 0 && buildAt < baseAt && baseAt < parentsAt && parentsAt < publishAt,
    "the exact maximumLevel + 3 schedule is build, base, ordered parents, publish");
  assert.match(fineLevelSetSummaryWGSL,
    /let cold=directory\[9\]!=PUBLISHED[\s\S]*select\(pageDelta\[0\],b\[0\],cold\)/,
    "only the unpublished construction generation may enumerate the full active worklist");
  assert.match(fineLevelSetSummaryWGSL,
    /fineDirtyContains[\s\S]*stageFineOnlyCarry[\s\S]*publicDirtySummary/,
    "private fine-only carry and public corrected-coarse union must remain separate");
  assert.match(fineLevelSetSummaryWGSL,
    /fn recomputeFineSummaryBase[\s\S]*let fineRecord=inLevel;[\s\S]*fn recomputeFineSummaryParents[\s\S]*let inLevel=enabled&&key>=p\.levelOffset&&key<p\.levelOffset\+p\.levelKeyCount/,
    "coarse-dirty base and ancestor keys must refresh the live fine cache before the corrected-coarse union");
  assert.match(fineLevelSetSummaryWGSL,
    /fn coarseSummaryAt[\s\S]*COARSE_AUTHORITY\|CENTER_COMPLETE,bitcast<u32>\(e\.phi\)[\s\S]*fn publicDirtySummary[\s\S]*fineCenterComplete=\(value\.flags&CENTER_COMPLETE\)==CENTER_COMPLETE[\s\S]*select\(coarseValue\.centerPhi,fineCenter,fineCenterComplete\)/,
    "pressure classification must prefer a complete narrow-band fine centre and fall back to current coarse phi");
  assert.match(fineLevelSetSummaryWGSL,
    /fn combineSummary\(left:Entry,right:Entry\)->Entry\{return Entry\(left\.key,/,
    "parent aggregation must preserve the sorted destination key rather than adopting a child key");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /prepareFineSummaryWork|summarizeFineBricks|mergeCoarsePhiSummaries|scanFineSummarySegments|publishFineSummaryRuns/,
    "the deleted full-live rebuild path must not remain as a fallback");
  assert.match(fineLevelSetSummaryWGSL,
    /prepareFineSummaryDelta[\s\S]*workState\[3\]=count;workState\[4\]=padded[\s\S]*publishSortDispatch\(padded\)/,
    "the builder publishes the exact power-of-two compact prefix as an indirect extent");
  assert.match(fineLevelSetSummaryWGSL,
    /sortFineSummaryTiles[\s\S]*sortTile\[lid\][\s\S]*mergeFineSummaryRuns[\s\S]*mergeRunPartition/,
    "sorting uses parallel shared-memory tiles and deterministic merge-path passes");
  assert.match(fineLevelSetSummaryWGSL,
    /publishSortDispatch\(items:u32\)[\s\S]*items>runWidth[\s\S]*writeSortDispatch\(1u\+passIndex,select\(0u,groups,enabled\)\)[\s\S]*\(activeMerges&1u\)!=0u/,
    "inactive merge widths must publish zero-work indirect dispatches and canonicalize only odd ping-pong parity");
  assert.match(fineLevelSetSummaryWGSL,
    /mergeFineSummaryScratchToRecords[\s\S]*records\[index\]=sortScratch\[index\]/,
    "odd active merge counts restore the exact sorted prefix to the canonical records arena");
  assert.match(encode,
    /dispatchWorkgroupsIndirect\(this\.indirect,24\+index\*12\)[\s\S]*24\+this\.plan\.mergePassCount\*12/,
    "every merge width and the conditional canonicalization use their own GPU-authored extent");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /@compute[^]*?fn (?:emitFineSummaryDirtyAncestors|emitCorrectedCoarseDirtySummaries|sortFineSummaryDelta|compactFineOnlySummaryCarry|compactFineOnlySummaryDirty|prepareFineOnlySummaryMerge|mergeFineOnlySummaryCandidate|validateAndCommitFineOnlySummaryCandidate|compactFineSummaryCarry|compactFineSummaryDirty|prepareFineSummaryMerge|mergeFineSummaryCandidate|validateAndCommitFineSummaryCandidate)\b/,
    "all serial/mutable setup and publication entry points must stay deleted");
  assert.match(fineLevelSetSummaryWGSL,
    /publishFineSummaryDelta[\s\S]*stageValidateFineOnlyCandidate\(lid\);storageBarrier\(\);workgroupBarrier\(\);[\s\S]*stageGlobalCarry/,
    "the private fine commit must become visible before the corrected-coarse union is built");
  assert.match(fineLevelSetSummaryWGSL, /let fineRecord=inLevel;/,
    "coarse-dirty keys must refresh the private fine cache before the public fine/coarse merge");
  assert.match(fineLevelSetSummaryWGSL,
    /fn recomputeFineSummaryParents[\s\S]*if\(lid<8u\)[\s\S]*parentChildren\[lid\]=item[\s\S]*workgroupBarrier\(\)[\s\S]*for\(var child=0u;child<8u/,
    "parent summary children and center taps must use eight lanes before the exact ordered fold");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /fn recomputeFineSummaryParents[\s\S]*?\{if\(lid!=0u\)\{return;\}/,
    "parent summary workgroups must not leave 63 lanes idle");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /histogramFineSummaryDelta|prefixFineSummaryDelta|scatterFineSummaryDelta|SortP|sortP|histograms/,
    "radix pipelines and their backing storage must stay deleted");
  assert.match(fineLevelSetSummaryWGSL, /pageDelta\[15\]==PAGE_DELTA_VALID/,
    "recurring summaries consume the topology producer's one-valued delta sentinel");
  assert.match(fineLevelSetSummaryWGSL,
    /fn finePage\(key:u32\)[\s\S]*directoryBase=5u\+p\.pageCapacity[\s\S]*b\[directoryBase\+key\]/,
    "base and center summaries use the generation-validated direct page directory");
  const finePageBody = fineLevelSetSummaryWGSL.match(
    /fn finePage\(key:u32\)->u32\{[\s\S]*?\n\}/,
  )?.[0] ?? "";
  assert.doesNotMatch(finePageBody, /var low=|while\(low<high\)/,
    "summary taps must not repeatedly binary-search the compact fine worklist");
  assert.doesNotMatch(fineLevelSetSummaryWGSL, /pageDelta\[15\]==PUBLISHED/,
    "the topology page-delta ABI cannot inherit the summary directory's high-bit sentinel");
  assert.doesNotMatch(fineLevelSetSummaryWGSL,
    /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/);
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
  assert.equal(observed.size, 6,
    "the small audit plan must exercise prepare, emit, tiled-sort, recompute, and publication");
  for (const [entryPoint, bindings] of observed) {
    assert.deepEqual(bindings, reachableSummaryBindings(entryPoint),
      `${entryPoint} host bindings must equal transitive WGSL reachability`);
  }
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
      const logicalKeys = Array.from(generation.worklistWords.slice(5, 5 + generation.activeCount),
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
    const topKey = summaries.plan.levelOffsets[summaries.plan.maximumLevel]; let top: number | undefined;
    for (let slot = 0; slot < words[2]; slot += 1) if (words[16 + slot * 8] === topKey) { top = 16 + slot * 8; break; }
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
