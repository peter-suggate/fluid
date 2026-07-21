import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  FineLevelSetBrickOracle,
  packFineLevelSetBrickKey,
  planFineLevelSetBricks,
} from "../lib/octree-fine-levelset-bricks";
import {
  WebGPUFineLevelSetBricks,
  fineLevelSetBrickSamplingWGSL,
} from "../lib/webgpu-octree-fine-levelset-bricks";
import { fineLevelSetGPUQueryTransportWGSL, makeFineLevelSetTransportWGSL, WebGPUFineLevelSetTransport,
  unpackFineLevelSetGPUTransportControl } from "../lib/webgpu-octree-fine-levelset-transport";
import {
  WebGPUFineLevelSetRedistance,
  fineLevelSetRedistanceWGSL,
  unpackFineLevelSetGPURedistanceControl,
} from "../lib/webgpu-octree-fine-levelset-redistance";
import {
  FINE_LEVELSET_TOPOLOGY_ERROR,
  FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON,
  fineLevelSetLeafSeedWGSL,
  makeFineLevelSetTopologyWGSL,
  planFineLevelSetLeafBrickBounds,
  planFineLevelSetTopologyBand,
  WebGPUFineLevelSetLeafSeeds,
  WebGPUFineLevelSetTopology,
  unpackFineLevelSetGPUTopologyControl,
} from "../lib/webgpu-octree-fine-levelset-topology";

test("fine transport diagnostics decode the exact first invalid velocity position", () => {
  const bytes = new ArrayBuffer(64);
  const words = new Uint32Array(bytes);
  const floats = new Float32Array(bytes);
  words.set([3, 2, 41, 0, 17, 5, 7, 11, 13, 19, 0x0120_0004, 0x0800_0004, 23]);
  floats.set([1.25, 2.5, 3.75], 13);

  assert.deepEqual(unpackFineLevelSetGPUTransportControl(words), {
    departureOutsideBand: 3,
    nonfiniteVelocity: 2,
    processed: 41,
    committed: false,
    extrapolatedVelocity: 17,
    maximumDisplacementFineCells: 5,
    faceBandUnavailable: 7,
    velocityUnavailable: 11,
    invalidVelocityStatus: 13,
    nonpositiveVelocityResult: 19,
    velocityStatusReasonOr: 0x0120_0004,
    firstInvalidVelocityStatus: 0x0800_0004,
    firstInvalidVelocityLocalIndex: 23,
    firstInvalidVelocityPosition: [1.25, 2.5, 3.75],
  });
});

test("fine topology dilation covers displacement, interpolation, redistance, and a safety brick", () => {
  assert.deepEqual(planFineLevelSetTopologyBand(4, {
    maximumBacktraceFineCells: 8,
    interpolationSupportFineCells: 1,
    redistanceBandFineCells: 32,
  }), {
    maximumBacktraceFineCells: 8,
    interpolationSupportFineCells: 1,
    redistanceBandFineCells: 32,
    safetyBrickRings: 1,
    requiredFineCells: 41,
    dilationBrickRings: 12,
  });
  assert.throws(() => planFineLevelSetTopologyBand(4, {
    maximumBacktraceFineCells: 1, interpolationSupportFineCells: 1,
    redistanceBandFineCells: 4, safetyBrickRings: 0,
  }), /at least one publication safety ring/);
});

test("fine topology discovers and dilates pages in parallel with an exact staged one-ring bootstrap", () => {
  const wgsl = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  ).replace(/\s+/g, "");
  const encode = WebGPUFineLevelSetTopology.prototype.encode.toString().replace(/\s+/g, "");

  assert.doesNotMatch(wgsl, /@compute@workgroup_size\(1\)fndiscoverDesired/,
    "interface discovery must not scan every resident page in one invocation");
  assert.match(wgsl, /@compute@workgroup_size\(64\)fnclearDesiredGeneration/);
  assert.match(wgsl, /@compute@workgroup_size\(64\)fndiscoverInterfaceBricks/);
  assert.match(wgsl, /@compute@workgroup_size\(64\)fninsertExternalSeeds/);
  assert.match(wgsl, /@compute@workgroup_size\(64\)fndilateDesiredRing/);
  assert.match(wgsl,
    /fninsertDesired\(key:u32\)[\s\S]*atomicCompareExchangeWeak\(&targetA\[slot\*2u\],INVALID,key\)/,
    "parallel discovery must atomically deduplicate desired pages");
  assert.match(wgsl,
    /fnbeginDesiredDilation\(\)\{targetB\[0\]=0u;targetB\[1\]=min\(atomicLoad\(&control\[2\]\),params\.pageCapacity\);\}/);
  assert.match(wgsl,
    /fndilateDesiredRing[\s\S]*letlayerStart=targetB\[0\];letlayerEnd=targetB\[1\];[\s\S]*targetB\[5u\+layerStart\+work\]/,
    "each pass must consume only the previous frontier, not pages appended during that pass");
  assert.match(wgsl,
    /fnadvanceDesiredDilation\(\)\{letpriorEnd=targetB\[1\];targetB\[0\]=priorEnd;targetB\[1\]=min\(atomicLoad\(&control\[2\]\),params\.pageCapacity\);\}/);
  assert.match(wgsl,
    /rawCount>params\.pageCapacity\|\|rawCount>available[\s\S]*atomicOr\(&control\[0\],CAPACITY\);atomicMax\(&control\[6\],max\(rawCount,params\.pageCapacity\+1u\)\)/,
    "truncated external seed lists must fail closed with a strict capacity lower bound");

  assert.match(encode, /initialDilationBrickRings=bandPlan\.safetyBrickRings/,
    "Section 5 bootstrap remains the interface plus its explicit safety ring");
  assert.match(encode,
    /for\(letring=0;ring<initialDilationBrickRings;ring\+=1\)[\s\S]*this\.dilatePipeline[\s\S]*this\.advanceDilationPipeline/,
    "host pass boundaries must advance one exact Chebyshev frontier per ring");
  assert.match(encode,
    /this\.beginDilationPipeline[\s\S]*?,1,\[0,6,7\]\)/,
    "beginDesiredDilation reads params.pageCapacity as well as the worklist/control buffers");
  assert.match(encode,
    /this\.advanceDilationPipeline[\s\S]*?,1,\[0,6,7\]\)/,
    "advanceDesiredDilation must bind the params buffer used to clip its next frontier");
});

test("B4 leaf seeds cover all eight factor-8 bricks per finest cell", () => {
  const factor4 = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [6, 5, 4],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8 });
  const factor8 = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [6, 5, 4],
    finestCellWidth: 1, fineFactor: 8, brickResolution: 4, maximumResidentBricks: 8 });
  assert.deepEqual(planFineLevelSetLeafBrickBounds(factor4, [2, 1, 1], 1), {
    first: [2, 1, 1], last: [2, 1, 1], bricksPerFinestCell: 1, brickCount: 1,
  });
  assert.deepEqual(planFineLevelSetLeafBrickBounds(factor8, [2, 1, 1], 1), {
    first: [4, 2, 2], last: [5, 3, 3], bricksPerFinestCell: 2, brickCount: 8,
  });
  assert.deepEqual(planFineLevelSetLeafBrickBounds(factor8, [1, 1, 1], 2), {
    first: [2, 2, 2], last: [5, 5, 5], bricksPerFinestCell: 2, brickCount: 64,
  });
  assert.throws(() => planFineLevelSetLeafBrickBounds(factor8, [5, 4, 3], 2), /outside/);

  assert.match(fineLevelSetLeafSeedWGSL,
    /let first=origin\*params\.header\.x\/params\.header\.y;[\s\S]*last\/=params\.header\.y/,
    "seed publication must map leaf bounds by fineFactor / brickResolution on every axis");
});

test("fine topology telemetry labels exact requirements and overflow lower bounds", () => {
  const published = unpackFineLevelSetGPUTopologyControl([0, 7, 35, 35, 1, 0, 7, 0]);
  assert.equal(published.requiredDesiredBricks, 35);
  assert.equal(published.requiredDesiredBricksExact, true);
  assert.equal(published.dilationBrickRings, 7);
  assert.equal(published.downstreamFinalizeReason, 0);
  const overflow = unpackFineLevelSetGPUTopologyControl([1, 7, 32, 0, 1, 1, 33, 0]);
  assert.equal(overflow.desiredBricks, 32);
  assert.equal(overflow.requiredDesiredBricks, 33);
  assert.equal(overflow.requiredDesiredBricksExact, false);
  assert.equal(overflow.dilationBrickRings, 0);
  const downstream = unpackFineLevelSetGPUTopologyControl([16, 7, 32, 32, 1, 1, 0,
    FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON.redistance | FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON.volume]);
  assert.equal(downstream.downstreamFinalizeReason, 6);
});

test("fine topology rollback snapshot cannot alias Section 5 transport or fast-march scratch", () => {
  const constructor = WebGPUFineLevelSetTopology.toString().replace(/\s+/g, "");
  const encode = WebGPUFineLevelSetTopology.prototype.encode.toString().replace(/\s+/g, "");
  const finalize = WebGPUFineLevelSetTopology.prototype.encodeFinalizePublication.toString().replace(/\s+/g, "");
  assert.match(constructor,
    /current\.flags!==next\.flags[\s\S]*current\.workB!==next\.workB[\s\S]*current\.rollbackPhi!==next\.rollbackPhi[\s\S]*current\.rollbackPhi===current\.workA[\s\S]*current\.rollbackPhi===current\.workB/,
    "A/B topology must reject an aliased or generation-local rollback buffer");
  assert.match(encode, /binding:10,resource:resource\(this\.current\.rollbackPhi\)/,
    "the signed pre-transaction phi must use its dedicated rollback channel");
  assert.doesNotMatch(encode, /binding:10,resource:resource\(this\.current\.work[AB]\)/,
    "transport/distance/request scratch cannot carry rollback authority");
  assert.match(finalize, /binding:10,resource:resource\(this\.current\.rollbackPhi\)/,
    "a rejected downstream generation must restore the protected signed snapshot");
});

test("fine topology keeps cold failure unpublished and separates recurring support from affine cold seeds", () => {
  const shader = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  ).replace(/\s+/g, "");
  assert.match(shader,
    /fninsertExternalSeeds[\s\S]*if\(currentPublished&&\(tagged==INVALID\|\|\(tagged&RECURRING_SUPPORT\)==0u\)\)\{return;\}/,
    "a published generation admits only explicitly tagged recurring support keys");
  assert.match(shader,
    /fnexternalSeedPhi[\s\S]*params\.affineSeeds==0u\|\|currentFinePublished\(\)[\s\S]*return3\.402823e38/,
    "recurring pages initialize from coarse phi rather than a compact affine bootstrap plane");
  assert.match(shader,
    /fnrollbackFailedGeneration[\s\S]*if\(!currentPublished\)\{targetB\[0\]=0u;targetB\[1\]=params\.nextGeneration;targetB\[2\]=0u;targetB\[3\]=0u;targetB\[4\]=0u;atomicStore\(&control\[4\],0u\);atomicStore\(&control\[5\],1u\);return;\}/,
    "failure before a first valid fine generation must remain explicitly unpublished");
  assert.match(shader,
    /letcount=min\(sourceC\[0\],params\.pageCapacity\)[\s\S]*atomicStore\(&control\[4\],1u\);atomicStore\(&control\[5\],1u\)/,
    "only recurring rejection may publish the retagged previous fine authority");
});

test("fine topology binds exactly the resources reachable from every compute entry point", () => {
  const shader = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  );
  const bindings = new Map<string, number>();
  for (const match of shader.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+([A-Za-z_]\w*)/g,
  )) bindings.set(match[2], Number(match[1]));

  const bodies = new Map<string, string>();
  for (const match of shader.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    const open = shader.indexOf("{", match.index);
    assert.notEqual(open, -1, `function ${match[1]} must have a body`);
    let depth = 0; let close = -1;
    for (let at = open; at < shader.length; at += 1) {
      if (shader[at] === "{") depth += 1;
      if (shader[at] === "}" && --depth === 0) { close = at; break; }
    }
    assert.notEqual(close, -1, `function ${match[1]} must close its body`);
    bodies.set(match[1], shader.slice(open + 1, close));
  }
  const entryPoints = [...shader.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z_]\w*)/g,
  )].map((match) => match[1]);
  const reachableBindings = (entryPoint: string): number[] => {
    const pending = [entryPoint]; const reached = new Set<string>(); const used = new Set<number>();
    while (pending.length > 0) {
      const name = pending.pop()!;
      if (reached.has(name)) continue;
      reached.add(name);
      const body = bodies.get(name);
      assert.notEqual(body, undefined, `reachable function ${name} must exist`);
      for (const [global, binding] of bindings) {
        if (new RegExp(`\\b${global}\\b`).test(body!)) used.add(binding);
      }
      for (const callee of bodies.keys()) {
        if (!reached.has(callee) && new RegExp(`\\b${callee}\\s*\\(`).test(body!)) pending.push(callee);
      }
    }
    return [...used].sort((a, b) => a - b);
  };

  const observed = new Map<string, number[]>();
  const buffer = {} as GPUBuffer;
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 },
    queue: { writeBuffer() {} },
    createBuffer: () => ({}),
    createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint,
      getBindGroupLayout: () => ({ entryPoint: compute.entryPoint }),
    }),
    createBindGroup: ({ layout, entries }: { layout: { entryPoint: string }; entries: { binding: number }[] }) => {
      const actual = entries.map(({ binding }) => binding).sort((a, b) => a - b);
      const prior = observed.get(layout.entryPoint);
      if (prior) assert.deepEqual(actual, prior, `${layout.entryPoint} bind contract changed between encodes`);
      observed.set(layout.entryPoint, actual); return {};
    },
  } as unknown as GPUDevice;
  const shared = {
    flags: { kind: "flags" } as unknown as GPUBuffer,
    phi: { kind: "phi" } as unknown as GPUBuffer,
    workA: { kind: "workA" } as unknown as GPUBuffer,
    workB: { kind: "workB" } as unknown as GPUBuffer,
    rollbackPhi: { kind: "rollbackPhi" } as unknown as GPUBuffer,
  };
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [1, 1, 1],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 1 });
  const current = { ...shared, generation: 1, generationSlot: 0 as const, params: buffer,
    hash: buffer, metadata: buffer, worklist: buffer, plan };
  const next = { ...shared, generation: 2, generationSlot: 1 as const, params: buffer,
    hash: buffer, metadata: buffer, worklist: buffer, plan };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8 } });
  try {
    const topology = new WebGPUFineLevelSetTopology(device, current, next,
      "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}");
    topology.encode(encoder);
    topology.encodeFinalizePublication(encoder, { redistance: buffer });
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }

  assert.deepEqual([...observed.keys()].sort(), [...entryPoints].sort(),
    "host encoding must exercise every topology compute entry point");
  for (const entryPoint of entryPoints) {
    assert.deepEqual(observed.get(entryPoint), reachableBindings(entryPoint),
      `${entryPoint} bind group must equal its transitive WGSL resource reachability`);
  }
});

test("fine-brick sampling WGSL uses bounded lookup, generation validation, and explicit coarse fallback", () => {
  assert.match(fineLevelSetBrickSamplingWGSL, /probe<32u/);
  assert.match(fineLevelSetBrickSamplingWGSL, /probe>=params\.maximumHashProbes/);
  assert.match(fineLevelSetBrickSamplingWGSL, /metadata\[base\+2u\]!=params\.generation/);
  assert.match(fineLevelSetBrickSamplingWGSL, /Result\(coarsePhi,0u/);
  assert.doesNotMatch(fineLevelSetBrickSamplingWGSL, /octree.*row/i);
});

test("transport hook requires an injected octree velocity sampler and factor-ratio tracing", () => {
  assert.throws(() => makeFineLevelSetTransportWGSL(""), /sampleOctreeVelocity/);
  const source = makeFineLevelSetTransportWGSL("fn sampleOctreeVelocity(position:vec3f)->vec3f{return position*0.0;}");
  const closedSource = makeFineLevelSetTransportWGSL(
    "fn sampleOctreeVelocity(position:vec3f)->vec3f{return position*0.0;}", "closed-neumann");
  assert.match(source, /const CLOSED_DOMAIN_BOUNDARY:bool=false/);
  assert.match(closedSource, /const CLOSED_DOMAIN_BOUNDARY:bool=true/);
  assert.match(source, /segment<8u/);
  assert.match(source, /segment>=params\.fineFactor/);
  assert.match(source, /params\.timestep\/f32\(params\.fineFactor\)/);
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /abs\(phi\[index\]\)>=chunk\.transportBandDistance/,
    "transport must leave the outer valid band available only as backtrace/interpolation support");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /if\(chunk\.closedDomainBoundary==0u&&\(any\(raw<vec3f\(0\.\)\)\|\|any\(raw>sampleMax\)\)\)\{return vec2f\(0\.\);\}/,
    "wall ghosts require an explicit closed-domain policy");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /workA\[index\]=phi\[index\];\s*if\(abs\(phi\[index\]\)>=chunk\.transportBandDistance\)\{return;\}/,
    "support-only samples must preserve phi before the shared scratch is committed");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /status&0x08000000u[\s\S]*control\.faceBandUnavailable[\s\S]*control\.velocityUnavailable/,
    "transport diagnostics must distinguish face-band coverage from unavailable Stage-B velocity");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /atomicLoad\(&control\.velocityUnavailable\)==0u\)\{atomicStore\(&control\.committed,1u\)/,
    "an unavailable velocity must remain a hard transport publication failure");
  const publication = makeFineLevelSetTopologyWGSL(
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}",
  ).replace(/\s+/g, "");
  assert.match(publication,
    /transportValid=arrayLength\(&transportControl\)>=4u&&transportControl\[3\]!=0u/,
    "fine publication requires the Section 5-backed transport commit bit");
  assert.match(publication,
    /select\(0u,8u,!transportValid\)/,
    "an unavailable Section 5 transport must be visible as the exact downstream rejection reason");
  assert.match(source, /atomicLoad\(&control\.departureOutsideBand\)!=0u/);
  assert.doesNotMatch(source, /velocity(?:Texture|Grid|Buffer)/);
});

test("fine redistance applies its inclusive residual tolerance at telemetry precision", () => {
  assert.match(fineLevelSetRedistanceWGSL,
    /let residual=u32\([\s\S]*atomicMax\(&control\.residualScaled,residual\);[\s\S]*if\(residual>u32\(p\.tolerance\*1000000\.\)\)/);
  assert.doesNotMatch(fineLevelSetRedistanceWGSL, /if\(residual>p\.tolerance\)/);
});

test("every fine redistance pipeline names an existing WGSL compute entry point", () => {
  const pipelineEntryPoints = [...WebGPUFineLevelSetRedistance.toString()
    .matchAll(/pipeline\("([A-Za-z0-9_]+)"\)/g)]
    .map((match) => match[1]);
  const shaderEntryPoints = [...fineLevelSetRedistanceWGSL
    .matchAll(/@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z0-9_]+)/g)]
    .map((match) => match[1]);

  assert.equal(pipelineEntryPoints.length, 20,
    "the constructor should expose the complete bounded fine-lattice fast-march pipeline sequence");
  assert.deepEqual(new Set(pipelineEntryPoints), new Set(shaderEntryPoints),
    "pipeline construction and WGSL compute entry points must remain in exact parity");
});

test("fine redistance binds exactly the resources reachable from each compute entry point", () => {
  const observed = new Map<string, Set<number>>();
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 },
    queue: { writeBuffer() {} },
    createBuffer: () => ({}),
    createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint,
      getBindGroupLayout: () => ({ entryPoint: compute.entryPoint }),
    }),
    createBindGroup: ({ layout, entries }: { layout: { entryPoint: string }; entries: { binding: number }[] }) => {
      observed.set(layout.entryPoint, new Set(entries.map(({ binding }) => binding)));
      return {};
    },
  } as unknown as GPUDevice;
  const buffer = {} as GPUBuffer;
  const source = {
    generation: 1,
    hash: buffer, metadata: buffer, worklist: buffer, flags: buffer, phi: buffer, workA: buffer, workB: buffer,
    rollbackPhi: buffer,
    plan: {
      fineFactor: 4, brickResolution: 4, brickDimensions: [1, 1, 1], sampleDimensions: [4, 4, 4],
      samplesPerBrick: 64, hashCapacity: 2, maximumHashProbes: 2, maximumResidentBricks: 1,
      fineCellWidth: 0.25,
    },
  };
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {}, end() {} };
  const encoder = { clearBuffer() {}, beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    for (const fineFactor of [4, 8] as const) {
      new WebGPUFineLevelSetRedistance(device, {
        ...source, plan: { ...source.plan, fineFactor },
      } as never).encode(encoder, { bandCells: 1 });
    }
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }

  const expected: Record<string, number[]> = {
    initializeControl: [0, 3, 8], prepareActiveDispatch: [0, 3, 9],
    initializeDistances: [0, 2, 3, 4, 5, 6, 8], seedDistances: [0, 2, 3, 4, 5, 6, 8],
    marchBucket: [0, 2, 3, 4, 6, 8], requestPages: [0, 2, 3, 4, 6, 7, 8],
    prepareRequestDispatch: [0, 8, 9], deduplicateRequests: [0, 7, 8],
    classifyRequests: [0, 1, 4, 7, 8], initializeRequestedPages: [0, 2, 4, 5, 6, 7, 8],
    linkRequestedPages: [0, 1, 2, 7, 8], copyPublicationTable: [0, 1, 7, 8],
    reservePublicationSlots: [0, 7, 8], installReverseLinks: [0, 2, 7, 8],
    publishRequestedPages: [0, 1, 3, 7, 8], finishActivation: [0, 3, 8],
    advanceBucket: [8], validateDistances: [0, 2, 3, 4, 6, 8],
    finalizeDistances: [0, 3, 8], commitDistances: [0, 2, 3, 4, 5, 6, 8],
  };
  assert.deepEqual(Object.fromEntries([...observed].map(([entryPoint, bindings]) =>
    [entryPoint, [...bindings].sort((a, b) => a - b)])), expected);
});

test("factor-4 product redistance preserves every causal dispatch in 86 compute passes", () => {
  const passes: string[][] = [];
  let currentPipeline = "";
  let scratchClearCount = 0;
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: 65_535 }, queue: { writeBuffer() {} },
    createBuffer: () => ({}), createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({
      entryPoint: compute.entryPoint, getBindGroupLayout: () => ({}),
    }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const buffer = {} as GPUBuffer;
  const source = { generation: 1, hash: buffer, metadata: buffer, worklist: buffer,
    flags: buffer, phi: buffer, workA: buffer, workB: buffer, rollbackPhi: buffer,
    plan: { fineFactor: 4, brickResolution: 4, brickDimensions: [1, 1, 1], sampleDimensions: [4, 4, 4],
      samplesPerBrick: 64, hashCapacity: 2, maximumHashProbes: 2, maximumResidentBricks: 1,
      fineCellWidth: 0.25 } };
  const encoder = {
    clearBuffer(target: GPUBuffer) { if (target === source.workB) scratchClearCount += 1; },
    beginComputePass() {
      const commands: string[] = []; passes.push(commands);
      return {
        setPipeline(pipeline: { entryPoint: string }) { currentPipeline = pipeline.entryPoint; },
        setBindGroup() {}, dispatchWorkgroups() { commands.push(currentPipeline); },
        dispatchWorkgroupsIndirect() { commands.push(currentPipeline); }, end() {},
      };
    },
  } as unknown as GPUCommandEncoder;
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  try {
    // Product default: 4 interface cells * factor 4, plus factor + one
    // transport/interpolation support cell at publication = 21 fine cells.
    new WebGPUFineLevelSetRedistance(device, source as never).encode(encoder, { bandCells: 21 });
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }

  const march = ["marchBucket", "requestPages"];
  const activation = ["prepareRequestDispatch", "deduplicateRequests", "classifyRequests",
    "initializeRequestedPages", "linkRequestedPages", "copyPublicationTable", "reservePublicationSlots",
    "installReverseLinks", "publishRequestedPages", "finishActivation", "prepareActiveDispatch", "advanceBucket"];
  assert.equal(passes.length, 86, "21 fine cells require 42 half-cell buckets, two passes each, plus endpoints");
  assert.equal(scratchClearCount, 42, "each causal bucket retains its dedupe/publication scratch clear boundary");
  assert.deepEqual(passes[0], ["initializeControl", "prepareActiveDispatch", "initializeDistances", "seedDistances"]);
  for (let bucket = 0; bucket < 42; bucket += 1) {
    assert.deepEqual(passes[1 + bucket * 2], march);
    assert.deepEqual(passes[2 + bucket * 2], activation);
  }
  assert.deepEqual(passes.at(-1), ["validateDistances", "finalizeDistances", "commitDistances"]);
  assert.equal(passes.flat().length, 595, "pass batching must not remove or repeat any algorithm dispatch");
});

test("factor-4/factor-8 B4 redistance is distance-ordered FMM with march-driven pages and strict band validity", () => {
  assert.doesNotMatch(fineLevelSetRedistanceWGSL, /relaxDistances|Jacobi|fn pop\(|fn push\(|marchDistances/,
    "fixed whole-band relaxation is not the Section 5 fine-grid march");
  assert.match(fineLevelSetRedistanceWGSL,
    /fn bucketUpper\(\)->f32\{return f32\(atomicLoad\(&control\.bucket\)\+1u\)\*\(\.5\*p\.fineWidth\);\}/,
    "half-cell buckets stay below the h/sqrt(3) minimum 3-D upwind increment");
  assert.match(fineLevelSetRedistanceWGSL, /@compute @workgroup_size\(64\)fn marchBucket/,
    "distance work must run in parallel rather than one capacity-wide invocation");
  assert.doesNotMatch(fineLevelSetRedistanceWGSL, /activateRequests|@workgroup_size\(1\)fn .*for\(var request/,
    "page activation must not retain a serial capacity-wide fallback entry point");
  assert.match(fineLevelSetRedistanceWGSL,
    /@compute @workgroup_size\(64\)fn deduplicateRequests[\s\S]*@compute @workgroup_size\(64\)fn initializeRequestedPages/,
    "requests must be deduplicated and initialized with bounded parallel kernels");
  const encode = WebGPUFineLevelSetRedistance.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /dispatchWorkgroupsIndirect\(this\.indirect,offset\)/);
  assert.match(encode, /indirectRun\(this\.marchPipeline/,
    "each bucket must dispatch from the live active-page count");
  assert.match(encode, /options\.bandCells\*2/,
    "both quality factors march a bounded number of half-cell buckets");
  const encodeSource = WebGPUFineLevelSetRedistance.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encodeSource, /fineFactor!==4&&this\.source\.plan\.fineFactor!==8/,
    "factor eight must run the same global fine-coordinate FMM instead of failing at the product gate");
  assert.match(fineLevelSetRedistanceWGSL,
    /\(flags\[index\]&FRONTIER\)!=0u&&distance\[index\]\+p\.fineWidth<bandDistance\(\)/,
    "only the accepted causal frontier may request another page inside the physical band");
  assert.match(encode,
    /this\.initializeRequestPipeline[\s\S]*this\.linkRequestPipeline[\s\S]*this\.reservePublicationPipeline[\s\S]*this\.installLinksPipeline[\s\S]*this\.publishRequestPipeline/,
    "page payloads and forward/reverse links must complete before hash publication");
  assert.match(fineLevelSetRedistanceWGSL,
    /atomicStore\(&pageHash\[slot\*2u\+1u\],id\);atomicStore\(&pageHash\[slot\*2u\],key\)/,
    "the page ID must be visible before its hash key publication fence");
  assert.match(fineLevelSetRedistanceWGSL,
    /fn finishActivation\(\)[\s\S]*atomicStore\(&worklist\[0\],count\)/,
    "the live worklist count must publish only after all page hash entries complete");
  assert.match(fineLevelSetRedistanceWGSL,
    /if\(d>=bandDistance\(\)\|\|\(flags\[flat\]&KNOWN\)==0u\)\{flags\[flat\]=0u;\}/,
    "allocated hazard pages outside the completed narrow band must lose fine authority");

  const topologyEncode = WebGPUFineLevelSetTopology.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(topologyEncode, /initialDilationBrickRings=bandPlan\.safetyBrickRings/,
    "each A/B target starts from interface blocks plus exactly the requested one-ring hazard support");

  const topologyWGSL = makeFineLevelSetTopologyWGSL("fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}");
  assert.match(topologyWGSL,
    /sourceB\[slot\*2u\+1u\]=id;sourceC\[base\]=id;[\s\S]*sourceD\[5u\+id\]=id/,
    "dynamic activeCount allocation relies on topology's explicit dense physical-ID ABI");
  assert.match(topologyWGSL, /for\(var word=0u;word<10u;word\+=1u\)\{sourceC\[base\+word\]=INVALID;\}/,
    "the target generation must clear every metadata row before dense reassignment");
});

test("Dawn factor-4 bucket FMM activates a physical band beyond the initial one-ring topology", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-levelset checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice(); device.pushErrorScope("validation");
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [6, 1, 1],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 6 });
  const oracle = new FineLevelSetBrickOracle(plan);
  oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(plan, [2, 0, 0])], ([x]) => x - 2.5);
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const current = owner.uploadGeneration(oracle.exportGPUGeneration());
  const target = owner.prepareGPUGeneration(2);
  const topology = new WebGPUFineLevelSetTopology(device, current, target,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-2.5;}");
  const redistance = new WebGPUFineLevelSetRedistance(device, target);
  const readback = device.createBuffer({ size: 68, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const topologyEncoder = device.createCommandEncoder();
  topology.encode(topologyEncoder, undefined, [], { maximumBacktraceFineCells: 0,
    interpolationSupportFineCells: 0, redistanceBandFineCells: 7, safetyBrickRings: 1 });
  device.queue.submit([topologyEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  if (process.env.FLUID_FINE_FMM_REPORT === "1") console.error("fine-fmm-dynamic topology-complete");
  const topologyValidationError = await device.popErrorScope(); assert.equal(topologyValidationError, null);
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder();
  redistance.encode(encoder, { bandCells: 7, residualTolerance: 1 });
  encoder.copyBufferToBuffer(redistance.control, 0, readback, 0, 48);
  encoder.copyBufferToBuffer(target.worklist, 0, readback, 48, 20);
  const started = performance.now(); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  if (process.env.FLUID_FINE_FMM_REPORT === "1") console.error("fine-fmm-dynamic redistance-complete");
  const elapsedMs = performance.now() - started;
  const validationError = await device.popErrorScope(); assert.equal(validationError, null);
  await readback.mapAsync(GPUMapMode.READ); const bytes = readback.getMappedRange().slice(0); readback.unmap();
  const control = unpackFineLevelSetGPURedistanceControl(new Uint32Array(bytes, 0, 12));
  const worklist = new Uint32Array(bytes, 48, 5);
  assert.equal(control.flags, 0); assert.equal(control.firstError, 0xffff_ffff);
  assert.equal(control.unresolvedCells, 0); assert.equal(control.committed, true);
  assert.ok(control.activatedPages > 0, "the causal front must allocate beyond the initial topology ring");
  assert.ok(control.finalPages > control.initialPages);
  assert.equal(worklist[0], control.finalPages);
  assert.equal(worklist[1], 2);
  if (process.env.FLUID_FINE_FMM_REPORT === "1") {
    console.error(`fine-fmm-dynamic ${JSON.stringify({ elapsedMs, ...control })}`);
  }
  readback.destroy(); redistance.destroy(); topology.destroy(); owner.destroy(); device.destroy();
});

test("Dawn atomically rolls back a downstream-rejected fine generation and accepts a retry", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-publication checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [1, 1, 1],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 1 });
  const oracle = new FineLevelSetBrickOracle(plan);
  oracle.publishInterfaceAndRing([packFineLevelSetBrickKey(plan, [0, 0, 0])], ([x]) => x - 0.5);
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const current = owner.uploadGeneration(oracle.exportGPUGeneration());
  const target = owner.prepareGPUGeneration(2);
  const topology = new WebGPUFineLevelSetTopology(device, current, target,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-0.5;}");
  const redistanceControl = device.createBuffer({ size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const run = async (redistanceWords: Uint32Array) => {
    device.queue.writeBuffer(redistanceControl, 0, new Uint32Array(redistanceWords));
    const readback = device.createBuffer({ size: 44,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    topology.encode(encoder, undefined, [], undefined, true);
    topology.encodeFinalizePublication(encoder, { redistance: redistanceControl });
    encoder.copyBufferToBuffer(topology.control, 0, readback, 0, 32);
    encoder.copyBufferToBuffer(target.worklist, 0, readback, 32, 8);
    encoder.copyBufferToBuffer(target.phi, 0, readback, 40, 4);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const bytes = readback.getMappedRange().slice(0);
    readback.unmap(); readback.destroy(); return bytes;
  };

  const rejected = await run(new Uint32Array(4));
  assert.equal(await device.popErrorScope(), null);
  const rejectedControl = unpackFineLevelSetGPUTopologyControl(new Uint32Array(rejected, 0, 8));
  assert.notEqual(rejectedControl.flags & FINE_LEVELSET_TOPOLOGY_ERROR.downstreamPublication, 0,
    `rejected publication control ${JSON.stringify(rejectedControl)}`);
  assert.equal(rejectedControl.published, true, "the target slot must publish only the restored prior authority");
  assert.equal(rejectedControl.rolledBack, true);
  assert.deepEqual([...new Uint32Array(rejected, 32, 2)], [1, 2]);
  assert.ok(new Float32Array(rejected, 40, 1)[0] < 0, "rollback must restore the prior finite payload");

  const retried = await run(new Uint32Array([0, 0, 1, 1]));
  const retriedControl = unpackFineLevelSetGPUTopologyControl(new Uint32Array(retried, 0, 8));
  assert.equal(retriedControl.flags, 0);
  assert.equal(retriedControl.published, true);
  assert.equal(retriedControl.rolledBack, false);
  assert.deepEqual([...new Uint32Array(retried, 32, 2)], [1, 2]);

  redistanceControl.destroy(); topology.destroy(); owner.destroy(); device.destroy();
});

test("Dawn indexes global factor-4 bricks and returns coarse phi for missing/outside bricks", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-levelset checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const shaderModule = device.createShaderModule({ code: fineLevelSetBrickSamplingWGSL });
  assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
  const transportModule = device.createShaderModule({ code: makeFineLevelSetTransportWGSL(
    "fn sampleOctreeVelocity(position:vec3f)->vec3f{return position*0.0;}",
  ) });
  assert.deepEqual((await transportModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);

  for (const fineFactor of [4, 8] as const) {
    const chunkPlan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2],
      finestCellWidth: 1, fineFactor, brickResolution: 4, maximumResidentBricks: 2 });
    const chunkOwner = new WebGPUFineLevelSetBricks(device, chunkPlan);
    const chunkSource = chunkOwner.initializeEmptyGPUGeneration(1); const chunkCapacity = 32;
    assert.equal((chunkCapacity * 16) % device.limits.minStorageBufferOffsetAlignment, 0);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const results = device.createBuffer({ size: chunkCapacity * 16, usage: storage });
    const statuses = device.createBuffer({ size: chunkCapacity * 4, usage: storage });
    const sampleControl = device.createBuffer({ size: 32, usage: storage }); const offsets: number[] = [];
    const prepass = { source: { results, statuses, control: sampleControl, queryCapacity: chunkCapacity },
      encodeFromPositions(_encoder: GPUCommandEncoder, positions: GPUBuffer | GPUBufferBinding,
        _headers: GPUBuffer, _rowVelocities: GPUBuffer, options: { queryCount?: number }) {
        assert.equal(options.queryCount, chunkCapacity); assert.ok("buffer" in positions); offsets.push(positions.offset ?? 0);
      } };
    const chunkTransport = new WebGPUFineLevelSetTransport(device, chunkSource, prepass as never);
    const headers = device.createBuffer({ size: 48, usage: storage });
    const rowVelocities = device.createBuffer({ size: 16, usage: storage });
    const chunkReadback = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const chunkEncoder = device.createCommandEncoder();
    chunkTransport.encode(chunkEncoder, { timestep: 0.1, headers, rowVelocities, dimensions: [2, 2, 2],
      physicalCellSize: 1, maximumLeafSize: 1, generation: 1 });
    chunkEncoder.copyBufferToBuffer(chunkTransport.control, 0, chunkReadback, 0, 32);
    device.queue.submit([chunkEncoder.finish()]); await device.queue.onSubmittedWorkDone();
    await chunkReadback.mapAsync(GPUMapMode.READ);
    const chunkControl = unpackFineLevelSetGPUTransportControl(new Uint32Array(chunkReadback.getMappedRange().slice(0)));
    chunkReadback.unmap(); const chunks = (chunkPlan.maximumResidentBricks * chunkPlan.samplesPerBrick) / chunkCapacity;
    assert.equal(offsets.length, fineFactor * chunks);
    assert.deepEqual(offsets, Array(offsets.length).fill(0),
      "chunked transport must reuse one bounded trajectory allocation");
    assert.deepEqual(chunkControl, { departureOutsideBand: 0, nonfiniteVelocity: 0, processed: 0,
      committed: true, extrapolatedVelocity: 0, maximumDisplacementFineCells: 0,
      faceBandUnavailable: 0, velocityUnavailable: 0 });
    chunkReadback.destroy(); headers.destroy(); rowVelocities.destroy(); chunkTransport.destroy();
    results.destroy(); statuses.destroy(); sampleControl.destroy(); chunkOwner.destroy();
  }

  const plan = planFineLevelSetBricks({
    domainOrigin: [1, 2, 3], finestCellDimensions: [2, 2, 2], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8,
  });
  const oracle = new FineLevelSetBrickOracle(plan);
  const key = packFineLevelSetBrickKey(plan, [0, 0, 0]);
  oracle.publishInterfaceAndRing([key], ([x]) => x - 1.5);
  const page = oracle.pageForKey(key); assert.ok(page); page.phi[0] = -9;
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const source = owner.uploadGeneration(oracle.exportGPUGeneration());
  assert.equal(source.generation, 1);
  assert.equal(source.generationSlot, 1);

  const queryData = new Float32Array([
    1.125, 2.125, 3.125, 111,
    1 + 1.125, 2 + 1.125, 3 + 1.125, 123,
    0, 0, 0, 456,
  ]);
  const queries = device.createBuffer({ size: queryData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(queries, 0, queryData);
  const results = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "sampleQueries" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: source.params } },
    { binding: 1, resource: { buffer: source.hash } },
    { binding: 2, resource: { buffer: source.metadata } },
    { binding: 3, resource: { buffer: source.flags } },
    { binding: 4, resource: { buffer: source.phi } },
    { binding: 5, resource: { buffer: queries } },
    { binding: 6, resource: { buffer: results } },
  ] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(results, 0, readback, 0, 48);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0); readback.unmap();
  const values = new Float32Array(bytes); const words = new Uint32Array(bytes);
  assert.equal(values[0], -9); assert.equal(words[1], 1);
  assert.equal(values[4], 123); assert.equal(words[5], 0);
  assert.equal(values[8], 456); assert.equal(words[9], 0);

  const nextSource = owner.prepareGPUGeneration(2);
  const topology = new WebGPUFineLevelSetTopology(device, source, nextSource,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-1.5;}");
  const topologyReadback = device.createBuffer({ size: 52, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const topologyEncoder = device.createCommandEncoder(); topology.encode(topologyEncoder);
  topologyEncoder.copyBufferToBuffer(topology.control, 0, topologyReadback, 0, 32);
  topologyEncoder.copyBufferToBuffer(nextSource.worklist, 0, topologyReadback, 32, 20);
  device.queue.submit([topologyEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await topologyReadback.mapAsync(GPUMapMode.READ);
  const topologyWords = new Uint32Array(topologyReadback.getMappedRange().slice(0)); topologyReadback.unmap();
  const topologyControl = unpackFineLevelSetGPUTopologyControl(topologyWords);
  assert.equal(topologyControl.flags, 0);
  assert.equal(topologyControl.published, true);
  assert.ok(topologyControl.interfaceBricks > 0);
  assert.equal(topologyWords[9], 2);
  assert.equal(topologyWords[8], topologyControl.desiredBricks);

  const redistance = new WebGPUFineLevelSetRedistance(device, nextSource);
  const redistanceReadback = device.createBuffer({ size: 52, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const redistanceEncoder = device.createCommandEncoder();
  redistance.encode(redistanceEncoder, { bandCells: 2, residualTolerance: 1 });
  redistanceEncoder.copyBufferToBuffer(redistance.control, 0, redistanceReadback, 0, 48);
  redistanceEncoder.copyBufferToBuffer(nextSource.phi, 0, redistanceReadback, 48, 4);
  device.queue.submit([redistanceEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await redistanceReadback.mapAsync(GPUMapMode.READ);
  const redistanceBytes = redistanceReadback.getMappedRange().slice(0); redistanceReadback.unmap();
  const redistanceControl = unpackFineLevelSetGPURedistanceControl(new Uint32Array(redistanceBytes, 0, 12));
  assert.ok(redistanceControl.seedCount > 0);
  assert.equal(redistanceControl.unresolvedCells, 0);
  assert.equal(redistanceControl.committed, true);
  assert.equal(redistanceControl.flags, 0);
  assert.equal(redistanceControl.firstError, 0xffff_ffff);
  assert.ok(redistanceControl.initialPages > 0);
  assert.ok(redistanceControl.finalPages >= redistanceControl.initialPages);
  assert.ok(redistanceControl.acceptedCells >= redistanceControl.seedCount);
  assert.ok(new Float32Array(redistanceBytes, 48, 1)[0] < 0);
  if (process.env.FLUID_FINE_FMM_REPORT === "1") {
    console.error(`fine-fmm-control ${JSON.stringify(redistanceControl)}`);
  }

  // A topology safety guard may be much wider than the requested redistance
  // band. The fixed band-sized iteration budget must still commit by storing
  // finite saturated distances outside that band.
  const guardPlan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
    finestCellDimensions: [20, 1, 1], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 20 });
  const guardOracle = new FineLevelSetBrickOracle(guardPlan);
  const guardKeys = Array.from({ length: guardPlan.brickDimensions[0] }, (_, x) =>
    packFineLevelSetBrickKey(guardPlan, [x, 0, 0]));
  guardOracle.publishInterfaceAndRing(guardKeys, ([x]) => x - 0.5);
  const guardOwner = new WebGPUFineLevelSetBricks(device, guardPlan);
  const guardSource = guardOwner.uploadGeneration(guardOracle.exportGPUGeneration());
  const guardRedistance = new WebGPUFineLevelSetRedistance(device, guardSource);
  const guardReadback = device.createBuffer({ size: 28,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const guardEncoder = device.createCommandEncoder();
  guardRedistance.encode(guardEncoder, { bandCells: 2, residualTolerance: 1 });
  guardEncoder.copyBufferToBuffer(guardRedistance.control, 0, guardReadback, 0, 16);
  guardEncoder.copyBufferToBuffer(guardSource.phi,
    (guardPlan.maximumResidentBricks - 1) * guardPlan.samplesPerBrick * 4,
    guardReadback, 16, 4);
  guardEncoder.copyBufferToBuffer(guardSource.flags,
    (guardPlan.maximumResidentBricks - 1) * guardPlan.samplesPerBrick * 4,
    guardReadback, 20, 4);
  guardEncoder.copyBufferToBuffer(guardSource.flags, 0, guardReadback, 24, 4);
  device.queue.submit([guardEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await guardReadback.mapAsync(GPUMapMode.READ);
  const guardBytes = guardReadback.getMappedRange().slice(0); guardReadback.unmap();
  const guardControl = unpackFineLevelSetGPURedistanceControl(new Uint32Array(guardBytes, 0, 4));
  assert.ok(guardControl.seedCount > 0);
  assert.equal(guardControl.unresolvedCells, 0);
  assert.equal(guardControl.committed, true);
  assert.equal(new Float32Array(guardBytes, 16, 1)[0], 0.5);
  assert.equal(new Uint32Array(guardBytes, 20, 1)[0], 0,
    "the allocated outside-band guard must use coarse-octree phi fallback");
  assert.notEqual(new Uint32Array(guardBytes, 24, 1)[0] & 1, 0,
    "the actual narrow-band interface must remain fine-level-set authority");
  guardReadback.destroy(); guardRedistance.destroy(); guardOwner.destroy();

  const smallPlan = planFineLevelSetBricks({
    domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 4,
  });
  const smallOracle = new FineLevelSetBrickOracle(smallPlan);
  smallOracle.publishInterfaceAndRing([packFineLevelSetBrickKey(smallPlan, [0, 0, 0])], ([x]) => x - 0.5);
  const smallOwner = new WebGPUFineLevelSetBricks(device, smallPlan);
  const smallCurrent = smallOwner.uploadGeneration(smallOracle.exportGPUGeneration());
  const smallNext = smallOwner.prepareGPUGeneration(2);
  const overflowTopology = new WebGPUFineLevelSetTopology(device, smallCurrent, smallNext,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-0.5;}");
  const overflowReadback = device.createBuffer({ size: 36, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const overflowEncoder = device.createCommandEncoder(); overflowTopology.encode(overflowEncoder);
  overflowEncoder.copyBufferToBuffer(overflowTopology.control, 0, overflowReadback, 0, 32);
  overflowEncoder.copyBufferToBuffer(smallNext.worklist, 0, overflowReadback, 32, 4);
  device.queue.submit([overflowEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await overflowReadback.mapAsync(GPUMapMode.READ);
  const overflowWords = new Uint32Array(overflowReadback.getMappedRange().slice(0)); overflowReadback.unmap();
  const overflowControl = unpackFineLevelSetGPUTopologyControl(overflowWords);
  assert.notEqual(overflowControl.flags, 0);
  assert.equal(overflowControl.requiredDesiredBricks, smallCurrent.plan.maximumResidentBricks + 1);
  assert.equal(overflowControl.requiredDesiredBricksExact, false);
  assert.equal(overflowControl.published, true);
  assert.equal(overflowControl.rolledBack, true);
  assert.equal(overflowWords[8], smallCurrent.plan.maximumResidentBricks);

  const bootstrapOwner = new WebGPUFineLevelSetBricks(device, plan);
  const emptyCurrent = bootstrapOwner.initializeEmptyGPUGeneration(1);
  const bootstrapNext = bootstrapOwner.prepareGPUGeneration(2);
  const leafBuffer = new ArrayBuffer(48); const leafWords = new Uint32Array(leafBuffer);
  const leafFloats = new Float32Array(leafBuffer); leafWords[0] = 0; leafWords[1] = 1;
  // Bootstrap must copy interface values, not merely keys: this affine plane
  // crosses zero inside the first globally keyed factor-4 brick.
  leafFloats[4] = 0; leafFloats[5] = 1;
  const bootstrapLeaves = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bootstrapCandidates = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bootstrapCandidateControl = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bootstrapLeaves, 0, leafWords);
  device.queue.writeBuffer(bootstrapCandidates, 0, new Uint32Array([0, 2]));
  device.queue.writeBuffer(bootstrapCandidateControl, 0, new Uint32Array([1, 1, 1, 1]));
  const leafSeeds = new WebGPUFineLevelSetLeafSeeds(device, bootstrapNext);
  const bootstrapTopology = new WebGPUFineLevelSetTopology(device, emptyCurrent, bootstrapNext,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-0.5;}");
  const bootstrapReadback = device.createBuffer({ size: 48 + 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const bootstrapEncoder = device.createCommandEncoder();
  const seedSource = leafSeeds.encode(bootstrapEncoder, { buffer: bootstrapLeaves }, { buffer: bootstrapCandidates },
    { buffer: bootstrapCandidateControl });
  bootstrapTopology.encode(bootstrapEncoder, seedSource);
  bootstrapEncoder.copyBufferToBuffer(leafSeeds.buffer, 0, bootstrapReadback, 0, 8);
  bootstrapEncoder.copyBufferToBuffer(bootstrapTopology.control, 0, bootstrapReadback, 8, 32);
  bootstrapEncoder.copyBufferToBuffer(bootstrapNext.worklist, 0, bootstrapReadback, 40, 8);
  bootstrapEncoder.copyBufferToBuffer(bootstrapNext.phi, 0, bootstrapReadback, 48, 256);
  device.queue.submit([bootstrapEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await bootstrapReadback.mapAsync(GPUMapMode.READ);
  const bootstrapWords = new Uint32Array(bootstrapReadback.getMappedRange().slice(0)); bootstrapReadback.unmap();
  assert.deepEqual([...bootstrapWords.slice(0, 2)], [1, 0]);
  const bootstrapControl = unpackFineLevelSetGPUTopologyControl(bootstrapWords.slice(2, 10));
  assert.equal(bootstrapControl.flags, 0);
  // A block 1-ring is the complete 3D Chebyshev neighborhood, clipped at
  // this domain corner (2 x 2 x 2 bricks).
  assert.equal(bootstrapControl.desiredBricks, 8);
  assert.equal(bootstrapControl.dilationBrickRings, 1);
  assert.equal(bootstrapControl.requiredDesiredBricksExact, true);
  assert.equal(bootstrapControl.published, true);
  assert.deepEqual([...bootstrapWords.slice(10, 12)], [8, 2]);
  const bootstrapPhi = new Float32Array(bootstrapWords.buffer, 48, 64);
  assert.ok([...bootstrapPhi].some((value) => value < 0));
  assert.ok([...bootstrapPhi].some((value) => value > 0));

  redistance.destroy(); redistanceReadback.destroy(); topology.destroy(); topologyReadback.destroy();
  overflowTopology.destroy(); overflowReadback.destroy(); smallOwner.destroy();
  leafSeeds.destroy(); bootstrapTopology.destroy(); bootstrapReadback.destroy(); bootstrapOwner.destroy();
  bootstrapLeaves.destroy(); bootstrapCandidates.destroy(); bootstrapCandidateControl.destroy();
  owner.destroy(); queries.destroy(); results.destroy(); readback.destroy(); device.destroy();
});
