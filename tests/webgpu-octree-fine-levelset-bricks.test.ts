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
  FINE_LEVELSET_FMM_MAX_DIAGNOSTIC_SAMPLES,
  WebGPUFineLevelSetRedistance,
  fineLevelSetJFACPTWGSL,
  fineLevelSetRedistanceWGSL,
  planFineLevelSetJFAStrides,
  resolveFineLevelSetRedistanceMethod,
  unpackFineLevelSetGPURedistanceControl,
} from "../lib/webgpu-octree-fine-levelset-redistance";
import {
  FINE_LEVELSET_TOPOLOGY_ERROR,
  FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON,
  fineLevelSetLeafSeedWGSL,
  makeFineLevelSetTopologyWGSL,
  FINE_LEVELSET_DIRECT_DILATION_MAXIMUM_BRICKS,
  planFineLevelSetLeafBrickBounds,
  planFineLevelSetChebyshevFloodPasses,
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

test("fine topology takes the larger trajectory/redistance radius plus the paper one-ring", () => {
  assert.deepEqual(planFineLevelSetTopologyBand(4, {
    maximumBacktraceFineCells: 8,
    interpolationSupportFineCells: 1,
    redistanceBandFineCells: 32,
  }), {
    maximumBacktraceFineCells: 8,
    interpolationSupportFineCells: 1,
    redistanceBandFineCells: 32,
    safetyBrickRings: 1,
    requiredFineCells: 32,
    dilationBrickRings: 9,
  });
  assert.throws(() => planFineLevelSetTopologyBand(4, {
    maximumBacktraceFineCells: 1, interpolationSupportFineCells: 1,
    redistanceBandFineCells: 4, safetyBrickRings: 0,
  }), /at least one publication safety ring/);
});

test("fine topology discovers and pre-dilates the complete support with a logarithmic Chebyshev flood", () => {
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
  assert.match(wgsl, /@compute@workgroup_size\(64\)fndilateDesiredFromSeeds/);
  assert.match(wgsl,
    /fninsertDesired\(key:u32\)[\s\S]*atomicCompareExchangeWeak\(&targetA\[slot\*2u\],INVALID,key\)/,
    "parallel discovery must atomically deduplicate desired pages");
  assert.match(wgsl,
    /fnbeginDesiredDilation\(\)\{targetB\[0\]=0u;targetB\[1\]=min\(atomicLoad\(&control\[2\]\),params\.pageCapacity\);atomicStore\(&control\[8\],targetB\[1\]\);\}/,
    "the exact interface/endpoint prefix must be frozen before allocation-halo dilation");
  assert.match(wgsl,
    /fndilateDesiredRing[\s\S]*letradius=targetB\[0\];letlayerEnd=targetB\[1\];[\s\S]*letexpansion=min\(radius\+1u,params\.dilationBrickRings-radius\)[\s\S]*targetB\[5u\+work\]/,
    "each pass expands the complete prior Chebyshev ball by a doubling radius");
  assert.match(wgsl,
    /fnadvanceDesiredDilation\(\)[\s\S]*targetB\[0\]=radius\+expansion;targetB\[1\]=min\(atomicLoad\(&control\[2\]\),params\.pageCapacity\);\}/);
  assert.match(wgsl,
    /rawCount>params\.pageCapacity\|\|rawCount>available[\s\S]*atomicOr\(&control\[0\],CAPACITY\);atomicMax\(&control\[6\],max\(rawCount,params\.pageCapacity\+1u\)\)/,
    "truncated external seed lists must fail closed with a strict capacity lower bound");

  assert.match(encode, /dilationBrickRings=bandPlan\.dilationBrickRings/,
    "topology must allocate transport, interpolation, redistance, and safety support before redistance");
  assert.match(encode,
    /logicalBrickCount<=FINE_LEVELSET_DIRECT_DILATION_MAXIMUM_BRICKS[\s\S]*this\.directDilationPipeline[\s\S]*floodPasses=planFineLevelSetChebyshevFloodPasses\(dilationBrickRings\)[\s\S]*this\.dilatePipeline[\s\S]*this\.advanceDilationPipeline/,
    "mini lattices use one direct expansion while larger lattices retain logarithmic flooding");
  assert.match(wgsl,
    /fndilateDesiredFromSeeds[\s\S]*seedCount=targetB\[1\][\s\S]*completeCapacity=params\.pageCapacity>=logicalBricks[\s\S]*insertDesired/,
    "direct dilation must consume only the frozen seed prefix and preserve capacity overflow detection");
  assert.match(encode,
    /this\.beginDilationPipeline[\s\S]*?,1,\[0,6,7\]\)/,
    "beginDesiredDilation reads params.pageCapacity as well as the worklist/control buffers");
  assert.match(encode,
    /this\.advanceDilationPipeline[\s\S]*?,1,\[0,6,7\]\)/,
    "advanceDesiredDilation must bind the params buffer used to clip its next frontier");
  assert.equal(encode.match(/beginComputePass/g)?.length, 1,
    "the launch-bound topology chain must remain in one ordered compute pass");
  const finalize = WebGPUFineLevelSetTopology.prototype.encodeFinalizePublication.toString().replace(/\s+/g, "");
  assert.equal(finalize.match(/beginComputePass/g)?.length, 1,
    "publication validation and failure-only rollback must share one ordered compute pass");
  assert.equal(planFineLevelSetChebyshevFloodPasses(0), 0);
  assert.equal(planFineLevelSetChebyshevFloodPasses(1), 1);
  assert.equal(planFineLevelSetChebyshevFloodPasses(6), 3);
  assert.equal(planFineLevelSetChebyshevFloodPasses(12), 4);
  assert.equal(FINE_LEVELSET_DIRECT_DILATION_MAXIMUM_BRICKS, 15 ** 3);
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
    /let first=origin\*params\.header\.x\/params\.header\.y;[\s\S]*last=min\(last\/params\.header\.y/,
    "seed publication must map leaf bounds by fineFactor / brickResolution on every axis");
  assert.match(fineLevelSetLeafSeedWGSL,
    /scanBlockBase\(\)->u32\{return max\(params\.scan\.y,params\.scan\.w\)\+2u;\}/,
    "the scan total/base footer must not overlap the candidate-row eligibility bitset");
});

test("fine topology telemetry labels exact requirements and overflow lower bounds", () => {
  const published = unpackFineLevelSetGPUTopologyControl([0, 7, 35, 35, 1, 0, 7, 0]);
  assert.equal(published.requiredDesiredBricks, 35);
  assert.equal(published.requiredDesiredBricksExact, true);
  assert.equal(published.dilationBrickRings, 7);
  assert.equal(published.downstreamFinalizeReason, 0);
  assert.equal(published.interfaceSeedBricks, undefined,
    "legacy eight-word diagnostic snapshots remain decodable");
  const prefixed = unpackFineLevelSetGPUTopologyControl([0, 7, 35, 35, 1, 0, 7, 0, 11]);
  assert.equal(prefixed.interfaceSeedBricks, 11);
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
    /fncurrentFinePopulated\(\)->bool\{returncurrentFinePublished\(\)&&currentWorklist\[0\]>0u;\}/,
    "a structurally published empty A/B slot remains a cold bootstrap source");
  assert.match(shader,
    /fninsertExternalSeeds[\s\S]*letrecurring=currentFinePopulated\(\)[\s\S]*if\(recurring&&!endpoint\)\{return;\}/,
    "a published generation admits only explicitly tagged recurring support keys");
  assert.match(shader,
    /fnexternalSeedPhi[\s\S]*params\.affineSeeds==0u\|\|currentFinePopulated\(\)[\s\S]*return3\.402823e38/,
    "recurring pages initialize from coarse phi rather than a compact affine bootstrap plane");
  assert.match(shader,
    /fnexternalAffineInterfaceBrick[\s\S]*for\(varcorner=0u;corner<8u;corner\+=1u\)[\s\S]*returnminimum<=0\.0&&maximum>=0\.0/,
    "cold affine keys enter the pre-dilation prefix only when their page contains a zero crossing");
  assert.match(shader,
    /letfirst=vec3f\(brick\*params\.brickResolution\)\/f32\(params\.fineFactor\);letlast=vec3f\(\(brick\+vec3u\(1u\)\)\*params\.brickResolution\)\/f32\(params\.fineFactor\)/,
    "interface admission must include shared page boundaries instead of testing sample centres only");
  const alignedFace = 1;
  const centreBounds = [[0.125, 0.875], [1.125, 1.875]] as const;
  const supportBounds = [[0, 1], [1, 2]] as const;
  assert.ok(centreBounds.every(([first, last]) => !((first - alignedFace) <= 0 && (last - alignedFace) >= 0)),
    "the regression fixture must reproduce centre-only rejection on both sides of an aligned face");
  assert.ok(supportBounds.every(([first, last]) => (first - alignedFace) <= 0 && (last - alignedFace) >= 0),
    "both page supports must admit their shared zero face for deterministic one-ring ownership");
  assert.match(shader,
    /if\(!recurring&&!endpoint&&!externalAffineInterfaceBrick\(key\)\)\{return;\}/,
    "the published-empty t0 path cannot collapse to a zero interface prefix or seed the entire affine leaf set");
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
    const largePlan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
      finestCellDimensions: [17, 17, 17], finestCellWidth: 1, fineFactor: 4,
      brickResolution: 4, maximumResidentBricks: 1 });
    const largeCurrent = { ...current, plan: largePlan };
    const largeNext = { ...next, plan: largePlan };
    new WebGPUFineLevelSetTopology(device, largeCurrent, largeNext,
      "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}").encode(encoder);
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
    /if\(chunk\.closedDomainBoundary==0u&&\(any\(raw<vec3f\(0\.\)\)\|\|raw\.x>sampleMax\.x\|\|raw\.z>sampleMax\.z\|\|\(raw\.y>sampleMax\.y&&chunk\.openTopBoundary==0u\)\)\)\{return vec3f\(0\.,0\.,1\.\);\}/,
    "wall ghosts require an explicit closed-domain or authored open-top policy");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /workA\[index\]=phi\[index\];\s*if\(abs\(phi\[index\]\)>=chunk\.transportBandDistance\)\{return;\}/,
    "support-only samples must preserve phi before the shared scratch is committed");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /status&0x08000000u[\s\S]*flags\|=FACE_UNAVAILABLE[\s\S]*flags\|=INVALID_STATUS\|VELOCITY_UNAVAILABLE/,
    "transport diagnostics must distinguish face-band coverage from unavailable Stage-B velocity");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /control\.committed=select\(0u,1u,s0\[0\]==0u&&s1\[0\]==0u&&s7\[0\]==0u\)/,
    "an unavailable velocity must remain a hard transport publication failure");
  assert.doesNotMatch(fineLevelSetGPUQueryTransportWGSL, /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
    "recurring fine transport must reduce deterministic per-query outcomes without atomics");
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
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /value\.y==0\.\)[\s\S]*outcomes\[local\]=vec2u\(packOutcome\(flags,extrapolated,displacement\),0x04000000u\|u32\(value\.z\)\)[\s\S]*bitcast<u32>\(positions\[i\]\.x\)/,
    "a rejected departure must retain its failure class and first physical coordinate for sparse-band forensics");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /chunk\.closedDomainBoundary==0u&&\(any\(below\)\|\|above\.x\|\|above\.z\|\|\(above\.y&&chunk\.openTopBoundary==0u\)\)/,
    "closed-wall Neumann extension must absorb exterior integration drift while strict boundaries reject it");
  assert.match(fineLevelSetGPUQueryTransportWGSL,
    /above\.y&&chunk\.openTopBoundary==0u[\s\S]*extendBoundary=chunk\.closedDomainBoundary!=0u\|\|chunk\.openTopBoundary!=0u/,
    "only an authored open ceiling may extend an exterior top characteristic onto the boundary sample");
  assert.doesNotMatch(source, /velocity(?:Texture|Grid|Buffer)/);
});

test("fine redistance applies its inclusive residual tolerance at telemetry precision", () => {
  assert.match(fineLevelSetRedistanceWGSL,
    /let residual=u32\([\s\S]*atomicMax\(&control\.residualScaled,residual\);[\s\S]*if\(residual>u32\(p\.tolerance\*1000000\.\)\)/);
  assert.doesNotMatch(fineLevelSetRedistanceWGSL, /if\(residual>p\.tolerance\)/);
});

test("fine redistance exposes fixed JFA-CPT strides and a selectable FMM oracle", () => {
  assert.equal(FINE_LEVELSET_FMM_MAX_DIAGNOSTIC_SAMPLES, 256,
    "the backend-gated oracle must stay limited to four B4 diagnostic pages");
  assert.deepEqual(planFineLevelSetJFAStrides(21), [32, 16, 8, 4, 2, 1, 1]);
  assert.deepEqual(planFineLevelSetJFAStrides(1), [1, 1]);
  assert.equal(resolveFineLevelSetRedistanceMethod(undefined), "jfa-cpt");
  assert.equal(resolveFineLevelSetRedistanceMethod("fmm"), "fmm");
  assert.throws(() => resolveFineLevelSetRedistanceMethod("heap"), /Unknown/);
  assert.match(fineLevelSetJFACPTWGSL,
    /d<bestD\|\|\(d==bestD&&seedStableKey\(candidate\)<seedStableKey\(best\)\)/,
    "equal-distance propagation must choose the stable global sample key");
  assert.match(fineLevelSetJFACPTWGSL,
    /var bestD=LARGE;if\(best!=INVALID\)\{let delta=point-materializedClosestPoint\(best\);bestD=dot\(delta,delta\);\}/,
    "each flood must cache the current winner's distance instead of recomputing its closest point per candidate");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn resolvedDistance\(seed:u32,q:vec3u\)->f32\{if\(seed==INVALID\)\{return bandDistance\(\);\}return length/,
    "a reachable distance must remain unclamped so beyond-band seeds cannot masquerade as exact-cutoff samples");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn resolvedSeed\(index:u32\)->u32\{return select\(workB\[index\],workA\[index\],p\.distanceInB!=0u\);\}/,
    "the final distance channel must retain the opposite channel's closest-point identity");
  assert.match(fineLevelSetJFACPTWGSL,
    /if\(resolvedSeed\(index\)==INVALID\|\|d>bandDistance\(\)\)\{flags\[index\]=0u;\}/,
    "the closed cutoff may publish only a sample reached from a real interface seed");
  assert.doesNotMatch(fineLevelSetJFACPTWGSL, /fn betterSeed/,
    "candidate comparison must evaluate each closest point only once");
  assert.match(fineLevelSetJFACPTWGSL,
    /flags\[index\]\|=INTERFACE\|\(closest<<SAMPLE_FLAG_BITS\)/,
    "seeding must materialize the subcell closest point into otherwise-unused sample flag bits");
  assert.match(fineLevelSetJFACPTWGSL,
    /fn materializedClosestPoint\(index:u32\)[^}]*flags\[index\]>>SAMPLE_FLAG_BITS[^}]*CP_FRACTION_MASK/,
    "flood comparisons must read the cached closest point without re-walking neighboring phi");
  assert.doesNotMatch(fineLevelSetJFACPTWGSL,
    /fn materializedClosestPoint\([^}]*sampleIndex|fn materializedClosestPoint\([^}]*pageOf/,
    "cached closest-point lookup must not perform a sparse hash lookup");
  assert.match(fineLevelSetJFACPTWGSL,
    /flags\[index\]&INTERFACE\)!=0u\)\{seed=index;\}/,
    "interface seeds must retain the FMM oracle's subcell zero-crossing samples");
  assert.match(fineLevelSetJFACPTWGSL,
    /seed==INVALID&&abs\(bitcast<f32>\(phi\[index\]\)\)<bandDistance\(\)[\s\S]*control\.unresolved/,
    "a JFA propagation miss inside transported narrow-band support must fail publication");
});

test("sparse diagonal JFA landing gaps deterministically reject narrow-band publication", () => {
  const brickResolution = 4; const dimensions = [6, 6, 1] as const;
  const nearRing = [[0, 0, 0], [1, 0, 0], [0, 1, 0]] as const;
  const farRing = [[4, 4, 0], [3, 4, 0], [5, 4, 0], [4, 3, 0], [4, 5, 0]] as const;
  const residentBricks = new Set([...nearRing, ...farRing].map((q) => q.join(",")));
  const residentSamples: [number, number, number][] = [];
  for (let z = 0; z < dimensions[2] * brickResolution; z += 1) {
    for (let y = 0; y < dimensions[1] * brickResolution; y += 1) {
      for (let x = 0; x < dimensions[0] * brickResolution; x += 1) {
        if (residentBricks.has([Math.floor(x / 4), Math.floor(y / 4), Math.floor(z / 4)].join(","))) {
          residentSamples.push([x, y, z]);
        }
      }
    }
  }
  const key = (q: readonly number[]) => q.join(",");
  let reachable = new Set(residentSamples
    .filter(([x, y]) => x < 8 && y < 8)
    .map(key));
  for (const stride of planFineLevelSetJFAStrides(7)) {
    const next = new Set(reachable);
    for (const q of residentSamples) {
      let found = false;
      for (let dz = -1; dz <= 1 && !found; dz += 1) {
        for (let dy = -1; dy <= 1 && !found; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (reachable.has(key([q[0] + dx * stride, q[1] + dy * stride, q[2] + dz * stride]))) {
              found = true; break;
            }
          }
        }
      }
      if (found) next.add(key(q));
    }
    reachable = next;
  }
  assert.equal(reachable.has(key([16, 16, 0])), false,
    "the disconnected diagonal island has no resident landing chain from the interface seeds");
  assert.match(fineLevelSetJFACPTWGSL,
    /seed==INVALID&&abs\(bitcast<f32>\(phi\[index\]\)\)<bandDistance\(\)\)\{atomicAdd\(&control\.unresolved,1u\);\}/,
    "an unreachable transported narrow-band sample must reject finalization");
  assert.match(fineLevelSetJFACPTWGSL,
    /atomicLoad\(&control\.unresolved\)==0u[\s\S]*atomicStore\(&control\.committed,1u\)/,
    "publication cannot commit while a sparse propagation gap is unresolved");
});

test("every constructed fine redistance pipeline names an existing WGSL compute entry point", () => {
  const pipelineEntryPoints = [...WebGPUFineLevelSetRedistance.toString()
    .matchAll(/pipeline\("([A-Za-z0-9_]+)"\)/g)]
    .map((match) => match[1]);
  const shaderEntryPoints = [...fineLevelSetRedistanceWGSL
    .matchAll(/@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z0-9_]+)/g)]
    .map((match) => match[1]);

  assert.equal(pipelineEntryPoints.length, 9);
  assert.ok(pipelineEntryPoints.every((entryPoint) => shaderEntryPoints.includes(entryPoint)));
  assert.doesNotMatch(WebGPUFineLevelSetRedistance.toString(),
    /this\.(?:request|prepareRequest|dedupe|classify|initializeRequest|linkRequest|copyPublication|reservePublication|installLinks|publishRequest|finishActivation)Pipeline/,
    "the FMM oracle must not construct the old mid-march allocator chain");
  const jfaEntryPoints = [...WebGPUFineLevelSetRedistance.toString()
    .matchAll(/jfaPipeline\("([A-Za-z0-9_]+)"\)/g)].map((match) => match[1]);
  const jfaShaderEntryPoints = [...fineLevelSetJFACPTWGSL
    .matchAll(/@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z0-9_]+)/g)].map((match) => match[1]);
  assert.deepEqual(new Set(jfaEntryPoints), new Set(jfaShaderEntryPoints));
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
      } as never).encode(encoder, { bandCells: 2 });
    }
    assert.throws(() => new WebGPUFineLevelSetRedistance(device, {
      ...source, plan: { ...source.plan, brickDimensions: [2, 2, 2], sampleDimensions: [8, 8, 8] },
    } as never).encode(encoder, { bandCells: 2, method: "fmm" }),
    /limited to 256 logical samples/);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
  }

  const expected: Record<string, number[]> = {
    initializeJFAControl: [0, 3, 8], seedClosestPoints: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    jumpFloodAToB: [0, 1, 2, 3, 4, 6, 7], jumpFloodBToA: [0, 1, 2, 3, 4, 6, 7],
    resolveClosestPointsBToA: [0, 2, 3, 4, 5, 6, 7, 8],
    validateJFADistances: [0, 1, 2, 3, 4, 6, 7, 8],
    finalizeJFADistances: [0, 3, 8], commitJFADistances: [0, 2, 3, 4, 5, 6, 7, 8],
  };
  assert.deepEqual(Object.fromEntries([...observed].map(([entryPoint, bindings]) =>
    [entryPoint, [...bindings].sort((a, b) => a - b)])), expected);
});

test("factor-4 product JFA-CPT redistance is one pass with fixed dispatches", () => {
  const passes: string[][] = [];
  let currentPipeline = "";
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
    clearBuffer() {},
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

  assert.equal(passes.length, 1);
  assert.deepEqual(passes[0], ["initializeJFAControl", "seedClosestPoints",
    "jumpFloodAToB", "jumpFloodBToA", "jumpFloodAToB", "jumpFloodBToA",
    "jumpFloodAToB", "jumpFloodBToA", "jumpFloodAToB", "resolveClosestPointsBToA",
    "validateJFADistances", "finalizeJFADistances", "commitJFADistances"]);
  assert.equal(passes[0].slice(1, 10).length, 9,
    "the 21-cell distance transform is one seed, seven floods, and one resolve dispatch");
});

test("factor-4/factor-8 B4 redistance keeps fixed-resident FMM as an oracle", () => {
  assert.doesNotMatch(fineLevelSetRedistanceWGSL, /relaxDistances|Jacobi|fn pop\(|fn push\(|marchDistances/,
    "fixed whole-band relaxation is not the Section 5 fine-grid march");
  assert.match(fineLevelSetRedistanceWGSL,
    /fn bucketUpper\(\)->f32\{return f32\(atomicLoad\(&control\.bucket\)\+1u\)\*\(\.5\*p\.fineWidth\);\}/,
    "half-cell buckets stay below the h/sqrt(3) minimum 3-D upwind increment");
  assert.match(fineLevelSetRedistanceWGSL,
    /@compute @workgroup_size\(64\)fn snapshotKnownDistances[\s\S]*snapshot\[index\]=select\(LARGE,distance\[index\],\(flags\[index\]&KNOWN\)!=0u\);/,
    "each FMM bucket must freeze its causal input before the parallel update");
  assert.match(fineLevelSetRedistanceWGSL,
    /@compute @workgroup_size\(64\)fn marchBucket/,
    "the opt-in FMM oracle must not serialize the complete resident sample set onto one GPU lane");
  assert.doesNotMatch(fineLevelSetRedistanceWGSL, /activateRequests|@workgroup_size\(1\)fn .*for\(var request/,
    "page activation must not retain a serial capacity-wide fallback entry point");
  const encode = WebGPUFineLevelSetRedistance.prototype.encode.toString().replace(/\s+/g, "");
  assert.doesNotMatch(encode, /dispatchWorkgroupsIndirect|this\.indirect/,
    "the validation oracle uses the same fixed-capacity direct dispatch shape as JFA");
  assert.match(encode, /run\(this\.snapshotPipeline,[\s\S]*this\.source\.plan\.maximumResidentBricks,pass\)[\s\S]*run\(this\.marchPipeline,[\s\S]*this\.source\.plan\.maximumResidentBricks,pass\)/,
    "each oracle bucket must snapshot and march the resident pages in parallel");
  assert.match(encode, /options\.bandCells\*2/,
    "both quality factors march a bounded number of half-cell buckets");
  const encodeSource = WebGPUFineLevelSetRedistance.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encodeSource, /fineFactor!==4&&this\.source\.plan\.fineFactor!==8/,
    "factor eight must run the same global fine-coordinate FMM instead of failing at the product gate");
  assert.doesNotMatch(encode, /requestPipeline|dedupePipeline|publishRequestPipeline|clearBuffer\(this\.source\.workB/,
    "redistance must never mutate fixed topology residency");
  assert.match(fineLevelSetRedistanceWGSL,
    /if\(d>=bandDistance\(\)\|\|\(flags\[flat\]&KNOWN\)==0u\)\{flags\[flat\]=0u;\}/,
    "allocated hazard pages outside the completed narrow band must lose fine authority");

  const topologyEncode = WebGPUFineLevelSetTopology.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(topologyEncode, /dilationBrickRings=bandPlan\.dilationBrickRings/,
    "each A/B target starts with the complete redistance support resident");

  const topologyWGSL = makeFineLevelSetTopologyWGSL("fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x;}");
  assert.match(topologyWGSL,
    /sourceB\[slot\*2u\+1u\]=id;sourceC\[base\]=id;[\s\S]*sourceD\[5u\+id\]=id/,
    "dynamic activeCount allocation relies on topology's explicit dense physical-ID ABI");
  assert.match(topologyWGSL, /for\(var word=0u;word<10u;word\+=1u\)\{sourceC\[base\+word\]=INVALID;\}/,
    "the target generation must clear every metadata row before dense reassignment");
});

test("opt-in Dawn reproducer: factor-4 fixed-resident FMM consumes the pre-dilated physical band", {
  skip: !process.env.WEBGPU_NODE_MODULE || process.env.FLUID_RUN_FINE_FMM_ORACLE !== "1"
    ? "set WEBGPU_NODE_MODULE and FLUID_RUN_FINE_FMM_ORACLE=1 for the backend-gated FMM reproducer" : false,
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice(); device.pushErrorScope("validation");
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [4, 1, 1],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 4 });
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
  const topologyValidationError = await device.popErrorScope();
  assert.equal(topologyValidationError?.message ?? null, null);
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder();
  redistance.encode(encoder, { bandCells: 7, residualTolerance: 1, method: "fmm" });
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
  assert.equal(control.activatedPages, 0, "redistance may not allocate pages");
  assert.equal(control.finalPages, control.initialPages);
  assert.equal(control.initialPages, plan.maximumResidentBricks,
    "the topology flood must make the complete bounded domain resident first");
  assert.equal(worklist[0], control.finalPages);
  assert.equal(worklist[1], 2);
  if (process.env.FLUID_FINE_FMM_REPORT === "1") {
    console.error(`fine-fmm-dynamic ${JSON.stringify({ elapsedMs, ...control })}`);
  }
  readback.destroy(); redistance.destroy(); topology.destroy(); owner.destroy(); device.destroy();
});

test("opt-in Dawn backend reproducer: sparse diagonal JFA support gap dispatch", {
  skip: !process.env.WEBGPU_NODE_MODULE || process.env.FLUID_RUN_SPARSE_JFA_GAP !== "1"
    ? "set WEBGPU_NODE_MODULE and FLUID_RUN_SPARSE_JFA_GAP=1 for the Dawn sparse-gap reproducer" : false,
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice(); device.pushErrorScope("validation");
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [6, 6, 2],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 16 });
  const oracle = new FineLevelSetBrickOracle(plan);
  const near = packFineLevelSetBrickKey(plan, [0, 0, 0]);
  const diagonal = packFineLevelSetBrickKey(plan, [4, 4, 0]);
  // Two axial one-rings form disconnected diagonal islands. Only the near
  // island crosses zero; the far island deliberately claims transported
  // narrow-band values so a missed CPT propagation must reject publication.
  oracle.publishInterfaceAndRing([near, diagonal], ([x, y]) =>
    x > 3 && y > 3 ? 0.25 : x - 0.375);
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const source = owner.uploadGeneration(oracle.exportGPUGeneration());
  const redistance = new WebGPUFineLevelSetRedistance(device, source);
  const readback = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  redistance.encode(encoder, { bandCells: 7, residualTolerance: 1 });
  encoder.copyBufferToBuffer(redistance.control, 0, readback, 0, 48);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  const validationError = await device.popErrorScope(); assert.equal(validationError, null);
  await readback.mapAsync(GPUMapMode.READ);
  const control = unpackFineLevelSetGPURedistanceControl(
    new Uint32Array(readback.getMappedRange().slice(0), 0, 12));
  readback.unmap();
  assert.ok(control.seedCount > 0);
  // This is the intended result once the Dawn/Metal backend fault is cleared;
  // the deterministic test above owns the always-on fail-closed contract.
  assert.ok(control.unresolvedCells > 0,
    "a narrow-band island unreachable through resident JFA landing points must be reported");
  assert.equal(control.committed, false);
  assert.equal(control.flags, 0, "a propagation gap is a publication rejection, not a shader fault");
  readback.destroy(); redistance.destroy(); owner.destroy(); device.destroy();
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
      encodeRowDescriptors() {},
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
  assert.ok(overflowControl.requiredDesiredBricks > smallCurrent.plan.maximumResidentBricks);
  assert.equal(overflowControl.requiredDesiredBricksExact, false);
  assert.equal(overflowControl.published, true);
  assert.equal(overflowControl.rolledBack, true);
  assert.equal(overflowWords[8], smallCurrent.plan.maximumResidentBricks);

  const bootstrapOwner = new WebGPUFineLevelSetBricks(device, plan);
  const emptyCurrent = bootstrapOwner.initializeEmptyGPUGeneration(1);
  const bootstrapNext = bootstrapOwner.prepareGPUGeneration(2);
  const leafBuffer = new ArrayBuffer(64); const leafWords = new Uint32Array(leafBuffer);
  const leafFloats = new Float32Array(leafBuffer); leafWords[3] = 1; leafWords[4] = 2;
  // Bootstrap must copy interface values, not merely keys: this affine plane
  // crosses zero inside the first globally keyed factor-4 brick.
  leafFloats[8] = 0; leafFloats[9] = 1;
  const bootstrapLeaves = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
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
