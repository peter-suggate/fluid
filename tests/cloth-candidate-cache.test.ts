import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { OgcContactDetector } from '../lib/cloth/contact/ogc-detector';
import { buildOgcEdges, buildOgcTopology } from '../lib/cloth/contact/ogc-topology';
import { buildOgcBVH } from '../lib/cloth/contact/bvh';
import { merge, sheet } from './cloth-contact-oracle';
import { candidateCacheWords } from '../lib/cloth/contact/candidate-cache-wgsl';
import { CLOTH_SEPARATION_CULL_WGSL } from '../lib/cloth/contact/separation-culling-wgsl';
import { closestPointTriangle, closestSegmentSegment, type V3 } from './cloth-contact-oracle';
import { ClothSystem, CLOTH_SCENE_STEP_S } from '../lib/cloth/system';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription } from '../lib/cloth/scenes/bridson-2002/scene-description';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
test('stackless BVH covers every face exactly once and its escape links leave each subtree', () => {
  const mesh=merge(sheet(8,8,0.01),sheet(5,5,0.01,[1,0,0]));
  const triangles=new Uint32Array(mesh.tris.flatMap(t=>[...t,0]));
  const rest=new Float32Array(mesh.x.flatMap(x=>[...x,0]));
  const topology=buildOgcTopology({vertexCount:mesh.x.length,triangles,edges:buildOgcEdges(triangles)},rest);
  const bvh=buildOgcBVH(topology,rest,128),faces:number[]=[],visited=new Set<number>();
  const walk=(node:number):number=>{
    assert.ok(!visited.has(node));visited.add(node);
    const b=12*node,count=bvh.words[b+8]!;
    const start=faces.length;
    if(count){for(let i=0;i<count;i++)faces.push(bvh.words[bvh.words[b+7]!+i]!);assert.equal(bvh.words[b+6],node+1);}
    else {const next=walk(node+1);assert.equal(bvh.words[b+9],next);assert.equal(walk(next),bvh.words[b+6]);}
    const subtreeFaces=new Set(faces.slice(start));let greatest=0;
    for(let e=0;e<topology.layout.edgeCount;e++)if(subtreeFaces.has(topology.words[topology.layout.edgesOffset+4*e+2]!))greatest=e+1;
    assert.equal(bvh.words[b+10],greatest,'subtree edge bound matches face ownership independently');
    return bvh.words[b+6]!;
  };
  assert.equal(walk(0),bvh.nodes);assert.equal(visited.size,bvh.nodes);
  assert.deepEqual(faces.sort((a,b)=>a-b),Array.from({length:mesh.tris.length},(_,i)=>i));
  assert.equal(bvh.levels.reduce((sum,l)=>sum+l.count,0),bvh.nodes);
});
test('candidate cache rejects invalid capacity and allowance', () => {
  for (const skin_m of [0, -1, NaN, Infinity]) assert.throws(() => candidateCacheWords(2, 3, { skin_m, facesPerQuery: 4 }));
  for (const facesPerQuery of [0, -1, 1.5, Infinity, 2 ** 32]) assert.throws(() => candidateCacheWords(2, 3, { skin_m: 0.01, facesPerQuery }));
  assert.equal(candidateCacheWords(2, 3, { skin_m: 0.01, facesPerQuery: 4 }), 92);
});

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('subgroups') ? ['subgroups'] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = []; device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); errors.push((e as GPUUncapturedErrorEvent).error.message); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); } finally { device.destroy(); }
}

test('GPU separating-axis rejection preserves crossing paths and agrees with independent swept distances', { skip: !gpu }, async () => withDevice(async device => {
  // Inputs are rounded to f32 before the Float64 oracle sees them, matching the GPU's actual geometry.
  const count = 2052, data = new Float32Array(count * 32), sides = new Uint32Array(count);
  let seed = 789123;
  const random = () => { seed = (Math.imul(seed,1664525)+1013904223) >>> 0; return seed/2**32; };
  for (let i=0;i<count;i++) {
    sides[i] = i%2 ? 2 : 1;
    const offset = i%7 === 0 ? 1000 : 0, common = i%5 === 0 ? 10 : 0;
    for (let k=0;k<4;k++) {
      const b=32*i+8*k;
      for(let j=0;j<3;j++) { data[b+j]=offset+0.2*(random()-0.5); data[b+4+j]=common+0.04*(random()-0.5); }
      data[b+3]=0.001+0.009*random(); data[b+7]=0.001*random();
    }
  }
  const put=(i:number,x:V3[],w:V3[],nA:number)=>{
    sides[i]=nA;
    for(let k=0;k<4;k++) {const b=32*i+8*k;data.set(x[k]!,b);data[b+3]=0.001;data.set(w[k]!,b+4);data[b+7]=0;}
  };
  // Endpoints are separated, but the midpoint intersects. Common translation must not hide it.
  put(0,[[0,0,0.02],[-0.1,-0.1,0],[0.1,-0.1,0],[0,0.1,0]],[[10,0,-0.04],[10,0,0],[10,0,0],[10,0,0]],1);
  put(1,[[-0.1,0,0.02],[0.1,0,0.02],[0,-0.1,0],[0,0.1,0]],[[10,0,-0.04],[10,0,-0.04],[10,0,0],[10,0,0]],2);
  put(2,[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],1);
  put(3,[[-0.1,0,0],[0.1,0,0],[-0.1,0,0.00099],[0.1,0,0.00099]],[[0,0,0],[0,0,0],[0,0,0],[0,0,0]],2);
  const make=(size:number,usage:number)=>device.createBuffer({size,usage});
  const input=make(data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  const side=make(sides.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
  const output=make(4*count,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
  const read=make(4*count,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
  try {
    device.queue.writeBuffer(input,0,data);device.queue.writeBuffer(side,0,sides);
    const code=`
@group(0) @binding(0) var<storage,read> data:array<vec4f>;
@group(0) @binding(1) var<storage,read> sides:array<u32>;
@group(0) @binding(2) var<storage,read_write> result:array<u32>;
struct Params { r:f32, gammaR:f32, sepEps:f32 }
const P=Params(0.001,0.9,0.000001);
fn xd(i:u32)->vec3f {return data[2u*i].xyz;}
fn predOf(i:u32)->vec3f {return data[2u*i+1u].xyz;}
fn envOf(i:u32)->f32 {return data[2u*i].w;}
fn pathStray(i:u32)->f32 {return data[2u*i+1u].w;}
fn datCaps(a:f32,b:f32)->f32 {return 0.5*P.gammaR*(a+b)+P.sepEps;}
fn datRound(s:f32)->f32 {return 1.52587890625e-5*s;}
fn absMax(x:vec3f)->f32 {return max(abs(x.x),max(abs(x.y),abs(x.z)));}
${CLOTH_SEPARATION_CULL_WGSL}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
  if(id.x>=${count}u){return;} let b=4u*id.x;
  result[id.x]=u32(separatedClothPair(vec4u(b,b+1u,b+2u,b+3u),sides[id.x]));
}`;
    const module=device.createShaderModule({code}),pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
    const bindings=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,side,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bindings);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();encoder.copyBufferToBuffer(output,0,read,0,4*count);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const result=new Uint32Array(read.getMappedRange()).slice();read.unmap();
    assert.deepEqual(Array.from(result.subarray(0,4)),[0,0,0,0],'crossing, degenerate and active pairs must survive');
    let rejected=0;
    for(let i=4;i<count;i++) if(result[i]) {
      rejected++; const nA=sides[i]!;
      const caps=[0,0],strays=[0,0];
      for(let k=0;k<4;k++){const group=k<nA?0:1;caps[group]=Math.max(caps[group]!,data[32*i+8*k+3]!);strays[group]=Math.max(strays[group]!,data[32*i+8*k+7]!);}
      const bound=0.45*(caps[0]!+caps[1]!)+strays[0]!+strays[1]!+1e-6;
      for(let step=0;step<=64;step++) {
        const x=Array.from({length:4},(_,k)=>Array.from({length:3},(_,j)=>data[32*i+8*k+j]!+step/64*data[32*i+8*k+4+j]!) as V3);
        let distance:number;
        if(nA===2) distance=closestSegmentSegment(x[0]!,x[1]!,x[2]!,x[3]!).d;
        else { const {bary}=closestPointTriangle(x[0]!,x[1]!,x[2]!,x[3]!);distance=Math.hypot(...x[0]!.map((v,j)=>v-bary.reduce((s,w,k)=>s+w*x[k+1]![j]!,0))); }
        assert.ok(distance>bound,`culled pair ${i} at ${step}/64: distance ${distance}, required ${bound}`);
        if(step===0)assert.ok(distance>0.001,'no omitted active contact');
      }
    }
    assert.ok(rejected>100,`exercise real rejections: ${rejected}`);
    console.log({separationCases:count,rejected});
  } finally {input.destroy();side.destroy();output.destroy();read.destroy();}
}));

test('tiny Bridson rotating sphere: cached candidates and overflow fallback preserve the trajectory', { skip: !gpu }, async () => withDevice(async device => {
  const run = async (facesPerQuery?: number) => {
    const scene = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 }));
    scene.cloth = scene.cloth!.map(d => ({ ...d, solver: { ...d.solver!, contactProjectionMetric: 'hvv' as const } }));
    const system = await ClothSystem.create(device, scene, { statusInterval: 1, contactBroadphase: facesPerQuery === 0 ? 'bvh' : 'grid', ...(facesPerQuery ? { contactCandidateCache: { skin_m: 0.005, facesPerQuery } } : {}) });
    const det = system.contactFor(scene.cloth![0]!.id)!;
    const track: number[][] = []; let hits = 0, builds = 0, overflows = 0, contacts = 0;
    try {
      for (let tick = 1; tick <= 24; tick++) {
        const [pose] = (await system.capture([tick * CLOTH_SCENE_STEP_S]))[0]!;
        assert.equal(system.failure, undefined, JSON.stringify(system.failure));
        track.push(Array.from(pose!.positions_m));
        const c = await det.readCounters(); contacts += c.bodyContacts;
        const h = await det.readCandidateCacheCounters();
        if (h) { builds += h.builds; hits += h.hits; overflows += h.overflows; }
      }
      console.log({ facesPerQuery, hits, builds, overflows, contacts });
      return { track, hits, builds, overflows, contacts };
    } finally { system.destroy(); }
  };
  const baseline = await run();
  for (const cap of [64, 1, 0]) {
      const cached = await run(cap);
      if (cap) assert.ok(cached.hits > 0 && cached.builds > 0, 'actually reuses candidates');
      if (cap === 1) assert.ok(cached.overflows > 0, 'exercises full-search fallback');
      assert.ok(cached.contacts > 0, 'touches moving sphere');
      let maxError = 0;
      for (let t = 0; t < baseline.track.length; t++) for (let i = 0; i < baseline.track[t]!.length; i++) maxError = Math.max(maxError, Math.abs(cached.track[t]![i]! - baseline.track[t]![i]!));
      assert.ok(maxError < 1e-5, `maximum trajectory difference ${maxError} m`);
    }
}));

test('conflict witnesses preserve tiny sphere trajectories and safety diagnostics', { skip: !gpu }, async () => withDevice(async device => {
  const run = async (contactConflictWitnesses: boolean, contactConflictWitnessSolids=false) => {
    const scene=bridson2002SceneDescription(bridson2002RotatingSphere({nodesAcross:16}));
    scene.cloth=scene.cloth!.map(d=>({...d,solver:{...d.solver,contactProjectionMetric:'hvv' as const}}));
    const system=await ClothSystem.create(device,scene,{liveOptimizations:true,contactConflictWitnesses,contactConflictWitnessSolids,statusInterval:1});
    const track=[];let skips=0,tests=0;
    try {
      for(let tick=1;tick<=24;tick++) {
        const pose=(await system.capture([tick*CLOTH_SCENE_STEP_S]))[0]![0]!;
        assert.equal(system.failure,undefined,JSON.stringify(system.failure));
        const detector=system.contactFor(scene.cloth![0]!.id)!;
        const c=await detector.readCounters(),diagnosis=await detector.readDiagnosis();
        skips+=c.witnessSkips;tests+=c.witnessTests;
        track.push({positions:Array.from(pose.positions_m),velocities:Array.from(pose.velocities_m_s!),held:Array.from(diagnosis.held),
          safety:Object.fromEntries(['status','held','zonePeak','conflicts','redos','levelFallback','intersections','solidIntersections','missingPairs','lagPairs','uncertified','uncovered'].map(k=>{const value=c[k as keyof typeof c];assert.equal(typeof value,'number',k);assert.ok(Number.isFinite(value),k);return [k,value];}))});
      }
      return {track,skips,tests};
    } finally {system.destroy();}
  };
  const baseline=await run(false),witness=await run(true);
  assert.deepEqual(witness.track,baseline.track);
  const solid=await run(true,true);assert.deepEqual(solid.track,baseline.track);
  assert.ok(solid.skips>0 && solid.tests>0,'solid witnesses actually skip doomed rounds');
  console.log({witnessSkips:witness.skips,witnessTests:witness.tests});
}));

test('live optimization opt-out and alternate solvers preserve the current pose and BDF2 history', { skip: !gpu }, async () => withDevice(async device => {
  let scene=bridson2002SceneDescription(bridson2002RotatingSphere({nodesAcross:16}));
  const system=await ClothSystem.create(device,scene,{liveOptimizations:true});
  try {
    const time=2*CLOTH_SCENE_STEP_S;
    const before=(await system.capture([time]))[0]![0]!;
    for(const patch of [{optimizations:false},{optimizations:true,contactBroadphase:'cached-grid' as const},{optimizations:true,contactBroadphase:'bvh' as const,multigrid:true},{multigrid:false,contactEdgeEvaluation:'unique' as const},{contactEdgeEvaluation:'pruned' as const,contactBvhLeafSize:2 as const},{contactEdgeEvaluation:'staged' as const},{contactEdgeEvaluation:'packet' as const},{contactVertexEvaluation:'direct' as const,lazyLineSearch:false},{contactVertexEvaluation:'packet' as const,lazyLineSearch:true,contactConflictReuse:'all' as const},{contactConflictReuse:'off' as const}]) {
      scene={...scene,cloth:scene.cloth!.map(d=>({...d,solver:{...d.solver,...patch}}))};
      system.adopt(scene);await system.settled();
      assert.equal(system.failure,undefined,JSON.stringify(system.failure));
      const after=(await system.capture([time]))[0]![0]!;
      assert.deepEqual(after.positions_m,before.positions_m);assert.deepEqual(after.velocities_m_s,before.velocities_m_s);
      assert.deepEqual({...after.solverState,time_s:0},{...before.solverState,time_s:0});
      assert.ok(Math.abs(system.time_s-time)<1e-8,'clock survives its f32 upload without a step reset');
    }
    // A second click can arrive while the first replacement still compiles.
    for(const optimizations of [false,true]) {
      scene={...scene,cloth:scene.cloth!.map(d=>({...d,solver:{...d.solver,optimizations}}))};system.adopt(scene);
    }
    await system.settled();assert.equal(system.failure,undefined,JSON.stringify(system.failure));
    const after=(await system.capture([time]))[0]![0]!;
    assert.deepEqual(after.positions_m,before.positions_m);assert.deepEqual({...after.solverState,time_s:0},{...before.solverState,time_s:0});assert.ok(Math.abs(system.time_s-time)<1e-8);
  } finally {system.destroy();}
}));

// Deliberately drive detection rounds without a solver: move bounds within their allowance, then
// outside it, shrink/re-expand a face, and start a new timestep. Compare the actual contact and DAT
// records to a full search after every round, not just the final positions of a trajectory.
test('GPU candidate coverage: motion, invalidation, shrink/expand and timestep reset match full detection', { skip: !gpu }, async () => withDevice(async device => {
  const mesh = merge(sheet(2, 2, 0.01), sheet(2, 2, 0.01, [0.003, 0.002, 0.0015]), sheet(2, 2, 0.01, [0.003, 0.002, 0.0077]), sheet(2, 2, 0.01, [1, 0, 0]));
  const pack = (x: typeof mesh.x) => new Float32Array(x.flatMap(p => [...p, 0]));
  const triangles = new Uint32Array(mesh.tris.flatMap(t => [...t, 0]));
  const topology = { vertexCount: mesh.x.length, triangles, edges: buildOgcEdges(triangles) };
  const storage = (a: Float32Array<ArrayBuffer>) => { const b = device.createBuffer({ size: a.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, a); return b; };
  const run = async (cache: boolean, separationCulling = false, moving = false, broadphase: 'grid' | 'bvh' = 'grid', bvhLeafSize = 4, uniqueEdgePairs = false, bvhEdgePruning = false, edgeStageCapacity?: number, dynamicEnvelope = false, truncation: 'planar' | 'isotropic' = 'planar', edgeStageBlockSize: 1|4|16 = 1, edgeStagePacket = false, vertexStage = false, vertexStagePacket = false) => {
    const speed = new Float32Array(4 * mesh.x.length);
    if (moving) for (let v=0;v<mesh.x.length;v++) { speed[4*v]=1; speed[4*v+2]=v<8 ? 0.2 : -0.2; }
    if(dynamicEnvelope) for(let v=0;v<mesh.x.length;v++) {speed[4*v+1]=0.02*(v%5);speed[4*v+2]+=0.01*(v%3);}
    const positions = storage(pack(mesh.x)), velocities = storage(speed);
    const det = await OgcContactDetector.create(device, topology, pack(mesh.x), {
      r: 0.002, kc: 1e5, mu: 0.3, epsV: 0.01, dt: 1/60, gravity: [0,0,0], rqBase: 0.003, rqMax: 0.02,
      envelopeScale: dynamicEnvelope ? 1 : 0, truncation, contactCapacity: 96, pairCapacity: 8192, gridCapacity: 8192, bucketCount: 4096, roundCap: 64, prediction: moving ? 'inertial' : 'none', separationCulling, broadphase, bvhLeafSize, uniqueEdgePairs, bvhEdgePruning, edgeStageCapacity, edgeStageBlockSize, edgeStagePacket, vertexStage, vertexStagePacket,
      ...(cache ? { candidateCache: { skin_m: 0.001, facesPerQuery: 32 } } : {}),
    }, { positions, velocities });
    const snapshots = []; const headers = []; const queues:string[][]=[];
    const pass = (fn: (p: GPUComputePassEncoder) => void) => { const e = device.createCommandEncoder(), p = e.beginComputePass(); fn(p); p.end(); device.queue.submit([e.finish()]); };
    try {
      pass(p => { det.encodeFrameBegin(p); det.encodeStepBegin(p); });
      const motions = [0, 0.0002, -0.0002, 0.006, 0.0062, 0.0062, 0, 0];
      for (let round = 0; round < motions.length; round++) {
        const x = mesh.x.map((p, i) => [p[0] + (i >= mesh.x.length/4 && i < mesh.x.length/2 ? motions[round]! : 0), p[1], p[2] - (i >= mesh.x.length/2 && i < 3*mesh.x.length/4 && round > 0 ? 0.0006 : 0) ] as [number,number,number]);
        if (round === 5) for (const p of x) p[0] *= 0.25;
        device.queue.writeBuffer(positions, 0, pack(x));
        // Re-arm one raw round, retaining the cache (no samples, four bucket-scan blocks).
        const n = Math.ceil(det.N/64), t = Math.ceil(det.T/64), q = Math.ceil((det.N+det.E)/64);
        const args = new Uint32Array([n,1,t,4,1,1,t,q,n].flatMap(w => [w,1,1]));
        device.queue.writeBuffer(det.args, 0, args);
        if (round === 7) pass(p => det.encodeStepBegin(p));
        pass(p => { for (const name of det.roundKernels) det.encodeRoundKernel(p, name); });
        const contacts = await det.readContactEntries();
        const lists = Array.from(contacts.counts, (n,v) => Array.from({ length:n }, (_,k) => Array.from(contacts.entries.subarray(4*(v*contacts.header[1]!+k),4*(v*contacts.header[1]!+k+1))).join(',')).sort());
        const pairs = await det.readPairs(); const bits = new Uint32Array(pairs.planes.buffer);
        const planes = Array.from({length:pairs.slots}, (_,i) => i).filter(i=>pairs.ids[4*i]!==0xffffffff).map(i=>[...pairs.ids.subarray(4*i,4*i+4),...bits.subarray(4*i,4*i+4)].join(',')).sort();
        const diag = await det.readDiagnosis();
        snapshots.push({ lists, planes, env: Array.from(diag.env), xdet: Array.from(diag.xdet) });
        headers.push(await det.readCandidateCacheCounters());
        const staged=await det.readEdgeStageCandidates();
        if(staged && !staged.overflow)queues.push(Array.from({length:staged.entries.length/2},(_,i)=>i).filter(i=>staged.entries[2*i]!==0xffffffff).map(i=>`${staged.entries[2*i]},${staged.entries[2*i+1]}`).sort());
      }
      if(edgeStageCapacity){
        const before=await det.readContactEntries(),beforePairs=await det.readPairs();
        device.queue.writeBuffer(det.args,0,new Uint32Array(27));
        pass(p=>{for(const name of det.roundKernels)det.encodeRoundKernel(p,name);});
        assert.deepEqual(await det.readContactEntries(),before,'inactive round cannot replay stale staged contacts');
        assert.deepEqual(await det.readPairs(),beforePairs,'inactive round cannot replay stale staged DAT pairs');
        const counters=await det.readCounters();
        assert.equal(counters.edgeStageOverflows>0,edgeStageCapacity===1,JSON.stringify({message:'tiny capacity exercises full overflow fallback',edgeStageCapacity,vertexStage,vertexStagePacket,moving,counters}));
      }
      return { snapshots, headers, queues };
    } finally { det.destroy(); positions.destroy(); velocities.destroy(); }
  };
  const full = await run(false), cached = await run(true);
  assert.deepEqual(cached.snapshots, full.snapshots);
  assert.ok(cached.snapshots[0]!.planes.length > 0, 'nonempty DAT comparison');
  assert.ok(cached.snapshots[0]!.lists.some(l => l.length), 'nonempty contact comparison');
  assert.ok(cached.headers[2]!.hits > 0, 'small motion reuses lists');
  assert.ok(cached.headers[3]!.resets > cached.headers[2]!.resets, 'larger motion expires face bounds');
  assert.ok(cached.headers[3]!.hits > cached.headers[2]!.hits, 'distant queries survive local invalidation');
  assert.ok(cached.headers[5]!.resets > cached.headers[4]!.resets, 'shrinkage expires face bounds');
  assert.ok(cached.headers[7]!.builds > 0 && cached.headers[7]!.hits === 0, 'new timestep cannot reuse old epoch');
  for (const moving of [false,true]) {
    const reference = moving ? await run(false,false,true) : full;
    for (const leafSize of [2,4,8,16]) {
      const hierarchical = await run(false,false,moving,'bvh',leafSize);
      assert.deepEqual(hierarchical.snapshots,reference.snapshots,`BVH leaf size ${leafSize} preserves all contacts and DAT planes`);
    }
    const unique = await run(false,false,moving,'bvh',4,true);
    assert.deepEqual(unique.snapshots,reference.snapshots,'single edge evaluation preserves both directed contacts and DAT planes');
    const pruned = await run(false,false,moving,'bvh',4,true,true);
    assert.deepEqual(pruned.snapshots,reference.snapshots,'edge ownership subtree pruning preserves contacts and DAT planes');
    for(const [capacity,block] of [[1,1],[8192,1],[1,4],[8192,4],[8192,16]] as const) {
      const staged=await run(false,false,moving,'bvh',4,true,true,capacity,false,'planar',block);
      assert.deepEqual(staged.snapshots,reference.snapshots,`edge staging capacity ${capacity} preserves contacts and real DAT planes`);
    }
    for(const cap of [1,8192]) {
      const vertices=await run(false,false,moving,'bvh',4,true,true,cap,false,'planar',4,false,true);
      assert.deepEqual(vertices.snapshots,reference.snapshots,`vertex staging ${cap} preserves contacts and DAT planes`);
      if(device.features.has('subgroups')){
        const packet=await run(false,false,moving,'bvh',4,true,true,cap,false,'planar',4,false,true,true);
        assert.deepEqual(packet.snapshots,reference.snapshots,`vertex packets ${cap} preserve contacts and DAT planes`);
      }
    }
    if(device.features.has('subgroups')){
      const scalar=await run(false,false,moving,'bvh',4,true,true,8192,false,'planar',4);
      const packet=await run(false,false,moving,'bvh',4,true,true,8192,false,'planar',4,true);
      assert.deepEqual(packet.queues,scalar.queues,'packet traversal preserves the exact candidate multiset');
      assert.deepEqual(packet.snapshots,reference.snapshots,'packet traversal preserves contacts and real DAT planes');
    }
    const culled = await run(false,true,moving);
    for(let i=0;i<reference.snapshots.length;i++) {
      const a=reference.snapshots[i]!,b=culled.snapshots[i]!;
      assert.deepEqual(b.lists,a.lists,'separation rejection preserves every active contact');
      assert.deepEqual(b.env,a.env); assert.deepEqual(b.xdet,a.xdet);
      const planes=new Set(a.planes);
      for(const plane of b.planes) assert.ok(planes.has(plane),'surviving DAT planes are unchanged');
    }
  }
  // Reverse face numbering and winding, then use unequal predicted displacements/envelopes.
  const records=Array.from({length:triangles.length/4},(_,t)=>Array.from(triangles.subarray(4*t,4*t+4))).reverse();
  triangles.set(records.flatMap(r=>[r[2]!,r[1]!,r[0]!,0]));
  topology.edges=buildOgcEdges(triangles);
  const asymmetric=await run(false,false,true,'grid',4,false,false,undefined,true);
  for(const [broadphase,cached] of [['grid',false],['grid',true],['bvh',false]] as const){
    const result=await run(cached,false,true,broadphase,4,true,broadphase==='bvh',undefined,true);
    assert.deepEqual(result.snapshots,asymmetric.snapshots,'ownership remains exact with reordered topology and unequal envelopes');
  }
  const stagedAsymmetric=await run(false,false,true,'bvh',4,true,true,8192,true);
  assert.deepEqual(stagedAsymmetric.snapshots,asymmetric.snapshots,'staging remains exact with unequal envelopes');
  const isotropic=await run(false,false,true,'bvh',4,false,false,undefined,true,'isotropic');
  const isotropicUnique=await run(false,false,true,'bvh',4,true,true,undefined,true,'isotropic');
  assert.deepEqual(isotropicUnique.snapshots,isotropic.snapshots,'isotropic queries retain both directed searches');

  if(device.features.has('subgroups')){
    // N=63 puts a valid edge in lane31 with an invalid leader, and leaves a whole
    // invalid subgroup in each boundary workgroup. Unused vertices stay far away.
    while(mesh.x.length<63)mesh.x.push([10+mesh.x.length,0,0]);
    topology.vertexCount=mesh.x.length;
    const scalar=await run(false,false,true,'bvh',4,true,true,8192,false,'planar',4);
    const packet=await run(false,false,true,'bvh',4,true,true,8192,false,'planar',4,true);
    assert.deepEqual(packet.queues,scalar.queues,'partial packets preserve every candidate even with invalid lane0');
    assert.deepEqual(packet.snapshots,scalar.snapshots,'invalid subgroups safely reconverge');
    const vertexPacket=await run(false,false,true,'bvh',4,true,true,8192,false,'planar',4,true,true,true);
    assert.deepEqual(vertexPacket.snapshots,scalar.snapshots,'partial vertex packets preserve coverage and safely reconverge');
    const overflow=await run(false,false,true,'bvh',4,true,true,1,false,'planar',4,true);
    assert.deepEqual(overflow.snapshots,scalar.snapshots,'packet overflow uses a complete scalar fallback');
  }

}));
