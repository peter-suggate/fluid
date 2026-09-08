import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { readFile } from "node:fs/promises";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12CurrentMapLayout, createSparseCM12CurrentMapWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-current-map.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();
const near = (a: number, b: number, tolerance = 3e-5) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);

(dawnModule ? test : test.skip)("production C2 current map interpolation, composition, and global orientation admission",
  { timeout: 120_000 }, async t => {
    await acquireWebGPUExclusiveLock("dawn-test", "current-map-carrier");
    let gpu: GPU | undefined, device: GPUDevice | undefined;
    const buffers: GPUBuffer[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const layout = createSparseCM12CurrentMapLayout(16, [4, 4, 4], 16, .5, 4);
      const points = [[.5, .7, 1.1], [1.5, 2, 2.5], [2.99999, 1.1, .7], [3, 1.1, .7], [3.00001, 1.1, .7], [9.125, 9.625, 9.375]];
      const module = device.createShaderModule({ label: "Production C2 current-map oracle", code: /* wgsl */ `
struct Parameters{frame:vec4f,velocity:vec4f,linear:vec4f}
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<uniform> p:Parameters;
@group(0) @binding(2) var<storage,read_write> failures:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> queries:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4f>;
fn cm12RetainedDensityAcceptedBank()->u32{return u32(state[0]);}
fn cm12CurrentMapNativeVelocity(point:vec3f)->vec4f{
  let relative=point-vec3f(2.0);
  return vec4f(p.velocity.xyz+p.linear.xyz*select(relative,relative.yzx,p.linear.w>0.5),1.0);
}
fn cm12CurrentMapVelocityBoundary(point:vec3f,velocity:vec3f)->vec3f{
  // A compact affine patch tests nonuniform flow without forcing an abrupt
  // nonzero velocity against the exact identity coefficient collar.
  let distance=abs(point-vec3f(2.0));
  return velocity*(1.0-smoothstep(12.0,16.0,max(distance.x,max(distance.y,distance.z))));
}
fn cm12CurrentMapInitializeVelocity(id:u32,sample:vec4f){cm12CurrentMapWrite(CM12_CURRENT_MAP_VELOCITY_BASE,id,sample.xyz);}
fn cm12CurrentMapCoefficientBoundaryPoint(point:vec3f)->vec3f{
  if(p.frame.z==0.0){return point;}
  var result=point;
  for(var axis=0u;axis<3u;axis++){
    if(point[axis]==-CM12_CURRENT_MAP_SPACING){result[axis]=-point[axis];}
    if(axis!=1u&&point[axis]==4.0+CM12_CURRENT_MAP_SPACING){result[axis]=8.0-point[axis];}
  }
  return result;
}
fn cm12CurrentMapCoefficientBoundaryValue(point:vec3f,value:vec3f)->vec3f{
  if(p.frame.z==0.0){return value;}
  var result=value;
  for(var axis=0u;axis<3u;axis++){
    if(point[axis]==-CM12_CURRENT_MAP_SPACING||(axis!=1u&&point[axis]==4.0+CM12_CURRENT_MAP_SPACING)){result[axis]=-result[axis];}
    if(point[axis]==0.0||(axis!=1u&&point[axis]==4.0)){result[axis]=0.0;}
  }
  return result;
}
fn cm12CurrentMapFail(code:u32,id:u32){atomicOr(&failures[0],1u<<code);atomicMin(&failures[1],id);}
fn cm12CurrentMapFailed()->bool{return atomicLoad(&failures[0])!=0u;}
${createSparseCM12CurrentMapWGSL(layout)}
@compute @workgroup_size(64)
fn testCommit(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x!=0u||cm12CurrentMapFailed()){return;}
  cm12CurrentMapCommitIncrement();state[0]=f32(1u-cm12RetainedDensityAcceptedBank());
}
@compute @workgroup_size(64)
fn queryCertificate(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x!=0u){return;}
  let cell=vec3i((vec3f(2.0)-CM12_CURRENT_MAP_ORIGIN)*CM12_CURRENT_MAP_INVERSE_SPACING);
  var magnitude=1.0;
  for(var z=0;z<4;z++){for(var y=0;y<4;y++){for(var x=0;x<4;x++){
    let value=abs(cm12CurrentMapControl(cell+vec3i(x-1,y-1,z-1),cm12CurrentMapCandidateBank()));
    magnitude=max(magnitude,max(value.x,max(value.y,value.z)));
  }}}
  let values=cm12CurrentMapBernstein(cell,cm12CurrentMapCandidateBank());
  output[0]=vec4f(select(0.0,1.0,cm12CurrentMapPatchCertified(values,magnitude)),
    select(0.0,1.0,cm12CurrentMapPositiveJacobian(values,CM12_CURRENT_MAP_INVERSE_SPACING,magnitude)),0.0,0.0);
}
@compute @workgroup_size(64)
fn queryDeparture(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=arrayLength(&queries)){return;}
  output[4u*gid.x]=vec4f(cm12CurrentMapDeparture(queries[gid.x].xyz),0.0);
}
@compute @workgroup_size(64)
fn queryMap(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=arrayLength(&queries)){return;}
  let value=cm12CurrentMapEvaluate(queries[gid.x].xyz,cm12CurrentMapCandidateBank());
  output[4u*gid.x]=vec4f(value.point,determinant(value.jacobian));
  for(var axis=0u;axis<3u;axis++){output[4u*gid.x+axis+1u]=vec4f(value.jacobian[axis],0.0);}
  let q=vec3i(floor(queries[gid.x].xyz));
  let range=cm12CurrentMapRangeOnFineSupport(q,cm12CurrentMapCandidateBank());
  output[4u*gid.x+1u].w=range[0].x;output[4u*gid.x+2u].w=range[1].x;
  output[4u*gid.x+3u].w=cm12CurrentMapBoundaryNormalBound(q,0u,0u,cm12CurrentMapCandidateBank());
}` });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(message => message.type === "error").map(message =>
        `${message.lineNum}:${message.linePos} ${message.message}`), []);
      const bindingLayout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindingLayout] });
      const entries = ["compileCurrentMapVelocity", "compileCurrentMapTraceSchedule", "sealCurrentMapTraceSchedule",
        "advanceCurrentMapNodes", "filterCurrentMapX", "filterCurrentMapY", "filterCurrentMapZ", "certifyCurrentMap",
        "compileCurrentMapPhysicalBoundary", "boundCurrentMapPhysicalBoundary", "queryMap", "queryDeparture", "queryCertificate", "publishCurrentMapIncrement", "testCommit"];
      const pipelines = new Map<string, GPUComputePipeline>();
      for (const entryPoint of entries) pipelines.set(entryPoint,
        await device.createComputePipelineAsync({ label: entryPoint, layout: pipelineLayout, compute: { module, entryPoint } }));
      const allocate = (size: number, usage: GPUBufferUsageFlags) => {
        const buffer = device!.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
      };
      const state = allocate(4 * layout.endWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const parameters = allocate(48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const failures = allocate(8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const queries = allocate(16 * points.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const output = allocate(64 * points.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const readback = allocate(output.size + 40, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(queries, 0, new Float32Array(points.flatMap(point => [...point, 0])));
      const bindings = device.createBindGroup({ layout: bindingLayout,
        entries: [state, parameters, failures, queries, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const run = async (advance: boolean, filterOnly = false, traceOnly = false, certificateOnly = false) => {
        device!.queue.writeBuffer(failures, 0, new Uint32Array([0, 0xffffffff]));
        const encoder = device!.createCommandEncoder();
        const dispatch = (entry: string, count: number) => {
          const pass = encoder.beginComputePass(); pass.setPipeline(pipelines.get(entry)!); pass.setBindGroup(0, bindings);
          pass.dispatchWorkgroups(Math.ceil(count / (["boundCurrentMapPhysicalBoundary", "sealCurrentMapTraceSchedule"].includes(entry) ? 1 : 64))); pass.end();
        };
        if (advance || traceOnly) {
          dispatch("compileCurrentMapVelocity", layout.nodeCount);
          dispatch("compileCurrentMapTraceSchedule", layout.lineCounts[0]);
          dispatch("sealCurrentMapTraceSchedule", 1);
        }
        if (advance) dispatch("advanceCurrentMapNodes", layout.nodeCount);
        if (advance || filterOnly) {
          for (let axis = 0; axis < 3; axis++) dispatch(["filterCurrentMapX", "filterCurrentMapY", "filterCurrentMapZ"][axis]!, layout.lineCounts[axis]!);
        }
        if (certificateOnly) dispatch("queryCertificate", 1);
        else if (traceOnly) dispatch("queryDeparture", points.length);
        else {
          dispatch("certifyCurrentMap", layout.cellCount); dispatch("compileCurrentMapPhysicalBoundary", layout.boundarySampleCount);
          dispatch("boundCurrentMapPhysicalBoundary", 6); dispatch("queryMap", points.length);
        }
        if (advance) { dispatch("publishCurrentMapIncrement", layout.nodeCount); dispatch("testCommit", 1); }
        encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
        encoder.copyBufferToBuffer(failures, 0, readback, output.size, 8);
        encoder.copyBufferToBuffer(state, 4 * layout.boundaryBoundsBaseWords, readback, output.size + 8, 24);
        encoder.copyBufferToBuffer(state, 4 * layout.traceSubstepCountBaseWords, readback, output.size + 32, 4);
        encoder.copyBufferToBuffer(state, 4 * layout.chainCountBaseWords, readback, output.size + 36, 4);
        device!.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const mapped = readback.getMappedRange();
        const values = new Float32Array(mapped, 0, output.size / 4).slice();
        const receipt = new Uint32Array(mapped, output.size, 2).slice();
        const boundary = new Float32Array(mapped, output.size + 8, 6).slice();
        const substepCount = new Float32Array(mapped, output.size + 32, 1)[0]!;
        const chainCount = new Float32Array(mapped, output.size + 36, 1)[0]!; readback.unmap();
        return { values, receipt, boundary, substepCount, chainCount };
      };
      await t.test("uniform and nonuniform saddle flow compose the simulated inverse map", async () => {
        for (const linear of [[0, 0, 0], [.15, -.1, .05]]) {
          const dt = .125, velocity = [-.75, .5, .25];
          device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
          device!.queue.writeBuffer(parameters, 0, new Float32Array([dt, 1, 0, 0, ...velocity, 0, ...linear, 0]));
          const diagonal = linear.map(a => 1 - dt * a + .5 * dt * dt * a * a);
          const shift = velocity.map((b, axis) => -dt * b + .5 * dt * dt * linear[axis]! * b);
          for (let step = 1; step <= 2; step++) {
            device!.queue.writeBuffer(state, 0, new Float32Array([(step - 1) % 2]));
            const { values, receipt, boundary, chainCount } = await run(true);
            assert.equal(receipt[0], 0, `linear=${linear} step=${step} fault=${receipt}`);
            assert.equal(chainCount, step, "one certified increment is archived by each successful commit");
            points.forEach((point, i) => {
              for (let axis = 0; axis < 3; axis++) {
                const expected = 2 + diagonal[axis]! ** step * (Math.fround(point[axis]!) - 2)
                  + shift[axis]! * (step === 1 ? 1 : 1 + diagonal[axis]!);
                near(values[16 * i + axis]!, expected);
                for (let row = 0; row < 3; row++) near(values[16 * i + 4 * (axis + 1) + row]!, row === axis ? diagonal[axis]! ** step : 0);
              }
              near(values[16 * i + 3]!, diagonal.reduce((product, value) => product * value ** step, 1));
              assert.ok(values[16 * i + 7]! <= values[16 * i]! && values[16 * i + 11]! >= values[16 * i]!, "mapped support range encloses query");
              const boundaryDisplacement = 2 - 2 * diagonal[0]! ** step + shift[0]! * (step === 1 ? 1 : 1 + diagonal[0]!);
              assert.ok(values[16 * i + 15]! >= Math.abs(boundaryDisplacement) - 3e-6, "face bound encloses normal displacement");
            });
            for (let axis = 0; axis < 3; axis++) for (let side = 0; side < 2; side++) {
              const x = side * layout.dimensions[axis];
              const displacement = 2 + diagonal[axis]! ** step * (x - 2)
                + shift[axis]! * (step === 1 ? 1 : 1 + diagonal[axis]!) - x;
              const inward = Math.max(0, side === 0 ? displacement : -displacement);
              assert.ok(boundary[2 * axis + side]! >= inward - 3e-6);
              // The generic open-boundary interval carries a conservative
              // arithmetic margin through each composed increment.
              near(boundary[2 * axis + side]!, inward, 1e-4);
            }
          }
        }
      });
      await t.test("deeper interval subdivision certifies an actual positive production patch", async () => {
        const fixture = JSON.parse(await readFile(new URL("./fixtures/sparse-cm12-current-map-positive-patch.json", import.meta.url), "utf8")) as
          { spacingFine: number; controls: number[][] };
        assert.equal(layout.spacingFine, fixture.spacingFine);
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
        const center = (2 + layout.padding) / layout.spacingFine;
        for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) {
          const id = center - 1 + layout.nodeDimensions[0] * (center + y - 1 + layout.nodeDimensions[1] * (center + z - 1));
          const row = fixture.controls.slice(4 * (y + 4 * z), 4 * (y + 4 * z + 1)).flat();
          device!.queue.writeBuffer(state, 4 * (layout.coefficientBaseWords[1] + 3 * id), new Float32Array(row));
        }
        const result = await run(false, false, false, true);
        assert.equal(result.values[0], 1, "all subdivided boxes have a positive determinant bound");
        assert.equal(result.values[1], 0, "the unsplit interval is inconclusive");
      });
      await t.test("distinct noncommuting increments preserve composition order and the full Jacobian", async () => {
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
        const dt = .125;
        const flows = [
          { velocity: [-.75, .5, .25], linear: [.15, -.1, .05], cyclic: false },
          { velocity: [.3, -.25, .1], linear: [-.05, .08, .12], cyclic: true },
        ];
        const multiply = (a: number[][], b: number[][]) => a.map(row => b[0]!.map((_, column) =>
          row.reduce((value, entry, inner) => value + entry * b[inner]![column]!, 0)));
        const apply = (a: number[][], value: number[]) => a.map(row => row.reduce((sum, entry, column) => sum + entry * value[column]!, 0));
        const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const transforms: Array<{ matrix: number[][]; shift: number[] }> = [];
        let expectedJacobian = identity;
        for (const flow of flows) {
          device!.queue.writeBuffer(parameters, 0, new Float32Array([dt, 1, 0, 0, ...flow.velocity, 0, ...flow.linear, Number(flow.cyclic)]));
          const a = identity.map((row, axis) => row.map((_, column) => column === (flow.cyclic ? (axis + 1) % 3 : axis) ? flow.linear[axis]! : 0));
          const square = multiply(a, a);
          const matrix = identity.map((row, axis) => row.map((entry, column) => entry - dt * a[axis]![column]! + .5 * dt * dt * square[axis]![column]!));
          const ab = apply(a, flow.velocity);
          const shift = flow.velocity.map((value, axis) => -dt * value + .5 * dt * dt * ab[axis]!);
          transforms.push({ matrix, shift }); expectedJacobian = multiply(expectedJacobian, matrix);
          const result = await run(true); assert.equal(result.receipt[0], 0);
          points.forEach((point, i) => {
            let expected = point.map(value => Math.fround(value) - 2);
            for (const transform of [...transforms].reverse()) expected = apply(transform.matrix, expected).map((value, axis) => value + transform.shift[axis]!);
            for (let axis = 0; axis < 3; axis++) {
              near(result.values[16 * i + axis]!, expected[axis]! + 2);
              for (let row = 0; row < 3; row++) near(result.values[16 * i + 4 * (axis + 1) + row]!, expectedJacobian[row]![axis]!);
            }
          });
        }
      });
      await t.test("a shared RK2 schedule keeps departures continuous across local travel thresholds", async () => {
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
        const dt = .125, linear = .5, velocity = 7.5;
        device!.queue.writeBuffer(parameters, 0, new Float32Array([dt, 1, 0, 0, velocity, 0, 0, 0, linear, 0, 0, 0]));
        const { values, receipt, substepCount } = await run(false, false, true);
        assert.equal(receipt[0], 0); assert.equal(substepCount, 3);
        const substep = dt / substepCount;
        const multiplier = 1 - substep * linear + .5 * substep * substep * linear * linear;
        const shift = -substep * velocity + .5 * substep * substep * linear * velocity;
        points.forEach((point, i) => {
          let expected = Math.fround(point[0]!);
          for (let step = 0; step < substepCount; step++) expected = 2 + multiplier * (expected - 2) + shift;
          near(values[16 * i]!, expected, 4e-6);
          near(values[16 * i + 1]!, Math.fround(point[1]!), 0);
          near(values[16 * i + 2]!, Math.fround(point[2]!), 0);
        });
        // These straddle speed*dt=1. Per-point ceil previously introduced a
        // finite 1-step/2-step jump where the velocity itself is continuous.
        near(values[16 * 4]! - values[16 * 2]!,
          multiplier ** substepCount * (Math.fround(points[4]![0]!) - Math.fround(points[2]![0]!)), 2e-6);
      });
      await t.test("local quasi-interpolation reproduces a cubic map and has compact spatial dependence", async () => {
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
        const nodal = new Float32Array(3 * layout.nodeCount);
        const polynomial = ([x, y, z]: number[]) => [.002 * x! ** 3, .001 * x! * y! + .002 * y! ** 3, .01 * z! ** 2];
        const taper = (value: number) => {
          const t = Math.max(0, Math.min(1, (Math.abs(value) - 4) / 8));
          return 1 - t ** 3 * (10 - 15 * t + 6 * t * t);
        };
        for (let z = 0; z < layout.nodeDimensions[2]; z++) for (let y = 0; y < layout.nodeDimensions[1]; y++) for (let x = 0; x < layout.nodeDimensions[0]; x++) {
          const relative = [x, y, z].map((value, axis) => layout.spacingFine * value + layout.originFine[axis]! - 2);
          const weight = relative.reduce((value, coordinate) => value * taper(coordinate), 1);
          const id = x + layout.nodeDimensions[0] * (y + layout.nodeDimensions[1] * z);
          nodal.set(polynomial(relative).map(value => weight * value), 3 * id);
        }
        device!.queue.writeBuffer(state, 4 * layout.nodalBaseWords, nodal);
        const cubic = await run(false, true); assert.equal(cubic.receipt[0], 0);
        points.slice(0, 5).forEach((point, i) => {
          const relative = point.map(value => Math.fround(value) - 2), [x, y, z] = relative;
          const displacement = polynomial(relative);
          const columns = [[1 + .006 * x! * x!, .001 * y!, 0], [0, 1 + .001 * x! + .006 * y! * y!, 0], [0, 0, 1 + .02 * z!]];
          for (let axis = 0; axis < 3; axis++) {
            near(cubic.values[16 * i + axis]!, Math.fround(point[axis]!) + displacement[axis]!, 3e-6);
            for (let row = 0; row < 3; row++) near(cubic.values[16 * i + 4 * (axis + 1) + row]!, columns[axis]![row]!, 3e-6);
          }
        });
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords)); nodal.fill(0);
        const centerCoordinate = (2 + layout.padding) / layout.spacingFine;
        const center = centerCoordinate + layout.nodeDimensions[0] * (centerCoordinate + layout.nodeDimensions[1] * centerCoordinate);
        nodal[3 * center + 1] = .02; device!.queue.writeBuffer(state, 4 * layout.nodalBaseWords, nodal);
        const local = await run(false, true); assert.equal(local.receipt[0], 0);
        const far = points.length - 1;
        for (let axis = 0; axis < 3; axis++) {
          assert.equal(local.values[16 * far + axis], Math.fround(points[far]![axis]!));
          for (let row = 0; row < 3; row++) assert.equal(local.values[16 * far + 4 * (axis + 1) + row], Number(row === axis));
        }
      });
      await t.test("candidate folding and a nonzero exterior collar reject publication", async () => {
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
        const center = Array(3).fill((2 + layout.padding) / layout.spacingFine) as number[];
        const id = center[0]! + layout.nodeDimensions[0] * (center[1]! + layout.nodeDimensions[1] * center[2]!);
        device!.queue.writeBuffer(state, 4 * (layout.coefficientBaseWords[1] + 3 * id), new Float32Array([8, 0, 0]));
        assert.ok((await run(false)).receipt[0]! & (1 << 3), "local noninvertibility must fail the global certificate");
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
        device!.queue.writeBuffer(state, 4 * layout.coefficientBaseWords[1], new Float32Array([.01, 0, 0]));
        assert.ok((await run(false)).receipt[0]! & (1 << 4), "identity exterior requires the exact coefficient collar");
      });
      await t.test("physical wall coefficient parity preserves planes and tangential slip in C2", async () => {
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
        device!.queue.writeBuffer(parameters, 0, new Float32Array([.125, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        const nodal = new Float32Array(3 * layout.nodeCount);
        for (let z = 0; z < layout.nodeDimensions[2]; z++) for (let y = 0; y < layout.nodeDimensions[1]; y++) for (let x = 0; x < layout.nodeDimensions[0]; x++) {
          const distance = Math.max(...[x, y, z].map((value, axis) => Math.abs(layout.spacingFine * value + layout.originFine[axis]! - 2)));
          const t = Math.max(0, Math.min(1, (distance - 5) / 7));
          const weight = 1 - t * t * (3 - 2 * t);
          const id = x + layout.nodeDimensions[0] * (y + layout.nodeDimensions[1] * z);
          nodal.set([.002 * weight, .003 * weight, .004 * weight], 3 * id);
        }
        device!.queue.writeBuffer(state, 4 * layout.nodalBaseWords, nodal);
        const wallPoints = [[0, .7, 1.1], [4, 2, 2.5], [2.9, 0, .7], [3, 1.1, 0], [3, 1.1, 4], [0, 0, 0]];
        device!.queue.writeBuffer(queries, 0, new Float32Array(wallPoints.flatMap(point => [...point, 0])));
        const result = await run(false, true); assert.equal(result.receipt[0], 0);
        wallPoints.forEach((point, i) => {
          for (let axis = 0; axis < 3; axis++) {
            if (point[axis] !== 0 && !(axis !== 1 && point[axis] === 4)) continue;
            near(result.values[16 * i + axis]!, point[axis]!, 3e-9);
            for (let tangent = 0; tangent < 3; tangent++) if (tangent !== axis) {
              near(result.values[16 * i + 4 * (axis + 1) + tangent]!, 0, 3e-9);
              near(result.values[16 * i + 4 * (tangent + 1) + axis]!, 0, 3e-9);
            }
          }
        });
        for (const face of [0, 1, 2, 4, 5]) assert.ok(result.boundary[face]! < 1e-6,
          `wall ${face} has only the conservative floating-point residual`);
        device!.queue.writeBuffer(queries, 0, new Float32Array(points.flatMap(point => [...point, 0])));
      });
      await t.test("a full map chain rejects before publication without advancing its count", async () => {
        device!.queue.writeBuffer(state, 0, new Float32Array(layout.endWords));
        device!.queue.writeBuffer(state, 4 * layout.chainCountBaseWords, new Float32Array([layout.chainCapacity]));
        device!.queue.writeBuffer(parameters, 0, new Float32Array([.125, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        const result = await run(true);
        assert.ok(result.receipt[0]! & (1 << 5), "capacity must reject before an out-of-bounds archive write");
        assert.equal(result.chainCount, layout.chainCapacity, "failed admission leaves the accepted chain intact");
      });
      await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
    } finally {
      for (const buffer of buffers) buffer.destroy(); device?.destroy(); if (gpu) live.delete(gpu);
      await releaseWebGPUExclusiveLock();
    }
  });
