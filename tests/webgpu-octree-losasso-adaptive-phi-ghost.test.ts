import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { octreeLosassoAdaptivePhiGhostWGSL }
  from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";

test("adaptive ghost refresh reconstructs base geometry before every classification exit", () => {
  const shader = octreeLosassoAdaptivePhiGhostWGSL;
  assert.doesNotMatch(shader, /1\.\s*\/\s*f\.inverseDistance|1\.\s*\/\s*face\.inverseDistance/,
    "conditioning must not use a coefficient written by a previous scalar generation");
  assert.match(shader, /var baseDistance=max\(negativeSpan,positiveSpan\)\*p\.originCell\.w/);
  assert.match(shader, /baseDistance=\.5\*\(negativeSpan\+positiveSpan\)\*p\.originCell\.w/,
    "conditioning must reconstruct the immutable graph-centre pressure dual");
  const restore = shader.indexOf("f.inverseDistance=1./baseDistance");
  assert.ok(restore >= 0 && restore < shader.indexOf("if(wet==INVALID)"),
  "a face that ceases crossing must publish its geometric coefficient before returning");
});

test("adaptive ghost refresh replaces rather than accumulates interface proximity", () => {
  const shader = octreeLosassoAdaptivePhiGhostWGSL;
  assert.match(shader, /f\.reserved&=~FACE_INTERFACE_NEARBY/);
  assert.match(shader,
    /f\.negativeRow<rows&&\(rowPhi\[f\.negativeRow\]\.w&INTERFACE\)!=0u/);
  assert.match(shader,
    /f\.positiveRow<rows&&\(rowPhi\[f\.positiveRow\]\.w&INTERFACE\)!=0u/);
  assert.match(shader, /if\(negativeNearby\|\|positiveNearby\)\{f\.reserved\|=FACE_INTERFACE_NEARBY;\}/);
});

test("adaptive ghost refresh preserves the separated closed-wall active set", () => {
  const shader = octreeLosassoAdaptivePhiGhostWGSL;
  assert.match(shader, /let separated=closed&&\(f\.reserved&FACE_SEPARATED\)!=0u/);
  assert.match(shader, /solid\|\|\(air==INVALID&&closed&&!separated\)/,
    "an ordinary closed wall remains Neumann");
  assert.match(shader,
    /if\(air==INVALID&&closed&&separated\)\{dual=max\(\.5,\.5\*wetSpan-\.5\)\*p\.originCell\.w;\}/);
  assert.match(shader, /else if\(separated\)\{f\.openFraction=1\.;\}/,
    "a separated wet lid face must be reopened as a ghost-fluid face");
});

test("adaptive ghost refresh retains a fractional rigid aperture across interface scaling", () => {
  const shader = octreeLosassoAdaptivePhiGhostWGSL;
  assert.match(shader, /let geometricOpen=clamp\(f\.openFraction,0\.,1\.\)/);
  assert.match(shader, /f\.inverseDistance=1\.\/baseDistance;f\.openFraction=geometricOpen/);
  assert.doesNotMatch(shader, /f\.inverseDistance=1\.\/ghostDistance;f\.openFraction=1\./,
    "ghost-fluid distance scaling must not reopen a geometry-conditioned interior face");
});

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("adaptive ghost GPU path preserves fractional open area on a liquid-air face", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for adaptive ghost validation",
}, async () => {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const make = (size: number, usage = storage) => device.createBuffer({ size, usage });

  const params = make(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const graph = make(128); const leaves = make(96); const rowPhi = make(32);
  const faceControl = make(32); const faces = make(32); const ghosts = make(16);
  const rowToLeaf = make(8); const readback = make(48,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  const paramsData = new ArrayBuffer(96);
  new Float32Array(paramsData)[7] = .05;
  new Uint32Array(paramsData)[18] = 1;
  device.queue.writeBuffer(params, 0, paramsData);
  const graphWords = new Uint32Array(32); graphWords[1] = 2; graphWords[28] = 2;
  device.queue.writeBuffer(graph, 0, graphWords);
  const leafWords = new Uint32Array(24); leafWords[3] = 1; leafWords[15] = 1;
  device.queue.writeBuffer(leaves, 0, leafWords);
  const rowWords = new Uint32Array(8); const rowFloats = new Float32Array(rowWords.buffer);
  rowFloats[0] = -.02; rowWords[3] = 4; rowFloats[4] = .03; rowWords[7] = 4;
  device.queue.writeBuffer(rowPhi, 0, rowWords);
  const faceControlWords = new Uint32Array(8); faceControlWords[2] = 1;
  device.queue.writeBuffer(faceControl, 0, faceControlWords);
  const faceWords = new Uint32Array(8); const faceFloats = new Float32Array(faceWords.buffer);
  faceWords.set([0, 1, 0, 0], 0); faceFloats[4] = 1; faceFloats[5] = 7;
  faceFloats[6] = .375; faceFloats[7] = 0;
  device.queue.writeBuffer(faces, 0, faceWords);
  device.queue.writeBuffer(rowToLeaf, 0, new Uint32Array([0, 1]));

  const shaderModule = device.createShaderModule({ code: octreeLosassoAdaptivePhiGhostWGSL });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module: shaderModule, entryPoint: "deriveGhosts" } });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    params, graph, leaves, rowPhi, faceControl, faces, ghosts, rowToLeaf,
  ].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(faces, 0, readback, 0, 32);
  encoder.copyBufferToBuffer(ghosts, 0, readback, 32, 16);
  device.pushErrorScope("validation"); device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  assert.equal((await device.popErrorScope())?.message, undefined);
  await readback.mapAsync(GPUMapMode.READ);
  const result = Uint32Array.from(new Uint32Array(readback.getMappedRange()));
  readback.unmap();
  const resultFloats = new Float32Array(result.buffer);
  assert.ok(Math.abs(resultFloats[5] - 50) < 1e-4,
    `expected theta-scaled inverse distance 50, received ${resultFloats[5]}`);
  assert.ok(Math.abs(resultFloats[6] - .375) < 1e-6,
    `expected preserved aperture .375, received ${resultFloats[6]}`);
  assert.ok(Math.abs(resultFloats[8] - .02) < 1e-6,
    `expected ghost distance .02, received ${resultFloats[8]}`);
  assert.equal(result[11], 1, "interior liquid-air ghost must remain valid");

  for (const buffer of [params, graph, leaves, rowPhi, faceControl, faces, ghosts,
    rowToLeaf, readback]) buffer.destroy();
  device.destroy();
});
