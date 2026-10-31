import type { SculptWallMesh } from "../lib/svo/features/primary-visibility/sculpt-wall-mesh";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";
import type { SceneDescription } from "../lib/core/model";
import type { SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { svoRenderScenePublication, type SvoRenderScenePublication } from "../lib/svo/features/scene-publication/svo-render-scene-publication";
import type { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { packSvoDryViewUniforms } from "./svo-dry-frame-harness";
import { writeFramePng } from "./write-frame-png";
import { resolveDisplayGrade } from "../lib/core/webgpu-lighting";
import { svoSceneLighting } from "../lib/svo/features/lighting-visibility/svo-dry-scene-lighting";

/** Production renderer A/B on identical edited geometry and moving views.
 * Readbacks/fences are diagnostic only and stay outside production editing. */
export async function benchmarkSculptWallCamera(input: {
  device: GPUDevice; renderer: SparseVoxelDrySceneRenderer; solver: WebGPULiveSvoScene;
  scene: SceneDescription; initialPublication: SvoRenderScenePublication; target: GPUTexture; uniforms: GPUBuffer;
  view: Parameters<typeof packSvoDryViewUniforms>[0]; outPath: string; cycles: number; adapterInfo: unknown;
}) {
  const { device, renderer, solver, scene, target, view, cycles } = input;
  assert.equal(scene.sceneId, "atlantic-lighthouse"); assert.ok(GPUPassTimestampRecorder.supported(device));
  const directory = dirname(input.outPath); mkdirSync(directory, { recursive: true });
  let publication = input.initialPublication, revision = 2;
  const patches: SolidWorldVoxelPatch[] = Array.from({ length: 16 }, (_, step) => {
    const rock = step >= 8, shift = step % 8;
    return { operation: "clear", kind: "ellipsoid", affectsScenery: true,
      minimum: rock ? [88 + shift, 8, 116] : [117 + shift, 31, 95],
      maximumExclusive: rock ? [100 + shift, 16, 128] : [127 + shift, 41, 105] };
  });
  async function publish(edits: SolidWorldVoxelPatch[]) {
    const next = { ...scene, solidVoxels: [...scene.solidVoxels, ...edits] };
    solver.stageSceneUpdate(next); publication = svoRenderScenePublication(next, revision, publication)!;
    assert.ok(renderer.publishScene({ ...publication.data, renderRevision: revision++ }));
    const encoder = device.createCommandEncoder(); solver.encodeSceneMaintenance(encoder);
    assert.ok(renderer.encode(encoder, target)); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  }
  async function frame(index: number, timed: boolean) {
    const camera = { ...view.camera, azimuth_rad: view.camera.azimuth_rad + .12 * Math.sin(index * .31),
      elevation_rad: view.camera.elevation_rad + .04 * Math.cos(index * .23) };
    const words = packSvoDryViewUniforms({ ...view, camera, cameraMoving: true });
    device.queue.writeBuffer(input.uniforms, 0, words); renderer.setRasterVisibilityView(words);
    const encoder = device.createCommandEncoder();
    const timer = timed ? new GPUPassTimestampRecorder(device, 512, "Edited lighthouse camera") : undefined;
    const start = performance.now(); assert.ok(renderer.encode(timer?.instrument(encoder) ?? encoder, target));
    const encode_ms = performance.now() - start;
    const resolved = timer?.resolve(encoder); device.queue.submit([encoder.finish()]);
    if (!timer) { await device.queue.onSubmittedWorkDone(); return undefined; }
    assert.ok(resolved, "camera movement must produce a new frame");
    const reading = await timer.read(); assert.ok(reading, "missing GPU timestamps invalidate the run");
    return { gpu_ms: reading.span_ms, encode_ms, passes: reading.passes };
  }
  async function textureBytes(texture: GPUTexture, bpp: number) {
    const pitch = Math.ceil(view.width * bpp / 256) * 256;
    const read = device.createBuffer({ size: pitch * view.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow: pitch }, [view.width, view.height]);
    device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(view.width * view.height * bpp), mapped = new Uint8Array(read.getMappedRange());
    for (let y=0;y<view.height;y++) bytes.set(mapped.subarray(y*pitch,y*pitch+view.width*bpp),y*view.width*bpp);
    read.unmap();read.destroy();return bytes;
  }
  const geometry = () => (renderer as unknown as { splitGeometry: GPUTexture }).splitGeometry;
  async function snapshot(name: string) {
    await frame(0,false);
    const bytes = await textureBytes(target,8); assert.ok(bytes.some(v=>v!==0),"empty GPU frames invalidate timings");
    writeFramePng(join(directory,`${name}.png`),{width:view.width,height:view.height,packedRows:new Uint32Array(bytes.buffer),grade:resolveDisplayGrade(svoSceneLighting(scene).grade)});
    return new Float32Array((await textureBytes(geometry(),16)).buffer);
  }
  const intact = await snapshot("intact");
  await publish(patches);
  // Finish the ordinary scene mesh maintenance too, so the A/B measures camera
  // movement rather than a different number of pending edit batches.
  for(let i=0;i<240;i++){
    const encoder=device.createCommandEncoder();solver.encodeSceneMaintenance(encoder);renderer.encode(encoder,target);device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
    await new Promise(resolve=>setTimeout(resolve,5));
    if(i>=4&&renderer.surfaceMeshStatus?.state==="ready"&&renderer.surfaceMeshStatus.drawn)break;
    assert.ok(i<239,`edited mesh did not settle: ${JSON.stringify(renderer.surfaceMeshStatus)}`);
  }
  const cache = (renderer as unknown as { sculptWallMesh: SculptWallMesh }).sculptWallMesh;
  let cachedVertices = 0, overflowPages = 0;
  for (const held of cache.pages.values()) {
    const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(held.args,0,read,0,32); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const args = new Uint32Array(read.getMappedRange());
    cachedVertices += args[0]!; if(args[0]!==args[4]) overflowPages++;
    read.unmap(); read.destroy();
  }
  assert.ok(cachedVertices > 0, "the benchmark must generate actual cut-wall triangles");
  assert.equal(overflowPages,0,"the benchmark must measure cached triangles, not overflow ray marching");
  const samples: Record<string, {gpu_ms:number;encode_ms:number}[]> = { raymarch: [], cached: [] };
  const captured = new Map<string,Float32Array>();
  const builtBefore = renderer.sculptWallStatistics.builtPages;
  for(const mode of ["raymarch","cached","cached","raymarch"] as const){
    renderer.setSculptWallMeshEnabled(mode==="cached");
    for(let i=-4;i<cycles;i++){
      const result=await frame(i,i>=0);
      if(result){assert.ok(!result.passes.some(p=>p.label==="Cache changed sculpt wall triangles"),"camera movement must not rebuild walls");samples[mode]!.push({gpu_ms:result.gpu_ms,encode_ms:result.encode_ms});}
    }
    if(!captured.has(mode))captured.set(mode,await snapshot(mode));
    console.log(JSON.stringify({phase:"sculpt-wall-camera",mode,samples:samples[mode]!.length,mesh:renderer.sculptWallStatistics}));
  }
  assert.equal(renderer.sculptWallStatistics.builtPages,builtBefore,"both moving-camera arms retain the baked wall mesh");
  renderer.setSculptWallMeshEnabled(true);
  const carved=await snapshot("carved");
  await publish([]);const undo=await snapshot("undo");
  await publish(patches);const redo=await snapshot("redo");
  const differences=(a:Float32Array,b:Float32Array)=>{
    let pixels=0,maxDepth=0;for(let p=0;p<a.length/4;p++){if([0,1,2,3].some(c=>Math.abs(a[p*4+c]!-b[p*4+c]!)>1e-4)){pixels++;const delta=Math.abs(a[p*4+3]!-b[p*4+3]!);if(Number.isFinite(delta))maxDepth=Math.max(maxDepth,delta);}}return {pixels,maxDepth};
  };
  const quantile=(a:number[],q:number)=>[...a].sort((a,b)=>a-b)[Math.min(a.length-1,Math.floor(a.length*q))]!;
  const result=Object.fromEntries(Object.entries(samples).map(([mode,values])=>[mode,{gpuMedian_ms:quantile(values.map(v=>v.gpu_ms),.5),gpuP95_ms:quantile(values.map(v=>v.gpu_ms),.95),cpuEncodeMedian_ms:quantile(values.map(v=>v.encode_ms),.5),samples:values}]));
  const parity={intactToCarved:differences(intact,carved),undo:differences(intact,undo),redo:differences(carved,redo),rayToTriangles:differences(captured.get("raymarch")!,captured.get("cached")!)};
  const report={scene:scene.sceneId,width:view.width,height:view.height,adapter:input.adapterInfo,stamps:patches.length,camera:view.camera,
    scope:"Complete production dry-renderer GPU spans during camera orbit; excludes fluid simulation, water and browser presentation. Same lighting, geometry and camera sequence in ABBA order. Four warmup frames per arm. Meshing excluded from steady camera timings.",
    result,parity,mesh:{...renderer.sculptWallStatistics,cachedTriangles:cachedVertices/3,overflowPages}};
  writeFileSync(input.outPath,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
  assert.ok(parity.intactToCarved.pixels>0,"the benchmark must actually see edits");
  assert.equal(parity.undo.pixels,0,"undo restores geometry");assert.equal(parity.redo.pixels,0,"redo restores cached wall geometry");
}
