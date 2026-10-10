import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandRedistanceWGSL } from "../lib/methods/uniform/uniform-narrow-band-redistance.wgsl";
import { narrowBandParticleSurfaceWGSL, NARROW_BAND_SEED_DEPTH, NARROW_BAND_SURFACE_RADIUS } from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { createUniformTroughScene } from "../lib/core/uniform-trough-scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { advanceUniform, readUniformFields } from "./helpers/uniform-geometric";

function shaderFunction(name: string, source = narrowBandRedistanceWGSL): string {
  const start = source.indexOf(`fn ${name}(`);
  assert.ok(start >= 0, name);
  let depth = 0;
  for (let i = source.indexOf("{", start); i < source.length; i++) {
    if (source[i] === "{") depth++;
    if (source[i] === "}" && --depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`Unclosed ${name}`);
}

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("NB distance searches retain the water plane at voxel walls", async () => {
  await withUniformDevice("NB voxel distance", async device => {
    const output = device.createBuffer({size: 16 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
    try {
      const module = device.createShaderModule({code: /* wgsl */`
const UM_D=vec3u(4);const nbCoarseOnly=false;
struct Params {hDt:vec4f}
const params=Params(vec4f(1));
@group(0) @binding(0) var<storage,read_write> output:array<vec4f>;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umSolidEnabled()->bool{return true;}
// A one-cell wall. Both its planes belong to open cells; its interior does not.
var<private> wall:u32=0u;
fn umCellOpen(c:vec3i)->f32{
 if(wall==0u){return select(1.0,0.0,c.x>=2);}
 if(wall==1u){return select(1.0,0.0,c.x<2);}
 return select(1.0,0.0,c.x==2);
}
fn umTileAt(c:vec3u)->u32{return 0u;}
fn umTileWidth(t:u32)->u32{return 1u;}
fn umLoadVertex(p:vec3u)->f32{
 if((wall==0u&&p.x>2u)||(wall==1u&&p.x<2u)){return 10.0;}
 return f32(p.y)-2.5;
}
fn umLoadCorner(p:vec3u)->f32{return umLoadVertex(p);}
fn umVertexSum8(v:array<f32,8>)->f32{var s=0.0;for(var k=0u;k<8u;k++){s+=v[k];}return s;}
${["nbOpenSurfaceCell", "nbSurfaceSegmentOpen", "nbTrilinear", "nbPhiGradient"].map(name => shaderFunction(name)).join("\n")}
const NB_SURFACE_RADIUS=${NARROW_BAND_SURFACE_RADIUS};
const NB_SEED_DEPTH=${NARROW_BAND_SEED_DEPTH};
${shaderFunction("nbPlanarSeedDepth", narrowBandParticleSurfaceWGSL)}
@compute @workgroup_size(1) fn main(){
 output[0]=nbPhiGradient(vec3f(2,1,2));
 wall=1u;output[1]=nbPhiGradient(vec3f(2,1,2));
 wall=2u;
 output[2]=vec4f(vec3f(nbOpenSurfaceCell(vec3f(2.25,1,2))),0);
 output[3]=vec4f(f32(nbSurfaceSegmentOpen(vec3f(1.75,1,2),vec3f(3.25,1,2))),
  f32(nbSurfaceSegmentOpen(vec3f(2,1,2),vec3f(2,2.5,2))),0,0);
 // Walk the production gradient toward the zero set at the positive-side
 // solid contact. This must reach y=2.5, without moving into the wall.
 wall=0u;var q=vec3f(2,1,2);
 for(var i=0u;i<5u;i++){
  let s=nbPhiGradient(q);let g2=dot(s.xyz,s.xyz);if(g2<1e-8){break;}
  let next=q-s.w*s.xyz/g2;if(!nbSurfaceSegmentOpen(q,next)){break;}q=next;
 }
 output[4]=vec4f(q,length(q-vec3f(2,1,2)));
 for(var i=0u;i<=10u;i++){
  let fraction=f32(i)*0.1;let site=vec3f(1.25,1.25,1.25);let d=site.y-(2.0+fraction);
  let depth=nbPlanarSeedDepth(site,d,vec3f(0,1,0));
  let lower=depth-fraction;let upper=lower+1.0;
  let a=sqrt(0.125+lower*lower)-NB_SURFACE_RADIUS;
  let b=sqrt(0.125+upper*upper)-NB_SURFACE_RADIUS;
  output[5u+i]=vec4f(mix(a,b,fraction),depth,0,0);
 }
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const pipeline = await device.createComputePipelineAsync({layout: "auto", compute: {module, entryPoint: "main"}});
      const group = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [{binding: 0, resource: {buffer: output}}]});
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0, group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
      const result = await readMixedBuffer(device, output);
      assert.deepEqual([...result.slice(0, 8)], [0, 1, 0, -1.5, 0, 1, 0, -1.5], "both wall orientations preserve the plane's gradient");
      assert.equal(result[8], -1, "wall interior has no liquid sampling cell");
      assert.deepEqual([...result.slice(12, 14)], [0, 1], "reject a search through a thin wall, allow motion along it");
      assert.deepEqual([...result.slice(16, 20)], [2, 2.5, 2, 1.5], "the nearest surface lies above the boundary vertex");
      for (let i = 0; i <= 10; i++) assert.ok(Math.abs(result[4 * (5 + i)]!) < 1e-6,
        `waterline fraction ${i / 10}: particle reconstruction preserves the nodal plane`);
    } finally { output.destroy(); }
  });
});

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("NB reconstruction keeps a stationary waterline against a voxel riser for four seconds", {timeout: 240_000}, async () => {
  await withUniformDevice("NB voxel riser reconstruction", async device => {
   for (const configuration of ["deep", "film", "ledge"]) for (const gravity of [0, -9.80665]) {
    // Exercise deep contact, an isolated 0.4-cell film, and a shallow ledge
    // next to deep particles. No inflow or initial motion supplies energy.
    const floor = configuration === "film" ? 8 : 4;
    const scene = createUniformTroughScene("settled-tank", 0.05), waterline = configuration === "deep" ? 0.431 : 0.42;
    scene.container.width_m = scene.container.height_m = scene.container.depth_m = 0.8;
    scene.container.fillFraction = waterline / 0.8;
    scene.fluid.initialLiquidVolumes = [];
    scene.fluid.gravity_m_s2 = {x: 0, y: gravity, z: 0};
    scene.fluid.surfaceTension_N_m = 0;
    scene.solidVoxels = [
      {operation: "fill", minimum: [0, 0, 0], maximumExclusive: [16, floor, 16]},
      {operation: "fill", minimum: [8, floor, 0], maximumExclusive: [16, 12, 16]},
    ];
    if (configuration === "ledge") scene.solidVoxels.push(
      {operation: "fill", minimum: [4, 4, 0], maximumExclusive: [8, 8, 16]});
    const solver = await uniformNarrowBandMethod.createSolverAsync!(device, scene, "balanced",
      {...uniformNarrowBandMethod.appDefaults, timeStep: "sixtieth", detailPolicy: "full"}, undefined, () => {}) as WebGPUUniformReferenceSolver;
    try {
      for (let frame = 1; frame <= 240; frame++) {
        await advanceUniform(solver, frame / 60);
        if (frame !== 1 && frame !== 60 && frame !== 240) continue;
        const fields = await readUniformFields(device, solver);
        let maxError = 0;
        for (let x = 2; x <= 8; x++) {
          const low = fields.vertex(x, 8, 8), high = fields.vertex(x, 9, 8);
          assert.ok(low <= 0 && high > 0, `${configuration}, frame ${frame}, x=${x}: keep the waterline crossing`);
          const height = 0.05 * (8 - low / (high - low));
          maxError = Math.max(maxError, Math.abs(height - waterline));
        }
        console.log(JSON.stringify({configuration, frame, gravity, maxError_m: maxError}));
        assert.ok(maxError < 0.00005,
          `${configuration}, frame ${frame}: a stationary voxel contact moved ${maxError / 0.05} cells`);
      }
    } finally { solver.destroy(); }
   }
  });
});
