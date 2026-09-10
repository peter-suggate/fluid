import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

/** Isolate the *production* density interpolation from page/cache/mesh paths.
 * The fixture remains inside one native vertical interface bracket in the ROI.
 */
export async function sampleCoarseBowlVolumeKernel(device: GPUDevice, nx: number, ny: number,
  nz: number, h: number, width: number, phase: number): Promise<Float32Array> {
  const source = await readFile(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
  const functions = ["interpolatedPresentationDensityAt", "presentationInterpolatedDensityPhi"].map(name => {
    const fn = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
    assert.ok(fn, `production ${name}`); return fn;
  }).join("\n");
  const code = `
struct Params{dimensions:vec4u,frame:vec4f}
const p=Params(vec4u(${nx},${ny},${nz},0),vec4f(0,${h},0,0));
const CM12_LIQUID_ISOVALUE=.5;
const cm12PresentationBrick=0u;
fn brickHasUnclippedWorldGeometry(brick:u32)->bool{_=brick;return false;}
// This fixture uses bounded interior cells, with the ordinary closed-tank
// canonicalization. Dynamic-world continuation is intentionally absent.
fn presentationCanonicalCoarseCoordinate(q:vec3i,scale:u32,brick:u32)->vec3i{
  _=brick;return clamp(q,vec3i(0),vec3i(p.dimensions.xyz/scale)-vec3i(1));
}
@group(0)@binding(0)var<storage,read_write>result:array<vec4f>;
fn presentationStencilDensityAt(q:vec3i,scale:u32,first:vec3i,dims:vec3u,fits:bool,offset:u32)->f32{
  _=first;_=dims;_=fits;_=offset;
  let w=f32(scale);let x=(f32(q.x)+.5)*w-${nx / 2 + phase};let z=(f32(q.z)+.5)*w-${nz / 2};
  // Exact moment of the same 1/8-finest-cell midpoint area quadrature.
  let height=17.3+.003*(x*x+.7*z*z+1.7*(w*w/12.-1./768.));
  return clamp((height-f32(q.y)*w)/w,0.,1.);
}
${functions}
@compute @workgroup_size(64)fn main(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=${nx * nz}u){return;}
  let x=gid.x%${nx}u;let z=gid.x/${nx}u;
  let expected=17.3+.003*(pow(f32(x)+.5-${nx / 2 + phase},2.)+.7*pow(f32(z)+.5-${nz / 2},2.));
  let y=i32(floor(expected-.5));
  let lo=presentationInterpolatedDensityPhi(vec3i(i32(x),y,i32(z)),${width}u,vec3i(0),vec3u(0),false,0u);
  let hi=presentationInterpolatedDensityPhi(vec3i(i32(x),y+1,i32(z)),${width}u,vec3i(0),vec3u(0),false,0u);
  result[gid.x]=vec4f(lo,hi,unpack2x16float(pack2x16float(vec2f(lo,hi))));
}`;
  device.pushErrorScope("validation");
  const shaderModule = device.createShaderModule({ label: "grid-imprint-isolated-volume-kernel", code });
  const compilation = await shaderModule.getCompilationInfo();
  assert.deepEqual(compilation.messages.filter(m => m.type === "error").map(m => `${m.lineNum}:${m.linePos} ${m.message}`), []);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: {
    module: shaderModule, entryPoint: "main",
  } });
  const output = device.createBuffer({ size: nx * nz * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(nx * nz / 64)); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, output.size); device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(readback.getMappedRange()).slice();
    const error = await device.popErrorScope(); assert.equal(error, null, error?.message);
    return values;
  } finally {
    if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); output.destroy();
  }
}
