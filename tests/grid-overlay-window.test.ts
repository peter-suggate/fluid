import assert from "node:assert/strict";
import test from "node:test";
import { gridOverlayShader } from "../lib/core/webgpu-grid-overlay";
import { gridStructure3DWGSL } from "../lib/core/grid-structure-3d.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

function functionSource(name: string): string {
  const start = gridOverlayShader.indexOf(`fn ${name}(`);
  assert.ok(start >= 0);
  let depth = 0;
  for (let i = gridOverlayShader.indexOf("{", start); i < gridOverlayShader.length; i++) {
    if (gridOverlayShader[i] === "{") depth++;
    if (gridOverlayShader[i] === "}" && --depth === 0) return gridOverlayShader.slice(start, i + 1);
  }
  throw new Error(`Unclosed WGSL function ${name}`);
}

test("grid structure stays at the same world position after storage-window recentering", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("grid overlay window", async device => {
    // Render the production traversal and frame transform against a tiny
    // topology fixture: the same 4h owners, readdressed by the live window.
    const module = device.createShaderModule({ code: `
struct View {gridInfo:vec4f,container:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,viewport:vec4f,debug:vec4f}
const u=View(vec4f(32,32,32,1),vec4f(4,4,4,0),vec4f(5,5,6,0),vec4f(0,2,0,0),vec4f(64,64,0,0),vec4f(0));
struct Layers {selection:vec4f};const layers=Layers(vec4f(0,0,0.7,0));
struct Lsv {global:vec4u};const sliceLsvP=Lsv(vec4u(2,0,0,0));
@group(0) @binding(0) var<uniform> window:vec4i;
fn umFramePage()->vec3i{return window.xyz;}
fn umPresentationEnabled()->bool{return true;}
fn udrInit(){}
fn cameraTanHalfFov()->f32{return 0.5;}
struct OverlayGridFrame {minimumFine:vec3i,dimensions:vec3i,boundsMin:vec3f,size:vec3f}
${functionSource("overlayGridFrame")}
${functionSource("volumeBoxEdgePixels")}
${functionSource("gridLinePaint")}
struct UMOwner {width:u32,origin:vec3u}
fn umOwnerAt(cell:vec3i)->UMOwner{
  let world=cell+32*umFramePage();
  if(any(world<vec3i(0))||any(world>=vec3i(32))){return UMOwner();}
  return UMOwner(4u,vec3u((cell/4)*4));
}
fn umOrigin(owner:UMOwner)->vec3u{return owner.origin;}
fn umStageBase()->u32{return 0xffffffffu;}
const UM_STAGE_TRANSPORT=0u;
fn umStagePreviousWidth(stage:u32,tile:vec3i)->u32{return 0u;}
fn umStageTileWidth(stage:u32,kind:u32,tile:vec3i)->u32{return 0u;}
fn umTilePresent(tile:vec3i)->bool{return false;}
fn umTileWidth(tile:u32)->u32{return 4u;}
fn umTileAt(tile:vec3u)->u32{return 0u;}
const LP_coarsened=vec3f(0);
fn sceneColor(c:vec3f)->vec3f{return c;}
fn displayColor(c:vec3f)->vec3f{return c/(1.0+c);}
struct RepresentedCell {lower:vec3f,upper:vec3f}
fn representedCell(cell:vec3i,dims:vec3i,origin:vec3i,low:vec3f,size:vec3f,adaptive:bool,tall:bool)->RepresentedCell{return RepresentedCell();}
struct BodySample {occupied:bool}
fn gridBodySample(bounds:RepresentedCell)->BodySample{return BodySample(false);}
struct VertexOutput {@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index) i:u32)->VertexOutput{
  let xy=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3))[i];
  return VertexOutput(vec4f(xy,0,1),xy*0.5+0.5);
}
${gridStructure3DWGSL}` });
    const pipeline = await device.createRenderPipelineAsync({ layout: "auto", vertex: { module, entryPoint: "vertexMain" }, fragment: { module, entryPoint: "fragmentGrid3D", targets: [{ format: "rgba8unorm" }] } });
    const window = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const target = device.createTexture({ size: [64, 64], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const read = device.createBuffer({ size: 64 * 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: window } }] });
    try {
      const render = async (frame: number[]) => {
        device.queue.writeBuffer(window, 0, new Int32Array([...frame, 0]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
        encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: 256 }, [64, 64]);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const pixels = new Uint8Array(read.getMappedRange()).slice(); read.unmap();
        return pixels;
      };
      const initial = await render([0, 0, 0]);
      assert.ok(initial.filter((value, i) => i % 4 === 3 && value > 0).length > 500, "initial grid must be visible");
      for (const frame of [[-27, -30, -28], [-26, -29, -27], [0, 0, 0]]) {
        const shifted = await render(frame);
        // A subpixel edge may round one byte differently after translation.
        assert.ok(shifted.every((value, i) => Math.abs(value - initial[i]!) <= 1), `grid moved or disappeared at frame ${frame}`);
      }
    } finally { read.destroy(); target.destroy(); window.destroy(); }
  });
});
