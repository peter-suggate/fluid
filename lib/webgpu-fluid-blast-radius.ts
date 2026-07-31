/**
 * The pressure dependency cone, flooded over the real sparse topology.
 *
 * `fluid-blast-radius.ts` grows the cone analytically over a dense hierarchy and
 * proves the structural fact: because the SPGrid pyramid bottoms out at one cell
 * and the correction is prolonged back down, one V-cycle makes every cell depend
 * on every other. That answer is topology-independent, which is exactly why it
 * cannot show what a given frame's topology does to the *fine-grid* reach.
 *
 * This is the other half. It floods the level-0 row graph as published — the
 * 18-direction power stencil, resolved through the owner map the renderer
 * already binds — and colours every row by its hop distance from a seed. What
 * that reveals is reach anisotropy: one smoother hop across a 32³ leaf moves
 * thirty-two finest cells, while inside a refined band it moves one. The cone is
 * therefore lopsided in a way no dense model can express, and the lopsidedness
 * is a property of the adaptivity the scene chose.
 *
 * Scope, stated plainly: this is the smoother/operator reach only. It
 * deliberately does not model restriction or prolongation, so it is the cone
 * *before* the coarse grid short-circuits it. Read it beside the analytic
 * report, not instead of it.
 *
 * Nothing here is encoded inside an accepted advance and no solver buffer is
 * written; the flood reads the owner map and owns its own hop field.
 */
import { octreeTechniqueSharedWGSL } from "./webgpu-octree-technique-shared";
import { PassBroker } from "./webgpu-pass-broker";

/** Rows unreached by the flood keep this sentinel rather than a large distance. */
export const FLUID_BLAST_RADIUS_UNREACHED = 0xffff_ffff;
/**
 * Relaxation rounds. Each is one full finest-grid dispatch, and the cone stops
 * being informative long before it saturates, so this is a bounded diagnostic
 * ceiling rather than a convergence criterion.
 */
export const FLUID_BLAST_RADIUS_ROUNDS = 24;
const CONFIG_BYTES = 32;

/** Types and constants; declared before the bindings that reference them. */
const blastRadiusTypesWGSL = /* wgsl */ `
struct BlastConfig { dimensions:vec3u, rowCapacity:u32, rounds:u32, pad0:u32, pad1:u32, pad2:u32 }
const UNREACHED:u32=${FLUID_BLAST_RADIUS_UNREACHED}u;

/**
 * The eighteen power-diagram neighbour directions, in the same order the
 * operator uses: six faces then twelve edges.
 */
fn blastDirection(index:u32)->vec3i {
  let d=array<vec3i,18>(
    vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),
    vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),
    vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));
  return d[index];
}
`;

/** Needs both `u` and `config`, so it follows the binding declarations. */
const blastRadiusSeedWGSL = /* wgsl */ `
/**
 * The cell the cone is seeded from, derived from the slice controls already in
 * the shared uniform: the slice plane picks one axis and the domain centre
 * supplies the other two, so moving the existing slider sweeps the seed. In the
 * volume presentation there is no plane, so the seed is the domain centre.
 */
fn blastSeedCell(dimensions:vec3u)->vec3u {
  let extent=vec3f(max(dimensions,vec3u(1u)));
  let axis=i32(round(u.debug.x));
  let fraction=clamp(u.debug.y,0.0,0.999999);
  var cell=vec3f(floor(extent*0.5));
  if(axis==1){cell.z=floor(fraction*extent.z);}
  else if(axis==2){cell.x=floor(fraction*extent.x);}
  else if(axis==3){cell.y=floor(fraction*extent.y);}
  return vec3u(clamp(cell,vec3f(0.0),extent-vec3f(1.0)));
}
`;

export const fluidBlastRadiusFloodShader = /* wgsl */ `
struct Uniforms {
  viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f,
  options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f,
}
${blastRadiusTypesWGSL}
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var ownerRows:texture_3d<u32>;
@group(0) @binding(2) var<storage,read_write> hops:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> config:BlastConfig;
${blastRadiusSeedWGSL}

fn ownerAt(cell:vec3i)->u32 {
  if(any(cell<vec3i(0))||any(cell>=vec3i(config.dimensions))){return UNREACHED;}
  let row=textureLoad(ownerRows,cell,0).x;
  return select(UNREACHED,row,row<config.rowCapacity);
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid:vec3u) {
  if(gid.x>=config.rowCapacity){return;}
  atomicStore(&hops[gid.x],UNREACHED);
}

@compute @workgroup_size(1)
fn seed() {
  let row=ownerAt(vec3i(blastSeedCell(config.dimensions)));
  // A seed landing on a dry or absent cell leaves every row unreached, which
  // the view draws as an explicit empty cone rather than as a silent zero.
  if(row==UNREACHED){return;}
  atomicStore(&hops[row],0u);
}

/**
 * One relaxation round over the finest grid.
 *
 * Every cell reads its own row's current distance and offers distance + 1 to the
 * rows owning its eighteen neighbours. Cells interior to a row resolve to that
 * same row and contribute nothing, so the pass is exactly a relaxation over the
 * row adjacency graph. Reading and writing the same field within a round can
 * only propagate a value further in one pass; every value written is the length
 * of some real path, so the atomic minimum converges to the true hop distance
 * from below the round count rather than overshooting it.
 */
@compute @workgroup_size(4,4,4)
fn relax(@builtin(global_invocation_id) gid:vec3u) {
  let cell=vec3i(gid);
  if(any(cell>=vec3i(config.dimensions))){return;}
  let row=ownerAt(cell);
  if(row==UNREACHED){return;}
  let distance=atomicLoad(&hops[row]);
  if(distance==UNREACHED||distance>=config.rounds){return;}
  for(var index=0u;index<18u;index+=1u){
    let neighbour=ownerAt(cell+blastDirection(index));
    if(neighbour==UNREACHED||neighbour==row){continue;}
    atomicMin(&hops[neighbour],distance+1u);
  }
}
`;

export const fluidBlastRadiusOverlayShader = /* wgsl */ `
${octreeTechniqueSharedWGSL}
${blastRadiusTypesWGSL}
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var ownerRows:texture_3d<u32>;
@group(0) @binding(2) var<storage,read> hops:array<u32>;
@group(0) @binding(3) var<uniform> config:BlastConfig;
${blastRadiusSeedWGSL}

fn hopAt(point:vec3f)->u32 {
  let extent=vec3f(max(config.dimensions,vec3u(1u)));
  let coordinate=worldToFine(point);
  if(any(coordinate<vec3f(0.0))||any(coordinate>=extent)){return UNREACHED;}
  let row=textureLoad(ownerRows,vec3i(floor(coordinate)),0).x;
  if(row>=config.rowCapacity||row>=arrayLength(&hops)){return UNREACHED;}
  return hops[row];
}

fn hopColor(distance:u32)->vec3f {
  if(distance==0u){return vec3f(1.0,1.0,1.0);}
  let t=clamp(f32(distance)/max(f32(config.rounds),1.0),0.0,1.0);
  if(t<0.34){return mix(vec3f(0.10,0.28,0.92),vec3f(0.06,0.78,0.80),t/0.34);}
  if(t<0.67){return mix(vec3f(0.06,0.78,0.80),vec3f(0.28,0.82,0.28),(t-0.34)/0.33);}
  return mix(vec3f(0.28,0.82,0.28),vec3f(0.98,0.35,0.08),(t-0.67)/0.33);
}

fn blastSample(point:vec3f)->vec4f {
  let distance=hopAt(point);
  // Rows the cone never reached are drawn faintly so the cone reads as a shape
  // against the domain rather than as the only thing present.
  if(distance==UNREACHED){return vec4f(0.04,0.08,0.22,0.05);}
  if(distance==0u){return vec4f(1.0,1.0,1.0,0.95);}
  // Near hops are the informative part; far ones fade so the frontier is legible.
  let t=clamp(f32(distance)/max(f32(config.rounds),1.0),0.0,1.0);
  return vec4f(hopColor(distance),0.55*(1.0-t)+0.10);
}

fn blastVolume(uv:vec2f)->vec4f {
  let ray=cameraRay(uv);
  let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);
  let maximum=minimum+u.container.xyz;
  let interval=boxInterval(ray,minimum,maximum);
  if(interval.y<=interval.x){discard;}
  let steps=traversalSteps(ray,interval);
  let dt=(interval.y-interval.x)/f32(steps);
  var accum=vec4f(0.0);
  for(var i=0u;i<512u;i+=1u){
    if(i>=steps||accum.a>0.985){break;}
    let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);
    let sample=blastSample(point);
    if(sample.a<=0.001){continue;}
    accum=composite(accum,sample.rgb,sample.a*volumeOpacity());
  }
  return finishVolume(accum);
}

@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  if(i32(round(u.debug.x))==4){return blastVolume(input.uv);}
  let hit=sliceRay(input.uv);
  if(hit.w<=0.0){discard;}
  let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);
  let maximum=minimum+u.container.xyz;
  if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}
  let sample=blastSample(hit.xyz);
  if(sample.a<=0.001){discard;}
  return vec4f(displayColor(sample.rgb),sample.a);
}
`;

export interface FluidBlastRadiusRows {
  readonly dimensions: readonly [number, number, number];
  readonly rowCapacity: number;
}

/**
 * Owns the hop field and the pipelines that fill and draw it.
 *
 * The flood is re-run every time the view is drawn: the seed follows the slice
 * control and the topology is republished each accepted step, so a cached field
 * would silently describe a previous frame's adaptivity.
 */
export class WebGPUFluidBlastRadius {
  private clearPipeline?: GPUComputePipeline;
  private seedPipeline?: GPUComputePipeline;
  private relaxPipeline?: GPUComputePipeline;
  private drawPipeline?: GPURenderPipeline;
  private hops?: GPUBuffer;
  private config?: GPUBuffer;
  private floodGroup?: GPUBindGroup;
  private drawGroup?: GPUBindGroup;
  private floodLayout?: GPUBindGroupLayout;
  private drawLayout?: GPUBindGroupLayout;
  private ownerRows?: GPUTexture;
  private rows?: FluidBlastRadiusRows;
  private capacity = 0;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
    private readonly uniformBuffer: GPUBuffer,
  ) {}

  private async compile(label: string, code: string): Promise<GPUShaderModule> {
    const shaderModule = this.device.createShaderModule({ label, code });
    const compilation = await shaderModule.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    }
    return shaderModule;
  }

  async initialize(): Promise<void> {
    const [flood, overlay] = await Promise.all([
      this.compile("Fluid blast-radius flood", fluidBlastRadiusFloodShader),
      this.compile("Fluid blast-radius overlay", fluidBlastRadiusOverlayShader),
    ]);
    // Explicit layouts, not `layout: "auto"`.
    //
    // An automatic layout only carries the bindings its *entry point* reaches:
    // `clear` never reads the camera uniform or the owner map, `relax` never
    // reads the uniform, and the overlay declares the hop field read-only where
    // the flood declares it atomic. Four derived layouts are therefore four
    // different layouts, and the single bind group below cannot satisfy any but
    // the one it was built against — which fails validation and, because an
    // invalid bind group poisons the encoder, takes the whole frame down with
    // it. Declaring the set once makes every pipeline agree by construction.
    this.floodLayout = this.device.createBindGroupLayout({
      label: "Fluid blast-radius flood bindings",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });
    // The same four resources, seen by the fragment stage: the hop field is read
    // -only here, which is a different binding type and so a different layout.
    // The shared vertex stage reads nothing, so nothing is visible to it.
    this.drawLayout = this.device.createBindGroupLayout({
      label: "Fluid blast-radius overlay bindings",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    const floodPipelineLayout = this.device.createPipelineLayout({
      label: "Fluid blast-radius flood", bindGroupLayouts: [this.floodLayout],
    });
    const compute = (entryPoint: string) => this.device.createComputePipelineAsync({
      label: `Fluid blast-radius ${entryPoint}`, layout: floodPipelineLayout,
      compute: { module: flood, entryPoint },
    });
    [this.clearPipeline, this.seedPipeline, this.relaxPipeline, this.drawPipeline] = await Promise.all([
      compute("clear"), compute("seed"), compute("relax"),
      this.device.createRenderPipelineAsync({
        label: "Fluid blast-radius overlay",
        layout: this.device.createPipelineLayout({
          label: "Fluid blast-radius overlay", bindGroupLayouts: [this.drawLayout],
        }),
        vertex: { module: overlay, entryPoint: "vertexMain" },
        fragment: {
          module: overlay, entryPoint: "fragmentMain",
          targets: [{
            format: this.targetFormat,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            },
          }],
        },
        primitive: { topology: "triangle-list" },
      }),
    ]);
    this.rebuild();
  }

  setSource(ownerRows: GPUTexture | undefined, rows: FluidBlastRadiusRows | undefined): void {
    if (this.ownerRows === ownerRows && this.rows === rows) return;
    this.ownerRows = ownerRows;
    this.rows = rows;
    this.rebuild();
  }

  private rebuild(): void {
    this.floodGroup = undefined;
    this.drawGroup = undefined;
    const rows = this.rows;
    const ownerRows = this.ownerRows;
    if (!rows || !ownerRows || !this.floodLayout || !this.drawLayout) return;
    const capacity = Math.max(1, rows.rowCapacity);
    if (this.capacity !== capacity || !this.hops) {
      this.hops?.destroy();
      this.capacity = capacity;
      this.hops = this.device.createBuffer({
        label: "Fluid blast-radius hop field", size: capacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    if (!this.config) {
      this.config = this.device.createBuffer({
        label: "Fluid blast-radius config", size: CONFIG_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.config, 0, new Uint32Array([
      rows.dimensions[0], rows.dimensions[1], rows.dimensions[2], capacity,
      FLUID_BLAST_RADIUS_ROUNDS, 0, 0, 0,
    ]));
    const view = ownerRows.createView({ dimension: "3d" });
    const entries = (layout: GPUBindGroupLayout): GPUBindGroup => this.device.createBindGroup({
      label: "Fluid blast-radius bindings", layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: view },
        { binding: 2, resource: { buffer: this.hops! } },
        { binding: 3, resource: { buffer: this.config! } },
      ],
    });
    this.floodGroup = entries(this.floodLayout);
    this.drawGroup = entries(this.drawLayout);
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView): boolean {
    if (this.destroyed) return false;
    const rows = this.rows;
    if (!rows || !this.floodGroup || !this.drawGroup
      || !this.clearPipeline || !this.seedPipeline || !this.relaxPipeline || !this.drawPipeline) {
      return false;
    }
    const broker = new PassBroker(encoder);
    const flood = broker.compute({ label: "Flood fluid blast radius" });
    flood.setPipeline(this.clearPipeline);
    flood.setBindGroup(0, this.floodGroup);
    flood.dispatchWorkgroups(Math.ceil(this.capacity / 64));
    flood.setPipeline(this.seedPipeline);
    flood.dispatchWorkgroups(1);
    flood.setPipeline(this.relaxPipeline);
    const groups = rows.dimensions.map((extent) => Math.max(1, Math.ceil(extent / 4)));
    for (let round = 0; round < FLUID_BLAST_RADIUS_ROUNDS; round += 1) {
      flood.dispatchWorkgroups(groups[0], groups[1], groups[2]);
    }
    broker.fence("fluid blast radius flooded");

    const pass = encoder.beginRenderPass({
      label: "Fluid blast-radius overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this.drawPipeline);
    pass.setBindGroup(0, this.drawGroup);
    pass.draw(3);
    pass.end();
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.hops?.destroy();
    this.config?.destroy();
  }
}
