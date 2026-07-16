import type { TallCellLayout } from "./tall-cell-grid";

/**
 * Hierarchical velocity extrapolation for the restricted tall-cell grid,
 * following the tall-cell paper Section 3.3.1 (Eq 8/9): after a two-cell
 * narrow band of fine neighbor passes (the paper uses the Jeong et al.
 * Eikonal solver there), velocities are swept fine-to-coarse — a coarse
 * sample is known when any of its fine samples is known, averaging the known
 * ones — and then coarse-to-fine, filling remaining unknown fine samples by
 * interpolating the coarser level. After both sweeps every sample of the
 * finest grid carries a velocity, so large semi-Lagrangian traces far from
 * the liquid read a crude but valid estimate instead of zero.
 *
 * Velocity samples travel as rgba32f with w = 1 marking "known", matching
 * the extrapolateVelocity kernel's convention in tall-cell-kernels.ts.
 * Coarse levels are tall-cell grids themselves: bases follow Eq 9 with the
 * terrain fixed at zero (ceil of the 2x2 maximum halved) and the packed band
 * halves per level, so coarse grids always cover the finer ones.
 */
export const tallCellExtrapolationShader = /* wgsl */ `
struct LevelParams {
  sourceDims: vec4f,       // packed dims of the source level (x, packedY, z, fineNy)
  destinationDims: vec4f,  // packed dims of the destination level (x, packedY, z, fineNy)
}
@group(0) @binding(0) var sourceVelocity: texture_3d<f32>;
@group(0) @binding(1) var sourceBase: texture_2d<f32>;
@group(0) @binding(2) var destinationBase: texture_2d<f32>;
@group(0) @binding(3) var destinationVelocity: texture_storage_3d<rgba32float, write>;
@group(0) @binding(4) var<uniform> level: LevelParams;
@group(0) @binding(5) var destinationBaseOut: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var destinationVelocityIn: texture_3d<f32>;

fn sourcePackedDims()->vec3i{return vec3i(vec3f(level.sourceDims.xyz));}
fn destinationPackedDims()->vec3i{return vec3i(vec3f(level.destinationDims.xyz));}
fn sourceFineY()->i32{return i32(level.sourceDims.w);}
fn destinationFineY()->i32{return i32(level.destinationDims.w);}
fn sourceBaseAt(x:i32,z:i32)->i32{let d=sourcePackedDims();if(x<0||x>=d.x||z<0||z>=d.z){return 0;}return i32(round(textureLoad(sourceBase,vec2i(x,z),0).x));}
fn destinationBaseAt(x:i32,z:i32)->i32{let d=destinationPackedDims();if(x<0||x>=d.x||z<0||z>=d.z){return 0;}return i32(round(textureLoad(destinationBase,vec2i(x,z),0).x));}
// World-cell lookup on the source level through the packed tall mapping.
// Velocity inside a tall cell follows the paper's Eq 5 linear interpolation
// between the endpoint dofs (matching validVelocityCell in the solver
// kernels) when both endpoints are known; otherwise the nearer endpoint owns
// the row so unknown data never contaminates the blend.
fn sourceSampleAt(x:i32,y:i32,z:i32)->vec4f{
  let d=sourcePackedDims();
  if(x<0||x>=d.x||z<0||z>=d.z||y<0||y>=sourceFineY()){return vec4f(0.0);}
  let base=sourceBaseAt(x,z);
  if(y<base&&base>0){
    let bottom=textureLoad(sourceVelocity,vec3i(x,0,z),0);
    let top=textureLoad(sourceVelocity,vec3i(x,1,z),0);
    if(bottom.w>0.5&&top.w>0.5){let t=clamp(f32(y)/f32(max(base-1,1)),0.0,1.0);return vec4f(mix(bottom.xyz,top.xyz,t),1.0);}
    if(y==base-1){return top;}
    return bottom;
  }
  let packedY=2+y-base;
  if(packedY<2||packedY>=d.y){return vec4f(0.0);}
  return textureLoad(sourceVelocity,vec3i(x,packedY,z),0);
}
fn destinationWorldY(id:vec3i)->i32{
  let base=destinationBaseAt(id.x,id.z);
  if(id.y==0){return 0;}
  if(id.y==1){return max(0,base-1);}
  return base+id.y-2;
}
fn destinationActive(id:vec3i)->bool{
  let base=destinationBaseAt(id.x,id.z);
  if(id.y<2){return base>0;}
  return base+id.y-2<destinationFineY();
}

// Eq 9 with zero terrain: the coarse tall height is the ceiling of half the
// maximum fine height over the 2x2 footprint, clamped so the coarse band can
// represent the remaining fine column.
@compute @workgroup_size(8,8,1)
fn downsampleExtrapolationBase(@builtin(global_invocation_id) gid:vec3u){
  let d=destinationPackedDims();
  if(gid.x>=u32(d.x)||gid.y>=u32(d.z)){return;}
  let x=i32(gid.x);let z=i32(gid.y);
  var maximum=0;
  for(var dz=0;dz<2;dz+=1){for(var dx=0;dx<2;dx+=1){maximum=max(maximum,sourceBaseAt(2*x+dx,2*z+dz));}}
  let layers=d.y-2;
  let base=clamp((maximum+1)/2,0,max(0,destinationFineY()-layers));
  textureStore(destinationBaseOut,vec2i(x,z),vec4f(f32(base)));
}

// Fine-to-coarse: a coarse sample is known when any of the eight fine world
// cells it covers is known; its velocity averages the known ones.
@compute @workgroup_size(4,4,4)
fn downsampleVelocity(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);let d=destinationPackedDims();
  if(any(id<vec3i(0))||any(id>=d)){return;}
  if(!destinationActive(id)){textureStore(destinationVelocity,id,vec4f(0.0));return;}
  let worldY=destinationWorldY(id);
  var sum=vec3f(0.0);var known=0.0;
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let sample=sourceSampleAt(2*id.x+dx,2*worldY+dy,2*id.z+dz);
    if(sample.w>0.5){sum+=sample.xyz;known+=1.0;}
  }}}
  if(known>0.0){textureStore(destinationVelocity,id,vec4f(sum/known,1.0));}
  else{textureStore(destinationVelocity,id,vec4f(0.0));}
}

// Coarse-to-fine: fill unknown destination samples by interpolating the
// coarser (source) level at the destination's world position, renormalizing
// over known coarse samples. Known samples pass through unchanged.
@compute @workgroup_size(4,4,4)
fn fillUnknownVelocity(@builtin(global_invocation_id) gid:vec3u){
  let id=vec3i(gid);let d=destinationPackedDims();
  if(any(id<vec3i(0))||any(id>=d)){return;}
  if(!destinationActive(id)){textureStore(destinationVelocity,id,vec4f(0.0));return;}
  let current=textureLoad(destinationVelocityIn,id,0);
  if(current.w>0.5){textureStore(destinationVelocity,id,current);return;}
  // Destination world position in coarse (source) cell coordinates: the
  // coarse spacing is twice the fine spacing.
  let position=vec3f(f32(id.x)+0.5,f32(destinationWorldY(id))+0.5,f32(id.z)+0.5)*0.5-vec3f(0.5);
  let cell=vec3i(floor(position));let f=fract(position);
  var sum=vec3f(0.0);var weight=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let sample=sourceSampleAt(cell.x+offset.x,cell.y+offset.y,cell.z+offset.z);
    if(sample.w<=0.5){continue;}
    let w=select(1.0-f.x,f.x,offset.x==1)*select(1.0-f.y,f.y,offset.y==1)*select(1.0-f.z,f.z,offset.z==1);
    sum+=sample.xyz*w;weight+=w;
  }
  if(weight>1e-8){textureStore(destinationVelocity,id,vec4f(sum/weight,1.0));}
  else{textureStore(destinationVelocity,id,vec4f(current.xyz,0.0));}
}
`;

interface ExtrapolationLevel {
  nx: number; packedNy: number; nz: number; fineNy: number;
  velocityA: GPUTexture; velocityB: GPUTexture; base: GPUTexture; params: GPUBuffer;
}

export class TallCellVelocityHierarchy {
  private readonly levels: ExtrapolationLevel[] = [];
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private downsampleBasePipeline: GPUComputePipeline;
  private downsamplePipeline: GPUComputePipeline;
  private fillPipeline: GPUComputePipeline;
  private downBaseGroups: GPUBindGroup[] = [];
  private downGroups: GPUBindGroup[] = [];
  private upGroups: Array<{ group: GPUBindGroup; level: ExtrapolationLevel }> = [];
  private fineFillGroup: GPUBindGroup;
  private readonly fineLevelDims: { nx: number; packedNy: number; nz: number };
  private readonly ownedBuffers: GPUBuffer[] = [];
  private readonly ownedTextures: GPUTexture[] = [];
  readonly allocatedBytes: number;

  /**
   * @param fineVelocityIn  the seeded finest-level field (neighbor-pass
   *                        output; w = 1 marks known samples)
   * @param fineVelocityOut the texture receiving the fully extrapolated
   *                        finest-level field
   * @param fineBase        the finest-level column bases
   */
  constructor(
    device: GPUDevice,
    layout: TallCellLayout,
    fineVelocityIn: GPUTexture,
    fineVelocityOut: GPUTexture,
    fineBase: GPUTexture
  ) {
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    let bytes = 0;
    let nx = layout.nx, nz = layout.nz, fineNy = layout.fineNy, layers = layout.packedNy - 2;
    // Paper Sec 3.3.1: L = log2 of the smallest extent; stop while every
    // extent stays at least two cells so trilinear stencils remain valid.
    while (Math.min(Math.floor(nx / 2), Math.floor(nz / 2), Math.floor(fineNy / 2)) >= 2 && this.levels.length < 8) {
      nx = Math.max(2, Math.floor(nx / 2)); nz = Math.max(2, Math.floor(nz / 2));
      fineNy = Math.max(2, Math.floor(fineNy / 2)); layers = Math.max(2, Math.floor(layers / 2));
      const packedNy = Math.min(fineNy, layers) + 2;
      const texture = (label: string) => device.createTexture({ label, size: [nx, packedNy, nz], dimension: "3d", format: "rgba32float", usage });
      const velocityA = texture(`Extrapolation velocity A L${this.levels.length + 1}`);
      const velocityB = texture(`Extrapolation velocity B L${this.levels.length + 1}`);
      const base = device.createTexture({ label: `Extrapolation base L${this.levels.length + 1}`, size: [nx, nz], format: "r32float", usage });
      const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.levels.push({ nx, packedNy, nz, fineNy, velocityA, velocityB, base, params });
      this.ownedTextures.push(velocityA, velocityB, base); this.ownedBuffers.push(params);
      bytes += nx * packedNy * nz * 32 + nx * nz * 4 + 32;
    }
    this.allocatedBytes = bytes;
    // Aliasing a texture as sampled and storage in one pass is invalid even
    // when the shader ignores a binding, so unused slots bind these dummies.
    const dummySampled3d = device.createTexture({ label: "Extrapolation dummy sampled", size: [1, 1, 1], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const dummyStorage3d = device.createTexture({ label: "Extrapolation dummy storage", size: [1, 1, 1], dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.STORAGE_BINDING });
    const dummySampled2d = device.createTexture({ label: "Extrapolation dummy sampled 2d", size: [1, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const dummyStorage2d = device.createTexture({ label: "Extrapolation dummy storage 2d", size: [1, 1], format: "r32float", usage: GPUTextureUsage.STORAGE_BINDING });
    this.ownedTextures.push(dummySampled3d, dummyStorage3d, dummySampled2d, dummyStorage2d);
    this.bindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    const module = device.createShaderModule({ label: "Tall-cell extrapolation hierarchy", code: tallCellExtrapolationShader });
    void module.getCompilationInfo().then((info) => {
      for (const message of info.messages) if (message.type === "error") console.error(`Extrapolation WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
    }).catch(() => { /* Device loss is handled by the renderer. */ });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    this.downsampleBasePipeline = pipeline("downsampleExtrapolationBase");
    this.downsamplePipeline = pipeline("downsampleVelocity");
    this.fillPipeline = pipeline("fillUnknownVelocity");

    const fineLevel = {
      nx: layout.nx, packedNy: layout.packedNy, nz: layout.nz, fineNy: layout.fineNy, base: fineBase
    };
    this.fineLevelDims = { nx: layout.nx, packedNy: layout.packedNy, nz: layout.nz };
    const dims = (level: { nx: number; packedNy: number; nz: number; fineNy: number }) => [level.nx, level.packedNy, level.nz, level.fineNy];
    const paramsFor = (source: { nx: number; packedNy: number; nz: number; fineNy: number }, destination: { nx: number; packedNy: number; nz: number; fineNy: number }) => {
      const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, new Float32Array([...dims(source), ...dims(destination)]));
      this.ownedBuffers.push(buffer);
      return buffer;
    };
    const group = (entries: { sourceVelocity: GPUTexture; sourceBase: GPUTexture; destinationBase: GPUTexture; velocityOut: GPUTexture; params: GPUBuffer; baseOut: GPUTexture; velocityInOwn: GPUTexture }) =>
      device.createBindGroup({ layout: this.bindGroupLayout, entries: [
        { binding: 0, resource: entries.sourceVelocity.createView() },
        { binding: 1, resource: entries.sourceBase.createView() },
        { binding: 2, resource: entries.destinationBase.createView() },
        { binding: 3, resource: entries.velocityOut.createView() },
        { binding: 4, resource: { buffer: entries.params } },
        { binding: 5, resource: entries.baseOut.createView() },
        { binding: 6, resource: entries.velocityInOwn.createView() }
      ] });
    // Down sweep (fine -> coarse): bases first, then known-velocity averages
    // into each level's A texture. The finest source is the seeded field.
    for (let index = 0; index < this.levels.length; index += 1) {
      const source = index === 0
        ? { ...fineLevel, velocityRead: fineVelocityIn }
        : { ...this.levels[index - 1], velocityRead: this.levels[index - 1].velocityA };
      const destination = this.levels[index];
      const shared = paramsFor(source, destination);
      this.downBaseGroups.push(group({ sourceVelocity: dummySampled3d, sourceBase: source.base, destinationBase: dummySampled2d, velocityOut: dummyStorage3d, params: shared, baseOut: destination.base, velocityInOwn: dummySampled3d }));
      this.downGroups.push(group({ sourceVelocity: source.velocityRead, sourceBase: source.base, destinationBase: destination.base, velocityOut: destination.velocityA, params: shared, baseOut: dummyStorage2d, velocityInOwn: dummySampled3d }));
    }
    // Up sweep (coarse -> fine): fill each level's unknowns from the next
    // coarser level into its B texture. The coarsest level is never filled
    // (its A texture is the source), and the finest fill reads the seeded
    // field and writes the caller's output texture.
    for (let index = this.levels.length - 2; index >= 0; index -= 1) {
      const source = this.levels[index + 1];
      const sourceRead = index + 1 === this.levels.length - 1 ? source.velocityA : source.velocityB;
      const destination = this.levels[index];
      this.upGroups.push({
        group: group({ sourceVelocity: sourceRead, sourceBase: source.base, destinationBase: destination.base, velocityOut: destination.velocityB, params: paramsFor(source, destination), baseOut: dummyStorage2d, velocityInOwn: destination.velocityA }),
        level: destination
      });
    }
    const fineSource = this.levels.length === 0
      ? { level: { ...fineLevel }, read: fineVelocityIn, base: fineBase }
      : { level: this.levels[0], read: this.levels.length === 1 ? this.levels[0].velocityA : this.levels[0].velocityB, base: this.levels[0].base };
    this.fineFillGroup = group({ sourceVelocity: fineSource.read, sourceBase: fineSource.base, destinationBase: fineBase, velocityOut: fineVelocityOut, params: paramsFor(fineSource.level, fineLevel), baseOut: dummyStorage2d, velocityInOwn: fineVelocityIn });
  }

  /** Encode the hierarchy sweeps. The finest seeded field must already be in
   * the constructor's fineVelocityIn texture; the fully extrapolated result
   * lands in fineVelocityOut (known samples pass through unchanged). */
  encode(encoder: GPUCommandEncoder) {
    for (let index = 0; index < this.levels.length; index += 1) {
      const level = this.levels[index];
      const bases = encoder.beginComputePass();
      bases.setPipeline(this.downsampleBasePipeline); bases.setBindGroup(0, this.downBaseGroups[index]);
      bases.dispatchWorkgroups(Math.ceil(level.nx / 8), Math.ceil(level.nz / 8), 1);
      bases.end();
      const sweep = encoder.beginComputePass();
      sweep.setPipeline(this.downsamplePipeline); sweep.setBindGroup(0, this.downGroups[index]);
      sweep.dispatchWorkgroups(Math.ceil(level.nx / 4), Math.ceil(level.packedNy / 4), Math.ceil(level.nz / 4));
      sweep.end();
    }
    for (const { group, level } of this.upGroups) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.fillPipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(level.nx / 4), Math.ceil(level.packedNy / 4), Math.ceil(level.nz / 4));
      pass.end();
    }
    const fine = encoder.beginComputePass();
    fine.setPipeline(this.fillPipeline); fine.setBindGroup(0, this.fineFillGroup);
    fine.dispatchWorkgroups(Math.ceil(this.fineLevelDims.nx / 4), Math.ceil(this.fineLevelDims.packedNy / 4), Math.ceil(this.fineLevelDims.nz / 4));
    fine.end();
  }

  destroy() {
    for (const texture of this.ownedTextures) texture.destroy();
    for (const buffer of this.ownedBuffers) buffer.destroy();
  }
}
