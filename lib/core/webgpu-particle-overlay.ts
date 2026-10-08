import { fieldVisualization, type Visualization } from "./visualization-registry";
import {
  LAYER_PALETTE, LAYER_PARTICLE_AGITATION_SCALE, LAYER_PARTICLE_MOTION_FLOOR, LAYER_PARTICLE_SPEED_SCALE, PARTICLE_LEGEND,
  type ParticleView,
} from "./visual-layers";
import { simpleWaterMurkLength_m, simpleWaterShadingWGSL, type SimpleWaterInterfaces } from "./simple-water-shading";

/**
 * Draws a particle method's own particles as shaded spheres over the frame.
 *
 * The surface render shows what the particles were reconstructed into; this
 * shows the samples themselves: where a narrow band keeps them and where it
 * has none, how evenly a cell is populated, where they bunch or thin out.
 *
 * Each sphere is an impostor: one camera-facing quad per particle whose
 * fragment stage rebuilds the sphere's normal and depth, so the spheres shade,
 * intersect and occlude one another through a depth buffer this pass owns.
 * Opaque scenery hides them through the dry scene's depth. The shaded water
 * does not: the particles are inside it and the liquid is what they are drawn
 * over. Simple water is a murky body, so there each sphere is fogged by the
 * liquid in front of it and takes the surface's own shading over the top, from
 * the interfaces the water composite just rastered. A sphere the body hides
 * outright is dropped in the vertex stage and costs no fragments.
 *
 * The Motion view paints energy as opacity instead of speed as colour: a calm
 * sample is nearly transparent, one at full scale white and opaque, so the
 * picture is of where the liquid would be white. Speed does not make water
 * white, and no whitewater model takes it to: Ihmsen et al. 2012 count the
 * velocity a sample has relative to its neighbours, Wretborn et al. 2022 the
 * Reynolds stress of its fluctuation about the local mean near the surface. A
 * FLIP record carries that fluctuation, its own velocity less the grid's, and
 * its energy is the measure: liquid falling, sliding or streaming as one body
 * has none however fast it goes, and liquid colliding, shearing or breaking
 * has. A sample no grid carries has no mean to differ from. Its measure is its
 * kinetic energy less what falling gives, the part of its velocity that runs
 * with gravity left out, so a droplet thrown sideways or upward lights up and
 * one that is only dropping does not. A faint calm grain must not hide a
 * lively one behind it, so those samples write no depth, and a solver reorders its records
 * every step, so they cannot be blended over one another either: the picture
 * would change with the order. Each adds its optical depth to an offscreen
 * target, and one composite draws the sum: as opaque as the depths together
 * make it, in their depth-weighted mean colour, whatever order they came in.
 * They are also not drawn as separate spheres, which strobe as a fast flow
 * carries the seed lattice across the pixels: each is a wide smooth footprint
 * that overlaps its neighbours', so a sheet of samples reads as a sheet.
 *
 * Records are read straight out of the solver's particle buffer. There is no
 * readback and no staging copy. A solver whose live count exists only on the
 * GPU publishes where that word is, and it is copied into this pass's indirect
 * draw arguments on the queue, so the draw covers exactly the live prefix.
 */

/** Where a solver keeps its particles, and how to read one record. */
export interface GPUFluidParticleSource {
  /**
   * Particle records, `strideFloats` f32 each: position in floats 0..2,
   * a positive float 3 while the record is live, velocity (m/s) in 4..6.
   */
  readonly buffer: GPUBuffer;
  readonly strideFloats: number;
  /** Records the buffer holds. The draw never reads past it. */
  readonly capacity: number;
  /** Metres per position unit. Positions are tank-local with a corner origin. */
  readonly positionScale_m: readonly [number, number, number];
  /** Sphere radius: a little under half the seed spacing. */
  readonly radius_m: number;
  /**
   * The live records are the prefix this u32 counts. Absent: every one of
   * `capacity` records may be live.
   */
  readonly liveCount?: { readonly buffer: GPUBuffer; readonly byteOffset: number };
  /**
   * Where a record also keeps the grid's side of the sample, as float offsets:
   * the grid's velocity at it (m/s, three floats), its signed depth in position
   * units (negative inside the liquid), and a float that is one while no grid
   * cell carries it. A sample counts in full down to `surfaceDepth`, in whole
   * position units, and fades out over as much again. Absent: no sample has a
   * grid to differ from.
   */
  readonly grid?: {
    readonly velocityFloat: number; readonly depthFloat: number; readonly ballisticFloat: number;
    readonly surfaceDepth: number;
  };
}

export const PARTICLE_OVERLAY_UNIFORM_BYTES = 176;
/** Sphere radii a Motion sample's footprint spans; the shader says why. */
const PARTICLE_MOTION_REACH = 3;
/** The Motion view's sum: summed by the blend, so a float format that blends. */
const PARTICLE_MOTION_FORMAT: GPUTextureFormat = "rgba16float";

export interface ParticleOverlayCamera {
  readonly position_m: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly up: readonly [number, number, number];
  /** tan(verticalFieldOfView / 2). */
  readonly tanHalfFov: number;
  readonly aspect: number;
}

export interface ParticleOverlayFrame {
  readonly camera: ParticleOverlayCamera;
  readonly viewportWidth: number;
  readonly viewportHeight: number;
  readonly container_m: readonly [number, number, number];
  /** Reversed-Z near plane the scene depth was written with. */
  readonly depthNear_m: number;
  /** 1 draws opaque spheres. */
  readonly opacity?: number;
  /** Absent draws the speed ramp. */
  readonly view?: ParticleView;
  /** The Motion view leaves falling out of a ballistic sample's energy. Absent: it counts. */
  readonly gravity_m_s2?: readonly [number, number, number];
  /** Present when Simple water was composited into the target this frame. */
  readonly water?: SimpleWaterInterfaces;
}

const paletteWGSL = (name: string, rgb: readonly number[]) =>
  `const ${name}:vec3f=vec3f(${rgb.map(n => (n / 255).toFixed(8)).join(",")});`;

const particleOverlayUniformsWGSL = /* wgsl */ `
struct ParticleOverlayUniforms {
  // xyz camera position, w tan(halfFov)
  cameraPosition:vec4f,
  // xyz camera forward, w aspect
  cameraForward:vec4f,
  // xyz camera right, w opacity
  cameraRight:vec4f,
  // xyz camera up, w sphere radius in metres
  cameraUp:vec4f,
  // xy viewport pixels, z reversed-Z near metres, w scene depth valid
  viewport:vec4f,
  // xyz metres per position unit, w full-scale speed in m/s
  scale:vec4f,
  // xyz world position of the tank's minimum corner
  origin:vec4f,
  // x Simple water's murk length in metres, zero when there is none to be under
  water:vec4f,
  // xyz the unit direction of gravity, zero where there is none, w full-scale fluctuation in m/s
  gravity:vec4f,
  // x record stride in floats, y record capacity, z one in the Motion view
  records:vec4u,
  // Float offsets into a record: x the grid's velocity, y depth, z the ballistic
  // flag; w the depth a sample counts to in full, zero where records keep none
  grid:vec4u,
}
`;

export const particleOverlayShader = /* wgsl */ `
${particleOverlayUniformsWGSL}
@group(0) @binding(0) var<uniform> overlay:ParticleOverlayUniforms;
@group(0) @binding(1) var<storage, read> particles:array<f32>;
@group(0) @binding(2) var sceneDepth:texture_depth_2d;
@group(0) @binding(3) var waterFront:texture_2d<f32>;
@group(0) @binding(4) var waterBack:texture_2d<f32>;
@group(0) @binding(5) var waterRearFront:texture_2d<f32>;
@group(0) @binding(6) var waterRearBack:texture_2d<f32>;
@group(0) @binding(7) var waterFrontNormal:texture_2d<f32>;

struct VertexOut {
  @builtin(position) position:vec4f,
  // rgb the sphere's colour, a its opacity before the layer's own.
  @location(0) @interpolate(flat) color:vec4f,
  // Unit-disc coordinate across the quad.
  @location(1) disc:vec2f,
  // x view depth of the sphere centre, y drawn radius, both in metres.
  @location(2) @interpolate(flat) sphere:vec2f,
  // x liquid in front of the sphere centre in metres, y one behind a Simple surface.
  @location(3) @interpolate(flat) water:vec2f,
}

struct FragmentOut {
  @location(0) color:vec4f,
  @builtin(frag_depth) depth:f32,
}

${paletteWGSL("PARTICLE_SLOW", LAYER_PALETTE.particleSlow)}
${paletteWGSL("PARTICLE_MID", LAYER_PALETTE.particleMid)}
${paletteWGSL("PARTICLE_FAST", LAYER_PALETTE.particleFast)}
${paletteWGSL("PARTICLE_STILL", LAYER_PALETTE.particleStill)}
${paletteWGSL("PARTICLE_RUSHING", LAYER_PALETTE.particleRushing)}
const PARTICLE_MOTION_FLOOR:f32=${LAYER_PARTICLE_MOTION_FLOOR};
// The most one Motion sample can cover: its optical depth has to stay finite.
const PARTICLE_MOTION_OPAQUE:f32=0.98;
// A Motion sample is drawn this many sphere radii wide. A sphere's radius is a
// little under half the seed spacing, so this reaches past the neighbouring samples.
const PARTICLE_MOTION_REACH:f32=${PARTICLE_MOTION_REACH};
// A sphere smaller than this would alias into crawling noise, so a distant
// cloud is drawn slightly fat instead.
const PARTICLE_MINIMUM_PIXELS:f32=0.75;
// Murk lengths of liquid past which the least absorbed channel passes under
// 1.5%: the sphere would change no pixel, so it is not drawn.
const PARTICLE_HIDDEN_MURK_LENGTHS:f32=7.0;
${simpleWaterShadingWGSL}

// The Motion view's measure of the record at a float offset, zero to one.
fn particleAgitation(at:u32,velocity:vec3f)->f32 {
  if (overlay.grid.w>0u && particles[at+overlay.grid.z]!=1.0) {
    // The energy of the sample's velocity about the grid's, which is the mean
    // of the samples around it. Entrainment happens at the surface, so a
    // sample counts less the deeper it sits.
    let mean=vec3f(particles[at+overlay.grid.x],particles[at+overlay.grid.x+1u],particles[at+overlay.grid.x+2u]);
    let fluctuation=velocity-mean;
    let full=max(overlay.gravity.w,1e-6);
    let reach=f32(overlay.grid.w);
    let surface=clamp(2.0+particles[at+overlay.grid.y]/reach,0.0,1.0);
    return clamp(dot(fluctuation,fluctuation)/(full*full),0.0,1.0)*surface;
  }
  // Kinetic energy as a fraction of the full-scale speed's, without the
  // speed a sample has along gravity: that much it got by dropping.
  let scale=max(overlay.scale.w,1e-6);
  let fall=max(dot(velocity,overlay.gravity.xyz),0.0);
  return clamp((dot(velocity,velocity)-fall*fall)/(scale*scale),0.0,1.0);
}

fn particleColor(at:u32,velocity:vec3f)->vec4f {
  let scale=max(overlay.scale.w,1e-6);
  if (overlay.records.z==1u) {
    let t=particleAgitation(at,velocity);
    return vec4f(mix(PARTICLE_STILL,PARTICLE_RUSHING,t),mix(PARTICLE_MOTION_FLOOR,1.0,t));
  }
  let t=clamp(length(velocity)/scale,0.0,1.0);
  return vec4f(mix(mix(PARTICLE_SLOW,PARTICLE_MID,clamp(2.0*t,0.0,1.0)),PARTICLE_FAST,clamp(2.0*t-1.0,0.0,1.0)),1.0);
}

// Distance from the eye to an interface texel, negative where none was drawn.
fn waterInterfaceDistance(positions:texture_2d<f32>,pixel:vec2i)->f32 {
  let texel=textureLoad(positions,pixel,0);
  return select(-1.0,length(texel.xyz-overlay.cameraPosition.xyz),texel.a>=0.5);
}

@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex:u32,
  @builtin(instance_index) instance:u32,
)->VertexOut {
  var corners=array<vec2f,4>(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(-1.0,1.0),vec2f(1.0,1.0));
  var output:VertexOut;
  output.color=vec4f(0.0);
  output.disc=corners[vertexIndex];
  output.sphere=vec2f(0.0);
  output.water=vec2f(0.0);
  // A collapsed quad: a retired record, one past the buffer, or a hidden sphere.
  output.position=vec4f(0.0,0.0,0.0,1.0);
  if (instance>=overlay.records.y) { return output; }
  let at=overlay.records.x*instance;
  if (!(particles[at+3u]>0.0)) { return output; }
  let local=vec3f(particles[at],particles[at+1u],particles[at+2u]);
  let velocity=vec3f(particles[at+4u],particles[at+5u],particles[at+6u]);
  let world=local*overlay.scale.xyz+overlay.origin.xyz;
  let relative=world-overlay.cameraPosition.xyz;
  let view=vec3f(dot(relative,overlay.cameraRight.xyz),
    dot(relative,overlay.cameraUp.xyz),dot(relative,overlay.cameraForward.xyz));
  let near=overlay.viewport.z;
  let tangent=max(overlay.cameraPosition.w,1e-4);
  let viewport=max(overlay.viewport.xy,vec2f(1.0));
  let pixelsPerMetre=0.5*viewport.y/(max(view.z,near)*tangent);
  let reach=select(1.0,PARTICLE_MOTION_REACH,overlay.records.z==1u);
  let radius=max(overlay.cameraUp.w*reach,PARTICLE_MINIMUM_PIXELS/pixelsPerMetre);
  // The whole sphere stays in front of the near plane, so its depth is valid.
  if (!(view.z-radius>near)) { return output; }
  let perspective=1.0/(view.z*tangent);
  let aspect=max(overlay.cameraForward.w,1e-4);
  if (overlay.water.x>0.0) {
    // One lookup per sphere, at its centre: the liquid in front of it barely
    // changes across a footprint that is a few pixels wide.
    let uv=vec2f(0.5+0.5*view.x*perspective/aspect,0.5-0.5*view.y*perspective);
    let pixel=vec2i(clamp(uv,vec2f(0.0),vec2f(0.9999))*vec2f(textureDimensions(waterFront)));
    let front=waterInterfaceDistance(waterFront,pixel);
    let centre=length(relative);
    if (front>0.0 && centre>front) {
      let back=waterInterfaceDistance(waterBack,pixel);
      var path=centre-front;
      // Past the nearest interval's exit: the gap behind it is air.
      if (back>front && centre>back) {
        path=simpleWaterPath(centre,front,back,waterInterfaceDistance(waterRearFront,pixel),
          waterInterfaceDistance(waterRearBack,pixel));
      }
      if (path-radius>PARTICLE_HIDDEN_MURK_LENGTHS*overlay.water.x) { return output; }
      output.water=vec2f(path,1.0);
    }
  }
  let corner=view.xy+corners[vertexIndex]*radius;
  output.position=vec4f(corner.x*perspective/aspect,corner.y*perspective,0.0,1.0);
  output.color=particleColor(at,velocity);
  output.sphere=vec2f(view.z,radius);
  return output;
}

struct SphereSample {
  color:vec3f,
  // Cosine between the sphere's normal and the eye at this fragment.
  towardCamera:f32,
  viewDepth_m:f32,
}

// One fragment of a sphere, discarded off its disc and behind the scenery.
fn sphereSample(input:VertexOut)->SphereSample {
  let radial=dot(input.disc,input.disc);
  if (radial>1.0 || input.sphere.y<=0.0) { discard; }
  let towardCamera=sqrt(1.0-radial);
  let viewDepth_m=input.sphere.x-input.sphere.y*towardCamera;
  let near=overlay.viewport.z;
  if (overlay.viewport.w>0.5) {
    // Reversed-Z: depth = near / viewDepth, and zero means nothing was drawn.
    let stored=textureLoad(sceneDepth,vec2u(input.position.xy),0);
    if (stored>0.0 && viewDepth_m>near/stored) { discard; }
  }
  let forward=overlay.cameraForward.xyz;
  let normal=normalize(overlay.cameraRight.xyz*input.disc.x+overlay.cameraUp.xyz*input.disc.y
    -forward*towardCamera);
  // The water's own key, wrapped past the terminator, under a sky that is
  // brighter than the bounce: the lit side says which way is up and the unlit
  // side still reads as round. The rim is darkened as the neighbours of a
  // packed sphere would shade it, which is what separates one grain from the
  // next when each is a few pixels wide.
  let wrap=0.5+0.5*dot(normal,SIMPLE_WATER_KEY);
  let light=(mix(0.30,0.50,0.5+0.5*normal.y)+0.62*wrap*wrap)*(0.66+0.34*towardCamera);
  let highlight=pow(max(dot(normal,normalize(SIMPLE_WATER_KEY-forward)),0.0),40.0);
  var color=input.color.rgb*light+vec3f(0.20*highlight);
  // Motion spheres are unlit, so a fast one is white on every side.
  if (overlay.records.z==1u) { color=input.color.rgb; }
  // The cap of a sphere breaking the surface stays dry; the rest goes under.
  let travel=input.water.x-input.sphere.y*towardCamera;
  if (input.water.y>0.5 && travel>0.0) {
    let viewport=max(overlay.viewport.xy,vec2f(1.0));
    let screen=input.position.xy/viewport;
    let surface=textureLoad(waterFrontNormal,vec2u(screen*vec2f(textureDimensions(waterFrontNormal))),0);
    let tangent=overlay.cameraPosition.w;
    let rd=normalize(forward+overlay.cameraRight.xyz*((2.0*screen.x-1.0)*overlay.cameraForward.w*tangent)
      +overlay.cameraUp.xyz*((1.0-2.0*screen.y)*tangent));
    var facing=-rd;
    if (surface.a>1e-4 && dot(surface.xyz,surface.xyz)>1e-8) { facing=normalize(surface.xyz); }
    if (dot(facing,rd)>0.0) { facing=-facing; }
    color=mix(color,simpleWaterOver(color,travel,facing,rd,overlay.water.x),
      clamp(2.0*travel/input.sphere.y,0.0,1.0));
  }
  return SphereSample(color,towardCamera,viewDepth_m);
}

@fragment fn fragmentMain(input:VertexOut)->FragmentOut {
  let sphere=sphereSample(input);
  let near=overlay.viewport.z;
  var output:FragmentOut;
  output.color=vec4f(sphere.color,input.color.a*clamp(overlay.cameraRight.w,0.0,1.0));
  output.depth=clamp(near/max(sphere.viewDepth_m,near),0.0,1.0);
  return output;
}

// The Motion view's accumulation, summed by the blend: rgb the colour weighted
// by the fragment's optical depth, a that depth.
//
// A sample is not drawn as its sphere here. Samples are seeded on a lattice and
// a fast flow carries that lattice a good part of its spacing every frame, so
// a picture of separate dots strobes: it alternates between the lattice and
// its shifted copy. Each sample is instead a smooth footprint that falls to
// nothing at its neighbours, which a lattice of them sums to an even sheet, and
// it carries the optical depth its sphere would have, spread over that width.
@fragment fn fragmentMotion(input:VertexOut)->@location(0) vec4f {
  let sphere=sphereSample(input);
  let footprint=sphere.towardCamera*sphere.towardCamera;
  let depth=-log(1.0-min(input.color.a,PARTICLE_MOTION_OPAQUE))
    *(2.0/(PARTICLE_MOTION_REACH*PARTICLE_MOTION_REACH))*footprint*footprint;
  return vec4f(sphere.color*depth,depth);
}
`;

/** Draws the Motion view's accumulated optical depth over the frame. */
export const particleMotionCompositeShader = /* wgsl */ `
${particleOverlayUniformsWGSL}
@group(0) @binding(0) var<uniform> overlay:ParticleOverlayUniforms;
@group(0) @binding(1) var accumulated:texture_2d<f32>;

@vertex fn vertexMain(@builtin(vertex_index) vertexIndex:u32)->@builtin(position) vec4f {
  var corners=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
  return vec4f(corners[vertexIndex],0.0,1.0);
}

@fragment fn fragmentMain(@builtin(position) position:vec4f)->@location(0) vec4f {
  let sum=textureLoad(accumulated,vec2u(position.xy),0);
  if (!(sum.a>0.0)) { discard; }
  return vec4f(sum.rgb/sum.a,(1.0-exp(-sum.a))*clamp(overlay.cameraRight.w,0.0,1.0));
}
`;

/** One instanced draw over the solver's resident particle records. */
export class ParticleOverlay {
  private pipeline?: GPURenderPipeline;
  /** The Motion view's two: spheres summed into `motion`, and that sum drawn over the frame. */
  private motionPipeline?: GPURenderPipeline;
  private compositePipeline?: GPURenderPipeline;
  private compositeLayout?: GPUBindGroupLayout;
  private motion?: GPUTexture;
  private motionView?: GPUTextureView;
  private compositeGroup?: GPUBindGroup;
  private layout?: GPUBindGroupLayout;
  private uniforms?: GPUBuffer;
  private indirect?: GPUBuffer;
  private fallbackDepth?: GPUTexture;
  private fallbackDepthView?: GPUTextureView;
  private fallbackWater?: GPUTexture;
  private fallbackWaterViews?: SimpleWaterInterfaces;
  private depth?: GPUTexture;
  private depthView?: GPUTextureView;
  /** One bind group per particle buffer: a solver may alternate two. */
  private readonly bindGroups = new Map<GPUBuffer, GPUBindGroup>();
  private boundSceneDepth?: GPUTextureView;
  private boundWater?: SimpleWaterInterfaces;
  private source?: GPUFluidParticleSource;
  private destroyed = false;
  private readonly uniformData = new ArrayBuffer(PARTICLE_OVERLAY_UNIFORM_BYTES);
  private readonly uniformF32 = new Float32Array(this.uniformData);
  private readonly uniformU32 = new Uint32Array(this.uniformData);

  constructor(
    private readonly device: GPUDevice,
    private readonly targetFormat: GPUTextureFormat,
  ) {}

  async initialize(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: "Fluid particle overlay", code: particleOverlayShader,
    });
    const compositeModule = this.device.createShaderModule({
      label: "Fluid particle overlay motion composite", code: particleMotionCompositeShader,
    });
    for (const module of [shaderModule, compositeModule]) {
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === "error");
      if (errors.length > 0) {
        throw new Error(`${module.label}:\n${errors
          .map((error) => `${error.lineNum}:${error.linePos} ${error.message}`)
          .join("\n")}`);
      }
    }
    this.uniforms = this.device.createBuffer({
      label: "Fluid particle overlay uniforms",
      size: PARTICLE_OVERLAY_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // vertexCount, instanceCount, firstVertex, firstInstance. The instance
    // count is the only word a frame rewrites.
    this.indirect = this.device.createBuffer({
      label: "Fluid particle overlay draw arguments",
      size: 16, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.indirect, 0, new Uint32Array([4, 0, 0, 0]));
    this.fallbackDepth = this.device.createTexture({
      label: "Fluid particle overlay scene depth fallback",
      size: [1, 1], format: "depth32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.fallbackDepthView = this.fallbackDepth.createView();
    // Bound while there is no Simple water; the murk length is zero then and
    // no stage reads it.
    this.fallbackWater = this.device.createTexture({
      label: "Fluid particle overlay water fallback",
      size: [1, 1], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    const fallbackWaterView = this.fallbackWater.createView();
    this.fallbackWaterViews = {
      frontPosition: fallbackWaterView, frontNormal: fallbackWaterView, backPosition: fallbackWaterView,
      rearFrontPosition: fallbackWaterView, rearBackPosition: fallbackWaterView,
    };
    const interfaceTexture = { sampleType: "unfilterable-float" } as const;
    this.layout = this.device.createBindGroupLayout({
      label: "Fluid particle overlay bindings",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "depth" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, texture: interfaceTexture },
        { binding: 4, visibility: GPUShaderStage.VERTEX, texture: interfaceTexture },
        { binding: 5, visibility: GPUShaderStage.VERTEX, texture: interfaceTexture },
        { binding: 6, visibility: GPUShaderStage.VERTEX, texture: interfaceTexture },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: interfaceTexture },
      ],
    });
    this.compositeLayout = this.device.createBindGroupLayout({
      label: "Fluid particle overlay motion composite bindings",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: interfaceTexture },
      ],
    });
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const over: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    const sum: GPUBlendComponent = { srcFactor: "one", dstFactor: "one" };
    [this.pipeline, this.motionPipeline, this.compositePipeline] = await Promise.all([
      this.device.createRenderPipelineAsync({
        label: "Fluid particle overlay", layout,
        vertex: { module: shaderModule, entryPoint: "vertexMain" },
        fragment: { module: shaderModule, entryPoint: "fragmentMain", targets: [{ format: this.targetFormat, blend: over }] },
        primitive: { topology: "triangle-strip" },
        // Reversed-Z, as the scene depth is: nearer is greater, and the clear is 0.
        depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "greater" },
      }),
      // No depth: every sphere reaches the sum. Scenery still hides them: that
      // test is the fragment stage's own, against the scene depth.
      this.device.createRenderPipelineAsync({
        label: "Fluid particle overlay motion", layout,
        vertex: { module: shaderModule, entryPoint: "vertexMain" },
        fragment: { module: shaderModule, entryPoint: "fragmentMotion", targets: [{ format: PARTICLE_MOTION_FORMAT, blend: { color: sum, alpha: sum } }] },
        primitive: { topology: "triangle-strip" },
      }),
      this.device.createRenderPipelineAsync({
        label: "Fluid particle overlay motion composite",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] }),
        vertex: { module: compositeModule, entryPoint: "vertexMain" },
        fragment: { module: compositeModule, entryPoint: "fragmentMain", targets: [{ format: this.targetFormat, blend: over }] },
      }),
    ]);
  }

  get ready(): boolean {
    return Boolean(this.pipeline && this.motionPipeline && this.compositePipeline && this.uniforms && this.layout && this.indirect);
  }

  setSource(source: GPUFluidParticleSource | undefined): void {
    this.source = source;
  }

  private bindGroupFor(
    buffer: GPUBuffer, sceneDepth: GPUTextureView | undefined, frameWater: SimpleWaterInterfaces | undefined,
  ): GPUBindGroup {
    const view = sceneDepth ?? this.fallbackDepthView!;
    const water = frameWater ?? this.fallbackWaterViews!;
    if (view !== this.boundSceneDepth || water !== this.boundWater) {
      this.bindGroups.clear(); this.boundSceneDepth = view; this.boundWater = water;
    }
    let group = this.bindGroups.get(buffer);
    if (!group) {
      // A third buffer means a new solver: the cached two are its predecessor's.
      if (this.bindGroups.size >= 2) this.bindGroups.clear();
      group = this.device.createBindGroup({
        label: "Fluid particle overlay bind group",
        layout: this.layout!,
        entries: [
          { binding: 0, resource: { buffer: this.uniforms! } },
          { binding: 1, resource: { buffer } },
          { binding: 2, resource: view },
          { binding: 3, resource: water.frontPosition },
          { binding: 4, resource: water.backPosition },
          { binding: 5, resource: water.rearFrontPosition },
          { binding: 6, resource: water.rearBackPosition },
          { binding: 7, resource: water.frontNormal },
        ],
      });
      this.bindGroups.set(buffer, group);
    }
    return group;
  }

  private depthViewFor(width: number, height: number): GPUTextureView {
    if (!this.depth || this.depth.width !== width || this.depth.height !== height) {
      this.depth?.destroy();
      this.depth = this.device.createTexture({
        label: "Fluid particle overlay depth",
        size: [width, height], format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.depthView = this.depth.createView();
    }
    return this.depthView!;
  }

  private motionViewFor(width: number, height: number): GPUTextureView {
    if (!this.motion || this.motion.width !== width || this.motion.height !== height) {
      this.motion?.destroy();
      this.motion = this.device.createTexture({
        label: "Fluid particle overlay motion sum",
        size: [width, height], format: PARTICLE_MOTION_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.motionView = this.motion.createView();
      this.compositeGroup = this.device.createBindGroup({
        label: "Fluid particle overlay motion composite bind group",
        layout: this.compositeLayout!,
        entries: [
          { binding: 0, resource: { buffer: this.uniforms! } },
          { binding: 1, resource: this.motionView },
        ],
      });
    }
    return this.motionView!;
  }

  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    sceneDepth: GPUTextureView | undefined,
    frame: ParticleOverlayFrame,
  ): boolean {
    const source = this.source;
    if (!this.ready || !source || source.capacity === 0) return false;
    const { camera } = frame;
    const width = Math.max(1, Math.round(frame.viewportWidth));
    const height = Math.max(1, Math.round(frame.viewportHeight));
    const f = this.uniformF32, u = this.uniformU32;
    f.set([...camera.position_m, Math.max(1e-4, camera.tanHalfFov)], 0);
    f.set([...camera.forward, Math.max(1e-4, camera.aspect)], 4);
    f.set([...camera.right, Math.max(0, Math.min(1, frame.opacity ?? 1))], 8);
    f.set([...camera.up, source.radius_m], 12);
    f.set([width, height, Math.max(1e-6, frame.depthNear_m), sceneDepth ? 1 : 0], 16);
    f.set([...source.positionScale_m, LAYER_PARTICLE_SPEED_SCALE], 20);
    f.set([-0.5 * frame.container_m[0], 0, -0.5 * frame.container_m[2], 0], 24);
    f.set([frame.water ? simpleWaterMurkLength_m(frame.container_m) : 0, 0, 0, 0], 28);
    const gravity = frame.gravity_m_s2 ?? [0, 0, 0], pull = Math.hypot(...gravity) || 1;
    f.set([gravity[0] / pull, gravity[1] / pull, gravity[2] / pull, LAYER_PARTICLE_AGITATION_SCALE], 32);
    const motion = frame.view === "motion", grid = source.grid;
    u.set([source.strideFloats, source.capacity, motion ? 1 : 0, 0], 36);
    u.set(grid ? [grid.velocityFloat, grid.depthFloat, grid.ballisticFloat, grid.surfaceDepth] : [0, 0, 0, 0], 40);
    this.device.queue.writeBuffer(this.uniforms!, 0, this.uniformData);
    if (source.liveCount) {
      encoder.copyBufferToBuffer(source.liveCount.buffer, source.liveCount.byteOffset, this.indirect!, 4, 4);
    }
    const pass = encoder.beginRenderPass(motion ? {
      label: "Fluid particle overlay motion",
      colorAttachments: [{
        view: this.motionViewFor(width, height), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store",
      }],
    } : {
      label: "Fluid particle overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: {
        view: this.depthViewFor(width, height),
        depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "discard",
      },
    });
    pass.setPipeline(motion ? this.motionPipeline! : this.pipeline!);
    pass.setBindGroup(0, this.bindGroupFor(source.buffer, sceneDepth, frame.water));
    if (source.liveCount) pass.drawIndirect(this.indirect!, 0);
    else pass.draw(4, source.capacity);
    pass.end();
    if (motion) {
      const composite = encoder.beginRenderPass({
        label: "Fluid particle overlay motion composite",
        colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
      });
      composite.setPipeline(this.compositePipeline!);
      composite.setBindGroup(0, this.compositeGroup!);
      composite.draw(3);
      composite.end();
    }
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniforms?.destroy();
    this.indirect?.destroy();
    this.fallbackDepth?.destroy();
    this.fallbackWater?.destroy();
    this.depth?.destroy();
    this.motion?.destroy();
    this.bindGroups.clear();
    this.uniforms = undefined;
    this.indirect = undefined;
    this.fallbackDepth = undefined;
    this.fallbackWater = undefined;
    this.fallbackWaterViews = undefined;
    this.boundWater = undefined;
    this.depth = undefined;
    this.motion = undefined;
    this.motionView = undefined;
    this.compositeGroup = undefined;
    this.pipeline = undefined;
    this.motionPipeline = undefined;
    this.compositePipeline = undefined;
    this.source = undefined;
  }
}

/** The same spheres as a single field view, for methods on the one-view row. */
export const particleOverlayVisualizations: readonly Visualization[] = Object.freeze([
  fieldVisualization({
    kind: "field", id: "fluid-particles/spheres", pass: "Fluid particles",
    label: "Particles",
    description: "The method's own particles as shaded spheres, coloured by speed. Shows the samples the surface is reconstructed from: how evenly cells are populated and where particles bunch or thin out.",
    source: "Solver-resident particle records, drawn on the GPU without readback",
    mode: "particles", axis: "volume", planeless: true, icon: "tracers",
    swatch: "#8f8cdb",
    legend: PARTICLE_LEGEND.map(entry => ({ swatch: entry.color, mark: "point" as const, label: entry.label })),
  }),
]);
