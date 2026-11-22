/**
 * The brick engine's detail engines: regions of a scene run again at h / 2 or h / 4 by a nested brick engine, driven
 * by the parent at the region's boundary (brick-detail.wgsl.ts's relaxation shell), so a scene keeps its resolution
 * and gains particle-scale fidelity where it matters (the authored detail regions and the placement heuristic,
 * brick-detail-regions.ts).
 *
 * - brickDetailSpecs: each detail box's fine scene (brick-detail-scene.ts), its setup (the ordinary scene adapter at
 *   h / factor) and its coupling layout (BrickLayoutOptions.detail);
 * - BrickDetailExchange: the parent's state on the box's shell, written into the fine engine's `src` after every
 *   parent step (one dispatch, a node a lane: nbVertexPhi and nbFaceU, the parent's owner readers, at the parent's
 *   vertices about the box's faces).
 * The order a step (BrickNBFlipSolver.advanceTo): the parent's step, the exchange, the fine step. The fine engine
 * relaxes toward the parent's state at the step's end (t + dt) on its way from t: the shell is a boundary condition
 * taken at the new time, as an inflow's prescribed velocity is. One way: the parent runs as if the detail engine were
 * not there, and the renderer draws the fine surface inside the box in place of the parent's.
 */
import type { SceneDescription } from "../../core/model";
import { fluidLatticeFarPhi } from "../../core/initial-liquid-lattice";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { BrickLayout, boxWGSL, brickAbiWGSL, resolveBox } from "./brick-abi";
import { brickDetailBoxes, type DetailBox } from "./brick-detail-regions";
import { brickDetailFrame, brickDetailSceneGaps, brickDetailSubScene } from "./brick-detail-scene";
import { DETAIL_AT, DETAIL_DEPTH, brickDetailNodes, type BrickDetailLayout } from "./brick-detail.wgsl";
import { BrickEngine, type Vec3 } from "./brick-engine";
import { BrickPresentation } from "./brick-present";
import { BrickDetailSurface } from "./brick-detail-surface";
import type { PressureSolverFactory } from "./brick-pressure-slot";
import type { BrickStagesFactory } from "./brick-stages";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { brickStaticCutBricks } from "./brick-static.wgsl";
import { solidWorldForFluidLattice } from "../../core/solid-world";
import { brickInitialExtent, brickSceneSetup, type BrickSceneSetup } from "./brick-scene";
import { FH, VOL_UNIT } from "./velocity-kernels.wgsl";

export interface BrickDetailSpec {
  readonly box: DetailBox;
  /** The fine engine's setup (h / factor, the box's own scene). */
  readonly setup: BrickSceneSetup;
  readonly layout: BrickDetailLayout;
  /** The box's low corner in the parent's world (m): the fine lattice's origin there. */
  readonly low_m: readonly [number, number, number];
  readonly farPhi_m: number;
}

/** The coupled faces of a box (bit 2a + side): every face off the parent's container cells. */
export function detailCoupledFaces(box: DetailBox, cells: Vec3): number {
  let mask = 0;
  for (let a = 0; a < 3; a++) {
    if (box.lo[a]! > 0) mask |= 1 << (2 * a);
    if (box.hi[a]! < cells[a]!) mask |= 1 << (2 * a + 1);
  }
  return mask;
}

/**
 * Every detail engine of the scene, or [] with none; an explicit refusal (throws) when the scene asks for detail the
 * engine cannot run (brickDetailSceneGaps, a fine setup's gaps).
 */
export function brickDetailSpecs(scene: SceneDescription, setup: BrickSceneSetup, values: Readonly<Record<string, unknown>>): BrickDetailSpec[] {
  const cells = sceneLatticeDimensions(scene) as unknown as Vec3;
  const extent = brickInitialExtent({ lattice: setup.lattice, ...(setup.box ? { box: setup.box } : {}), ...(setup.farPhi !== undefined ? { farPhi: setup.farPhi } : {}), liquid: setup.liquid, ...(setup.initial ? { initial: setup.initial } : {}) });
  const boxes = brickDetailBoxes(scene, cells, { top: extent.top, rest: extent.cells / (cells[0] * cells[2]), sources: setup.sources !== undefined });
  return boxes.map(box => {
    const gaps = brickDetailSceneGaps(scene, box);
    if (gaps.length) throw new Error(`NB-FLIP brick engine cannot run scene "${scene.sceneId ?? "(unnamed)"}"'s detail: ${gaps.join("; ")}`);
    const sub = brickDetailSubScene(scene, box);
    const fine = brickSceneSetup(sub, values);
    if (!fine.setup || fine.gaps.length) throw new Error(`NB-FLIP detail engine ${box.origins.join("+")}: ${fine.gaps.join("; ")}`);
    const nodes = [0, 1, 2].map(a => box.hi[a]! - box.lo[a]! + 3) as unknown as [number, number, number];
    const layout: BrickDetailLayout = { factor: box.factor, nodes, coupled: detailCoupledFaces(box, cells) };
    brickDetailNodes(layout);
    return { box, setup: fine.setup, layout, low_m: brickDetailFrame(scene, box).low_m, farPhi_m: fluidLatticeFarPhi(sub) };
  });
}

/** The exchange's WGSL on the parent's layout (exported for the tests). */
export function brickDetailExchangeWGSL(parent: BrickLayout, spec: Pick<BrickDetailSpec, "box" | "layout">, step: { readonly dt_s: number; readonly h_m: number }): string {
  const n = spec.layout.nodes, m = brickDetailNodes(spec.layout), c = spec.layout.coupled;
  const bits = (side: number) => `vec3<bool>(${[0, 1, 2].map(a => ((c >> (2 * a + side)) & 1) !== 0).join(",")})`;
  return resolveBox(parent, /* wgsl */`
${brickAbiWGSL(parent, { set: { name: "nbSet", atomic: false }, dir: { name: "dir", atomic: false } })}
@group(0) @binding(0) var<storage,read> nbSet:array<u32>;
@group(0) @binding(1) var<storage,read> dir:array<u32>;
@group(0) @binding(2) var<storage,read> phi:array<u32>;
@group(0) @binding(3) var<storage,read> vel:array<f32>;
@group(0) @binding(4) var<storage,read_write> src:array<f32>;
@group(0) @binding(5) var<storage,read_write> fineSet:array<atomic<u32>>;
${brickOwnerWGSL({ phi: "phi", vel: "vel", box: boxWGSL(parent) })}
const XD_LO:vec3i=vec3i(${spec.box.lo.join(",")});const XD_N:vec3i=vec3i(${n.join(",")});const XD_D:i32=${DETAIL_DEPTH};
const XD_M:u32=${m}u;const XD_R:f32=${spec.layout.factor}.0;const XD_AT:u32=${DETAIL_AT}u;
const XD_CLO:vec3<bool>=${bits(0)};const XD_CHI:vec3<bool>=${bits(1)};
// A parent face's liquid flux, m/s to fine cells a step: dt h_p² / h_f³ = dt / h_p r³.
const XD_FLUX:f32=${step.dt_s / step.h_m * spec.layout.factor ** 3};
const XD_BUDGET:u32=${FH.VOL_BUDGET}u;const XD_UNIT:f32=${VOL_UNIT}.0;const XD_HIGH:f32=${2 ** 32 / VOL_UNIT}.0;
var<workgroup> xdSum:array<f32,64>;
// The node of slab index i (brickDetailIndex's inverse).
fn xdNode(i:u32)->vec3i{
  let N=XD_N;let D=XD_D;var k=i32(i);
  let sx=D*N.y*N.z;
  if(k<2*sx){let side=k/sx;let r=k%sx;let x=r%D;return vec3i(select(x,N.x-D+x,side==1),(r/D)%N.y,r/(D*N.y));}
  k-=2*sx;let mx=N.x-2*D;let sy=mx*D*N.z;
  if(k<2*sy){let side=k/sy;let r=k%sy;let y=(r/mx)%D;return vec3i(D+r%mx,select(y,N.y-D+y,side==1),r/(mx*D));}
  k-=2*sy;let my=N.y-2*D;let sz=mx*my*D;
  let side=k/sz;let r=k%sz;let z=r/(mx*my);return vec3i(D+r%mx,D+(r/mx)%my,select(z,N.z-D+z,side==1));
}
// Node n's liquid inflow (fine cells a step) where it is a coupled face's: the face's velocity times its wetted
// fraction (the parent's φ at the face's centre, its four corners' mean), into the box.
fn xdInflow(n:vec3i,v:vec3i,u:vec3f)->f32{
  var flux=0.0;
  for(var a=0u;a<3u;a++){
    let b=(a+1u)%3u;let c=(a+2u)%3u;
    if(n[b]<1||n[b]>XD_N[b]-3||n[c]<1||n[c]>XD_N[c]-3){continue;}
    let low=XD_CLO[a]&&n[a]==1;let high=XD_CHI[a]&&n[a]==XD_N[a]-2;
    if(!low&&!high){continue;}
    var eb=vec3i(0);eb[b]=1;var ec=vec3i(0);ec[c]=1;
    let phiC=0.25*(nbVertexPhi(v)+nbVertexPhi(v+eb)+nbVertexPhi(v+ec)+nbVertexPhi(v+eb+ec));
    flux+=select(-1.0,1.0,low)*u[a]*clamp(0.5-phiC,0.0,1.0);
  }
  return flux*XD_FLUX;
}
@compute @workgroup_size(64) fn detailExchange(@builtin(global_invocation_id) g:vec3u,@builtin(num_workgroups) nw:vec3u,@builtin(local_invocation_index) lid:u32){
  let i=g.x+64u*nw.x*g.y;
  var inflow=0.0;
  if(i<XD_M){
    let n=xdNode(i);let v=XD_LO-vec3i(1)+n;let at=XD_AT+4u*i;
    let u=vec3f(nbFaceU(v,0u),nbFaceU(v,1u),nbFaceU(v,2u));
    src[at]=nbVertexPhi(v)*XD_R;
    src[at+1u]=u.x;src[at+2u]=u.y;src[at+3u]=u.z;
    inflow=xdInflow(n,v,u);
  }
  // The step's inflow joins the fine engine's volume budget (velocity-kernels.wgsl.ts FH.VOL_BUDGET, 64-bit fixed
  // point in VOL_UNIT a cell): a workgroup's sum, added with the low word's carry (two's complement: an outflow
  // subtracts).
  xdSum[lid]=inflow;workgroupBarrier();
  for(var s=32u;s>0u;s>>=1u){if(lid<s){xdSum[lid]+=xdSum[lid+s];}workgroupBarrier();}
  if(lid==0u&&xdSum[0]!=0.0){
    let f=xdSum[0];let units=floor(f/XD_HIGH);let low=u32(min(round((f-units*XD_HIGH)*XD_UNIT),4294967295.0));
    let old=atomicAdd(&fineSet[XD_BUDGET],low);
    atomicAdd(&fineSet[XD_BUDGET+1u],bitcast<u32>(i32(units))+select(0u,1u,old+low<old));
  }
}`);
}

/** The parent's state on a fine engine's shell, a dispatch after every parent step (and one before the fine boot). */
export class BrickDetailExchange {
  private group?: GPUBindGroup;
  private constructor(private readonly device: GPUDevice, private readonly parent: BrickEngine, private readonly pipeline: GPUComputePipeline, private readonly bindLayout: GPUBindGroupLayout, private readonly groups: [number, number], private readonly label: string) {}

  static async create(device: GPUDevice, parent: BrickEngine, spec: Pick<BrickDetailSpec, "box" | "layout">): Promise<BrickDetailExchange> {
    const code = brickDetailExchangeWGSL(parent.layout, spec, parent.options), label = `nb-brick.detailExchange.${spec.box.origins.join("+")}`;
    const module = device.createShaderModule({ label, code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(`NB-FLIP detail exchange does not compile: ${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
    const C = GPUShaderStage.COMPUTE;
    const layout = device.createBindGroupLayout({ label, entries: [
      ...[0, 1, 2, 3].map(binding => ({ binding, visibility: C, buffer: { type: "read-only-storage" as const } })),
      ...[4, 5].map(binding => ({ binding, visibility: C, buffer: { type: "storage" as const } }))] });
    const pipeline = await device.createComputePipelineAsync({ label, layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "detailExchange" } });
    const workgroups = Math.ceil(brickDetailNodes(spec.layout) / 64), x = Math.min(workgroups, 32768);
    return new BrickDetailExchange(device, parent, pipeline, layout, [x, Math.ceil(workgroups / x)], label);
  }

  /** Bind the fine engine's `src` and `set` (once, before the first encode). */
  bind(src: GPUBuffer, set: GPUBuffer) {
    const p = this.parent;
    this.group = this.device.createBindGroup({ label: this.label, layout: this.bindLayout, entries: [
      { binding: 0, resource: { buffer: p.buffer("set") } }, { binding: 1, resource: { buffer: p.buffer("dir") } },
      { binding: 2, resource: { buffer: p.buffer("phi") } }, { binding: 3, resource: { buffer: p.buffer("vel") } },
      { binding: 4, resource: { buffer: src } }, { binding: 5, resource: { buffer: set } }] });
  }

  encode(encoder: GPUCommandEncoder) {
    if (!this.group) throw new Error("NB-FLIP detail exchange: encoded before bind");
    const pass = encoder.beginComputePass({ label: "nb-brick.detailExchange" });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.group); pass.dispatchWorkgroups(this.groups[0], this.groups[1]); pass.end();
  }

  /** Encode and submit alone. */
  submit() {
    const enc = this.device.createCommandEncoder({ label: this.label });
    this.encode(enc); this.device.queue.submit([enc.finish()]);
  }
}

/** What a detail engine shares with its parent's build. */
export interface BrickDetailBuild {
  readonly stages: BrickStagesFactory;
  readonly solver: PressureSolverFactory;
  readonly values: Readonly<Record<string, unknown>>;
}

/** One detail engine: the fine brick engine of a box, its exchange from the parent, and its presentation. */
export class BrickDetail {
  private constructor(readonly spec: BrickDetailSpec, readonly engine: BrickEngine, readonly exchange: BrickDetailExchange, readonly presentation: BrickPresentation, readonly surface: BrickDetailSurface) {}

  static async create(device: GPUDevice, parent: BrickEngine, spec: BrickDetailSpec, build: BrickDetailBuild): Promise<BrickDetail> {
    const s = spec.setup;
    if (Math.abs(s.dt_s - parent.options.dt_s) > 1e-12) throw new Error(`NB-FLIP detail engine: its step ${s.dt_s} s is not the parent's ${parent.options.dt_s} s`);
    const exchange = await BrickDetailExchange.create(device, parent, spec);
    const engine = await BrickEngine.create(device, {
      lattice: s.lattice, ...(s.box ? { box: s.box } : {}), ...(s.openTop ? { openTop: true } : {}), ...(s.openSides ? { openSides: true } : {}),
      farPhi: s.farPhi, h_m: s.h_m, dt_s: s.dt_s, gravity_mps2: s.gravity_mps2, capacities: s.capacities, liquid: s.liquid, initial: s.initial, initialVelocity: s.initialVelocity,
      // The volume budget is the box's: its boot volume, then every step the liquid the parent moves across the coupled
      // faces (the exchange adds it), so the box holds what the parent delivers.
      solver: build.solver, stages: build.stages, values: build.values, label: s.sceneId,
      fluid: { density_kg_m3: s.density_kg_m3, viscosity_Pa_s: s.viscosity_Pa_s, surfaceTension_N_m: s.surfaceTension_N_m },
      particles: { bandDepth: s.particleBandDepth },
      ...(s.staticMask ? { staticMask: s.staticMask } : {}), ...(s.staticsDeferred ? { staticsDeferred: true } : {}),
      detail: spec.layout,
      primeDetail: buffer => { exchange.bind(buffer("src"), buffer("set")); exchange.submit(); },
    });
    const presentation = await BrickPresentation.create(engine, spec.farPhi_m, s.presentCells);
    const cells = (s.presentCells ?? s.lattice) as unknown as [number, number, number], lo = spec.low_m;
    const surface = await BrickDetailSurface.create(device, presentation, {
      cells, bricks: s.lattice.map(n => n >> 3) as unknown as [number, number, number], h_m: s.h_m, origin_m: lo, coupled: spec.layout.coupled, far_m: spec.farPhi_m,
    }, [lo, [0, 1, 2].map(a => lo[a]! + cells[a]! * s.h_m) as unknown as [number, number, number]]);
    return new BrickDetail(spec, engine, exchange, presentation, surface);
  }

  /** The step after the parent's: the parent's new state on the shell, then the fine step (one queue, in order). */
  advance() { this.exchange.submit(); this.engine.advance(1); }

  /** The live voxel edit's preflight on this engine (its own mask, the box's cut of `scene`): throws, changing nothing. */
  validateSolids(scene: SceneDescription) {
    const mask = this.spec.setup.staticMask;
    if (!mask) throw new Error(`detail engine ${this.spec.box.origins.join("+")} was built with no static mask`);
    const trial = mask.clone();
    try { trial.update(solidWorldForFluidLattice(brickDetailSubScene(scene, this.spec.box))); }
    catch (error) { throw new Error(`This voxel edit needs more of detail engine ${this.spec.box.origins.join("+")}'s solid mask than it was built with (${error instanceof Error ? error.message : String(error)})`); }
    const cut = brickStaticCutBricks(trial), k = this.engine.layout.cap.k;
    if (cut > k) throw new Error(`This voxel edit cuts ${cut} of detail engine ${this.spec.box.origins.join("+")}'s bricks, over its ${k} K slots`);
  }

  /** Fold the live voxel edit (validateSolids first) into this engine's mask and restamp what moved. */
  foldSolids(scene: SceneDescription) {
    const mask = this.spec.setup.staticMask!;
    const lo: [number, number, number] = [Infinity, Infinity, Infinity], hi: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    const ranges = mask.update(solidWorldForFluidLattice(brickDetailSubScene(scene, this.spec.box)), (x, y, z) => {
      lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); lo[2] = Math.min(lo[2], z);
      hi[0] = Math.max(hi[0], x + 1); hi[1] = Math.max(hi[1], y + 1); hi[2] = Math.max(hi[2], z + 1);
    });
    if (ranges && hi[0] > lo[0]) this.engine.restampStatics(mask.words, ranges, lo, hi);
  }

  /** The presentation's copy, then its mesh. */
  present(encoder: GPUCommandEncoder) { this.presentation.encode(encoder); this.surface.encode(encoder); }

  destroy() { this.surface.destroy(); this.presentation.destroy(); this.engine.destroy(); }
}
