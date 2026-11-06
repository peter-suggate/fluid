import type { ClothFrameIdentity, ClothMaterial, ClothPhysicsOptions, ClothState, ClothTopologyData } from '../../cloth/types';
import type { ClothImpulsePublication } from '../../cloth/coupling';
import { porousTensor, porousPermeability } from '../../cloth/porous-material';
import { UniformGPUOrderingKernels } from './uniform-gpu-radix-sort';
import { CLOTH_COLOR_ROUNDS, CLOTH_DRAG_COLORS, CLOTH_STENCIL_STATUS, UniformClothGPUStencil, UniformClothStencilKernels, clothStencilTopology } from './uniform-cloth-gpu-stencil';
import { CLOTH_CELLS_STATUS, UniformClothCellsKernels, UniformClothGPUCells } from './uniform-cloth-gpu-cells';
import {clothCellWeights,clothCellKey} from './uniform-porous-capacity';
import { UniformClothPorePressure } from './uniform-cloth-pore-pressure';
import {uniformMixedSolidWGSL} from './uniform-mixed-solid.wgsl';
import { UniformClothTensorDrag } from './uniform-cloth-tensor-drag';
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, type UniformDetailGroup } from './uniform-detail-fields';
import type { UniformMixedOwnership } from './uniform-mixed-ownership';
import { uniformMixedTopologyWGSL } from './uniform-mixed-topology.wgsl';
import type { UniformRegionHeat } from './uniform-region-heat';
import type { UniformMixedSolid } from './uniform-mixed-solid.wgsl';

/** Actual production NB-FLIP fields. `velocity` is the forced MAC field,
 * after the FLIP snapshot and before pressure. It is never a proxy fluid. */
export interface UniformClothGridFields {
  readonly device:GPUDevice;
  readonly ownership: UniformMixedOwnership;
  readonly velocity: GPUTexture;
  readonly volume: GPUTexture;
  readonly projectedVelocity: GPUTexture;
  readonly phi:GPUTexture;
  readonly waterBudget:GPUBuffer;
  readonly solid?:UniformMixedSolid;
  readonly porousCapacity:GPUTexture;
  readonly setPorousCapacity:(texture?:GPUTexture)=>void;
  readonly invalidateLocalWater:()=>void;
  readonly setPressureMixtureFaces: (faces?: GPUTexture) => void;
  readonly heat: UniformRegionHeat;
  readonly origin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
}

export interface UniformClothStep {
  /** Effective hard constraints; source attachment flags remain finite forces. */
  readonly kinematicFixed?:ArrayLike<number>;
  /** Same-step nonfluid force predictor; accepted state is not integrated here. */
  readonly solidPredictorVelocity_m_s?:ArrayLike<number>;
  readonly state: ClothState;
  readonly data: ClothTopologyData;
  readonly materials: readonly ClothMaterial[];
  readonly identity: ClothFrameIdentity;
  readonly dt_s: number;
  readonly nonlinearDrag?: boolean;
  readonly physics?:ClothPhysicsOptions;
  /** Hard geometry can move this step: apertures come from the current collider poses. */
  readonly currentHardGeometry?:boolean;
}

interface Anchor { p: [number, number, number]; links: [number[], number[], number[]] }
export interface PreparedUniformClothStencil {
  anchors: Int32Array;
  coefficients: Float32Array;
  occupiedVolumes_m3: Float32Array;
  ranges: Uint32Array;
  gather: Uint32Array;
  coordinates: readonly [number, number, number][];
  identity: ClothFrameIdentity;
  dt_s: number;
  density_kg_m3: number;
  vertices: number;
  exponent:number;
  linkWeights:Float32Array;
  linkFaces:Uint32Array;
  vertexOccupiedVolume_m3:Float64Array;
  poreTensor:Float64Array;
  poreCoefficient_Pa:Float64Array;
  vertexMicro:Float64Array;
}
type Prepared=PreparedUniformClothStencil;

/** Deterministic h-MAC stencil builder. It splits each structural mass
 * between its interpolation faces independently for each velocity component.
 * Pair solves followed by mass-weighted recombination are dissipative.
 * Fine ownership is checked on the GPU; an unsupported stencil is fatal. */
export function prepareUniformClothStencil(fields: Pick<UniformClothGridFields, 'origin_m' | 'cellSize_m'>, step: UniformClothStep): Prepared {
  const { state, data, materials } = step, vertices = state.dryMass_kg.length;
  if (!(step.dt_s > 0) || !Number.isFinite(step.dt_s) || state.positions_m.length !== 3 * vertices
    || data.restPositions_m.length !== 3 * vertices || state.velocities_m_s.length !== 3 * vertices) throw new Error('Invalid cloth grid step');
  if (state.generation !== step.identity.generation || state.topologyRevision !== step.identity.topologyRevision) throw new Error('Stale cloth grid topology');
  const predictor=step.solidPredictorVelocity_m_s??state.velocities_m_s;
  if(predictor.length!==3*vertices||Array.from(predictor).some(v=>!Number.isFinite(v)))throw new Error('Invalid cloth nonfluid velocity predictor');
  const density = materials[0]?.waterDensity_kg_m3;
  if (!(density! > 0) || materials.some(m => m.waterDensity_kg_m3 !== density)) throw new Error('Cloth coupling requires one water density');
  if (fields.cellSize_m.some(h => !Number.isFinite(h) || h <= 0)) throw new Error('Invalid cloth grid cell size');
  const geometricVolume=new Float64Array(vertices),poreTensor=new Float64Array(9*vertices),poreCoefficient_Pa=new Float64Array(vertices),vertexMicro=new Float64Array(2*vertices);
  // The tensor and capillary coefficient depend only on the element: one
  // evaluation is shared by that element's vertices (same values).
  const elementPore = (material: ClothMaterial, axis: [number, number, number], shape: number) => {
    const pore=porousTensor(2,1,axis,shape),radius=porousPermeability(material.solidFraction,material.fiberDiameter_m).capillaryRadius_m;
    const cosine=Math.abs(material.contactAngle_rad-Math.PI/2)<1e-14?0:Math.cos(material.contactAngle_rad);
    return {pore,cosine,radius};
  };
  const addMaterial = (v: number, material: ClothMaterial, element: {pore:ArrayLike<number>;cosine:number;radius:number}, volume: number) => {
    geometricVolume[v]!+=volume;vertexMicro[2*v]!+=volume*material.fiberDiameter_m;vertexMicro[2*v+1]!+=volume*material.waterViscosity_Pa_s;
    const pore=element.pore;for(let k=0;k<9;k++)poreTensor[9*v+k]!+=volume*pore[k]!;
    poreCoefficient_Pa[v]!+=volume*material.surfaceTension_N_m*element.cosine/element.radius;
  };
  for (let t = 0; t < data.triangles.length / 3; t++) {
    const ids = data.triangles.slice(3 * t, 3 * t + 3), material = materials[data.triangleMaterials?.[t] ?? 0];
    if (!material) throw new Error('Missing cloth triangle material');
    const [a, b, c] = ids as [number, number, number];
    const u = [0, 1, 2].map(k => state.positions_m[3 * b + k]! - state.positions_m[3 * a + k]!);
    const v = [0, 1, 2].map(k => state.positions_m[3 * c + k]! - state.positions_m[3 * a + k]!);
    const n: [number, number, number] = [u[1]! * v[2]! - u[2]! * v[1]!, u[2]! * v[0]! - u[0]! * v[2]!, u[0]! * v[1]! - u[1]! * v[0]!];
    const twiceArea = Math.hypot(...n);
    if (!(twiceArea > 0)) throw new Error('Degenerate cloth face cannot exchange fluid');
    const element=elementPore(material, n, 1);for (const id of ids) addMaterial(id, material, element, twiceArea * material.thickness_m / 6);
  }
  for (let rod = 0; rod + 1 < data.rodOffsets.length; rod++) {
    const material = materials[data.rodMaterials?.[rod] ?? 0];
    if (!material) throw new Error('Missing cloth rod material');
    for (let j = data.rodOffsets[rod]!; j + 1 < data.rodOffsets[rod + 1]!; j++) {
      const a = data.rodIndices[j]!, b = data.rodIndices[j + 1]!;
      const tangent = [0, 1, 2].map(k => state.positions_m[3 * b + k]! - state.positions_m[3 * a + k]!) as [number, number, number];
      const length = Math.hypot(...tangent);
      if (!(length > 0)) throw new Error('Degenerate cloth rod cannot exchange fluid');
      const halfVolume = .5 * Math.PI * material.rodRadius_m * (material.rodRadiusB_m??material.rodRadius_m) * length;
      const element=elementPore(material, tangent, 0);addMaterial(a, material, element, halfVolume); addMaterial(b, material, element, halfVolume);
    }
  }
  for(let v=0;v<vertices;v++)if(geometricVolume[v]!>0){poreCoefficient_Pa[v]!/=geometricVolume[v]!;vertexMicro[2*v]!/=geometricVolume[v]!;vertexMicro[2*v+1]!/=geometricVolume[v]!;for(let k=0;k<9;k++)poreTensor[9*v+k]!/=geometricVolume[v]!;}
  const faceKey=new Float64Array(16),faceX=new Int32Array(16),faceY=new Int32Array(16),faceZ=new Int32Array(16),faceWeight=new Float64Array(16);
  const anchors = new Map<number, Anchor>(), linkCoefficients: number[] = [], linkVertices: number[] = [], linkVolumes: number[] = [],weights:number[]=[];
  for (let v = 0; v < vertices; v++) {
    const mass = state.dryMass_kg[v]! + density! * state.waterVolume_m3[v]!;
    const fixed = step.physics?.solveSolid===false||(((step.kinematicFixed??data.fixed)?.[v] ?? 0) & 1) !== 0;
    if (!(mass > 0) && !fixed) throw new Error('Free cloth vertex has no mass');
    const cellWeights = geometricVolume[v]===0 ? [] : clothCellWeights(fields,state.positions_m.subarray(3*v,3*v+3));
    for (let a = 0; a < 3; a++) {
      if (geometricVolume[v]===0) continue;
      // The pressure face partition is the average of the two native cell
      // partitions. Use that SAME adjoint basis for mass and drag transfer.
      // Along the face axis this has three entries, rather than a different
      // independently rasterized eight-point MAC material fraction.
      // Insertion-ordered face accumulation (at most 16 faces; linear search).
      let faces=0;
      for(const {p:cell,weight:cellWeight} of cellWeights){
       const weight=.5*cellWeight;
       for(let offset=-1;offset<=0;offset++){const x=cell[0]+(a===0?offset:0),y=cell[1]+(a===1?offset:0),z=cell[2]+(a===2?offset:0),key=clothCellKey(x,y,z);
        let f=0;while(f<faces&&faceKey[f]!==key)f++;
        if(f<faces)faceWeight[f]!+=weight;else{faceKey[faces]=key;faceX[faces]=x;faceY[faces]=y;faceZ[faces]=z;faceWeight[faces]=weight;faces++;}}
      }
      for (let f=0;f<faces;f++) {
        const weight=faceWeight[f]!;
        if (weight <= 1e-12) continue;
        const key = faceKey[f]!; let anchor = anchors.get(key);
        if (!anchor) { anchor = { p:[faceX[f]!,faceY[f]!,faceZ[f]!], links: [[], [], []] }; anchors.set(key, anchor); }
        const index = linkVertices.length;
        // Drag is evaluated at deposited GRID packing, including vf=1
        // controls whose subresolution grid psi remains below one. The two
        // reserved lanes are not a second vertex-local constitutive model.
        linkCoefficients.push(fixed ? 0 : weight * mass,0,0,predictor[3 * v + a]!);
        linkVolumes.push(weight * (state.solidVolume_m3[v]! + state.waterVolume_m3[v]!) * (step.physics?.pressureOnSolid===false?-1:1));
        weights.push(weight);
        linkVertices.push(3 * v + a); anchor.links[a]!.push(index);
      }
    }
  }
  const ordered = [...anchors.values()].sort((a, b) => a.p[2] - b.p[2] || a.p[1] - b.p[1] || a.p[0] - b.p[0]);
  const links=linkVertices.length,records = new Int32Array(8 * ordered.length), coefficients = new Float32Array(4 * links), occupiedVolumes_m3 = new Float32Array(links);
  const linkWeights=new Float32Array(links),linkFaces=new Uint32Array(links),linkOwner=new Uint32Array(links);
  // Per-coordinate link lists by counting sort: within each coordinate the
  // links keep increasing face-major order, exactly as sequential pushes.
  const ranges = new Uint32Array(6 * vertices), gather = new Uint32Array(links);
  let at = 0;
  ordered.forEach((anchor, index) => {
    const p=anchor.p,l=anchor.links,base=8*index;records[base]=p[0];records[base+1]=p[1];records[base+2]=p[2];records[base+3]=at;records[base+4]=l[0].length;records[base+5]=l[1].length;records[base+6]=l[2].length;records[base+7]=0;
    for (let axis=0;axis<3;axis++)for (const original of l[axis]!) {
      for(let j=0;j<4;j++)coefficients[4*at+j]=linkCoefficients[4*original+j]!; occupiedVolumes_m3[at] = linkVolumes[original]!;linkWeights[at]=weights[original]!;linkFaces[at]=3*index+axis;const owner=linkVertices[original]!;linkOwner[at++]=owner;ranges[2*owner+1]!++;
    }
  });
  for (const coefficient of coefficients) if (!Number.isFinite(coefficient)) throw new Error('Cloth coupling exceeds f32 representable range');
  at = 0;
  for(let index=0;index<3*vertices;index++){ranges[2*index]=at;at+=ranges[2*index+1]!;}
  const fill=new Uint32Array(3*vertices);
  for(let link=0;link<links;link++){const owner=linkOwner[link]!;gather[ranges[2*owner]!+fill[owner]!++]=link;}
  return { anchors: records, coefficients, occupiedVolumes_m3, ranges, gather, coordinates: ordered.map(a => a.p), identity: { ...step.identity },
    dt_s: step.dt_s, density_kg_m3: density!, vertices,exponent:step.physics?.dragExponent??1.6,linkWeights,linkFaces,vertexOccupiedVolume_m3:Float64Array.from(state.solidVolume_m3,(v,i)=>v+state.waterVolume_m3[i]!),poreTensor,poreCoefficient_Pa,vertexMicro };
}

/** Status words (32 bytes): unsupported anchors, first unsupported anchor,
 * anchors visited, wet faces, aperture status bits, and the first
 * unsupported anchor's lattice coordinate (bits). */
const STATUS_BYTES=32;
/** The readback header: status, then the GPU stencil's and cells' counts. */
const READ_HEADER=STATUS_BYTES+32;
/** cornerApertures status bits (status word 4). */
export const CLOTH_APERTURE_STATUS={unowned:1,coverage:2} as const;
const STENCIL_ERRORS:readonly [number,string][]=[[CLOTH_STENCIL_STATUS.outsideLattice,'the cloth spans more cells than 32-bit face keys address'],[CLOTH_STENCIL_STATUS.masslessFreeVertex,'a free cloth vertex has no mass'],
 [CLOTH_STENCIL_STATUS.noOpenSupport,'cloth has no open local cell support; collision/admission must be resolved'],[CLOTH_STENCIL_STATUS.degenerateElement,'a degenerate cloth element cannot exchange fluid'],
 [CLOTH_STENCIL_STATUS.anchorCapacity,'the stencil exceeds its anchor capacity'],[CLOTH_STENCIL_STATUS.colorBudget,`drag coloring needs more than ${CLOTH_DRAG_COLORS} colors`],[CLOTH_STENCIL_STATUS.colorRounds,`drag coloring did not finish in ${CLOTH_COLOR_ROUNDS} rounds`]];
const CELL_ERRORS:readonly [number,string][]=[[CLOTH_CELLS_STATUS.noOpenSupport,'cloth has no open local cell support inside the lattice'],[CLOTH_CELLS_STATUS.fiberFraction,'grid drag fiber fraction is outside (0,1)'],
 [CLOTH_CELLS_STATUS.permeability,'grid drag permeability is not positive'],[CLOTH_CELLS_STATUS.poreCapacity,'cloth material fills a cell: refine the actual porous support']];

/** One cloth topology's persistent GPU tables. */
interface CouplingTables {
  readonly data:ClothTopologyData;readonly materials:readonly ClothMaterial[];readonly vertices:number;
  readonly stencil:UniformClothGPUStencil;readonly cells:UniformClothGPUCells;readonly mixture:GPUTexture;
  readonly corrected:GPUBuffer;readonly impulses:GPUBuffer;readonly pressureSample:GPUBuffer;readonly vertexImpulse:GPUBuffer;
  readonly status:GPUBuffer;readonly params:GPUBuffer;readonly apertureParams:GPUBuffer;readonly readback:GPUBuffer;
  readonly gatherGroup:GPUBindGroup;readonly pressureSampleGatherGroup:GPUBindGroup;readonly supportGroup:GPUBindGroup;readonly apertureGroup:GPUBindGroup;
}
interface PreparedExchange {readonly identity:ClothFrameIdentity;readonly dt_s:number;readonly vertices:number;readonly vertexOccupiedVolume_m3:Float64Array}

/** Cloth / real WebGPU NB-FLIP bridge. The exchange stencil, its drag
 * coloring and the material cell tables are built on the GPU from resident
 * fields each step (prepareUniformClothStencil and the cell oracles are the
 * CPU parity references). Only cloth state goes up; one readback per step
 * returns the impulses and every status word. */
export class UniformClothGridCoupling {
  diagnostics:readonly number[]=[];
  readonly id = 'physics.cloth.porous-grid-drag';
  private readonly solveLayout: GPUBindGroupLayout;
  private readonly storeLayout: GPUBindGroupLayout;
  private readonly gatherLayout: GPUBindGroupLayout;
  private readonly supportLayout: GPUBindGroupLayout;
  private readonly pressureLayout: GPUBindGroupLayout;
  private readonly apertureLayout: GPUBindGroupLayout;
  private solvePipeline?: GPUComputePipeline;
  private firstUnsupportedPipeline?: GPUComputePipeline;
  private storePipeline?: GPUComputePipeline;
  private gatherPipeline?: GPUComputePipeline;
  private supportPipeline?: GPUComputePipeline;
  private pressurePipeline?: GPUComputePipeline;
  private aperturePipeline?: GPUComputePipeline;
  private currentAperturePipeline?: GPUComputePipeline;
  private ordering?: UniformGPUOrderingKernels;
  private stencilKernels?: UniformClothStencilKernels;
  private cellKernels?: UniformClothCellsKernels;
  private tables?: CouplingTables;
  private solveGroup?: UniformDetailGroup;
  private storeGroup?: UniformDetailGroup;
  private pressureGroup?: UniformDetailGroup;
  private tensorDrag:UniformClothTensorDrag;
  private porePressure:UniformClothPorePressure;
  private prepared?: PreparedExchange;
  private encoded = false;
  private projected = false;

  constructor(private readonly device: GPUDevice, readonly fields: UniformClothGridFields) {
    if(device!==fields.device)throw new Error('Cloth coupling must use the field-owning device wrapper for native field addressing');
    this.tensorDrag=new UniformClothTensorDrag(fields);
    this.porePressure=new UniformClothPorePressure(fields);
    const buffer = (binding: number, read = false): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: read ? 'read-only-storage' : 'storage' } });
    this.solveLayout = uniformDetailBindLayout(device, { entries: [buffer(0, true), buffer(1, true), buffer(2), buffer(3), buffer(4),
      { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, buffer(8, true),
      {binding:9,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba32float',viewDimension:'2d'}}] });
    this.storeLayout = uniformDetailBindLayout(device, { entries: [buffer(0, true), buffer(1, true),
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '3d' } }] });
    this.gatherLayout = device.createBindGroupLayout({ entries: [buffer(0, true), buffer(1, true), buffer(2, true), buffer(3)] });
    this.supportLayout = device.createBindGroupLayout({ entries: [buffer(0, true), buffer(1), buffer(2)] });
    this.pressureLayout = uniformDetailBindLayout(device, { entries: [buffer(0,true),buffer(1,true),buffer(2),{binding:3,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d'}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},buffer(6)] });
    this.apertureLayout = device.createBindGroupLayout({ entries: [{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},buffer(1,true),buffer(2),buffer(3)] });
  }

  async initialize(): Promise<void> {
    const topology = uniformMixedTopologyWGSL(this.fields.ownership.capacity, 0);
    const solid=this.fields.solid;
    const anchor = 'struct ClothAnchor { p:vec3i, first:u32, counts:vec3u, unused:u32 }; struct ClothAnchors { count:vec4u, unused:vec4u, items:array<ClothAnchor> };';
    const mixture = 'struct MixtureFace {p:vec4i,mobility:vec4f,mixture:vec4f,fluid:vec4f}; struct MixtureFaces {header:vec4u,faces:array<MixtureFace>};';
    const support = this.device.createShaderModule({ label: 'Cloth local support wake', code: topology + anchor + this.fields.heat.promoteWGSL(1, 1, 2) + /* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:ClothAnchors;
@compute @workgroup_size(64) fn wake(@builtin(global_invocation_id) gid:vec3u) {
 let i=gid.x;if(i>=anchors.count.x){return;}
 // Wake the finite stencil and one-cell derivative closure, never the domain.
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
 let p=anchors.items[i].p-32*umFramePage()+vec3i(x,y,z);let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));
 if(umKeyStored(tile)){rhPromote(umPageOfTile(tile));}
 }}}
}` });
    const solve = uniformDetailModule(this.device, { label: 'Cloth native pressure mixture preparation', code: topology + uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count) + /* wgsl */`
${anchor}
@group(1) @binding(0) var<storage,read> anchors:ClothAnchors;
@group(1) @binding(1) var<storage,read> links:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> corrected:array<vec4f>;
@group(1) @binding(3) var<storage,read_write> impulses:array<f32>;
@group(1) @binding(4) var<storage,read_write> status:array<atomic<u32>>;
@group(1) @binding(5) var velocity:texture_3d<f32>;
@group(1) @binding(6) var volume:texture_3d<f32>;
@group(1) @binding(7) var<uniform> physics:vec4f;
@group(1) @binding(8) var<storage,read> occupied:array<f32>;
${mixture}
@group(1) @binding(9) var pressure:texture_storage_2d<rgba32float,write>;
fn pressureStore(i:u32,value:vec4f){let width=textureDimensions(pressure).x;textureStore(pressure,vec2i(i32(i%width),i32(i/width)),value);}
fn supported(p:vec3i)->bool {
 let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));return umTileWidth(tile)==1u&&umKeyStored(tile);
}
@compute @workgroup_size(64) fn preparePressureMixture(@builtin(global_invocation_id) gid:vec3u) {
 let i=gid.x;if(i==0u){pressureStore(0u,bitcast<vec4f>(vec4u(anchors.count.x,0u,0u,0u)));}if(i>=anchors.count.x){return;}
 atomicAdd(&status[2],1u);
 var a=anchors.items[i];a.p-=32*umFramePage();var valid=supported(a.p);
 for(var axis=0u;axis<3u;axis++){if(a.counts[axis]!=0u){var q=a.p;q[axis]+=1;valid=valid&&supported(q);}}
 if(!valid){atomicAdd(&status[0],1u);atomicMin(&status[1],i);corrected[i]=vec4f(0.0);return;}
 var value=/*c*/textureLoad(velocity,a.p,0);var first=a.first;
 var pressureFace=MixtureFace(vec4i(a.p,0),vec4f(1.0),value,value);
 for(var axis=0u;axis<3u;axis++) {
  let count=a.counts[axis];if(count==0u){continue;}
  var q=a.p;q[axis]+=1;
  // Hard-boundary velocity enforcement is a collider reaction, not liquid
  // pressure traction on fabric. Closed faces retain the identity material
  // record and publish no projected-minus-forced cloth pressure impulse.
  if(umPressureFaceV(a.p,axis)<=0.0){first+=count;continue;}
  let wet=clamp(0.5*(/*c*/textureLoad(volume,a.p,0).x+/*c*/textureLoad(volume,q,0).x),0.0,1.0);
  if(wet>0.0){atomicAdd(&status[3],1u);}
  var solidFraction=0.0;for(var j=0u;j<count;j++){solidFraction+=abs(occupied[first+j])/physics.z;}
  if(solidFraction>=1.0){atomicAdd(&status[0],1u);atomicMin(&status[1],i);first+=count;continue;}
  let mass=physics.y*physics.z*wet;
  if(mass<=1e-30){first+=count;continue;}
  first+=count;
  var mixtureVelocity=(1.0-solidFraction)*value[axis];var mobility=1.0-solidFraction;
  for(var j=0u;j<count;j++){
   let k=first-count+j;let link=links[k];let fraction=abs(occupied[k])/physics.z;
   let solidVelocity=link.w;if(link.x>0.0&&occupied[k]>0.0){mobility+=physics.y*physics.z*fraction*fraction/link.x;}
   mixtureVelocity+=fraction*solidVelocity;
  }
  pressureFace.mobility[axis]=mobility;pressureFace.mixture[axis]=mixtureVelocity;pressureFace.fluid[axis]=value[axis];
  pressureFace.p.w|=i32(1u<<axis);
 }
 corrected[i]=value;
 pressureStore(1u+4u*i,bitcast<vec4f>(pressureFace.p));pressureStore(2u+4u*i,pressureFace.mobility);
 pressureStore(3u+4u*i,pressureFace.mixture);pressureStore(4u+4u*i,pressureFace.fluid);
}
// The first unsupported anchor's coordinate, for the fatal report.
@compute @workgroup_size(1) fn firstUnsupported(){
 let i=atomicLoad(&status[1]);if(i<anchors.count.x){let p=anchors.items[i].p;atomicStore(&status[5],bitcast<u32>(p.x));atomicStore(&status[6],bitcast<u32>(p.y));atomicStore(&status[7],bitcast<u32>(p.z));}
}` });
    const pressure = uniformDetailModule(this.device, {label:'Cloth pressure response gather', code:topology+anchor+mixture+/* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:ClothAnchors;
@group(1) @binding(1) var<storage,read> occupied:array<f32>;
@group(1) @binding(2) var<storage,read_write> impulses:array<f32>;
@group(1) @binding(3) var mixture:texture_2d<f32>;
fn mixtureWord(i:u32)->vec4f{let width=textureDimensions(mixture).x;return textureLoad(mixture,vec2i(i32(i%width),i32(i/width)),0);}
@group(1) @binding(4) var velocity:texture_3d<f32>;
@group(1) @binding(5) var<uniform> physics:vec4f;
@group(1) @binding(6) var<storage,read_write> pressureSample:array<f32>;
@compute @workgroup_size(64) fn pressureResponse(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=anchors.count.x){return;}var a=anchors.items[i];a.p-=32*umFramePage();let flags=bitcast<vec4i>(mixtureWord(1u+4u*i)).w;let fluid=mixtureWord(4u+4u*i);
 let projected=/*c*/textureLoad(velocity,a.p,0);var first=a.first;
 for(var axis=0u;axis<3u;axis++){
  // A dry face has the identity table and contributes no pressure response.
  if((u32(flags)&(1u<<axis))!=0u){
   let delta=projected[axis]-fluid[axis];
   for(var j=0u;j<a.counts[axis];j++){let k=first+j;impulses[k]+=physics.y*max(occupied[k],0.0)*delta;pressureSample[k]=abs(occupied[k])*delta;}
  }
  first+=a.counts[axis];
 }
}`});
    const store = uniformDetailModule(this.device, { label: 'Cloth NB-FLIP face commit', code: topology + /* wgsl */`
${anchor}
@group(1) @binding(0) var<storage,read> anchors:ClothAnchors;
@group(1) @binding(1) var<storage,read> corrected:array<vec4f>;
@group(1) @binding(2) var output:texture_storage_3d<rgba32float,write>;
fn supported(p:vec3i)->bool {
 let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));return umTileWidth(tile)==1u&&umKeyStored(tile);
}
@compute @workgroup_size(64) fn commitDrag(@builtin(global_invocation_id) gid:vec3u) {
 let i=gid.x;if(i>=anchors.count.x){return;}
 var a=anchors.items[i];a.p-=32*umFramePage();if(!supported(a.p)){return;}
 for(var axis=0u;axis<3u;axis++){if(a.counts[axis]!=0u){var q=a.p;q[axis]+=1;if(!supported(q)){return;}}}
 textureStore(output,a.p,corrected[i]);
}` });
    const gather = this.device.createShaderModule({ label: 'Cloth impulse gather', code: /* wgsl */`
@group(0) @binding(0) var<storage,read> ranges:array<vec2u>;
@group(0) @binding(1) var<storage,read> indices:array<u32>;
@group(0) @binding(2) var<storage,read> impulses:array<f32>;
@group(0) @binding(3) var<storage,read_write> vertexImpulse:array<f32>;
@compute @workgroup_size(64) fn gather(@builtin(global_invocation_id) gid:vec3u) {
 let i=gid.x;if(i>=arrayLength(&vertexImpulse)){return;}let range=ranges[i];var sum=0.0;
 for(var j=0u;j<range.y;j++){sum+=impulses[indices[range.x+j]];}vertexImpulse[i]=sum;
}` });
    // The hard aperture of every vertex's eight trilinear corner cells at the
    // accepted (or, with moving colliders, current) hard geometry: the same
    // native query as the support sample. A cell beyond the world box's walls
    // is closed; a cell without resident h ownership is fatal.
    const apertureCode=/* wgsl */`
struct Params {origin:vec4f,inverseCell:vec4f,dims:vec4u,misc:vec4f};
@group(1) @binding(0) var<uniform> params:Params;
@group(1) @binding(1) var<storage,read> positions:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> apertures:array<f32>;
@group(1) @binding(3) var<storage,read_write> status:array<atomic<u32>>;
@compute @workgroup_size(64) fn cornerApertures(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=8u*params.dims.w){return;}let v=i/8u;let c=i%8u;
 let q=(positions[v].xyz-params.origin.xyz)*params.inverseCell.xyz-vec3f(0.5);let f=q-floor(q);let o=vec3u(c&1u,(c>>1u)&1u,(c>>2u)&1u);
 let w=select(1.0-f.x,f.x,o.x==1u)*select(1.0-f.y,f.y,o.y==1u)*select(1.0-f.z,f.z,o.z==1u);
 apertures[i]=0.0;if(w<=0.0){return;}
 let p=vec3i(floor(q))+vec3i(o)-32*umFramePage();
 if(umWorldSolid((p>>vec3u(5u))+umFramePage())){return;}
 let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));
 if(!umKeyStored(tile)||umTileWidth(tile)!=1u){atomicOr(&status[4],${CLOTH_APERTURE_STATUS.unowned}u);return;}
 let open=umCellOpen(p);if(umRigidCoverageFailure.x!=0u){atomicOr(&status[4],${CLOTH_APERTURE_STATUS.coverage}u);return;}apertures[i]=open;
}`;
    const aperture=uniformDetailModule(this.device,{label:'Cloth corner hard apertures',code:topology+uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count)+apertureCode});
    const currentAperture=uniformDetailModule(this.device,{label:'Cloth corner current-geometry apertures',code:topology+uniformMixedSolidWGSL(solid?2:undefined,undefined,false,false,true,true)+apertureCode});
    for (const module of [solve, store, gather, support, pressure, aperture, currentAperture]) {
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join('\n'));
    }
    const solveLayout=this.device.createPipelineLayout({ bindGroupLayouts: [this.fields.ownership.bindLayout, this.solveLayout,...(solid?[solid.tileLayout]:[])] });
    [this.solvePipeline, this.firstUnsupportedPipeline, this.storePipeline, this.gatherPipeline, this.aperturePipeline, this.currentAperturePipeline] = await Promise.all([
      uniformDetailPipeline(this.device,this.fields.ownership,{ layout: solveLayout, compute: { module: solve, entryPoint: 'preparePressureMixture' } }),
      uniformDetailPipeline(this.device,this.fields.ownership,{ layout: solveLayout, compute: { module: solve, entryPoint: 'firstUnsupported' } }),
      uniformDetailPipeline(this.device,this.fields.ownership,{ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.fields.ownership.bindLayout, this.storeLayout] }), compute: { module: store, entryPoint: 'commitDrag' } }),
      this.device.createComputePipelineAsync({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.gatherLayout] }), compute: { module: gather, entryPoint: 'gather' } }),
      uniformDetailPipeline(this.device,this.fields.ownership,{layout:this.device.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.apertureLayout,...(solid?[solid.tileLayout]:[])]}),compute:{module:aperture,entryPoint:'cornerApertures'}}),
      uniformDetailPipeline(this.device,this.fields.ownership,{layout:this.device.createPipelineLayout({bindGroupLayouts:[this.fields.ownership.bindLayout,this.apertureLayout,...(solid?[solid.currentGeometryLayout]:[])]}),compute:{module:currentAperture,entryPoint:'cornerApertures'}}),
    ]);
    this.supportPipeline = await this.device.createComputePipelineAsync({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.fields.ownership.bindLayout, this.supportLayout] }), compute: { module: support, entryPoint: 'wake' } });
    this.pressurePipeline = await uniformDetailPipeline(this.device,this.fields.ownership,{ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.fields.ownership.bindLayout, this.pressureLayout] }), compute: { module: pressure, entryPoint: 'pressureResponse' } });
    [this.ordering,this.stencilKernels,this.cellKernels]=await Promise.all([UniformGPUOrderingKernels.create(this.device),UniformClothStencilKernels.create(this.device),UniformClothCellsKernels.create(this.device)]);
    await Promise.all([this.tensorDrag.initialize(),this.porePressure.initialize()]);
  }

  /** The persistent tables of one topology, built once (one-time setup). */
  private tablesFor(step: UniformClothStep): CouplingTables {
    const V=step.state.dryMass_kg.length,held=this.tables;
    if(held&&held.data===step.data&&held.materials===step.materials&&held.vertices===V)return held;
    if(!this.ordering||!this.stencilKernels||!this.cellKernels)throw new Error('Cloth grid coupling is not initialized');
    this.releaseTables();
    const d=this.device,topology=clothStencilTopology(step.data,step.materials,V);
    const stencil=new UniformClothGPUStencil(this.stencilKernels,this.ordering,topology),cells=new UniformClothGPUCells(this.cellKernels,this.ordering,stencil);
    const L=stencil.sections.links,A=stencil.anchorCapacity,graph=stencil.sections.graph;
    const create=(size:number,label:string,usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST)=>{
      if(size>d.limits.maxStorageBufferBindingSize)throw new Error(`Cloth ${label} exceeds storage binding capacity`);
      return d.createBuffer({label:`Cloth ${label}`,size:Math.max(16,size),usage});
    };
    const width=Math.min(1024,d.limits.maxTextureDimension2D),height=Math.ceil((1+4*A)/width);
    if(height>d.limits.maxTextureDimension2D)throw new Error('Cloth pressure mixture exceeds device texture capacity');
    const mixture=d.createTexture({label:'Cloth sparse pressure mixture',size:[width,height],format:'rgba32float',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING});
    const corrected=create(16*A,'corrected faces'),impulses=create(4*L,'pair impulses',GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC),pressureSample=create(4*L,'pressure velocity sample');
    const vertexImpulse=create(12*V,'vertex impulses',GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const status=create(STATUS_BYTES,'exchange status',GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC);
    const params=create(16,'exchange parameters',GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
    const readback=d.createBuffer({label:'Cloth impulse readback',size:READ_HEADER+36*V,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const ranges=stencil.graphBinding(graph.ranges,6*V),gather=stencil.graphBinding(graph.gather,L);
    const gatherGroup=d.createBindGroup({layout:this.gatherLayout,entries:[{binding:0,resource:ranges},{binding:1,resource:gather},{binding:2,resource:{buffer:impulses}},{binding:3,resource:{buffer:vertexImpulse}}]});
    const pressureSampleGatherGroup=d.createBindGroup({layout:this.gatherLayout,entries:[{binding:0,resource:ranges},{binding:1,resource:gather},{binding:2,resource:{buffer:pressureSample}},{binding:3,resource:{buffer:vertexImpulse}}]});
    const supportGroup=d.createBindGroup({layout:this.supportLayout,entries:[stencil.anchors,this.fields.heat.tiers,this.fields.heat.state].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const apertureGroup=d.createBindGroup({layout:this.apertureLayout,entries:[stencil.params,stencil.positions,stencil.cornerApertures,status].map((buffer,binding)=>({binding,resource:{buffer}}))});
    this.fields.setPressureMixtureFaces(mixture);
    this.tensorDrag.attach({stencil,cells,impulses,status});
    this.porePressure.attach(stencil.anchors,cells.poreTexture,A);
    return this.tables={data:step.data,materials:step.materials,vertices:V,stencil,cells,mixture,corrected,impulses,pressureSample,vertexImpulse,status,params,readback,
      apertureParams:stencil.params,gatherGroup,pressureSampleGatherGroup,supportGroup,apertureGroup};
  }

  prepare(step: UniformClothStep): void {
    if (!this.solvePipeline||!this.aperturePipeline||!this.currentAperturePipeline) throw new Error('Cloth grid coupling is not initialized');
    if (this.prepared) throw new Error('Finish the previous cloth-fluid exchange before preparing another');
    const { state, data, materials } = step, V = state.dryMass_kg.length;
    if (!(step.dt_s > 0) || !Number.isFinite(step.dt_s) || state.positions_m.length !== 3 * V
      || data.restPositions_m.length !== 3 * V || state.velocities_m_s.length !== 3 * V) throw new Error('Invalid cloth grid step');
    if (state.generation !== step.identity.generation || state.topologyRevision !== step.identity.topologyRevision) throw new Error('Stale cloth grid topology');
    if (!V) throw new Error('Cloth coupling has no structural elements');
    const predictor=step.solidPredictorVelocity_m_s??state.velocities_m_s;
    if(predictor.length!==3*V)throw new Error('Invalid cloth nonfluid velocity predictor');
    const density = materials[0]?.waterDensity_kg_m3;
    if (!(density! > 0) || materials.some(m => m.waterDensity_kg_m3 !== density)) throw new Error('Cloth coupling requires one water density');
    if (this.fields.cellSize_m.some(h => !Number.isFinite(h) || h <= 0)) throw new Error('Invalid cloth grid cell size');
    const t=this.tablesFor(step),d=this.device,fixed=step.kinematicFixed??data.fixed;
    // The cloth state goes up; nothing comes back until finish.
    const positions=new Float32Array(4*V),velocity=new Float32Array(4*V),scalars=new Float32Array(4*V);
    for(let v=0;v<V;v++){
      positions.set([state.positions_m[3*v]!,state.positions_m[3*v+1]!,state.positions_m[3*v+2]!,state.controlVolume_m3[v]!],4*v);
      const u0=predictor[3*v]!,u1=predictor[3*v+1]!,u2=predictor[3*v+2]!;
      if(!Number.isFinite(u0)||!Number.isFinite(u1)||!Number.isFinite(u2))throw new Error('Invalid cloth nonfluid velocity predictor');
      velocity.set([u0,u1,u2,0],4*v);
      scalars.set([state.dryMass_kg[v]!,state.waterVolume_m3[v]!,state.solidVolume_m3[v]!,((fixed?.[v]??0)&1)?1:0],4*v);
    }
    d.queue.writeBuffer(t.stencil.positions,0,positions);d.queue.writeBuffer(t.stencil.predictor,0,velocity);d.queue.writeBuffer(t.stencil.scalars,0,scalars);
    const exponent=step.physics?.dragExponent??1.6,solveSolid=step.physics?.solveSolid!==false;
    t.stencil.writeParameters({origin_m:this.fields.origin_m,cellSize_m:this.fields.cellSize_m,density_kg_m3:density!,pressureOnSolid:step.physics?.pressureOnSolid!==false,solveSolid});
    t.cells.writeParameters({origin_m:this.fields.origin_m,cellSize_m:this.fields.cellSize_m,density_kg_m3:density!,nonlinearDrag:(step.physics?.nonlinearDrag??step.nonlinearDrag)!==false,exponent});
    d.queue.writeBuffer(t.params,0,new Float32Array([step.dt_s,density!,this.fields.cellSize_m.reduce((a,b)=>a*b,1),exponent]));
    d.queue.writeBuffer(t.status,0,new Uint32Array([0,0xffffffff,0,0,0,0,0,0]));
    const e=d.createCommandEncoder({label:'Cloth GPU exchange stencil'});
    e.clearBuffer(t.impulses);e.clearBuffer(t.pressureSample);
    const pass=e.beginComputePass({label:'Cloth corner hard apertures'});pass.setPipeline(step.currentHardGeometry?this.currentAperturePipeline:this.aperturePipeline);
    pass.setBindGroup(0,this.fields.ownership.bindGroup);pass.setBindGroup(1,t.apertureGroup);
    if(this.fields.solid)pass.setBindGroup(2,step.currentHardGeometry?this.fields.solid.currentGeometryGroup:this.fields.solid.tileGroup);
    pass.dispatchWorkgroups(Math.ceil(8*V/64));pass.end();
    t.stencil.encode(e);t.cells.encode(e);
    d.queue.submit([e.finish()]);
    // Native-field groups bind the fields' current physical textures.
    const anchors=t.stencil.anchors;
    this.solveGroup = uniformDetailGroup(d, { layout: this.solveLayout, entries: [
      ...[anchors, t.stencil.coefficients, t.corrected, t.impulses, t.status].map((buffer, binding) => ({ binding, resource: { buffer } })),
      { binding: 5, resource: this.fields.velocity }, { binding: 6, resource: this.fields.volume }, { binding: 7, resource: { buffer: t.params } },
      { binding:8,resource:{buffer:t.stencil.occupied}},{binding:9,resource:t.mixture.createView()}] });
    this.storeGroup = uniformDetailGroup(d, { layout: this.storeLayout, entries: [{ binding: 0, resource: { buffer: anchors } }, { binding: 1, resource: { buffer: t.corrected } }, { binding: 2, resource: this.fields.velocity }] });
    this.pressureGroup = uniformDetailGroup(d,{layout:this.pressureLayout,entries:[...[anchors,t.stencil.occupied,t.impulses].map((buffer,binding)=>({binding,resource:{buffer}})),{binding:3,resource:t.mixture.createView()},{binding:4,resource:this.fields.projectedVelocity},{binding:5,resource:{buffer:t.params}},{binding:6,resource:{buffer:t.pressureSample}}]});
    this.tensorDrag.prepare({dt_s:step.dt_s,density_kg_m3:density!,exponent,solveSolid});
    this.porePressure.prepare(step,density!);
    this.prepared = {identity:{...step.identity},dt_s:step.dt_s,vertices:V,vertexOccupiedVolume_m3:Float64Array.from(state.solidVolume_m3,(s,i)=>s+state.waterVolume_m3[i]!)};
    this.encoded = false; this.projected = false;
  }

  encodeSupport(encoder: GPUCommandEncoder): void {
    const t=this.tables;
    if (!this.prepared || !t || !this.supportPipeline) throw new Error('Cloth support was not prepared');
    const pass = encoder.beginComputePass({ label: 'Cloth wake stencil regions' }); pass.setPipeline(this.supportPipeline);
    pass.setBindGroup(0, this.fields.ownership.bindGroup); pass.setBindGroup(1, t.supportGroup);
    pass.dispatchWorkgroups(Math.ceil(t.stencil.anchorCapacity / 64)); pass.end();
  }

  encodeForces(encoder: GPUCommandEncoder, dt_s: number): void {
    const p = this.prepared, t=this.tables;
    if (!p || !t || !this.solvePipeline || !this.firstUnsupportedPipeline || !this.storePipeline || !this.gatherPipeline || !this.solveGroup || !this.storeGroup) throw new Error('Cloth exchange was not prepared');
    if (this.encoded || Math.abs(p.dt_s - dt_s) > 1e-12 * p.dt_s) throw new Error('Cloth exchange must match exactly one fluid substep');
    this.porePressure.encode(encoder);
    this.tensorDrag.encode(encoder);
    const anchors=Math.ceil(t.stencil.anchorCapacity/64);
    for (const [label, pipeline, group] of [['pressure mixture', this.solvePipeline, this.solveGroup], ['commit faces', this.storePipeline, this.storeGroup]] as const) {
      const pass = encoder.beginComputePass({ label: `Cloth ${label}` }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.fields.ownership.bindGroup); pass.setBindGroup(1, group.group);if(pipeline===this.solvePipeline&&this.fields.solid)pass.setBindGroup(2,this.fields.solid.tileGroup); pass.dispatchWorkgroups(anchors);
      if(pipeline===this.solvePipeline){pass.setPipeline(this.firstUnsupportedPipeline);pass.dispatchWorkgroups(1);}
      pass.end();
    }
    const gather=encoder.beginComputePass({label:'Cloth drag impulse audit'});gather.setPipeline(this.gatherPipeline);gather.setBindGroup(0,t.gatherGroup);
    gather.dispatchWorkgroups(Math.ceil(3*p.vertices/64));gather.end();
    encoder.copyBufferToBuffer(t.vertexImpulse,0,t.readback,READ_HEADER+12*p.vertices,12*p.vertices);
    this.encoded = true;
  }

  encodeAfterProjection(encoder:GPUCommandEncoder):void {
    const p=this.prepared,t=this.tables;
    if(!p||!t||!this.encoded||this.projected||!this.pressurePipeline||!this.pressureGroup||!this.gatherPipeline)throw new Error('Cloth pressure exchange has no matching force stage');
    const response=encoder.beginComputePass({label:'Cloth projected pressure response'});response.setPipeline(this.pressurePipeline);
    response.setBindGroup(0,this.fields.ownership.bindGroup);response.setBindGroup(1,this.pressureGroup.group);response.dispatchWorkgroups(Math.ceil(t.stencil.anchorCapacity/64));response.end();
    const pass = encoder.beginComputePass({ label: 'Cloth collect water impulses' }); pass.setPipeline(this.gatherPipeline); pass.setBindGroup(0, t.gatherGroup);
    pass.dispatchWorkgroups(Math.ceil(3 * p.vertices / 64)); pass.end();
    encoder.copyBufferToBuffer(t.status, 0, t.readback, 0, STATUS_BYTES);encoder.copyBufferToBuffer(t.stencil.counts,0,t.readback,STATUS_BYTES,16);encoder.copyBufferToBuffer(t.cells.counts,0,t.readback,STATUS_BYTES+16,16);
    encoder.copyBufferToBuffer(t.vertexImpulse, 0, t.readback, READ_HEADER, 12 * p.vertices);
    const sample=encoder.beginComputePass({label:'Cloth ungated projected pressure sample'});sample.setPipeline(this.gatherPipeline);sample.setBindGroup(0,t.pressureSampleGatherGroup);
    sample.dispatchWorkgroups(Math.ceil(3*p.vertices/64));sample.end();encoder.copyBufferToBuffer(t.vertexImpulse,0,t.readback,READ_HEADER+24*p.vertices,12*p.vertices);
    this.projected = true;
  }

  async finish(): Promise<ClothImpulsePublication> {
    const p = this.prepared, read = this.tables?.readback;
    if (!p || !read || !this.projected) throw new Error('No projected cloth-fluid exchange to finish');
    try {
      await read.mapAsync(GPUMapMode.READ); const data = read.getMappedRange(), status = new Uint32Array(data, 0, 8), stencil = new Uint32Array(data, STATUS_BYTES, 4), cells = new Uint32Array(data, STATUS_BYTES + 16, 4);
      this.diagnostics=[...status,...stencil,...cells];
      if (status[4]! & CLOTH_APERTURE_STATUS.coverage) throw new Error('Cloth support hard-geometry integration did not converge');
      if (status[4]! & CLOTH_APERTURE_STATUS.unowned) throw new Error('Cloth support requires h ownership at every trilinear corner cell');
      for (const [bit, message] of STENCIL_ERRORS) if (stencil[2]! & bit) throw new Error(`Cloth GPU stencil: ${message}`);
      for (const [bit, message] of CELL_ERRORS) if (cells[2]! & bit) throw new Error(`Cloth GPU material cells: ${message}`);
      if (!stencil[1]) throw new Error('Cloth coupling has no structural elements');
      if (status[0]) throw new Error(`Cloth coupling requires resident h-MAC support: ${status[0]} unsupported anchors; first ${status[1]===0xffffffff?'(none indexed)':Array.from(new Int32Array(data,20,3))}`);
      const impulse_Ns = new Float32Array(data, READ_HEADER, 3 * p.vertices).slice();
      const dragImpulse_Ns = new Float32Array(data, READ_HEADER+12*p.vertices,3*p.vertices).slice();
      const pressureImpulse_Ns = Float32Array.from(impulse_Ns,(value,i)=>value-dragImpulse_Ns[i]!);
      if (impulse_Ns.some(x => !Number.isFinite(x))) throw new Error('Non-finite cloth fluid impulse');
      const pressureSample=new Float32Array(data,READ_HEADER+24*p.vertices,3*p.vertices);
      const fluidPressureAcceleration_m_s2=Float64Array.from(pressureSample,(value,i)=>value/Math.max(1e-30,p.vertexOccupiedVolume_m3[Math.floor(i/3)]!*p.dt_s));
      return { ...p.identity, impulse_Ns, dragImpulse_Ns, pressureImpulse_Ns, fluidPressureAcceleration_m_s2, mechanism: 'fluid' };
    } finally { if (read.mapState === 'mapped') read.unmap(); this.prepared = undefined; this.encoded = false; }
  }

  /** The resident per-link pair impulses, in the stencil's link order (diagnostics and parity tests). */
  get pairImpulses(): GPUBuffer|undefined { return this.tables?.impulses; }
  private releaseTables(): void {
    const t=this.tables;if(!t)return;this.tables=undefined;
    this.fields.setPressureMixtureFaces();t.mixture.destroy();t.cells.destroy();t.stencil.destroy();
    for(const b of [t.corrected,t.impulses,t.pressureSample,t.vertexImpulse,t.status,t.params,t.readback])b.destroy();
  }
  destroy(): void { this.porePressure.destroy();this.tensorDrag.destroy();this.releaseTables(); this.prepared = undefined; }
}
