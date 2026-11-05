import type { ClothFrameIdentity, ClothMaterial, ClothPhysicsOptions, ClothState, ClothTopologyData } from '../../cloth/types';
import type { ClothImpulsePublication } from '../../cloth/coupling';
import { porousTensor, porousPermeability } from '../../cloth/porous-material';
import {clothCellWeights} from './uniform-porous-capacity';
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
  const addMaterial = (v: number, material: ClothMaterial, axis: [number, number, number], shape: number, volume: number) => {
    geometricVolume[v]!+=volume;vertexMicro[2*v]!+=volume*material.fiberDiameter_m;vertexMicro[2*v+1]!+=volume*material.waterViscosity_Pa_s;
    const pore=porousTensor(2,1,axis,shape),radius=porousPermeability(material.solidFraction,material.fiberDiameter_m).capillaryRadius_m;
    for(let k=0;k<9;k++)poreTensor[9*v+k]!+=volume*pore[k]!;
    const cosine=Math.abs(material.contactAngle_rad-Math.PI/2)<1e-14?0:Math.cos(material.contactAngle_rad);
    poreCoefficient_Pa[v]!+=volume*material.surfaceTension_N_m*cosine/radius;
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
    for (const id of ids) addMaterial(id, material, n, 1, twiceArea * material.thickness_m / 6);
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
      addMaterial(a, material, tangent, 0, halfVolume); addMaterial(b, material, tangent, 0, halfVolume);
    }
  }
  for(let v=0;v<vertices;v++)if(geometricVolume[v]!>0){poreCoefficient_Pa[v]!/=geometricVolume[v]!;vertexMicro[2*v]!/=geometricVolume[v]!;vertexMicro[2*v+1]!/=geometricVolume[v]!;for(let k=0;k<9;k++)poreTensor[9*v+k]!/=geometricVolume[v]!;}
  const anchors = new Map<string, Anchor>(), linkCoefficients: number[][] = [], linkVertices: number[] = [], linkVolumes: number[] = [],weights:number[]=[];
  for (let v = 0; v < vertices; v++) {
    const mass = state.dryMass_kg[v]! + density! * state.waterVolume_m3[v]!;
    const fixed = step.physics?.solveSolid===false||(((step.kinematicFixed??data.fixed)?.[v] ?? 0) & 1) !== 0;
    if (!(mass > 0) && !fixed) throw new Error('Free cloth vertex has no mass');
    for (let a = 0; a < 3; a++) {
      if (geometricVolume[v]===0) continue;
      // The pressure face partition is the average of the two native cell
      // partitions. Use that SAME adjoint basis for mass and drag transfer.
      // Along the face axis this has three entries, rather than a different
      // independently rasterized eight-point MAC material fraction.
      const faceWeights=new Map<string,{p:[number,number,number];weight:number}>();
      for(const {p:cell,weight:cellWeight} of clothCellWeights(fields,state.positions_m.subarray(3*v,3*v+3))){
       const weight=.5*cellWeight;
       for(const offset of [-1,0]){const p=[...cell] as [number,number,number];p[a]!+=offset;const key=p.join(','),old=faceWeights.get(key);if(old)old.weight+=weight;else faceWeights.set(key,{p,weight});}
      }
      for (const {p,weight} of faceWeights.values()) {
        if (weight <= 1e-12) continue;
        const key = p.join(','); let anchor = anchors.get(key);
        if (!anchor) { anchor = { p, links: [[], [], []] }; anchors.set(key, anchor); }
        const index = linkCoefficients.length;
        // Drag is evaluated at deposited GRID packing, including vf=1
        // controls whose subresolution grid psi remains below one. The two
        // reserved lanes are not a second vertex-local constitutive model.
        linkCoefficients.push([fixed ? 0 : weight * mass,0,0,predictor[3 * v + a]!]);
        linkVolumes.push(weight * (state.solidVolume_m3[v]! + state.waterVolume_m3[v]!) * (step.physics?.pressureOnSolid===false?-1:1));
        weights.push(weight);
        linkVertices.push(3 * v + a); anchor.links[a]!.push(index);
      }
    }
  }
  const ordered = [...anchors.values()].sort((a, b) => a.p[2] - b.p[2] || a.p[1] - b.p[1] || a.p[0] - b.p[0]);
  const records = new Int32Array(8 * ordered.length), coefficients = new Float32Array(4 * linkCoefficients.length), occupiedVolumes_m3 = new Float32Array(linkVolumes.length);
  const linkWeights=new Float32Array(weights.length),linkFaces=new Uint32Array(weights.length);
  const vertexLinks = Array.from({ length: 3 * vertices }, () => [] as number[]);
  let at = 0;
  ordered.forEach((anchor, index) => {
    records.set([...anchor.p, at, ...anchor.links.map(links => links.length), 0], 8 * index);
    for (let axis=0;axis<3;axis++)for (const original of anchor.links[axis]!) {
      coefficients.set(linkCoefficients[original]!, 4 * at); occupiedVolumes_m3[at] = linkVolumes[original]!;linkWeights[at]=weights[original]!;linkFaces[at]=3*index+axis;vertexLinks[linkVertices[original]!]!.push(at++);
    }
  });
  for (const coefficient of coefficients) if (!Number.isFinite(coefficient)) throw new Error('Cloth coupling exceeds f32 representable range');
  const ranges = new Uint32Array(2 * vertexLinks.length), gather = new Uint32Array(linkCoefficients.length);
  at = 0;
  vertexLinks.forEach((links, index) => { ranges.set([at, links.length], 2 * index); gather.set(links, at); at += links.length; });
  return { anchors: records, coefficients, occupiedVolumes_m3, ranges, gather, coordinates: ordered.map(a => a.p), identity: { ...step.identity },
    dt_s: step.dt_s, density_kg_m3: density!, vertices,exponent:step.physics?.dragExponent??1.6,linkWeights,linkFaces,vertexOccupiedVolume_m3:Float64Array.from(state.solidVolume_m3,(v,i)=>v+state.waterVolume_m3[i]!),poreTensor,poreCoefficient_Pa,vertexMicro };
}

/** Correctness-first CPU cloth / real WebGPU NB-FLIP bridge. Only topology
 * stencils and compact impulses cross the CPU boundary; water stays on GPU. */
export class UniformClothGridCoupling {
  diagnostics:readonly number[]=[];
  readonly id = 'physics.cloth.porous-grid-drag';
  private readonly solveLayout: GPUBindGroupLayout;
  private readonly storeLayout: GPUBindGroupLayout;
  private readonly gatherLayout: GPUBindGroupLayout;
  private solvePipeline?: GPUComputePipeline;
  private storePipeline?: GPUComputePipeline;
  private gatherPipeline?: GPUComputePipeline;
  private supportPipeline?: GPUComputePipeline;
  private supportLayout: GPUBindGroupLayout;
  private supportGroup?: GPUBindGroup;
  private pressureLayout: GPUBindGroupLayout;
  private pressurePipeline?: GPUComputePipeline;
  private pressureGroup?: UniformDetailGroup;
  private projected = false;
  private mixtureTexture?:GPUTexture;
  private tensorDrag:UniformClothTensorDrag;
  private porePressure:UniformClothPorePressure;
  private buffers: GPUBuffer[] = [];
  private solveGroup?: UniformDetailGroup;
  private storeGroup?: UniformDetailGroup;
  private gatherGroup?: GPUBindGroup;
  private pressureSampleGatherGroup?:GPUBindGroup;
  private readback?: GPUBuffer;
  private prepared?: Prepared;
  private encoded = false;
  private status?: GPUBuffer;
  private impulse?: GPUBuffer;

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
  }

  async initialize(): Promise<void> {
    const topology = uniformMixedTopologyWGSL(this.fields.ownership.capacity, 0);
    const solid=this.fields.solid;
    const anchor = 'struct ClothAnchor { p:vec3i, first:u32, counts:vec3u, unused:u32 };';
    const mixture = 'struct MixtureFace {p:vec4i,mobility:vec4f,mixture:vec4f,fluid:vec4f}; struct MixtureFaces {header:vec4u,faces:array<MixtureFace>};';
    const support = this.device.createShaderModule({ label: 'Cloth local support wake', code: topology + anchor + this.fields.heat.promoteWGSL(1, 1, 2) + /* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:array<ClothAnchor>;
@compute @workgroup_size(64) fn wake(@builtin(global_invocation_id) gid:vec3u) {
 let i=gid.x;if(i>=arrayLength(&anchors)){return;}
 // Wake the finite stencil and one-cell derivative closure, never the domain.
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
 let p=anchors[i].p-32*umFramePage()+vec3i(x,y,z);let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));
 if(umKeyStored(tile)){rhPromote(umPageOfTile(tile));}
 }}}
}` });
    const solve = uniformDetailModule(this.device, { label: 'Cloth native pressure mixture preparation', code: topology + uniformMixedSolidWGSL(solid?2:undefined,solid?.coarse?.count) + /* wgsl */`
${anchor}
@group(1) @binding(0) var<storage,read> anchors:array<ClothAnchor>;
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
 let i=gid.x;if(i>=arrayLength(&anchors)){return;}
 atomicAdd(&status[2],1u);
 var a=anchors[i];a.p-=32*umFramePage();var valid=supported(a.p);
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
 if(i==0u){pressureStore(0u,bitcast<vec4f>(vec4u(arrayLength(&anchors),0u,0u,0u)));}
 pressureStore(1u+4u*i,bitcast<vec4f>(pressureFace.p));pressureStore(2u+4u*i,pressureFace.mobility);
 pressureStore(3u+4u*i,pressureFace.mixture);pressureStore(4u+4u*i,pressureFace.fluid);
}` });
    const pressure = uniformDetailModule(this.device, {label:'Cloth pressure response gather', code:topology+anchor+mixture+/* wgsl */`
@group(1) @binding(0) var<storage,read> anchors:array<ClothAnchor>;
@group(1) @binding(1) var<storage,read> occupied:array<f32>;
@group(1) @binding(2) var<storage,read_write> impulses:array<f32>;
@group(1) @binding(3) var mixture:texture_2d<f32>;
fn mixtureWord(i:u32)->vec4f{let width=textureDimensions(mixture).x;return textureLoad(mixture,vec2i(i32(i%width),i32(i/width)),0);}
@group(1) @binding(4) var velocity:texture_3d<f32>;
@group(1) @binding(5) var<uniform> physics:vec4f;
@group(1) @binding(6) var<storage,read_write> pressureSample:array<f32>;
@compute @workgroup_size(64) fn pressureResponse(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&anchors)){return;}var a=anchors[i];a.p-=32*umFramePage();let flags=bitcast<vec4i>(mixtureWord(1u+4u*i)).w;let fluid=mixtureWord(4u+4u*i);
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
@group(1) @binding(0) var<storage,read> anchors:array<ClothAnchor>;
@group(1) @binding(1) var<storage,read> corrected:array<vec4f>;
@group(1) @binding(2) var output:texture_storage_3d<rgba32float,write>;
fn supported(p:vec3i)->bool {
 let tile=512u*umBlockOf(p>>vec3u(5u))+umTileLocal(vec3u(p>>vec3u(2u)));return umTileWidth(tile)==1u&&umKeyStored(tile);
}
@compute @workgroup_size(64) fn commitDrag(@builtin(global_invocation_id) gid:vec3u) {
 let i=gid.x;if(i>=arrayLength(&anchors)){return;}
 var a=anchors[i];a.p-=32*umFramePage();if(!supported(a.p)){return;}
 for(var axis=0u;axis<3u;axis++){if(a.counts[axis]!=0u){var q=a.p;q[axis]+=1;if(!supported(q)){return;}}}
 textureStore(output,a.p,corrected[i]);
}` });
    const gather = this.device.createShaderModule({ label: 'Cloth impulse gather', code: /* wgsl */`
@group(0) @binding(0) var<storage,read> ranges:array<vec2u>;
@group(0) @binding(1) var<storage,read> indices:array<u32>;
@group(0) @binding(2) var<storage,read> impulses:array<f32>;
@group(0) @binding(3) var<storage,read_write> vertexImpulse:array<f32>;
@compute @workgroup_size(64) fn gather(@builtin(global_invocation_id) gid:vec3u) {
 let i=gid.x;if(i>=arrayLength(&ranges)){return;}let range=ranges[i];var sum=0.0;
 for(var j=0u;j<range.y;j++){sum+=impulses[indices[range.x+j]];}vertexImpulse[i]=sum;
}` });
    for (const module of [solve, store, gather, support, pressure]) {
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join('\n'));
    }
    [this.solvePipeline, this.storePipeline, this.gatherPipeline] = await Promise.all([
      uniformDetailPipeline(this.device,this.fields.ownership,{ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.fields.ownership.bindLayout, this.solveLayout,...(solid?[solid.tileLayout]:[])] }), compute: { module: solve, entryPoint: 'preparePressureMixture' } }),
      uniformDetailPipeline(this.device,this.fields.ownership,{ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.fields.ownership.bindLayout, this.storeLayout] }), compute: { module: store, entryPoint: 'commitDrag' } }),
      this.device.createComputePipelineAsync({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.gatherLayout] }), compute: { module: gather, entryPoint: 'gather' } }),
    ]);
    this.supportPipeline = await this.device.createComputePipelineAsync({ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.fields.ownership.bindLayout, this.supportLayout] }), compute: { module: support, entryPoint: 'wake' } });
    this.pressurePipeline = await uniformDetailPipeline(this.device,this.fields.ownership,{ layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.fields.ownership.bindLayout, this.pressureLayout] }), compute: { module: pressure, entryPoint: 'pressureResponse' } });
    await Promise.all([this.tensorDrag.initialize(),this.porePressure.initialize()]);
  }

  prepare(step: UniformClothStep): void {
    if (!this.solvePipeline) throw new Error('Cloth grid coupling is not initialized');
    if (this.prepared) throw new Error('Finish the previous cloth-fluid exchange before preparing another');
    const p = prepareUniformClothStencil(this.fields, step);
    if (!p.vertices || !p.coordinates.length) throw new Error('Cloth coupling has no structural elements');
    const maxGroups = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (Math.ceil(Math.max(p.coordinates.length, p.vertices * 3) / 64) > maxGroups) throw new Error('Cloth stencil exceeds the supported dispatch capacity');
    this.releaseBuffers();
    const create = (size: number, label: string, data?: ArrayBufferView, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => {
      if (size > this.device.limits.maxStorageBufferBindingSize) throw new Error(`Cloth ${label} exceeds storage binding capacity`);
      const buffer = this.device.createBuffer({ label: `Cloth ${label}`, size: Math.max(16, size), usage }); this.buffers.push(buffer);
      if (data) this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength); return buffer;
    };
    const anchors = create(p.anchors.byteLength, 'MAC anchors', p.anchors), links = create(p.coefficients.byteLength, 'drag links', p.coefficients);
    const occupied = create(p.occupiedVolumes_m3.byteLength,'occupied volume',p.occupiedVolumes_m3);
    const width=Math.min(1024,this.device.limits.maxTextureDimension2D),height=Math.ceil((1+4*p.coordinates.length)/width);
    if(height>this.device.limits.maxTextureDimension2D)throw new Error('Cloth pressure mixture exceeds device texture capacity');
    const mixture=this.device.createTexture({label:'Cloth sparse pressure mixture',size:[width,height],format:'rgba32float',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING});
    this.mixtureTexture=mixture;this.fields.setPressureMixtureFaces(mixture);
    const corrected = create(16 * p.coordinates.length, 'corrected faces'), impulses = create(4 * p.gather.length, 'pair impulses',undefined,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC), pressureSample=create(4*p.gather.length,'pressure velocity sample');
    const ranges = create(p.ranges.byteLength, 'vertex ranges', p.ranges), gather = create(p.gather.byteLength, 'vertex links', p.gather);
    this.status = create(16, 'exchange status', new Uint32Array([0, 0xffffffff, 0, 0]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.impulse = create(12 * p.vertices, 'vertex impulses', undefined, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const params = create(16, 'exchange parameters', new Float32Array([p.dt_s, p.density_kg_m3, this.fields.cellSize_m.reduce((a, b) => a * b, 1), p.exponent]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.readback = this.device.createBuffer({ label: 'Cloth impulse readback', size: 16 + 36 * p.vertices, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); this.buffers.push(this.readback);
    this.solveGroup = uniformDetailGroup(this.device, { layout: this.solveLayout, entries: [
      ...[anchors, links, corrected, impulses, this.status].map((buffer, binding) => ({ binding, resource: { buffer } })),
      { binding: 5, resource: this.fields.velocity }, { binding: 6, resource: this.fields.volume }, { binding: 7, resource: { buffer: params } },
      { binding:8,resource:{buffer:occupied}},{binding:9,resource:mixture.createView()}] });
    this.storeGroup = uniformDetailGroup(this.device, { layout: this.storeLayout, entries: [{ binding: 0, resource: { buffer: anchors } }, { binding: 1, resource: { buffer: corrected } }, { binding: 2, resource: this.fields.velocity }] });
    this.gatherGroup = this.device.createBindGroup({ layout: this.gatherLayout, entries: [ranges, gather, impulses, this.impulse].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.pressureSampleGatherGroup=this.device.createBindGroup({layout:this.gatherLayout,entries:[ranges,gather,pressureSample,this.impulse].map((buffer,binding)=>({binding,resource:{buffer}}))});
    this.supportGroup = this.device.createBindGroup({ layout: this.supportLayout, entries: [anchors, this.fields.heat.tiers, this.fields.heat.state].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.pressureGroup = uniformDetailGroup(this.device,{layout:this.pressureLayout,entries:[...[anchors,occupied,impulses].map((buffer,binding)=>({binding,resource:{buffer}})),{binding:3,resource:mixture.createView()},{binding:4,resource:this.fields.projectedVelocity},{binding:5,resource:{buffer:params}},{binding:6,resource:{buffer:pressureSample}}]});
    this.tensorDrag.prepare(p,step,{anchors,links,occupied,impulses,status:this.status});
    this.porePressure.prepare(p,step,anchors);
    this.prepared = p; this.encoded = false; this.projected = false;
  }

  encodeSupport(encoder: GPUCommandEncoder): void {
    if (!this.prepared || !this.supportPipeline || !this.supportGroup) throw new Error('Cloth support was not prepared');
    const pass = encoder.beginComputePass({ label: 'Cloth wake stencil regions' }); pass.setPipeline(this.supportPipeline);
    pass.setBindGroup(0, this.fields.ownership.bindGroup); pass.setBindGroup(1, this.supportGroup);
    pass.dispatchWorkgroups(Math.ceil(this.prepared.coordinates.length / 64)); pass.end();
  }

  encodeForces(encoder: GPUCommandEncoder, dt_s: number): void {
    const p = this.prepared;
    if (!p || !this.solvePipeline || !this.storePipeline || !this.gatherPipeline || !this.solveGroup || !this.storeGroup || !this.gatherGroup) throw new Error('Cloth exchange was not prepared');
    if (this.encoded || Math.abs(p.dt_s - dt_s) > 1e-12 * p.dt_s) throw new Error('Cloth exchange must match exactly one fluid substep');
    this.porePressure.encode(encoder);
    this.tensorDrag.encode(encoder);
    for (const [label, pipeline, group] of [['pressure mixture', this.solvePipeline, this.solveGroup], ['commit faces', this.storePipeline, this.storeGroup]] as const) {
      const pass = encoder.beginComputePass({ label: `Cloth ${label}` }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.fields.ownership.bindGroup); pass.setBindGroup(1, group.group);if(pipeline===this.solvePipeline&&this.fields.solid)pass.setBindGroup(2,this.fields.solid.tileGroup); pass.dispatchWorkgroups(Math.ceil(p.coordinates.length / 64)); pass.end();
    }
    const gather=encoder.beginComputePass({label:'Cloth drag impulse audit'});gather.setPipeline(this.gatherPipeline);gather.setBindGroup(0,this.gatherGroup);
    gather.dispatchWorkgroups(Math.ceil(3*p.vertices/64));gather.end();
    encoder.copyBufferToBuffer(this.impulse!,0,this.readback!,16+12*p.vertices,12*p.vertices);
    this.encoded = true;
  }

  encodeAfterProjection(encoder:GPUCommandEncoder):void {
    const p=this.prepared;
    if(!p||!this.encoded||this.projected||!this.pressurePipeline||!this.pressureGroup||!this.gatherPipeline||!this.gatherGroup)throw new Error('Cloth pressure exchange has no matching force stage');
    const response=encoder.beginComputePass({label:'Cloth projected pressure response'});response.setPipeline(this.pressurePipeline);
    response.setBindGroup(0,this.fields.ownership.bindGroup);response.setBindGroup(1,this.pressureGroup.group);response.dispatchWorkgroups(Math.ceil(p.coordinates.length/64));response.end();
    const pass = encoder.beginComputePass({ label: 'Cloth collect water impulses' }); pass.setPipeline(this.gatherPipeline); pass.setBindGroup(0, this.gatherGroup);
    pass.dispatchWorkgroups(Math.ceil(3 * p.vertices / 64)); pass.end();
    encoder.copyBufferToBuffer(this.status!, 0, this.readback!, 0, 16); encoder.copyBufferToBuffer(this.impulse!, 0, this.readback!, 16, 12 * p.vertices);
    const sample=encoder.beginComputePass({label:'Cloth ungated projected pressure sample'});sample.setPipeline(this.gatherPipeline);sample.setBindGroup(0,this.pressureSampleGatherGroup!);
    sample.dispatchWorkgroups(Math.ceil(3*p.vertices/64));sample.end();encoder.copyBufferToBuffer(this.impulse!,0,this.readback!,16+24*p.vertices,12*p.vertices);
    this.projected = true;
  }

  async finish(): Promise<ClothImpulsePublication> {
    const p = this.prepared, read = this.readback;
    if (!p || !read || !this.projected) throw new Error('No projected cloth-fluid exchange to finish');
    try {
      await read.mapAsync(GPUMapMode.READ); const data = read.getMappedRange(), status = new Uint32Array(data, 0, 4);
      this.diagnostics=Array.from(status);
      if (status[0]) throw new Error(`Cloth coupling requires resident h-MAC support: ${status[0]} unsupported anchors; first ${p.coordinates[status[1]!]}`);
      const impulse_Ns = new Float32Array(data, 16, 3 * p.vertices).slice();
      const dragImpulse_Ns = new Float32Array(data, 16+12*p.vertices,3*p.vertices).slice();
      const pressureImpulse_Ns = Float32Array.from(impulse_Ns,(value,i)=>value-dragImpulse_Ns[i]!);
      if (impulse_Ns.some(x => !Number.isFinite(x))) throw new Error('Non-finite cloth fluid impulse');
      const pressureSample=new Float32Array(data,16+24*p.vertices,3*p.vertices);
      const fluidPressureAcceleration_m_s2=Float64Array.from(pressureSample,(value,i)=>value/Math.max(1e-30,p.vertexOccupiedVolume_m3[Math.floor(i/3)]!*p.dt_s));
      return { ...p.identity, impulse_Ns, dragImpulse_Ns, pressureImpulse_Ns, fluidPressureAcceleration_m_s2, mechanism: 'fluid' };
    } finally { if (read.mapState === 'mapped') read.unmap(); this.prepared = undefined; this.encoded = false; }
  }

  private releaseBuffers(): void { this.fields.setPressureMixtureFaces();this.mixtureTexture?.destroy();this.mixtureTexture=undefined;for (const buffer of this.buffers) buffer.destroy(); this.buffers = []; }
  destroy(): void { this.porePressure.destroy();this.tensorDrag.destroy();this.releaseBuffers(); this.prepared = undefined; }
}
