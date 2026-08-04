import type { PassBroker } from "./webgpu-pass-broker";
import { octreeCompensatedF32WGSL } from "./octree-compensated-f32.wgsl";
import {
  OCTREE_LOSASSO_VCYCLE_SCHEDULE,
  type OctreeLosassoFirstOrderVCycle,
  type OctreeLosassoVCycleHierarchySource,
} from "./webgpu-octree-losasso-vcycle";

const vcycleWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Params { damping:f32, reserved0:u32, reserved1:u32, reserved2:u32 }
struct Face { negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,
  area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32 }
@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(1)var<storage,read> control:array<u32>;
@group(0)@binding(2)var<storage,read> offsets:array<u32>;
@group(0)@binding(3)var<storage,read> rowFaces:array<u32>;
@group(0)@binding(4)var<storage,read> faces:array<Face>;
@group(0)@binding(5)var<storage,read_write> b:array<f32>;
@group(0)@binding(6)var<storage,read_write> x:array<f32>;
@group(0)@binding(7)var<storage,read_write> output:array<f32>;
@group(0)@binding(8)var<storage,read> solve:array<u32>;
fn stopped()->bool{return solve[0]!=0u||solve[1]!=0u||control[3]!=1u;}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
${octreeCompensatedF32WGSL}
fn image(row:u32)->vec2f{
 if(row+1u>=arrayLength(&offsets)){return vec2f(3.402823e38,-1.);}let begin=offsets[row];let end=offsets[row+1u];
 // Coarse hierarchy rows retain fine boundary patches and may have thousands
 // of incidences. CSR order is fixed at publication, so compensated f32 is
 // deterministic and only malformed bounds need to fail closed here.
 if(begin>end||end>arrayLength(&rowFaces)){return vec2f(3.402823e38,-1.);}
 let centre=x[row];var imageSum=CompensatedF32(0.,0.);
 var diagonalSum=CompensatedF32(0.,0.);var valid=true;
 for(var cursor=begin;cursor<end;cursor+=1u){let faceId=rowFaces[cursor];if(faceId>=arrayLength(&faces)){valid=false;continue;}
  let face=faces[faceId];let coefficient=(face.openFraction*face.area)*face.inverseDistance;var difference=centre;
  if(face.negativeRow==row){if(face.positiveRow!=INVALID){if(face.positiveRow>=arrayLength(&x)){valid=false;continue;}difference-=x[face.positiveRow];}}
  else{if(face.negativeRow>=arrayLength(&x)){valid=false;continue;}difference-=x[face.negativeRow];}
  let term=coefficient*difference;if(!finite(coefficient)||coefficient<0.||!finite(term)){valid=false;continue;}
  diagonalSum=addCompensatedF32(diagonalSum,coefficient);
  imageSum=addCompensatedF32(imageSum,term);
 }
 let value=compensatedValue(imageSum);let diagonal=compensatedValue(diagonalSum);
 return select(vec2f(3.402823e38,-1.),vec2f(value,diagonal),valid&&finite(value)&&finite(diagonal)&&diagonal>0.);
}
@compute @workgroup_size(64)fn clearLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=0.;
}
@compute @workgroup_size(64)fn copyLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=b[row];
}
@compute @workgroup_size(64)fn jacobiLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=control[1]){return;}let pair=image(row);
 output[row]=select(0.,x[row]+p.damping*(b[row]-pair.x)/pair.y,pair.y>0.);
}
@compute @workgroup_size(64)fn residualLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=b[row]-image(row).x;
}
`;

const transferWGSL = /* wgsl */ `
@group(0)@binding(0)var<storage,read> fineControl:array<u32>;
@group(0)@binding(1)var<storage,read> coarseControl:array<u32>;
@group(0)@binding(2)var<storage,read> parents:array<u32>;
@group(0)@binding(3)var<storage,read> childOffsets:array<u32>;
@group(0)@binding(4)var<storage,read> childList:array<u32>;
@group(0)@binding(5)var<storage,read_write> input:array<f32>;
@group(0)@binding(6)var<storage,read_write> output:array<f32>;
@group(0)@binding(7)var<storage,read> solve:array<u32>;
${octreeCompensatedF32WGSL}
fn stopped()->bool{return solve[0]!=0u||solve[1]!=0u
 ||fineControl[3]!=1u||coarseControl[3]!=1u;}
@compute @workgroup_size(64)fn restrictLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=coarseControl[1]){return;}
 let begin=childOffsets[row];let end=childOffsets[row+1u];
 if(begin>end||end>fineControl[1]){output[row]=0.;return;}
 var sum=CompensatedF32(0.,0.);
 for(var cursor=begin;cursor<end;cursor+=1u){let child=childList[cursor];
  if(child<fineControl[1]){sum=addCompensatedF32(sum,input[child]);}
 }
 output[row]=compensatedValue(sum);
}
@compute @workgroup_size(64)fn prolongLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=fineControl[1]){return;}let parent=parents[row];
 if(parent<coarseControl[1]){
  output[row]+=input[parent];
 }
}
`;

function fusedSubL0WGSL(layout: NonNullable<
  OctreeLosassoVCycleHierarchySource["fusedSubL0"]>, levelCount: number): string {
  const vectorCapacities = layout.levelRowCapacities;
  const vectorBases: number[] = [];
  let vectorBase = 0;
  for (const capacity of vectorCapacities) {
    vectorBases.push(vectorBase);
    vectorBase += Math.ceil(capacity * 4 / 256) * 64;
  }
  return /* wgsl */ `
const INVALID:u32=0xffffffffu;
const LEVEL_COUNT:u32=${levelCount}u;
const VECTOR_CAPACITIES:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${vectorCapacities.map(value=>`${value}u`).join(",")});
const VECTOR_BASES:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${vectorBases.map(value=>`${value}u`).join(",")});
const TRANSITION_STRIDE:u32=${layout.transitionStrideWords}u;
const CONTROL_OFFSET:u32=${layout.controlOffsetWords}u;
const ROW_OFFSETS_OFFSET:u32=${layout.rowOffsetsOffsetWords}u;
const ROW_FACES_OFFSET:u32=${layout.rowFacesOffsetWords}u;
const FACES_OFFSET:u32=${layout.facesOffsetWords}u;
const PARENTS_OFFSET:u32=${layout.parentsOffsetWords}u;
const CHILD_OFFSETS_OFFSET:u32=${layout.childOffsetsOffsetWords}u;
const CHILD_LIST_OFFSET:u32=${layout.childListOffsetWords}u;
struct Params { damping:f32, reserved0:u32, reserved1:u32, reserved2:u32 }
struct Face { negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,
 area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32 }
@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(1)var<storage,read> hierarchy:array<u32>;
@group(0)@binding(2)var<storage,read_write> rhsVectors:array<f32>;
@group(0)@binding(3)var<storage,read_write> xAVectors:array<f32>;
@group(0)@binding(4)var<storage,read_write> xBVectors:array<f32>;
@group(0)@binding(5)var<storage,read_write> residualVectors:array<f32>;
@group(0)@binding(6)var<storage,read> solve:array<u32>;
${octreeCompensatedF32WGSL}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn transitionBase(level:u32)->u32{return (level-1u)*TRANSITION_STRIDE;}
fn levelControl(level:u32,word:u32)->u32{
 return hierarchy[transitionBase(level)+CONTROL_OFFSET+word];
}
fn vectorIndex(level:u32,row:u32)->u32{return VECTOR_BASES[level]+row;}
fn correctionValue(fromB:bool,level:u32,row:u32)->f32{
 if(fromB){return xBVectors[vectorIndex(level,row)];}
 return xAVectors[vectorIndex(level,row)];
}
fn loadFace(level:u32,faceId:u32)->Face{let base=transitionBase(level)+FACES_OFFSET+8u*faceId;
 return Face(hierarchy[base],hierarchy[base+1u],hierarchy[base+2u],hierarchy[base+3u],
  bitcast<f32>(hierarchy[base+4u]),bitcast<f32>(hierarchy[base+5u]),
  bitcast<f32>(hierarchy[base+6u]),bitcast<f32>(hierarchy[base+7u]));
}
fn image(level:u32,row:u32,fromB:bool)->vec2f{
 let base=transitionBase(level);let begin=hierarchy[base+ROW_OFFSETS_OFFSET+row];
 let end=hierarchy[base+ROW_OFFSETS_OFFSET+row+1u];
 let incidenceCapacity=2u*${layout.faceCapacity}u;
 if(begin>end||end>incidenceCapacity){return vec2f(3.402823e38,-1.);}
 let centre=correctionValue(fromB,level,row);
 var imageSum=CompensatedF32(0.,0.);var diagonalSum=CompensatedF32(0.,0.);var valid=true;
 for(var cursor=begin;cursor<end;cursor+=1u){let faceId=hierarchy[base+ROW_FACES_OFFSET+cursor];
  if(faceId>=${layout.faceCapacity}u){valid=false;continue;}let face=loadFace(level,faceId);
  let coefficient=(face.openFraction*face.area)*face.inverseDistance;var difference=centre;
  if(face.negativeRow==row){if(face.positiveRow!=INVALID){if(face.positiveRow>=VECTOR_CAPACITIES[level]){valid=false;continue;}
    difference-=correctionValue(fromB,level,face.positiveRow);}}
  else{if(face.negativeRow>=VECTOR_CAPACITIES[level]){valid=false;continue;}
   difference-=correctionValue(fromB,level,face.negativeRow);}
  let term=coefficient*difference;if(!finite(coefficient)||coefficient<0.||!finite(term)){valid=false;continue;}
  diagonalSum=addCompensatedF32(diagonalSum,coefficient);
  imageSum=addCompensatedF32(imageSum,term);
 }
 let value=compensatedValue(imageSum);let diagonal=compensatedValue(diagonalSum);
 return select(vec2f(3.402823e38,-1.),vec2f(value,diagonal),
  valid&&finite(value)&&finite(diagonal)&&diagonal>0.);
}
fn synchronize(){storageBarrier();workgroupBarrier();}
fn restrictInto(coarseLevel:u32,lane:u32,enabled:bool){
 let base=transitionBase(coarseLevel);let rows=select(0u,levelControl(coarseLevel,1u),enabled);
 for(var row=lane;row<rows;row+=256u){let begin=hierarchy[base+CHILD_OFFSETS_OFFSET+row];
  let end=hierarchy[base+CHILD_OFFSETS_OFFSET+row+1u];var sum=CompensatedF32(0.,0.);
  for(var cursor=begin;cursor<end;cursor+=1u){let child=hierarchy[base+CHILD_LIST_OFFSET+cursor];
   sum=addCompensatedF32(sum,residualVectors[vectorIndex(coarseLevel-1u,child)]);
  }
  rhsVectors[vectorIndex(coarseLevel,row)]=compensatedValue(sum);
  xAVectors[vectorIndex(coarseLevel,row)]=0.;
 }
 synchronize();
}
fn relaxLevel(level:u32,fromB:bool,lane:u32,enabled:bool){
 let rows=select(0u,levelControl(level,1u),enabled);
 for(var row=lane;row<rows;row+=256u){let pair=image(level,row,fromB);
  let value=select(0.,correctionValue(fromB,level,row)+p.damping*
    (rhsVectors[vectorIndex(level,row)]-pair.x)/pair.y,pair.y>0.);
  if(fromB){xAVectors[vectorIndex(level,row)]=value;}
  else{xBVectors[vectorIndex(level,row)]=value;}
 }
 synchronize();
}
fn formResidual(level:u32,lane:u32,enabled:bool){
 let rows=select(0u,levelControl(level,1u),enabled);
 for(var row=lane;row<rows;row+=256u){
  residualVectors[vectorIndex(level,row)]=rhsVectors[vectorIndex(level,row)]-image(level,row,false).x;
 }
 synchronize();
}
fn prolongInto(fineLevel:u32,lane:u32,enabled:bool){
 let coarseLevel=fineLevel+1u;let base=transitionBase(coarseLevel);
 let coarseRows=levelControl(coarseLevel,1u);
 let fineRows=select(0u,hierarchy[base+CHILD_OFFSETS_OFFSET+coarseRows],enabled);
 for(var row=lane;row<fineRows;row+=256u){let parent=hierarchy[base+PARENTS_OFFSET+row];
  if(parent<coarseRows){xAVectors[vectorIndex(fineLevel,row)]+=xAVectors[vectorIndex(coarseLevel,parent)];}
 }
 synchronize();
}
@compute @workgroup_size(256)
fn fusedSubL0(@builtin(local_invocation_index)lane:u32){
 var enabled=solve[0]==0u&&solve[1]==0u;
 for(var level=1u;level<LEVEL_COUNT;level+=1u){enabled=enabled&&levelControl(level,3u)==1u
  &&levelControl(level,1u)<=VECTOR_CAPACITIES[level];}
 restrictInto(1u,lane,enabled);
 for(var level=1u;level+1u<LEVEL_COUNT;level+=1u){
  relaxLevel(level,false,lane,enabled);relaxLevel(level,true,lane,enabled);
  formResidual(level,lane,enabled);restrictInto(level+1u,lane,enabled);
 }
 let bottom=LEVEL_COUNT-1u;
 for(var sweep=0u;sweep<8u;sweep+=1u){
  if((sweep&1u)==0u){relaxLevel(bottom,false,lane,enabled);}
  else{relaxLevel(bottom,true,lane,enabled);}
 }
 for(var level=bottom-1u;level>=1u;level-=1u){
  prolongInto(level,lane,enabled);relaxLevel(level,false,lane,enabled);
  relaxLevel(level,true,lane,enabled);
 }
 prolongInto(0u,lane,enabled);
}
`;
}

interface VectorSlice { readonly buffer:GPUBuffer;readonly offset:number;readonly size:number }

interface LevelPipelines {
  clear: GPUComputePipeline;
  copy: GPUComputePipeline;
  jacobi: GPUComputePipeline;
  residual: GPUComputePipeline;
}
interface TransferPipelines {
  restrict:GPUComputePipeline;prolong:GPUComputePipeline;
}

/** Plain Losasso first-order V-cycle. No alternate operator or boundary band exists. */
export class WebGPUOctreeLosassoVCycle implements OctreeLosassoFirstOrderVCycle {
  readonly backend = "losasso" as const;
  readonly operatorFamily = "closed-form-axis-face" as const;
  readonly levelOperator = "same-first-order-operator" as const;
  readonly boundaryBandSmoother = "absent" as const;
  readonly operatorOrder = 1 as const;
  readonly isSymmetricPositiveDefinite = true as const;
  readonly convergenceTail = "gpu-zero-indirect" as const;
  readonly schedule = OCTREE_LOSASSO_VCYCLE_SCHEDULE;
  readonly encodedSetupDispatchCount = 0;
  readonly encodedCorrectionDispatchCount: number;
  readonly allocatedBytes: number;
  readonly initializationTasks: readonly { label:string;run:()=>Promise<void> }[];

  private readonly params:GPUBuffer;
  private readonly vectorArenas:readonly GPUBuffer[];
  private readonly rhs:VectorSlice[]=[];
  private readonly xA:VectorSlice[]=[];
  private readonly xB:VectorSlice[]=[];
  private readonly residual:VectorSlice[]=[];
  private levelPipelines?:readonly LevelPipelines[];
  private transferPipelines?:readonly TransferPipelines[];
  private fusedSubL0Pipeline?:GPUComputePipeline;
  private readonly bindGroupCache=new Map<GPUComputePipeline,{
    entries:readonly {binding:number;resource:GPUBufferBinding}[];group:GPUBindGroup;
  }[]>();
  private readonly useFusedSubL0:boolean;
  private destroyed=false;

  constructor(private readonly device:GPUDevice,
    readonly hierarchy:OctreeLosassoVCycleHierarchySource) {
    if(hierarchy.levels.length<1||hierarchy.transfers.length!==hierarchy.levels.length-1){
      throw new RangeError("Losasso V-cycle needs a complete one-or-more-level hierarchy");
    }
    // The fixed-order result is covered by the bounded-f32 D4 gate; every
    // multi-level production hierarchy therefore uses the fused schedule.
    this.useFusedSubL0=Boolean(hierarchy.fusedSubL0&&hierarchy.levels.length>1);
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    this.params=device.createBuffer({label:"Losasso V-cycle Jacobi constants",size:16,
      usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(this.params,0,Float32Array.of(2/3,0,0,0));
    for(const [level,source] of hierarchy.levels.entries()){
      if(!Number.isSafeInteger(source.rowCapacity)||source.rowCapacity<1)throw new RangeError(`Losasso V-cycle level ${level} capacity is invalid`);
    }
    const levelVectorCapacities=hierarchy.fusedSubL0?.levelRowCapacities
      ?? hierarchy.levels.map(level=>level.rowCapacity);
    if(levelVectorCapacities.length!==hierarchy.levels.length
      ||levelVectorCapacities.some((capacity,level)=>!Number.isSafeInteger(capacity)
        ||capacity<1||capacity>hierarchy.levels[level]!.rowCapacity)){
      throw new RangeError("Losasso V-cycle packed level capacities are invalid");
    }
    const vectorOffsets:number[]=[];let vectorArenaBytes=0;
    for(const capacity of levelVectorCapacities){vectorOffsets.push(vectorArenaBytes);
      vectorArenaBytes+=Math.ceil(capacity*4/256)*256;}
    this.vectorArenas=Object.freeze(["rhs","correction A","correction B","residual"].map(name=>
      device.createBuffer({label:`Losasso V-cycle packed ${name}`,
        size:vectorArenaBytes,usage:storage})));
    const sections=[this.rhs,this.xA,this.xB,this.residual];
    for(const [kind,target] of sections.entries())for(let level=0;level<hierarchy.levels.length;level+=1){
      target.push(Object.freeze({buffer:this.vectorArenas[kind]!,
        offset:vectorOffsets[level]!,
        size:levelVectorCapacities[level]!*4}));
    }
    this.allocatedBytes=this.params.size+this.vectorArenas.reduce((sum,buffer)=>sum+buffer.size,0);
    // clear/copy + two pre + residual/restrict per descending level, eight
    // bottom sweeps, then prolong/two post per ascending level + final copy.
    this.encodedCorrectionDispatchCount=this.useFusedSubL0
      ?9:2*hierarchy.levels.length+7*(hierarchy.levels.length-1)+10;
    this.initializationTasks=[{label:"Compile plain Losasso first-order V-cycle",run:()=>this.initialize()}];
  }

  async initialize():Promise<void>{
    this.assertLive();if(this.levelPipelines)return;
    const levelModule=this.device.createShaderModule({label:"Losasso V-cycle level shader",code:vcycleWGSL});
    this.levelPipelines=await Promise.all(this.hierarchy.levels.map(async(_,level)=>{
      const compile=(entryPoint:string)=>this.device.createComputePipelineAsync({
        label:`Losasso V-cycle L${level} ${entryPoint}`,layout:"auto",compute:{module:levelModule,entryPoint}});
      const [clear,copy,jacobi,residual]=await Promise.all(["clearLevel","copyLevel","jacobiLevel","residualLevel"].map(compile));
      return {clear,copy,jacobi,residual};
    }));
    this.transferPipelines=await Promise.all(this.hierarchy.transfers.map(async(_,level)=>{
      const transferModule=this.device.createShaderModule({label:`Losasso V-cycle transfer ${level} shader`,
        code:transferWGSL});
      const compile=(entryPoint:string)=>this.device.createComputePipelineAsync({
        label:`Losasso V-cycle transfer ${level} ${entryPoint}`,layout:"auto",compute:{module:transferModule,entryPoint}});
      const [restrict,prolong]=await Promise.all([
        "restrictLevel","prolongLevel"].map(compile));
      return {restrict,prolong};
    }));
    if(this.useFusedSubL0){
      const module=this.device.createShaderModule({label:"Losasso fused sub-L0 V-cycle shader",
        code:fusedSubL0WGSL(this.hierarchy.fusedSubL0!,this.hierarchy.levels.length)});
      this.fusedSubL0Pipeline=await this.device.createComputePipelineAsync({
        label:"Losasso fused sub-L0 V-cycle",layout:"auto",
        compute:{module,entryPoint:"fusedSubL0"}});
    }
  }

  encodeSetup():void{}

  private cachedBindGroup(pipeline:GPUComputePipeline,
    entries:readonly {binding:number;resource:GPUBufferBinding}[]):GPUBindGroup{
    const candidates=this.bindGroupCache.get(pipeline)??[];
    const cached=candidates.find(candidate=>candidate.entries.length===entries.length
      &&candidate.entries.every((entry,index)=>{const wanted=entries[index]!;
        return entry.binding===wanted.binding&&entry.resource.buffer===wanted.resource.buffer
          &&entry.resource.offset===wanted.resource.offset&&entry.resource.size===wanted.resource.size;}));
    if(cached)return cached.group;
    const stableEntries=entries.map(entry=>({binding:entry.binding,resource:{...entry.resource}}));
    const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:stableEntries});
    candidates.push({entries:stableEntries,group});this.bindGroupCache.set(pipeline,candidates);return group;
  }

  encodeCorrection(broker:PassBroker,input:{rhs:GPUBuffer;correction:GPUBuffer;
    solverControl:GPUBuffer;rowCount:GPUBuffer}):void{
    this.assertLive();if(!this.levelPipelines||!this.transferPipelines)throw new Error("Losasso V-cycle is not initialized");
    type BufferInput=GPUBuffer|VectorSlice;
    const resource=(value:BufferInput):GPUBufferBinding=>"buffer" in value
      ?{buffer:value.buffer,offset:value.offset,size:value.size}:{buffer:value};
    const levelGroup=(level:number,pipeline:GPUComputePipeline,b:BufferInput,x:BufferInput,out:BufferInput)=>{
      const source=this.hierarchy.levels[level]!;
      const buffers=[this.params,source.control,source.rowFaceOffsets,source.rowFaces,source.faces,b,x,out,input.solverControl];
      const pipes=this.levelPipelines![level]!;
      const bindings=pipeline===pipes.clear?[1,7,8]
        :pipeline===pipes.copy?[1,5,7,8]
          :pipeline===pipes.residual?[1,2,3,4,5,6,7,8]
            :[0,1,2,3,4,5,6,7,8];
      return this.cachedBindGroup(pipeline,bindings.map(binding=>({binding,resource:resource(buffers[binding]!)})));
    };
    const dispatch=(pass:GPUComputePassEncoder,level:number,pipeline:GPUComputePipeline,group:GPUBindGroup)=>{
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);
      pass.dispatchWorkgroupsIndirect(this.hierarchy.levels[level]!.rowDispatch,0);
    };
    const pass=broker.compute({label:"Losasso V-cycle - initialize levels"});
    const fused=this.fusedSubL0Pipeline&&this.hierarchy.fusedSubL0
      &&this.hierarchy.levels.length>1;
    for(let level=0;level<(fused?1:this.hierarchy.levels.length);level+=1){
      const pipes=this.levelPipelines[level]!;
      if(!fused)dispatch(pass,level,pipes.clear,
        levelGroup(level,pipes.clear,this.rhs[level]!,this.xA[level]!,this.rhs[level]!));
      dispatch(pass,level,pipes.clear,levelGroup(level,pipes.clear,this.rhs[level]!,this.xA[level]!,this.xA[level]!));
    }
    const copy0=this.levelPipelines[0]!.copy;
    dispatch(pass,0,copy0,levelGroup(0,copy0,input.rhs,this.xA[0]!,this.rhs[0]!));
    if(fused){
      const pipes=this.levelPipelines[0]!;
      dispatch(pass,0,pipes.jacobi,levelGroup(0,pipes.jacobi,this.rhs[0]!,this.xA[0]!,this.xB[0]!));
      dispatch(pass,0,pipes.jacobi,levelGroup(0,pipes.jacobi,this.rhs[0]!,this.xB[0]!,this.xA[0]!));
      dispatch(pass,0,pipes.residual,levelGroup(0,pipes.residual,
        this.rhs[0]!,this.xA[0]!,this.residual[0]!));
      const fusedPipeline=this.fusedSubL0Pipeline!;
      const group=this.cachedBindGroup(fusedPipeline,[
        {binding:0,resource:{buffer:this.params}},
        {binding:1,resource:{buffer:this.hierarchy.fusedSubL0!.arena}},
        {binding:2,resource:{buffer:this.vectorArenas[0]!}},
        {binding:3,resource:{buffer:this.vectorArenas[1]!}},
        {binding:4,resource:{buffer:this.vectorArenas[2]!}},
        {binding:5,resource:{buffer:this.vectorArenas[3]!}},
        {binding:6,resource:{buffer:input.solverControl}},
      ]);
      pass.setPipeline(fusedPipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);
      dispatch(pass,0,pipes.jacobi,levelGroup(0,pipes.jacobi,this.rhs[0]!,this.xA[0]!,this.xB[0]!));
      dispatch(pass,0,pipes.jacobi,levelGroup(0,pipes.jacobi,this.rhs[0]!,this.xB[0]!,this.xA[0]!));
      dispatch(pass,0,copy0,levelGroup(0,copy0,this.xA[0]!,this.xA[0]!,input.correction));
      return;
    }
    for(let level=0;level<this.hierarchy.levels.length-1;level+=1){
      const pipes=this.levelPipelines[level]!;
      dispatch(pass,level,pipes.jacobi,levelGroup(level,pipes.jacobi,this.rhs[level]!,this.xA[level]!,this.xB[level]!));
      dispatch(pass,level,pipes.jacobi,levelGroup(level,pipes.jacobi,this.rhs[level]!,this.xB[level]!,this.xA[level]!));
      dispatch(pass,level,pipes.residual,levelGroup(level,pipes.residual,this.rhs[level]!,this.xA[level]!,this.residual[level]!));
      const transfer=this.hierarchy.transfers[level]!,transferPipelines=this.transferPipelines[level]!;
      const buffers=[this.hierarchy.levels[level]!.control,this.hierarchy.levels[level+1]!.control,
        transfer.fineParents,transfer.childOffsets,transfer.childList,this.residual[level]!,
        this.rhs[level+1]!,input.solverControl];
      const transferGroup=(pipeline:GPUComputePipeline,bindings:readonly number[])=>
        this.cachedBindGroup(pipeline,bindings.map(binding=>
          ({binding,resource:resource(buffers[binding]!)})));
      pass.setPipeline(transferPipelines.restrict);pass.setBindGroup(0,
        transferGroup(transferPipelines.restrict,[0,1,3,4,5,6,7]));
      pass.dispatchWorkgroupsIndirect(this.hierarchy.levels[level+1]!.rowDispatch,0);
    }
    const bottom=this.hierarchy.levels.length-1,bottomPipes=this.levelPipelines[bottom]!;
    for(let sweep=0;sweep<8;sweep+=1){
      const source=(sweep&1)===0?this.xA[bottom]!:this.xB[bottom]!;
      const target=(sweep&1)===0?this.xB[bottom]!:this.xA[bottom]!;
      dispatch(pass,bottom,bottomPipes.jacobi,levelGroup(bottom,bottomPipes.jacobi,this.rhs[bottom]!,source,target));
    }
    for(let level=bottom-1;level>=0;level-=1){
      const transfer=this.hierarchy.transfers[level]!,transferPipeline=this.transferPipelines[level]!.prolong;
      const buffers=[this.hierarchy.levels[level]!.control,this.hierarchy.levels[level+1]!.control,
        transfer.fineParents,transfer.childOffsets,transfer.childList,this.xA[level+1]!,
        this.xA[level]!,input.solverControl];
      const group=this.cachedBindGroup(transferPipeline,
        [0,1,2,5,6,7].map(binding=>({binding,resource:resource(buffers[binding]!)})));
      pass.setPipeline(transferPipeline);pass.setBindGroup(0,group);
      pass.dispatchWorkgroupsIndirect(transfer.fineRowDispatch,0);
      const pipes=this.levelPipelines[level]!;
      dispatch(pass,level,pipes.jacobi,levelGroup(level,pipes.jacobi,this.rhs[level]!,this.xA[level]!,this.xB[level]!));
      dispatch(pass,level,pipes.jacobi,levelGroup(level,pipes.jacobi,this.rhs[level]!,this.xB[level]!,this.xA[level]!));
    }
    const copyOut=this.levelPipelines[0]!.copy;
    dispatch(pass,0,copyOut,levelGroup(0,copyOut,this.xA[0]!,this.xA[0]!,input.correction));
  }

  destroy():void{if(this.destroyed)return;this.destroyed=true;this.params.destroy();
    for(const buffer of this.vectorArenas)buffer.destroy();
    this.bindGroupCache.clear();
    this.levelPipelines=undefined;this.transferPipelines=undefined;
    this.fusedSubL0Pipeline=undefined;}
  private assertLive(){if(this.destroyed)throw new Error("Losasso V-cycle is destroyed");}
}
