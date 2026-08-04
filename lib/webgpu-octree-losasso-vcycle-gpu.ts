import type { PassBroker } from "./webgpu-pass-broker";
import {
  createSignedRadix256F32ReductionWGSL,
  SIGNED_RADIX_256_F32_MAX_TERMS,
} from
  "./webgpu-exact-f32-reduction";
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
@group(0)@binding(5)var<storage,read> b:array<f32>;
@group(0)@binding(6)var<storage,read> x:array<f32>;
@group(0)@binding(7)var<storage,read_write> output:array<f32>;
@group(0)@binding(8)var<storage,read> solve:array<u32>;
fn stopped()->bool{return solve[0]!=0u||solve[1]!=0u||control[3]!=1u;}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn addExact(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
 if(magnitude==0u){return;}let rawExponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
 let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);let shift=select(3u,rawExponent+2u,rawExponent!=0u);
 let firstLimb=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=firstLimb+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
  if(byte!=0&&limb<36u){(*limbs)[limb]+=sign*byte;}}}
fn floorDiv256Local(value:i32)->vec2i{var carry=value/256;var digit=value-carry*256;if(digit<0){digit+=256;carry-=1;}return vec2i(carry,digit);}
fn exactValue(source:ptr<function,array<i32,36>>)->f32{var limbs=(*source);
 for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256Local(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}
 let negative=limbs[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=-limbs[limb];}
  for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256Local(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}}
 var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(limb*8u));}
 return select(magnitude,-magnitude,negative);}
fn image(row:u32)->vec2f{
 if(row+1u>=arrayLength(&offsets)){return vec2f(3.402823e38,-1.);}let begin=offsets[row];let end=offsets[row+1u];
 // Coarse hierarchy rows retain fine boundary patches and may have thousands
 // of incidences. Exact integer limbs remain permutation-independent for the
 // bounded hierarchy count, so only malformed CSR bounds fail closed here.
 if(begin>end||end>arrayLength(&rowFaces)){return vec2f(3.402823e38,-1.);}
 let centre=x[row];var exactImage:array<i32,36>;var exactDiagonal:array<i32,36>;var valid=true;
 for(var cursor=begin;cursor<end;cursor+=1u){let faceId=rowFaces[cursor];if(faceId>=arrayLength(&faces)){valid=false;continue;}
  let face=faces[faceId];let coefficient=(face.openFraction*face.area)*face.inverseDistance;var difference=centre;
  if(face.negativeRow==row){if(face.positiveRow!=INVALID){if(face.positiveRow>=arrayLength(&x)){valid=false;continue;}difference-=x[face.positiveRow];}}
  else{if(face.negativeRow>=arrayLength(&x)){valid=false;continue;}difference-=x[face.negativeRow];}
  let term=coefficient*difference;if(!finite(coefficient)||coefficient<0.||!finite(term)){valid=false;continue;}
  addExact(&exactDiagonal,coefficient);addExact(&exactImage,term);
 }
 let value=exactValue(&exactImage);let diagonal=exactValue(&exactDiagonal);
 return select(vec2f(3.402823e38,-1.),vec2f(value,diagonal),valid&&finite(value)&&finite(diagonal)&&diagonal>0.);
}
@compute @workgroup_size(64)fn clearLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(row>=control[1]){return;}output[row]=0.;
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

function transferWGSL(coarseRowCapacity:number,fineRowCapacity:number):string{return /* wgsl */ `
@group(0)@binding(0)var<storage,read> fineControl:array<u32>;
@group(0)@binding(1)var<storage,read> coarseControl:array<u32>;
@group(0)@binding(2)var<storage,read> parents:array<u32>;
@group(0)@binding(3)var<storage,read> weights:array<f32>;
@group(0)@binding(4)var<storage,read> input:array<f32>;
@group(0)@binding(5)var<storage,read_write> output:array<f32>;
@group(0)@binding(6)var<storage,read> solve:array<u32>;
${createSignedRadix256F32ReductionWGSL({group:0,binding:7,
  scalarCount:coarseRowCapacity,reductionLanes:64,
  livePartialCountExpression:"1u",maximumTermCount:fineRowCapacity})}
@group(0)@binding(8)var<storage,read> coarseInverseVolumes:array<f32>;
fn stopped()->bool{return solve[0]!=0u||solve[1]!=0u
 ||fineControl[3]!=1u||coarseControl[3]!=1u;}
@compute @workgroup_size(64)fn clearRestriction(@builtin(local_invocation_index)lane:u32){
 clearFixedPartial(0u,lane);
}
@compute @workgroup_size(64)fn depositRestriction(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=fineControl[1]){return;}let parent=parents[row];
 if(parent<coarseControl[1]){
  addFixedF32(0u,parent,(weights[row]*coarseInverseVolumes[parent])*input[row]);
 }
}
@compute @workgroup_size(64)fn finishRestriction(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=coarseControl[1]){return;}
 output[row]=fixedScalarValue(row);
}
@compute @workgroup_size(64)fn prolongLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=fineControl[1]){return;}let parent=parents[row];
 if(parent<coarseControl[1]){
  output[row]+=(weights[row]*coarseInverseVolumes[parent])*input[parent];
 }
}
`;}

interface LevelPipelines {
  clear: GPUComputePipeline;
  copy: GPUComputePipeline;
  jacobi: GPUComputePipeline;
  residual: GPUComputePipeline;
}
interface TransferPipelines {
  clear:GPUComputePipeline;deposit:GPUComputePipeline;
  finish:GPUComputePipeline;prolong:GPUComputePipeline;
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
  private readonly rhs:GPUBuffer[]=[];
  private readonly xA:GPUBuffer[]=[];
  private readonly xB:GPUBuffer[]=[];
  private readonly residual:GPUBuffer[]=[];
  private levelPipelines?:readonly LevelPipelines[];
  private transferPipelines?:readonly TransferPipelines[];
  private destroyed=false;

  constructor(private readonly device:GPUDevice,
    readonly hierarchy:OctreeLosassoVCycleHierarchySource) {
    if(hierarchy.levels.length<1||hierarchy.transfers.length!==hierarchy.levels.length-1){
      throw new RangeError("Losasso V-cycle needs a complete one-or-more-level hierarchy");
    }
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    this.params=device.createBuffer({label:"Losasso V-cycle Jacobi constants",size:16,
      usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(this.params,0,Float32Array.of(2/3,0,0,0));
    for(const [level,source] of hierarchy.levels.entries()){
      if(!Number.isSafeInteger(source.rowCapacity)||source.rowCapacity<1)throw new RangeError(`Losasso V-cycle level ${level} capacity is invalid`);
      if (source.rowFaces.size / 4 > SIGNED_RADIX_256_F32_MAX_TERMS) {
        throw new RangeError(`Losasso V-cycle level ${level} incidence capacity exceeds exact image capacity`);
      }
      const make=(name:string)=>device.createBuffer({label:`Losasso V-cycle L${level} ${name}`,
        size:source.rowCapacity*4,usage:storage});
      this.rhs.push(make("rhs"));this.xA.push(make("correction A"));
      this.xB.push(make("correction B"));this.residual.push(make("residual"));
    }
    this.allocatedBytes=this.params.size+[...this.rhs,...this.xA,...this.xB,...this.residual]
      .reduce((sum,buffer)=>sum+buffer.size,0);
    // clear/copy + two pre + residual/restrict per descending level, eight
    // bottom sweeps, then prolong/two post per ascending level + final copy.
    this.encodedCorrectionDispatchCount=2*hierarchy.levels.length
      +9*(hierarchy.levels.length-1)+10;
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
        code:transferWGSL(this.hierarchy.levels[level+1]!.rowCapacity,
          this.hierarchy.levels[level]!.rowCapacity)});
      const compile=(entryPoint:string)=>this.device.createComputePipelineAsync({
        label:`Losasso V-cycle transfer ${level} ${entryPoint}`,layout:"auto",compute:{module:transferModule,entryPoint}});
      const [clear,deposit,finish,prolong]=await Promise.all([
        "clearRestriction","depositRestriction","finishRestriction","prolongLevel"].map(compile));
      return {clear,deposit,finish,prolong};
    }));
  }

  encodeSetup():void{}

  encodeCorrection(broker:PassBroker,input:{rhs:GPUBuffer;correction:GPUBuffer;
    solverControl:GPUBuffer;rowCount:GPUBuffer}):void{
    this.assertLive();if(!this.levelPipelines||!this.transferPipelines)throw new Error("Losasso V-cycle is not initialized");
    const levelGroup=(level:number,pipeline:GPUComputePipeline,b:GPUBuffer,x:GPUBuffer,out:GPUBuffer)=>{
      const source=this.hierarchy.levels[level]!;
      const buffers=[this.params,source.control,source.rowFaceOffsets,source.rowFaces,source.faces,b,x,out,input.solverControl];
      const pipes=this.levelPipelines![level]!;
      const bindings=pipeline===pipes.clear?[1,7]
        :pipeline===pipes.copy?[1,5,7,8]
          :pipeline===pipes.residual?[1,2,3,4,5,6,7,8]
            :[0,1,2,3,4,5,6,7,8];
      return this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:bindings.map(binding=>({binding,resource:{buffer:buffers[binding]!}}))});
    };
    const dispatch=(pass:GPUComputePassEncoder,level:number,pipeline:GPUComputePipeline,group:GPUBindGroup)=>{
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);
      pass.dispatchWorkgroupsIndirect(this.hierarchy.levels[level]!.rowDispatch,0);
    };
    const pass=broker.compute({label:"Losasso V-cycle - initialize levels"});
    for(let level=0;level<this.hierarchy.levels.length;level+=1){
      const pipes=this.levelPipelines[level]!;
      dispatch(pass,level,pipes.clear,levelGroup(level,pipes.clear,this.rhs[level]!,this.xA[level]!,this.rhs[level]!));
      dispatch(pass,level,pipes.clear,levelGroup(level,pipes.clear,this.rhs[level]!,this.xA[level]!,this.xA[level]!));
    }
    const copy0=this.levelPipelines[0]!.copy;
    dispatch(pass,0,copy0,levelGroup(0,copy0,input.rhs,this.xA[0]!,this.rhs[0]!));
    for(let level=0;level<this.hierarchy.levels.length-1;level+=1){
      const pipes=this.levelPipelines[level]!;
      dispatch(pass,level,pipes.jacobi,levelGroup(level,pipes.jacobi,this.rhs[level]!,this.xA[level]!,this.xB[level]!));
      dispatch(pass,level,pipes.jacobi,levelGroup(level,pipes.jacobi,this.rhs[level]!,this.xB[level]!,this.xA[level]!));
      dispatch(pass,level,pipes.residual,levelGroup(level,pipes.residual,this.rhs[level]!,this.xA[level]!,this.residual[level]!));
      const transfer=this.hierarchy.transfers[level]!,transferPipelines=this.transferPipelines[level]!;
      const buffers=[this.hierarchy.levels[level]!.control,this.hierarchy.levels[level+1]!.control,
        transfer.fineParents,transfer.fineVolumes,this.residual[level]!,this.rhs[level+1]!,
        input.solverControl,transfer.restrictionPartials,transfer.coarseInverseVolumes];
      const transferGroup=(pipeline:GPUComputePipeline,bindings:readonly number[])=>
        this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:bindings.map(binding=>
          ({binding,resource:{buffer:buffers[binding]!}}))});
      pass.setPipeline(transferPipelines.clear);pass.setBindGroup(0,
        transferGroup(transferPipelines.clear,[7]));pass.dispatchWorkgroups(1);
      pass.setPipeline(transferPipelines.deposit);pass.setBindGroup(0,
        transferGroup(transferPipelines.deposit,[0,1,2,3,4,6,7,8]));
      pass.dispatchWorkgroupsIndirect(transfer.fineRowDispatch,0);
      pass.setPipeline(transferPipelines.finish);pass.setBindGroup(0,
        transferGroup(transferPipelines.finish,[0,1,5,6,7]));
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
        transfer.fineParents,transfer.fineVolumes,this.xA[level+1]!,this.xA[level]!,
        input.solverControl,transfer.restrictionPartials,transfer.coarseInverseVolumes];
      const group=this.device.createBindGroup({layout:transferPipeline.getBindGroupLayout(0),entries:
        [0,1,2,3,4,5,6,8].map(binding=>({binding,resource:{buffer:buffers[binding]!}}))});
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
    for(const buffer of [...this.rhs,...this.xA,...this.xB,...this.residual])buffer.destroy();
    this.levelPipelines=undefined;this.transferPipelines=undefined;}
  private assertLive(){if(this.destroyed)throw new Error("Losasso V-cycle is destroyed");}
}
