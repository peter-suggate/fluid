/** The bed-slope bake: the lane of the solid scratch from the voxel mask.
 *
 * A voxel ramp is a staircase, and a film on one tread has a level bed under
 * it. The lane gives each floor cell (an open cell on a solid voxel's top
 * face) the gradient of the surface the staircase stands for, so the forces
 * can add -g times it. UniformBedSlopeBake writes that lane on the GPU, from
 * the mask image the lane lies in, at scene build (encode) and after a solid
 * edit (encodeRegion). Nothing is read back and nothing is compiled or
 * allocated after initialize.
 *
 * The rule. Floor cells of 4-neighbour columns are linked when a film steps
 * from one to the other: level, or one cell up or down over a riser whose
 * upper air is open. A wall, a kerb two cells high and a pit edge link
 * nothing. A riser is a nosing: the face between the two cells, at the upper
 * cell's level. Every floor cell at level y learns the nearest nosing
 * (straight-line distance in the plane, reached along links) of each of the
 * levels y-1 (below its tread), y (its own tread's edge), y+1 (the riser it
 * runs up to) and y+2, and whether its tread has a drop of more than a cell
 * beside it. With d, u, d0 and u3 the distances to those of levels y, y+1,
 * y-1 and y+2, its bed height above its own top face is
 *
 *   d/(u+d)               between a nosing and a riser (the chord from one
 *                         to the other; run u+d of two cells or more)
 *   d/(d0-d)              on a top landing (no riser above), within a run
 *                         and a quarter of its nosing: the tread below's run
 *   1 - u/(u3-u)          on the last tread before a drop (no nosing of its
 *                         own), within a run and a quarter of the riser
 *
 * and it has none otherwise: a level floor, a lone step, a kerb, a pit edge,
 * a basin's floor and the run-out a staircase ends on are level. The heights
 * are then relaxed (each moves half way to the mean of its linked neighbours'
 * midpoints, an axis with one neighbour left out, so a height that is linear
 * never moves) and the lane is their gradient across the links: the centred
 * difference, or the one side that has a height. Below 1 in 64.5 it is zero
 * (a tread longer than 64 cells is a floor); above 0.5 by more than an eighth
 * it is zero (not a ramp), and up to there it is brought back to 0.5.
 *
 * A straight axis-aligned staircase of run n reads exactly 1/n on every cell
 * of every tread, its top landing and its last tread before a drop included:
 * the heights are linear along the run, so the relaxation and the centred
 * difference leave them alone. Curves and oblique runs are where the
 * relaxation earns its passes: the nearest nosing of a curved riser line
 * pivots about the line's ends.
 *
 * Why this and not a least-squares plane: a plane through top-face heights
 * needs sums over a connected neighbourhood, which no fixed number of local
 * passes yields (a sum is not idempotent, so it cannot be flooded), and it
 * reads 8/9 of the slope from three treads. A nearest nosing is a minimum:
 * it floods, in any order, to the same answer.
 *
 * Launches of a bake, one compute pass, fixed direct dispatches of 8x8
 * columns over the window: layers, links, `passes` spreads (96: a nosing
 * crosses one link a pass, and a 1:44 tread's far riser line is two runs
 * away), heights, `relax` relaxations (8), then the lane over the columns to
 * write. A full bake's window is the lattice. An edit's lane can change
 * within `reach` columns of it (passes + relax + 2), and those columns' state
 * is rebuilt from the mask within `reach` of them: the window. Links stop at
 * the window, so nothing outside it is read, and what a region bake writes is
 * what a full bake would.
 *
 * A column holds `layers` floor cells (4, at most 8), its lowest; a floor
 * cell above them stays level. */
import { uniformBedSlopeBakeWGSL } from "./uniform-bed-slope-bake.wgsl";

/** Region bakes one submit can hold: each takes a params record of its own. */
export const UNIFORM_BED_SLOPE_BAKE_REGIONS=8;
const RELAX=8;
/** Runs of two cells or more, in 1/128 cell. */
const MIN_RUN=256;
/** The band of an end tread past its run, in quarters. */
const BAND4=1;
/** A slope's square in (1/8192 cell a cell)^2: 127 is 1 in 64.5, 4608 nine eighths of a half. */
const MIN_SLOPE2=127*127,MAX_SLOPE2=4608*4608;

export interface UniformBedSlopeBakeOptions{
 /** Cells of the lattice; the lane covers its one-cell halo as well. */
 readonly size:readonly [number,number,number];
 /** The lane's first word in the mask image (uniformSolidScratchLayout). */
 readonly laneBaseWord:number;
 /** The lattice's low cell in the mask image's window (address.boxPage, in cells; default 0). */
 readonly origin?:readonly [number,number,number];
 /** Floor cells a column holds (default 4, at most 8). */
 readonly layers?:number;
 /** Spread passes, even (default 96). */
 readonly passes?:number;
}
export interface UniformBedSlopeBakeLayout{
 readonly size:readonly [number,number,number];
 readonly origin:readonly [number,number,number];
 /** Haloed cells: the lane's entries. */
 readonly cells:number;
 readonly laneBaseWord:number;
 /** Two cells a word. */
 readonly laneWords:number;
 readonly layers:number;
 readonly passes:number;
 readonly relax:number;
 /** Columns an edit's lane can change within, and columns past those whose state they read. */
 readonly reach:number;
 readonly columns:number;
 /** Launches of one bake. */
 readonly launches:number;
 /** Bytes of the bake's own buffers: nine words a slot, and the params records. */
 readonly scratchBytes:number;
 readonly minRun:number;readonly band4:number;readonly minSlope2:number;readonly maxSlope2:number;
}
export function uniformBedSlopeBakeLayout(options:UniformBedSlopeBakeOptions):UniformBedSlopeBakeLayout{
 const {size,laneBaseWord}=options,layers=options.layers??4,passes=options.passes??96;
 // A nosing is twelve bits an axis in half cells, a level eleven.
 if(size.length!==3||size.some(n=>!Number.isSafeInteger(n)||n<1)||size[0]>2047||size[1]>2046||size[2]>2047)throw new RangeError(`Uniform bed-slope bake: a lattice of ${size.join("x")} cells`);
 if(!Number.isSafeInteger(layers)||layers<1||layers>8)throw new RangeError(`Uniform bed-slope bake: ${layers} layers a column, not 1 to 8`);
 // Even: the spread ends in the buffer it started from. 256: a nosing's squared distance in 1/128 cell stays a u32.
 if(!Number.isSafeInteger(passes)||passes<2||passes>256||passes%2!==0)throw new RangeError(`Uniform bed-slope bake: ${passes} spread passes, not an even count from 2 to 256`);
 const cells=(size[0]+2)*(size[1]+2)*(size[2]+2),laneWords=Math.ceil(cells/2),columns=size[0]*size[2];
 if(!Number.isSafeInteger(laneBaseWord)||laneBaseWord<0)throw new RangeError(`Uniform bed-slope bake: lane base word ${laneBaseWord}`);
 const origin=options.origin??[0,0,0];
 if(origin.some(n=>!Number.isSafeInteger(n)||n<0))throw new RangeError(`Uniform bed-slope bake: origin ${origin.join(",")}`);
 return {size:[size[0],size[1],size[2]],origin:[origin[0],origin[1],origin[2]],cells,laneBaseWord,laneWords,layers,passes,relax:RELAX,reach:passes+RELAX+2,columns,
  launches:passes+RELAX+4,scratchBytes:36*layers*columns+256*(1+UNIFORM_BED_SLOPE_BAKE_REGIONS),minRun:MIN_RUN,band4:BAND4,minSlope2:MIN_SLOPE2,maxSlope2:MAX_SLOPE2};
}

/** An edited range of cells of the lattice (its halo's cells included: -1 and the size are cells). */
export interface UniformBedSlopeBakeBox{readonly minimum:readonly [number,number,number];readonly maximumExclusive:readonly [number,number,number]}
/** The columns a region bake of `box` covers, into `out` (eight words: the
 * window's x0, z0 and counts along x and z, then the same of the columns
 * whose lane is written). The written columns are the box grown by the reach
 * and cut to the lattice; the window is those grown by it again. */
export function uniformBedSlopeBakeRegion(layout:UniformBedSlopeBakeLayout,box:UniformBedSlopeBakeBox,out:Uint32Array):Uint32Array{
 const r=layout.reach;
 for(let i=0;i<2;i++){
  const axis=2*i,n=layout.size[axis]!,low=box.minimum[axis]!,high=box.maximumExclusive[axis]!;
  if(!Number.isSafeInteger(low)||!Number.isSafeInteger(high)||high<low)throw new RangeError(`Uniform bed-slope bake: a box from ${box.minimum.join(",")} to ${box.maximumExclusive.join(",")}`);
  const t0=Math.min(n,Math.max(0,low-r)),t1=Math.max(t0,Math.min(n,high+r)),w0=Math.max(0,t0-r),w1=Math.min(n,t1+r);
  out[i]=w0;out[2+i]=t1>t0?w1-w0:0;out[4+i]=t0;out[6+i]=t1-t0;
 }
 // Nothing to write along one axis is nothing to write.
 if(out[6]===0||out[7]===0)out[2]=out[3]=out[6]=out[7]=0;
 return out;
}
async function compile(device:GPUDevice,label:string,code:string):Promise<GPUShaderModule>{
 const module=device.createShaderModule({label,code});
 const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
 if(errors.length)throw new Error(`${label}\n${errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n")}`);
 return module;
}
interface Pipelines{readonly layers:GPUComputePipeline;readonly links:GPUComputePipeline;readonly spread:GPUComputePipeline;readonly height:GPUComputePipeline;
 readonly relaxOut:GPUComputePipeline;readonly relaxBack:GPUComputePipeline;readonly resolve:GPUComputePipeline}
interface Groups{readonly build:GPUBindGroup;readonly spreadOut:GPUBindGroup;readonly spreadBack:GPUBindGroup;readonly resolve:GPUBindGroup}

export class UniformBedSlopeBake{
 readonly layout:UniformBedSlopeBakeLayout;
 /** Bytes of the buffers the bake owns, allocated here and never again. */
 get scratchBytes():number{return this.layout.scratchBytes;}
 private readonly meta:GPUBuffer;
 private readonly a:GPUBuffer;
 private readonly b:GPUBuffer;
 /** Record 0 the whole lattice, then one a region bake of a submit. */
 private readonly params:GPUBuffer;
 private readonly offsets:readonly number[][];
 private readonly region=new Uint32Array(8);
 /** Region bakes encoded since the last submit: each holds a params record. */
 private pending=0;
 private pipelines?:Pipelines;
 private groups?:Groups;
 private started=false;
 private static readonly PASS:GPUComputePassDescriptor={label:"Uniform bed-slope bake"};
 constructor(private readonly device:GPUDevice,private readonly mask:GPUBuffer,options:UniformBedSlopeBakeOptions){
  const l=this.layout=uniformBedSlopeBakeLayout(options);
  if(4*(l.laneBaseWord+l.laneWords)>mask.size)throw new RangeError(`Uniform bed-slope bake: ${mask.label} of ${mask.size} bytes does not hold a lane of ${l.laneWords} words at word ${l.laneBaseWord}`);
  // COPY_DST: a full bake clears the lane, the halo's entries with it.
  if(!(mask.usage&GPUBufferUsage.STORAGE)||!(mask.usage&GPUBufferUsage.COPY_DST))throw new Error(`Uniform bed-slope bake: ${mask.label} needs STORAGE and COPY_DST usage`);
  if(device.limits.minUniformBufferOffsetAlignment>256)throw new RangeError(`Uniform bed-slope bake: uniform records ${device.limits.minUniformBufferOffsetAlignment} bytes apart, more than 256`);
  const slots=l.layers*l.columns,usage=GPUBufferUsage.STORAGE;
  this.meta=device.createBuffer({label:"Uniform bed-slope bake links",size:4*slots,usage});
  this.a=device.createBuffer({label:"Uniform bed-slope bake nosings A",size:16*slots,usage});
  this.b=device.createBuffer({label:"Uniform bed-slope bake nosings B",size:16*slots,usage});
  this.params=device.createBuffer({label:"Uniform bed-slope bake params",size:256*(1+UNIFORM_BED_SLOPE_BAKE_REGIONS),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(this.params,0,new Uint32Array([0,0,l.size[0],l.size[2],0,0,l.size[0],l.size[2]]));
  this.offsets=Array.from({length:1+UNIFORM_BED_SLOPE_BAKE_REGIONS},(_,i)=>[256*i]);
 }
 /** Compile the seven pipelines and bind them. */
 async initialize():Promise<void>{
  if(this.started)throw new Error("Uniform bed-slope bake is already initialized");
  this.started=true;
  const device=this.device,l=this.layout,COMPUTE=GPUShaderStage.COMPUTE;
  const params:GPUBindGroupLayoutEntry={binding:0,visibility:COMPUTE,buffer:{type:"uniform",hasDynamicOffset:true,minBindingSize:32}};
  const storage=(binding:number,type:GPUBufferBindingType):GPUBindGroupLayoutEntry=>({binding,visibility:COMPUTE,buffer:{type}});
  const build=device.createBindGroupLayout({label:"Uniform bed-slope bake build",entries:[params,storage(1,"read-only-storage"),storage(2,"storage"),storage(3,"storage"),storage(4,"storage")]});
  const spread=device.createBindGroupLayout({label:"Uniform bed-slope bake spread",entries:[params,storage(1,"read-only-storage"),storage(2,"read-only-storage"),storage(3,"storage")]});
  const resolve=device.createBindGroupLayout({label:"Uniform bed-slope bake resolve",entries:[params,storage(1,"storage"),storage(2,"read-only-storage"),storage(3,"read-only-storage")]});
  const [buildModule,spreadModule,resolveModule]=await Promise.all([
   compile(device,"Uniform bed-slope bake build",uniformBedSlopeBakeWGSL(l,"build")),
   compile(device,"Uniform bed-slope bake spread",uniformBedSlopeBakeWGSL(l,"spread")),
   compile(device,"Uniform bed-slope bake resolve",uniformBedSlopeBakeWGSL(l,"resolve")),
  ]);
  const pipeline=(layout:GPUBindGroupLayout,module:GPUShaderModule,entryPoint:string)=>device.createComputePipelineAsync({label:`Uniform bed-slope bake ${entryPoint}`,
   layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint}});
  const [layers,links,height,relaxOut,relaxBack,spreads,resolves]=await Promise.all([
   pipeline(build,buildModule!,"bsLayers"),pipeline(build,buildModule!,"bsLinks"),pipeline(build,buildModule!,"bsHeight"),
   pipeline(build,buildModule!,"bsRelaxOut"),pipeline(build,buildModule!,"bsRelaxBack"),pipeline(spread,spreadModule!,"bsSpread"),pipeline(resolve,resolveModule!,"bsResolve"),
  ]);
  this.pipelines={layers:layers!,links:links!,height:height!,relaxOut:relaxOut!,relaxBack:relaxBack!,spread:spreads!,resolve:resolves!};
  const record={binding:0,resource:{buffer:this.params,size:32}};
  const group=(label:string,layout:GPUBindGroupLayout,buffers:readonly GPUBuffer[])=>device.createBindGroup({label:`Uniform bed-slope bake ${label}`,layout,
   entries:[record,...buffers.map((buffer,i)=>({binding:1+i,resource:{buffer}}))]});
  this.groups={build:group("build",build,[this.mask,this.meta,this.a,this.b]),spreadOut:group("spread out",spread,[this.meta,this.a,this.b]),
   spreadBack:group("spread back",spread,[this.meta,this.b,this.a]),resolve:group("resolve",resolve,[this.mask,this.meta,this.b])};
 }
 /** The launches of one bake over the window and written columns of params record `slot`. */
 private launch(encoder:GPUCommandEncoder,slot:number,windowX:number,windowZ:number,writeX:number,writeZ:number):void{
  const p=this.pipelines!,g=this.groups!,l=this.layout,offsets=this.offsets[slot]!;
  const wx=Math.ceil(windowX/8),wz=Math.ceil(windowZ/8);
  const pass=encoder.beginComputePass(UniformBedSlopeBake.PASS);
  pass.setBindGroup(0,g.build,offsets);
  pass.setPipeline(p.layers);pass.dispatchWorkgroups(wx,wz);
  pass.setPipeline(p.links);pass.dispatchWorkgroups(wx,wz);
  pass.setPipeline(p.spread);
  for(let i=0;i<l.passes;i++){pass.setBindGroup(0,i&1?g.spreadBack:g.spreadOut,offsets);pass.dispatchWorkgroups(wx,wz);}
  pass.setBindGroup(0,g.build,offsets);
  pass.setPipeline(p.height);pass.dispatchWorkgroups(wx,wz);
  for(let i=0;i<l.relax;i++){pass.setPipeline(i&1?p.relaxBack:p.relaxOut);pass.dispatchWorkgroups(wx,wz);}
  pass.setPipeline(p.resolve);pass.setBindGroup(0,g.resolve,offsets);pass.dispatchWorkgroups(Math.ceil(writeX/8),Math.ceil(writeZ/8));
  pass.end();
 }
 /** The whole lane from the mask as the buffer holds it when the encoder
  * runs: at scene build, after the mask's upload. */
 encode(encoder:GPUCommandEncoder):void{
  if(!this.pipelines)throw new Error("Uniform bed-slope bake is not initialized");
  const l=this.layout;
  encoder.clearBuffer(this.mask,4*l.laneBaseWord,4*l.laneWords);
  this.launch(encoder,0,l.size[0],l.size[2],l.size[0],l.size[2]);
 }
 /** The lane again where an edit of `box` can have changed it: after the
  * mask's dirty words are uploaded. The box is the edited cells; the bake
  * grows it by its reach. Each holds a params record until its encoder has
  * run, so the caller says when it has submitted (submitted). One more than
  * UNIFORM_BED_SLOPE_BAKE_REGIONS before that would rewrite the first's
  * record under it: it is refused. */
 encodeRegion(encoder:GPUCommandEncoder,box:UniformBedSlopeBakeBox):void{
  if(!this.pipelines)throw new Error("Uniform bed-slope bake is not initialized");
  const r=this.region;uniformBedSlopeBakeRegion(this.layout,box,r);
  if(r[6]===0)return;
  if(this.pending===UNIFORM_BED_SLOPE_BAKE_REGIONS)throw new Error(`Uniform bed-slope bake: more than ${UNIFORM_BED_SLOPE_BAKE_REGIONS} region bakes encoded without a submit`);
  const slot=1+this.pending++;
  this.device.queue.writeBuffer(this.params,256*slot,r);
  this.launch(encoder,slot,r[2]!,r[3]!,r[6]!,r[7]!);
 }
 /** Every encoder given to encodeRegion so far has been submitted: a record
  * written from here on is queued behind them, so all are free again. */
 submitted():void{this.pending=0;}
 destroy():void{this.meta.destroy();this.a.destroy();this.b.destroy();this.params.destroy();}
}
