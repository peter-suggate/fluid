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
 * Sparse by page column (32x32 columns, x + 64 z of the lattice's pages). A
 * riser of one cell has a partly solid page in its own column or the one
 * beside it, so a cell with a bed height lies within passes + 1 columns of a
 * page column that holds a partly solid page: the lane's page columns are
 * those within laneRadius pages of one, and the bake keeps state only for
 * those within windowRadius (laneRadius, then the reach). A column with no
 * state is outside every window and a column with no lane reads zero, as a
 * full bake over the lattice leaves both. plan() places the two sets from
 * the mask; when an edit moves either, the next bake is a full one.
 *
 * The lane: a directory word a page column (its slot's first word in the
 * mask buffer, 0 for none), then the slots, `layers` words a column, a word a
 * floor cell, lowest first: (y + 1) << 16 | entry, 0 for a level one.
 *
 * A column holds `layers` floor cells (4, at most 8), its lowest; a floor
 * cell above them stays level. */
import { SOLID_MASK_EMPTY, SOLID_MASK_FULL, SOLID_MASK_WALL } from "../../core/solid-occupancy-mask";
import { uniformBedSlopeBakeWGSL } from "./uniform-bed-slope-bake.wgsl";

/** Region bakes one submit can hold: each takes a params record of its own. */
export const UNIFORM_BED_SLOPE_BAKE_REGIONS=8;
/** Floor cells a column of the solver's lane holds: umBedSlope reads this many words a column. */
export const UNIFORM_BED_SLOPE_LAYERS=4;
/** The lane's directory: a word a page column of the 64x64 window. */
export const UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS=4096;
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
 /** The lane directory's first word in the mask image (uniformSolidScratchLayout). */
 readonly laneBaseWord:number;
 /** The lane slots' first word in the mask buffer, and how many (default: one a page column of the lattice). */
 readonly laneSlotBaseWord?:number;
 readonly laneSlots?:number;
 /** Page columns the bake holds state for at once (default: every one of the lattice). */
 readonly windowSlots?:number;
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
 /** Haloed cells, and a dense lane of them two a word: what an emulation holds (never allocated here). */
 readonly cells:number;
 readonly denseLaneWords:number;
 /** Page columns of the lattice along x and z. */
 readonly pages:readonly [number,number];
 readonly laneBaseWord:number;
 readonly laneSlotBaseWord:number;
 readonly laneSlots:number;
 /** Words of a lane slot: `layers` a column. */
 readonly laneSlotWords:number;
 readonly windowSlots:number;
 /** Pages past a page column with a partly solid page that can hold a slope, and that the bake keeps state for. */
 readonly laneRadius:number;
 readonly windowRadius:number;
 readonly layers:number;
 readonly passes:number;
 readonly relax:number;
 /** Columns an edit's lane can change within, and columns past those whose state they read. */
 readonly reach:number;
 readonly columns:number;
 /** Launches of one bake. */
 readonly launches:number;
 /** Bytes of the bake's own buffers: nine words a slot of its window's columns, its directory and the params records. */
 readonly scratchBytes:number;
 readonly minRun:number;readonly band4:number;readonly minSlope2:number;readonly maxSlope2:number;
}
export function uniformBedSlopeBakeLayout(options:UniformBedSlopeBakeOptions):UniformBedSlopeBakeLayout{
 const {size,laneBaseWord}=options,layers=options.layers??UNIFORM_BED_SLOPE_LAYERS,passes=options.passes??96;
 // A nosing is twelve bits an axis in half cells, a level eleven.
 if(size.length!==3||size.some(n=>!Number.isSafeInteger(n)||n<1)||size[0]>2047||size[1]>2046||size[2]>2047)throw new RangeError(`Uniform bed-slope bake: a lattice of ${size.join("x")} cells`);
 if(!Number.isSafeInteger(layers)||layers<1||layers>8)throw new RangeError(`Uniform bed-slope bake: ${layers} layers a column, not 1 to 8`);
 // Even: the spread ends in the buffer it started from. 256: a nosing's squared distance in 1/128 cell stays a u32.
 if(!Number.isSafeInteger(passes)||passes<2||passes>256||passes%2!==0)throw new RangeError(`Uniform bed-slope bake: ${passes} spread passes, not an even count from 2 to 256`);
 const cells=(size[0]+2)*(size[1]+2)*(size[2]+2),pages:[number,number]=[Math.ceil(size[0]/32),Math.ceil(size[2]/32)];
 if(pages[0]>64||pages[1]>64)throw new RangeError(`Uniform bed-slope bake: ${pages.join("x")} page columns pass the 64x64 window`);
 const laneSlots=options.laneSlots??pages[0]*pages[1],windowSlots=options.windowSlots??pages[0]*pages[1],laneSlotBaseWord=options.laneSlotBaseWord??0;
 for(const [name,n] of [["lane base word",laneBaseWord],["lane slot base word",laneSlotBaseWord],["lane slots",laneSlots],["window slots",windowSlots]] as const)
  if(!Number.isSafeInteger(n)||n<0)throw new RangeError(`Uniform bed-slope bake: ${name} ${n}`);
 const origin=options.origin??[0,0,0];
 if(origin.some(n=>!Number.isSafeInteger(n)||n<0))throw new RangeError(`Uniform bed-slope bake: origin ${origin.join(",")}`);
 const reach=passes+RELAX+2,laneRadius=Math.ceil((passes+1)/32),columns=1024*windowSlots;
 return {size:[size[0],size[1],size[2]],origin:[origin[0],origin[1],origin[2]],cells,denseLaneWords:Math.ceil(cells/2),pages,laneBaseWord,laneSlotBaseWord,laneSlots,laneSlotWords:1024*layers,windowSlots,
  laneRadius,windowRadius:laneRadius+Math.ceil(reach/32),layers,passes,relax:RELAX,reach,columns,
  launches:passes+RELAX+4,scratchBytes:36*layers*columns+4*UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS+256*(1+UNIFORM_BED_SLOPE_BAKE_REGIONS),minRun:MIN_RUN,band4:BAND4,minSlope2:MIN_SLOPE2,maxSlope2:MAX_SLOPE2};
}

/** The page columns (x + 64 z) of the lattice that hold a partly solid page
 * of a mask image (SolidOccupancyMask: neither EMPTY, FULL nor a wall), the
 * lattice's low cell at `origin` cells of its window. */
export function uniformBedSlopePartialColumns(words:ArrayLike<number>,size:readonly [number,number,number],origin:readonly [number,number,number]=[0,0,0]):Uint8Array{
 const out=new Uint8Array(UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS),o=origin.map(n=>n>>5),py=Math.ceil(size[1]/32);
 for(let z=0;z<Math.ceil(size[2]/32);z++)for(let x=0;x<Math.ceil(size[0]/32);x++)for(let y=0;y<py&&!out[x+64*z];y++){
  const word=words[((x+o[0]!)&63)|(((y+o[1]!)&63)<<6)|(((z+o[2]!)&63)<<12)]!,code=word&0xffff;
  if(word!==SOLID_MASK_WALL&&code!==SOLID_MASK_EMPTY&&code!==SOLID_MASK_FULL)out[x+64*z]=1;
 }
 return out;
}
/** The page columns within `radius` pages of a flagged one, cut to the lattice's, and their count. */
function uniformBedSlopeGrown(partial:ArrayLike<number>,pages:readonly [number,number],radius:number):{flags:Uint8Array;count:number}{
 const flags=new Uint8Array(UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS);let count=0;
 for(let z=0;z<pages[1];z++)for(let x=0;x<pages[0];x++){
  if(!partial[x+64*z])continue;
  for(let j=Math.max(0,z-radius);j<=Math.min(pages[1]-1,z+radius);j++)for(let i=Math.max(0,x-radius);i<=Math.min(pages[0]-1,x+radius);i++)if(!flags[i+64*j]){flags[i+64*j]=1;count++;}
 }
 return {flags,count};
}
/** Slots for the lane and the window of `partial` (uniformBedSlopePartialColumns)
 * and of one more lone partly solid page column, for edits: never more than the lattice's page columns. */
export function uniformBedSlopeBakeCapacity(size:readonly [number,number,number],partial:ArrayLike<number>,passes?:number):{laneSlots:number;windowSlots:number}{
 const l=uniformBedSlopeBakeLayout({size,laneBaseWord:0,passes}),all=l.pages[0]*l.pages[1],spare=(r:number)=>(2*r+1)**2;
 return {laneSlots:Math.min(all,uniformBedSlopeGrown(partial,l.pages,l.laneRadius).count+spare(l.laneRadius)),
  windowSlots:Math.min(all,uniformBedSlopeGrown(partial,l.pages,l.windowRadius).count+spare(l.windowRadius))};
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
 /** A word a page column: its window slot plus one, 0 for none. */
 private readonly directory:GPUBuffer;
 /** The lane's and the window's page columns as last placed (plan). */
 private held?:{lane:Uint8Array;window:Uint8Array};
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
  if(4*(l.laneBaseWord+UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS)>mask.size||4*(l.laneSlotBaseWord+l.laneSlots*l.laneSlotWords)>mask.size)
   throw new RangeError(`Uniform bed-slope bake: ${mask.label} of ${mask.size} bytes does not hold a lane directory at word ${l.laneBaseWord} and ${l.laneSlots} slots of ${l.laneSlotWords} words at word ${l.laneSlotBaseWord}`);
  // COPY_DST: a full bake clears the lane, the halo's entries with it.
  if(!(mask.usage&GPUBufferUsage.STORAGE)||!(mask.usage&GPUBufferUsage.COPY_DST))throw new Error(`Uniform bed-slope bake: ${mask.label} needs STORAGE and COPY_DST usage`);
  if(device.limits.minUniformBufferOffsetAlignment>256)throw new RangeError(`Uniform bed-slope bake: uniform records ${device.limits.minUniformBufferOffsetAlignment} bytes apart, more than 256`);
  const slots=l.layers*Math.max(1,l.columns),usage=GPUBufferUsage.STORAGE;
  this.meta=device.createBuffer({label:"Uniform bed-slope bake links",size:4*slots,usage});
  this.a=device.createBuffer({label:"Uniform bed-slope bake nosings A",size:16*slots,usage});
  this.b=device.createBuffer({label:"Uniform bed-slope bake nosings B",size:16*slots,usage});
  this.directory=device.createBuffer({label:"Uniform bed-slope bake window directory",size:4*UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS,usage:usage|GPUBufferUsage.COPY_DST});
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
  const build=device.createBindGroupLayout({label:"Uniform bed-slope bake build",entries:[params,storage(1,"read-only-storage"),storage(2,"storage"),storage(3,"storage"),storage(4,"storage"),storage(5,"read-only-storage")]});
  const spread=device.createBindGroupLayout({label:"Uniform bed-slope bake spread",entries:[params,storage(1,"read-only-storage"),storage(2,"read-only-storage"),storage(3,"storage"),storage(4,"read-only-storage")]});
  const resolve=device.createBindGroupLayout({label:"Uniform bed-slope bake resolve",entries:[params,storage(1,"storage"),storage(2,"read-only-storage"),storage(3,"read-only-storage"),storage(4,"read-only-storage")]});
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
  const d=this.directory;
  this.groups={build:group("build",build,[this.mask,this.meta,this.a,this.b,d]),spreadOut:group("spread out",spread,[this.meta,this.a,this.b,d]),
   spreadBack:group("spread back",spread,[this.meta,this.b,this.a,d]),resolve:group("resolve",resolve,[this.mask,this.meta,this.b,d])};
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
 /** Place the lane's and the window's page columns for a mask whose partly
  * solid pages lie in the page columns `partial` flags
  * (uniformBedSlopePartialColumns): slots in ascending page column, the
  * lane directory written into the mask buffer. True when either set moved:
  * the lane is then a full bake's to write (encode). More page columns than
  * the slots sized at build is refused. */
 plan(partial:ArrayLike<number>):boolean{
  const l=this.layout,lane=uniformBedSlopeGrown(partial,l.pages,l.laneRadius),window=uniformBedSlopeGrown(partial,l.pages,l.windowRadius),held=this.held;
  if(held&&lane.flags.every((f,i)=>f===held.lane[i])&&window.flags.every((f,i)=>f===held.window[i]))return false;
  if(lane.count>l.laneSlots||window.count>l.windowSlots)throw new Error(`Uniform bed-slope bake: the solids need ${lane.count} lane and ${window.count} window page columns; the bake was built for ${l.laneSlots} and ${l.windowSlots}. Rebuild the solver to take this edit`);
  const laneWords=new Uint32Array(UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS),windowWords=new Uint32Array(UNIFORM_BED_SLOPE_LANE_DIRECTORY_WORDS);
  for(let i=0,a=0,b=0;i<laneWords.length;i++){if(lane.flags[i])laneWords[i]=l.laneSlotBaseWord+l.laneSlotWords*a++;if(window.flags[i])windowWords[i]=++b;}
  this.device.queue.writeBuffer(this.mask,4*l.laneBaseWord,laneWords);this.device.queue.writeBuffer(this.directory,0,windowWords);
  this.held={lane:lane.flags,window:window.flags};
  return true;
 }
 /** The whole lane from the mask as the buffer holds it when the encoder
  * runs: at scene build, after the mask's upload and a plan, and after an
  * edit's plan moved. */
 encode(encoder:GPUCommandEncoder):void{
  if(!this.pipelines)throw new Error("Uniform bed-slope bake is not initialized");
  if(!this.held)throw new Error("Uniform bed-slope bake has no plan");
  const l=this.layout;
  if(l.laneSlots)encoder.clearBuffer(this.mask,4*l.laneSlotBaseWord,4*l.laneSlots*l.laneSlotWords);
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
  if(!this.held)throw new Error("Uniform bed-slope bake has no plan");
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
 destroy():void{this.meta.destroy();this.a.destroy();this.b.destroy();this.directory.destroy();this.params.destroy();}
}
