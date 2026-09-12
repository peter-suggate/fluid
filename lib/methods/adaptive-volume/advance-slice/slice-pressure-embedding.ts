import { CM12_GHOST_FLUID_THETA_MIN, CM12_LIQUID_ISOVALUE,
  cm12GhostFluidTheta } from "../../../core/cm12-numerics";
import { buildSparseAtlasCompositeGrid, type SparseAtlasCompositeGrid,
  type SparseAtlasGradientRow } from "../sparse-atlas-composite-projection";
import type { SliceSceneSeed } from "./slice-scene-seed";
import type { SliceNumericalFields, SliceNumericalTopology } from "./slice-stage-numerics";
import type { SliceTopology } from "./slice-topology";
import { solveSlicePressurePCG, type SlicePressurePCGReceipt } from "./slice-pressure-pcg";
import { createSlicePressureAuthority, publishSlicePressureAuthority,
  type SlicePressureAuthority } from "./slice-pressure-authority";
import { sparseAtlasBrickKey,
  type SparseAdaptiveMassAtlas, type SparseAdaptiveMassBrick } from "../sparse-brick-atlas";

const f = Math.fround;
const add = (a:number,b:number)=>f(f(a)+f(b));
const mul = (a:number,b:number)=>f(f(a)*f(b));
const div = (a:number,b:number)=>f(f(a)/f(b));

export interface SlicePressureEmbedding {
  readonly grid: SparseAtlasCompositeGrid;
  /** Source-grid cell -> accepted 2-D compact cell, or -1 outside the slice authority. */
  readonly reducedCell: Int32Array;
  readonly reducedOwner: Int32Array;
  /** Accepted 2-D compact cell -> unique half-open centre-Z source cell. */
  readonly centreCell: Int32Array;
  readonly incidences: readonly (readonly number[])[];
  /** Source row -> unique projected 2-D row; -1 for virtual Z/unmapped. */
  readonly projectedRow: Int32Array;
  /** 2-D row -> unique source row incident to a centre-Z source cell. */
  readonly centreRow: Int32Array;
  /** First unmapped/ambiguous reduced cell or row; fail-closes the solve. */
  readonly mappingFault?: { readonly kind:"cell"|"row"; readonly id:number };
  readonly numericalTopology: SliceNumericalTopology;
  /** Literal PCM1/PCF1/PEI1 images over the retained three-dimensional graph. */
  pressureAuthority: SlicePressureAuthority;
  pressure: Float32Array;
  pressureMember: Uint8Array;
  rowActive: Uint8Array;
  rowTheta: Float32Array;
  densityBits: Uint32Array;
  capacityBits: Uint32Array;
  normalXBits: Uint32Array;
  normalYBits: Uint32Array;
  rowOpenBits: Uint32Array;
  cacheInitialized: boolean;
}

export interface SlicePressureEmbeddingReceipt {
  readonly diagonal: Float32Array;
  readonly rhs: Float32Array;
  readonly activeRows: Uint8Array;
  readonly theta: Float32Array;
  readonly pressureWeight: Float32Array;
  readonly virtualDiagonal: Float32Array;
  readonly virtualRhs: Float32Array;
  readonly virtualMember: Uint8Array;
  readonly dirtyRowCount: number;
  readonly mappingFault?: SlicePressureEmbedding["mappingFault"];
}

export interface SlicePressureEmbeddingSolveReceipt {
  readonly embedding: SlicePressureEmbedding;
  readonly prepared: SlicePressureEmbeddingReceipt;
  readonly solve: SlicePressurePCGReceipt;
}

function ownerTable(topology:SliceNumericalTopology):Int32Array{
  const [nx,ny]=topology.dimensions!,table=new Int32Array(nx*ny).fill(-1);
  for(const cell of topology.cells)for(let y=Math.floor(cell.minimum[1]);y<cell.maximum[1];y++)
    for(let x=Math.floor(cell.minimum[0]);x<cell.maximum[0];x++)if(x>=0&&x<nx&&y>=0&&y<ny)table[x+nx*y]=cell.id;
  return table;
}

function virtualNumericalTopology(grid:SparseAtlasCompositeGrid,
  incidences:readonly (readonly number[])[]):SliceNumericalTopology{
  // SlicePressureAuthority is dimension-agnostic at runtime; its public
  // topology type is shared with the 2-D numerical stages, so retain axis Z
  // through this single checked adapter rather than collapsing the graph.
  return {
    cells:grid.cells.map(cell=>({id:cell.id,stableId:cell.stableLeafId,
      minimum:[cell.minimumFine[0],cell.minimumFine[1]],maximum:[cell.maximumFine[0],cell.maximumFine[1]],
      center:[cell.centerFine[0],cell.centerFine[1]],widths:[cell.widthsFine[0],cell.widthsFine[1]],
      area:cell.volumeFineCells,brickKey:cell.brickKey})),
    rows:grid.gradientRows.map(row=>({id:row.id,kind:row.kind,
      axis:row.axis as 0|1,center:[row.centerFine[0],row.centerFine[1]],
      area:row.areaFineCells2,staticArea:row.areaFineCells2,distance:row.centerDistanceFine,
      dualWeight:row.dualWeight,staticDualWeight:row.dualWeight,terms:row.terms})),
    subfaces:[],incidences,
  };
}

export function createSlicePressureEmbedding(seed:SliceSceneSeed,topology:SliceTopology,
  numerical:SliceNumericalTopology,initialFields?:SliceNumericalFields,
  previous?:SlicePressureEmbedding):SlicePressureEmbedding|undefined{
  if(!seed.sourceAtlas)return undefined;
  const source=seed.sourceAtlas;
  // Candidate generations extrude their accepted XY rung/activity through the
  // retained source Z layout; source brick keys/slots and depth seams stay put.
  const bricks:SparseAdaptiveMassBrick[]=[];
  for(const xy of topology.bricks){
    if(xy.active===false)continue;
    const span=xy.spanBricks??1,existing=source.bricks.filter(brick=>
      brick.coordinate[0]===xy.coordinate[0]&&brick.coordinate[1]===xy.coordinate[1]
      &&(brick.spanBricks??1)===span);
    const append=(brick:SparseAdaptiveMassBrick)=>{
      if(xy.resolution===brick.resolution){bricks.push(brick);return;}
      const count=xy.resolution**3;
      bricks.push({...brick,resolution:xy.resolution,density:new Float64Array(count),
        gamma:new Float64Array(count).fill(1)});
    };
    if(existing.length){existing.forEach(append);continue;}
    // A policy-grown 2-D WDR page has no generation-zero source brick.  Its
    // retained pressure context is the same page extruded through each aligned
    // source-Z brick layer, initialized as explicit new air.
    for(let z=0;z<source.brickDimensions[2];z+=span){
      const coordinate=[xy.coordinate[0],xy.coordinate[1],z] as const,count=xy.resolution**3;
      append({key:sparseAtlasBrickKey(coordinate,source),coordinate,spanBricks:span,
        unclipped:true,resolution:xy.resolution,density:new Float64Array(count),
        gamma:new Float64Array(count).fill(1)});
    }
  }
  const directory=new Map(bricks.map(brick=>[brick.key,brick] as const)),directoriesBySpan=new Map<number,Map<number,SparseAdaptiveMassBrick>>();
  for(const brick of bricks){const span=brick.spanBricks??1,by=directoriesBySpan.get(span)??new Map();by.set(brick.key,brick);directoriesBySpan.set(span,by);}
  const atlas:SparseAdaptiveMassAtlas={...source,bricks,directory,directoriesBySpan,
    generation:topology.generation};
  const grid=buildSparseAtlasCompositeGrid(atlas),owners=ownerTable(numerical),nx=numerical.dimensions![0];
  const reducedCell=Int32Array.from(grid.cells,cell=>{
    const x=Math.max(0,Math.min(numerical.dimensions![0]-1,Math.floor(cell.centerFine[0]))),
      y=Math.max(0,Math.min(numerical.dimensions![1]-1,Math.floor(cell.centerFine[1])));
    return owners[x+nx*y]??-1;});
  const centreCell=new Int32Array(topology.cells.length).fill(-1),z=seed.viewport.centerCellZ+.5;
  for(const cell of grid.cells){const reduced=reducedCell[cell.id]!;if(reduced<0)continue;
    const target=topology.cells[reduced]!;
    if(cell.minimumFine[0]===target.minimumFine[0]&&cell.maximumFine[0]===target.maximumFine[0]
      &&cell.minimumFine[1]===target.minimumFine[1]&&cell.maximumFine[1]===target.maximumFine[1]
      &&z>=cell.minimumFine[2]&&z<cell.maximumFine[2])centreCell[reduced]=cell.id;
  }
  const incidences=Array.from({length:grid.cells.length},()=>[] as number[]);
  for(const row of grid.gradientRows)for(const term of row.terms)incidences[term.cellId]!.push(row.id);
  const projectedRow=new Int32Array(grid.gradientRows.length).fill(-1);
  for(const row of grid.gradientRows){if(row.axis===2)continue;const matches=numerical.rows.filter(candidate=>candidate.axis===row.axis
      &&candidate.center[0]===row.centerFine[0]&&candidate.center[1]===row.centerFine[1]);
    if(matches.length===1)projectedRow[row.id]=matches[0]!.id;}
  const centreRow=new Int32Array(numerical.rows.length).fill(-1);
  for(const row of grid.gradientRows){const projected=projectedRow[row.id]!;if(projected<0)continue;
    if(!row.terms.some(term=>{const reduced=reducedCell[term.cellId]!;return reduced>=0&&centreCell[reduced]===term.cellId;}))continue;
    centreRow[projected]=centreRow[projected]===-1?row.id:-2;
  }
  const missingCell=centreCell.findIndex(value=>value<0),missingRow=centreRow.findIndex(value=>value<0);
  const mappingFault=missingCell>=0?{kind:"cell" as const,id:missingCell}
    :missingRow>=0?{kind:"row" as const,id:missingRow}:undefined;
  const pressure=initialFields?Float32Array.from(grid.cells,cell=>{
    const reduced=reducedCell[cell.id]!;return reduced<0?0:initialFields.pressure[reduced]!;}):new Float32Array(grid.cells.length);
  const pressureMember=initialFields?Uint8Array.from(grid.cells,cell=>{
    const reduced=reducedCell[cell.id]!;return reduced<0?0:initialFields.pressureMember[reduced]!;}):new Uint8Array(grid.cells.length);
  const frozenIncidences=incidences.map(value=>Object.freeze(value));
  const virtualTopology=virtualNumericalTopology(grid,frozenIncidences);
  const stableHighWater=grid.cells.reduce((maximum,cell)=>Math.max(maximum,cell.stableLeafId+1),1);
  const brickHighWater=grid.atlas.bricks.reduce((maximum,brick)=>Math.max(maximum,brick.key+1),1);
  let pressureAuthority=previous&&previous.pressureAuthority.cellCapacity>=stableHighWater
    &&previous.pressureAuthority.rowCapacity>=grid.gradientRows.length
    &&previous.pressureAuthority.peiLayout.brickCapacity>=brickHighWater
    ?previous.pressureAuthority:createSlicePressureAuthority(virtualTopology,{cells:stableHighWater,
      rows:Math.max(1,grid.gradientRows.length),bricks:brickHighWater});
  if(previous&&pressureAuthority!==previous.pressureAuthority){
    const prior=previous.pressureAuthority.receipt;
    pressureAuthority={...pressureAuthority,receipt:{...pressureAuthority.receipt,
      topologyGeneration:prior.topologyGeneration,pcmCellGeneration:prior.pcmCellGeneration,
      pcmRowGeneration:prior.pcmRowGeneration,coefficientGeneration:prior.coefficientGeneration,
      executionGeneration:prior.executionGeneration}};
  }
  return {grid,reducedCell,reducedOwner:owners,centreCell,incidences:frozenIncidences,projectedRow,centreRow,mappingFault,
    numericalTopology:virtualTopology,pressureAuthority,
    pressure,pressureMember,
    rowActive:new Uint8Array(grid.gradientRows.length),rowTheta:new Float32Array(grid.gradientRows.length),
    densityBits:new Uint32Array(grid.cells.length),capacityBits:new Uint32Array(grid.cells.length),
    normalXBits:new Uint32Array(grid.cells.length),normalYBits:new Uint32Array(grid.cells.length),
    rowOpenBits:new Uint32Array(grid.gradientRows.length),
    cacheInitialized:false};
}

function matching2dRow(embedding:SlicePressureEmbedding,numerical:SliceNumericalTopology,row:SparseAtlasGradientRow){
  const projected=embedding.projectedRow[row.id]??-1;
  return projected>=0?numerical.rows[projected]:undefined;
}

const virtualRoundoff=(capacity:number)=>mul(9.5367431640625e-7,capacity);

function virtualSourceRate(embedding:SlicePressureEmbedding,numerical:SliceNumericalTopology,
  fields:SliceNumericalFields,source:number):number{
  const reduced=embedding.reducedCell[source]!;
  if(reduced<0)return 0;
  return mul(fields.sourceRate?.[reduced]??0,
    div(embedding.grid.cells[source]!.volumeFineCells,numerical.cells[reduced]!.area));
}

function virtualCapacity(embedding:SlicePressureEmbedding,fields:SliceNumericalFields,
  source:number,before=false):number{
  const reduced=embedding.reducedCell[source]!;
  if(reduced<0)return 0;
  return mul((before?fields.capacityBefore?.[reduced]:undefined)??fields.capacity[reduced]!,
    embedding.grid.cells[source]!.volumeFineCells);
}

function virtualPressureDensity(embedding:SlicePressureEmbedding,numerical:SliceNumericalTopology,
  fields:SliceNumericalFields,source:number):number{
  const reduced=embedding.reducedCell[source]!;
  if(reduced<0)return 0;
  let density=div(fields.density[reduced]!,Math.max(fields.capacity[reduced]!,1e-6));
  const dt=fields.frameDt??0,finalCapacity=virtualCapacity(embedding,fields,source),
    beforeCapacity=virtualCapacity(embedding,fields,source,true),sourceRate=virtualSourceRate(embedding,numerical,fields,source);
  if(fields.solidMotionActive&&finalCapacity<beforeCapacity&&fields.density[reduced]!>0){
    const rate=dt>0?div(f(finalCapacity-beforeCapacity),dt):0;
    density=Math.max(density,add(CM12_LIQUID_ISOVALUE,
      Math.min(.5,div(mul(-rate,dt),Math.max(finalCapacity,1e-8)))));
  }
  if(sourceRate>0)density=Math.max(density,add(CM12_LIQUID_ISOVALUE,
    Math.min(.5,div(mul(sourceRate,dt),Math.max(finalCapacity,1e-8)))));
  return f(density);
}

function pressureOpenFraction(row:SparseAtlasGradientRow,row2:ReturnType<typeof matching2dRow>,
  embedding:SlicePressureEmbedding,fields:SliceNumericalFields):number{
  if(row.axis===2){
    // The retained depth graph is a virtual extrusion of the 2-D state.  A
    // Z-normal face therefore has the same mean(old,new) aperture as its XY
    // footprint, matching the X/Y dynamic geometry schedule.
    let open=1,seen=false;
    for(const term of row.terms){const reduced=embedding.reducedCell[term.cellId]!;
      if(reduced<0)continue;
      const before=fields.capacityBefore?.[reduced]??fields.capacity[reduced]!,
        after=fields.capacityAfter?.[reduced]??fields.capacity[reduced]!;
      open=Math.min(open,mul(.5,add(before,after)));seen=true;}
    return f(seen?open:1);
  }
  if(!row2)return 1;
  return f(row2.kind==="closed-world"?(row2.separating?1:0):(row2.openFraction??1));
}

function rowFluidVelocity(row:SparseAtlasGradientRow,row2:ReturnType<typeof matching2dRow>,
  fields:SliceNumericalFields):number{
  if(row.axis===2||!row2)return 0;
  return f(fields.faceVelocity[row2.id]!-mul(1-(row2.openFraction??1),row2.solidVelocity??0));
}

function virtualMovingPredictedFill(embedding:SlicePressureEmbedding,seed:SliceSceneSeed,
  numerical:SliceNumericalTopology,fields:SliceNumericalFields,source:number):boolean{
  if(!fields.solidMotionActive)return false;
  let equation=0,correction=0;
  for(const rowId of embedding.incidences[source]??[]){
    const row=embedding.grid.gradientRows[rowId]!,own=row.terms.find(term=>term.cellId===source);
    if(row.axis===2&&row.kind==="sparse-air"&&seed.boundary.z!=="omitted")continue;
    if(!own)continue;
    const value=mul(own.coefficient,mul(row.dualWeight,
      rowFluidVelocity(row,matching2dRow(embedding,numerical,row),fields)));
    if(value>0){let supported=false;
      for(const term of row.terms){if(term.cellId===source||term.coefficient*own.coefficient>=0)continue;
        const reduced=embedding.reducedCell[term.cellId]!;
        if(reduced<0)continue;
        const volume=mul(fields.density[reduced]!,embedding.grid.cells[term.cellId]!.volumeFineCells),
          capacity=virtualCapacity(embedding,fields,term.cellId);
        supported||=volume>virtualRoundoff(capacity)||virtualSourceRate(embedding,numerical,fields,term.cellId)>0;
      }
      if(!supported)continue;
    }
    const adjusted=f(value-correction),next=add(equation,adjusted);
    correction=f(f(next-equation)-adjusted);equation=next;
  }
  const reduced=embedding.reducedCell[source]!;
  if(reduced<0)return false;
  const predicted=add(mul(fields.density[reduced]!,embedding.grid.cells[source]!.volumeFineCells),
    mul(fields.frameDt??0,equation)),capacity=virtualCapacity(embedding,fields,source);
  return predicted>=f(capacity-virtualRoundoff(capacity));
}

function integratedColumnHeight(embedding:SlicePressureEmbedding,numerical:SliceNumericalTopology,
  fields:SliceNumericalFields,x:number):readonly[number,number]{
  const [nx,ny]=numerical.dimensions!,sx=Math.max(0,Math.min(nx-1,x));
  let y=0,massHeight=0,previous=1,columnOpen=-1,sawOpen=false,sawLiquid=false,sawAir=false;
  while(y<ny){
    if((numerical.solidVoxelFractionAt?.(sx,y)??0)>=1)return [0,0];
    const owner=embedding.reducedOwner[sx+nx*y]??-1;let fill=0,width=1;
    if(owner<0)width=Math.max(1,Math.min(8-y%8,ny-y));
    else{
      const open=fields.capacity[owner]!;if(open<=1e-6)return [0,0];
      sawOpen=true;if(columnOpen<0)columnOpen=open;
      if(Math.abs(f(open-columnOpen))>1e-3)return [0,0];
      const source=embedding.centreCell[owner]!;
      fill=Math.max(0,Math.min(1,source<0?0:virtualPressureDensity(embedding,numerical,fields,source)));
      const cell=numerical.cells[owner]!;
      width=Math.max(1,Math.min(cell.widths[1]-y%cell.widths[1],ny-y));
    }
    if(fill>previous+.01)return [0,0];
    previous=fill;sawLiquid||=fill>1e-3;sawAir||=fill<1-1e-3;
    massHeight=add(massHeight,mul(fill,width));y+=width;
  }
  return [massHeight,sawOpen&&sawLiquid&&sawAir?1:0];
}

function planarColumnHeight(embedding:SlicePressureEmbedding,numerical:SliceNumericalTopology,
  fields:SliceNumericalFields,row:SparseAtlasGradientRow):readonly[number,number]{
  const nx=numerical.dimensions![0],centre=Math.max(0,Math.min(nx-1,Math.floor(row.centerFine[0])));
  let height=0,minimum=Number.MAX_VALUE,maximum=-Number.MAX_VALUE,valid=true;
  // The retained state is a virtual Z extrusion, so production's five X/Z
  // probes reduce to centre and the two distinct X neighbours.
  for(const offset of [0,-1,1]){const receipt=integratedColumnHeight(embedding,numerical,fields,
      Math.max(0,Math.min(nx-1,centre+offset)));
    if(offset===0)height=receipt[0];valid&&=receipt[1]>.5;
    minimum=Math.min(minimum,receipt[0]);maximum=Math.max(maximum,receipt[0]);
  }
  return [height,valid&&maximum-minimum<=.01?1:0];
}

/**
 * Execute production row classification on the retained source-atlas graph,
 * then publish its centre-Z equations in unit-depth 2-D units. This preserves
 * individual mixed Y/Z terms through coefficient squaring.
 */
export function prepareSlicePressureEmbedding(embedding:SlicePressureEmbedding,
  seed:SliceSceneSeed,numerical:SliceNumericalTopology,fields:SliceNumericalFields):SlicePressureEmbeddingReceipt{
  const {grid,reducedCell}=embedding,prior=embedding.pressureMember,
    member=new Uint8Array(grid.cells.length),
    activeRows=new Uint8Array(grid.gradientRows.length),theta=new Float32Array(grid.gradientRows.length),
    pressureWeight=new Float32Array(grid.gradientRows.length);
  if(embedding.mappingFault){
    fields.fault={stage:"pressure-embedding-map",index:embedding.mappingFault.id,
      observed:embedding.mappingFault.kind==="cell"?1:2,expected:0};
    return {diagonal:new Float32Array(numerical.cells.length),rhs:new Float32Array(numerical.cells.length),
      activeRows,theta,pressureWeight,virtualDiagonal:new Float32Array(grid.cells.length),
      virtualRhs:new Float32Array(grid.cells.length),virtualMember:member,dirtyRowCount:0,
      mappingFault:embedding.mappingFault};
  }
  const directDirty=new Uint8Array(grid.cells.length),dirtyCell=new Uint8Array(grid.cells.length),
    bitBuffer=new ArrayBuffer(4),bitFloat=new Float32Array(bitBuffer),bitWord=new Uint32Array(bitBuffer),
    bits=(value:number)=>{bitFloat[0]=value;return bitWord[0]!;};
  for(const cell of grid.cells){const reduced=reducedCell[cell.id]!;if(reduced<0)continue;
    const density=bits(fields.density[reduced]!),capacity=bits(fields.capacity[reduced]!),nx=bits(fields.interfaceNormal[2*reduced]!),ny=bits(fields.interfaceNormal[2*reduced+1]!);
    directDirty[cell.id]=!embedding.cacheInitialized||density!==embedding.densityBits[cell.id]||capacity!==embedding.capacityBits[cell.id]
      ||nx!==embedding.normalXBits[cell.id]||ny!==embedding.normalYBits[cell.id]||(fields.sourceRate?.[reduced]??0)!==0?1:0;
    embedding.densityBits[cell.id]=density;embedding.capacityBits[cell.id]=capacity;embedding.normalXBits[cell.id]=nx;embedding.normalYBits[cell.id]=ny;
  }
  const pressureDensity=(source:number)=>virtualPressureDensity(embedding,numerical,fields,source);
  for(const cell of grid.cells){const reduced=reducedCell[cell.id]!;if(reduced<0)continue;
    let submerged=prior[cell.id]!==0,neighbors=0;
    if(submerged)for(const rowId of embedding.incidences[cell.id]??[]){const row=grid.gradientRows[rowId]!;
      if(row.axis===2&&row.kind==="sparse-air"&&seed.boundary.z!=="omitted")continue;
      if(row.terms.length<2){submerged=false;break;}
      for(const term of row.terms)if(term.cellId!==cell.id){neighbors++;if(!prior[term.cellId]){submerged=false;break;}}
      if(!submerged)break;
    }
    submerged&&=neighbors>0;
    member[cell.id]=(pressureDensity(cell.id)>=CM12_LIQUID_ISOVALUE||submerged
      ||virtualMovingPredictedFill(embedding,seed,numerical,fields,cell.id)
      ||virtualSourceRate(embedding,numerical,fields,cell.id)>0)
      &&virtualCapacity(embedding,fields,cell.id)>1e-8?1:0;
    if(member[cell.id]!==prior[cell.id])directDirty[cell.id]=1;
  }
  dirtyCell.set(directDirty);
  for(const cell of grid.cells){const reduced=reducedCell[cell.id]!;
    if(reduced<0||fields.capacity[reduced]!<.999999)continue;
    const fill=div(fields.density[reduced]!,Math.max(fields.capacity[reduced]!,1e-8));
    if(fill<=0||fill>=1)continue;
    for(const rowId of embedding.incidences[cell.id]??[]){const row=grid.gradientRows[rowId]!,own=row.terms.find(term=>term.cellId===cell.id);
      if(!own)continue;
      if(row.terms.some(term=>term.cellId!==cell.id&&own.coefficient*term.coefficient<0&&directDirty[term.cellId])){
        dirtyCell[cell.id]=1;break;
      }
    }
  }
  // PCM marks a 64-row tile when any contributing cell changes, and static or
  // moving solid geometry forces global invalidation.  Reclassifying only the
  // individual row changes the observable f32 cache schedule.
  const dirtyTiles=new Uint8Array(Math.ceil(grid.gradientRows.length/64));
  const globalInvalidation=!embedding.cacheInitialized||!!seed.production?.solidWorld||!!fields.solidMotionActive;
  if(globalInvalidation)dirtyTiles.fill(1);
  else for(const row of grid.gradientRows){
    const row2=matching2dRow(embedding,numerical,row),open=bits(pressureOpenFraction(row,row2,embedding,fields));
    if(open!==embedding.rowOpenBits[row.id]||row.terms.some(term=>dirtyCell[term.cellId]))dirtyTiles[row.id>>>6]=1;
  }
  let dirtyRowCount=0;
  const partialRegion=numerical.cells.some(cell=>(cell.refinementRegionScale??1)>1)
    &&numerical.cells.some(cell=>(cell.refinementRegionScale??1)===1);
  for(const row of grid.gradientRows){
    const row2=matching2dRow(embedding,numerical,row),openFraction=pressureOpenFraction(row,row2,embedding,fields);
    embedding.rowOpenBits[row.id]=bits(openFraction);
    const dirty=dirtyTiles[row.id>>>6]!==0;
    if(!dirty){activeRows[row.id]=embedding.rowActive[row.id]!;theta[row.id]=embedding.rowTheta[row.id]!;
      pressureWeight[row.id]=mul(row.dualWeight,openFraction);continue;}
    dirtyRowCount++;
    // Z sparse-air ports are symmetry/depth-boundary rows in the reduced model.
    if(row.axis===2&&row.kind==="sparse-air"&&seed.boundary.z!=="omitted")continue;
    const closedWorld=row2?.kind==="closed-world",exterior=closedWorld||row.kind==="sparse-air";
    let gx=0,gy=0,go=0,gw=0;
    if(!closedWorld)for(const term of row.terms){const reduced=reducedCell[term.cellId]!;if(reduced<0)continue;
      const cell=numerical.cells[reduced]!,nx=fields.interfaceNormal[2*reduced]!,ny=fields.interfaceNormal[2*reduced+1]!,w=Math.abs(term.coefficient);
      if(add(mul(nx,nx),mul(ny,ny))<=.5)continue;
      gx=add(gx,mul(w,nx));gy=add(gy,mul(w,ny));
      go=add(go,mul(w,add(fields.interfaceOffset[reduced]!,add(mul(nx,f(cell.center[0]-row.centerFine[0])),mul(ny,f(cell.center[1]-row.centerFine[1]))))));gw=add(gw,w);
    }
    const gl=f(Math.sqrt(add(mul(gx,gx),mul(gy,gy))));let geometryValid=gw>1e-8&&gl>mul(1e-6,gw);
    if(geometryValid)for(const term of row.terms){const reduced=reducedCell[term.cellId]!,cell=grid.cells[term.cellId]!;
      if(reduced<0){geometryValid=false;break;}const phi=div(f(add(mul(gx,f(cell.centerFine[0]-row.centerFine[0])),mul(gy,f(cell.centerFine[1]-row.centerFine[1])))-go),gl);
      geometryValid&&=fields.capacity[reduced]!>=.999999&&(member[term.cellId]?phi<=0:phi>=0);
    }
    let liquid=0,air=0,lp=0,lw=0,ap=0,aw=0,liquidY=0,airY=0,fullGradient=0,liquidGradient=0;
    for(const term of row.terms){const reduced=reducedCell[term.cellId]!,cell=grid.cells[term.cellId]!;
      if(reduced<0)continue;const oldPhi=mul(f(CM12_LIQUID_ISOVALUE-pressureDensity(term.cellId)),exterior?1:cell.widthsFine[row.axis]);
      const phi=geometryValid?div(f(add(mul(gx,f(cell.centerFine[0]-row.centerFine[0])),mul(gy,f(cell.centerFine[1]-row.centerFine[1])))-go),gl):oldPhi,w=f(Math.abs(term.coefficient)),signed=mul(term.coefficient,phi);
      fullGradient=add(fullGradient,signed);
      if(member[term.cellId]){liquid++;lp=add(lp,mul(w,phi));lw=add(lw,w);liquidGradient=add(liquidGradient,signed);liquidY=add(liquidY,mul(w,cell.centerFine[1]));}
      else{air++;ap=add(ap,mul(w,phi));aw=add(aw,w);airY=add(airY,mul(w,cell.centerFine[1]));}
    }
    if(liquid===0)continue;
    if(exterior){ap=add(ap,mul(lw,row.exteriorPhi??.5));
      const ly=div(liquidY,Math.max(lw,1e-9)),direction=row.centerFine[1]>=ly?1:-1;
      airY=add(airY,mul(lw,add(ly,mul(direction,row.centerDistanceFine))));aw=add(aw,lw);}
    const cut=air>0||exterior;let value=cut?cm12GhostFluidTheta(div(lp,Math.max(lw,1e-9)),div(ap,Math.max(aw,1e-9)),1e-12):1;
    const acceleration=fields.accelerationFine??[0,0],gravityLength=f(Math.sqrt(add(mul(acceleration[0],acceleration[0]),mul(acceleration[1],acceleration[1]))));
    if(cut&&row.axis===1&&gravityLength>1e-6&&partialRegion&&acceleration[1]<0
      &&Math.abs(acceleration[0])<=mul(1e-6,gravityLength)){
      const height=planarColumnHeight(embedding,numerical,fields,row),ly=div(liquidY,Math.max(lw,1e-9)),ay=div(airY,Math.max(aw,1e-9));
      if(height[1]>.5&&ay>ly+1e-6&&height[0]>ly&&height[0]<ay)value=Math.max(CM12_GHOST_FLUID_THETA_MIN,
        Math.min(1,div(f(height[0]-ly),f(ay-ly))));
    }
    if(cut&&row.kind==="mixed-seam"){let factor=0;if(fullGradient!==0)factor=liquidGradient===0?div(1,CM12_GHOST_FLUID_THETA_MIN):Math.max(0,Math.min(div(1,CM12_GHOST_FLUID_THETA_MIN),div(fullGradient,liquidGradient)));value=factor>0?div(1,factor):0;}
    const weighted=mul(row.dualWeight,openFraction);if(!(weighted>1e-8))continue;
    pressureWeight[row.id]=weighted;theta[row.id]=f(value);activeRows[row.id]=1;
  }
  const virtualDiagonal=new Float32Array(grid.cells.length),virtualRhs=new Float32Array(grid.cells.length);
  for(const cell of grid.cells){if(!member[cell.id])continue;const axes=new Float32Array(3),fluxAxes=new Float32Array(6);
    for(const rowId of embedding.incidences[cell.id]??[]){const row=grid.gradientRows[rowId]!;if(!activeRows[rowId]||theta[rowId]!<=0)continue;
      const own=row.terms.find(term=>term.cellId===cell.id);if(!own)continue;const row2=matching2dRow(embedding,numerical,row),weight=pressureWeight[rowId]!;
      axes[row.axis]=add(axes[row.axis]!,div(mul(weight,mul(own.coefficient,own.coefficient)),theta[rowId]!));
      const velocity=rowFluidVelocity(row,row2,fields),value=mul(own.coefficient,mul(row.dualWeight,velocity)),side=2*row.axis+(own.coefficient>0?0:1);
      fluxAxes[side]=add(fluxAxes[side]!,value);
    }
    virtualDiagonal[cell.id]=add(add(axes[0]!,axes[1]!),axes[2]!);
    virtualRhs[cell.id]=add(add(add(Math.min(fluxAxes[0]!,fluxAxes[1]!),Math.max(fluxAxes[0]!,fluxAxes[1]!)),add(Math.min(fluxAxes[2]!,fluxAxes[3]!),Math.max(fluxAxes[2]!,fluxAxes[3]!))),add(Math.min(fluxAxes[4]!,fluxAxes[5]!),Math.max(fluxAxes[4]!,fluxAxes[5]!)));
    const reduced=reducedCell[cell.id]!;if(reduced>=0){virtualRhs[cell.id]=add(virtualRhs[cell.id],-(fields.capacityRate?.[reduced]??0)*cell.volumeFineCells);
      virtualRhs[cell.id]=add(virtualRhs[cell.id],virtualSourceRate(embedding,numerical,fields,cell.id));}
  }
  const diagonal=new Float32Array(numerical.cells.length),rhs=new Float32Array(numerical.cells.length);
  for(const reducedCellId of numerical.cells.map(cell=>cell.id)){
    const source=embedding.centreCell[reducedCellId]!;if(source<0||!member[source])continue;
    const axes=new Float32Array(3),fluxAxes=new Float32Array(6),depth=grid.cells[source]!.widthsFine[2];
    for(const rowId of embedding.incidences[source]??[]){const row=grid.gradientRows[rowId]!;if(!activeRows[rowId]||theta[rowId]!<=0)continue;
      const own=row.terms.find(term=>term.cellId===source);if(!own)continue;const row2=matching2dRow(embedding,numerical,row),
        weight=pressureWeight[rowId]!,d=div(mul(weight,mul(own.coefficient,own.coefficient)),theta[rowId]!);
      axes[row.axis]=add(axes[row.axis]!,d);
      const velocity=rowFluidVelocity(row,row2,fields),value=mul(own.coefficient,mul(row.dualWeight,velocity)),side=2*row.axis+(own.coefficient>0?0:1);
      fluxAxes[side]=add(fluxAxes[side]!,value);
    }
    diagonal[reducedCellId]=div(add(add(axes[0]!,axes[1]!),axes[2]!),depth);
    rhs[reducedCellId]=div(add(add(add(Math.min(fluxAxes[0]!,fluxAxes[1]!),Math.max(fluxAxes[0]!,fluxAxes[1]!)),add(Math.min(fluxAxes[2]!,fluxAxes[3]!),Math.max(fluxAxes[2]!,fluxAxes[3]!))),add(Math.min(fluxAxes[4]!,fluxAxes[5]!),Math.max(fluxAxes[4]!,fluxAxes[5]!))),depth);
    rhs[reducedCellId]=add(rhs[reducedCellId],-(fields.capacityRate?.[reducedCellId]??0)*numerical.cells[reducedCellId]!.area);
    rhs[reducedCellId]=add(rhs[reducedCellId],fields.sourceRate?.[reducedCellId]??0);
  }
  fields.pressureMember.fill(0);
  for(const cell of numerical.cells){const source=embedding.centreCell[cell.id]!;
    if(source>=0)fields.pressureMember[cell.id]=member[source]!;}
  fields.pressureRowMember=Uint8Array.from(numerical.rows,row=>{
    const source=embedding.centreRow[row.id]!;return source>=0?activeRows[source]!:0;});
  embedding.pressureMember=member;
  embedding.rowActive.set(activeRows);embedding.rowTheta.set(theta);embedding.cacheInitialized=true;
  const virtualFields:SliceNumericalFields={
    density:Float32Array.from(grid.cells,cell=>{const reduced=reducedCell[cell.id]!;
      return reduced<0?0:fields.density[reduced]!;}),
    gamma:new Float32Array(grid.cells.length),
    capacity:Float32Array.from(grid.cells,cell=>{const reduced=reducedCell[cell.id]!;
      return reduced<0?0:fields.capacity[reduced]!;}),
    cellVelocity:new Float32Array(2*grid.cells.length),faceVelocity:new Float32Array(grid.gradientRows.length),
    pressure:embedding.pressure,pressureRhs:virtualRhs,pressureDiagonal:virtualDiagonal,
    pressureMember:member,pressureRowMember:activeRows,extensionDepth:new Uint8Array(grid.cells.length),
    interfaceNormal:Float32Array.from({length:2*grid.cells.length},(_,at)=>{
      const source=at>>>1,reduced=reducedCell[source]!;return reduced<0?0:fields.interfaceNormal[2*reduced+(at&1)]!;}),
    interfaceOffset:new Float32Array(grid.cells.length),lowFlux:new Float32Array(),highFlux:new Float32Array(),
    limitedFlux:new Float32Array(),fault:null,
  };
  embedding.pressureAuthority=publishSlicePressureAuthority(embedding.pressureAuthority,
    embedding.numericalTopology,virtualFields,{active:activeRows,theta},grid.atlas.generation,
    {globalRowInvalidation:globalInvalidation});
  return {diagonal,rhs,activeRows,theta,pressureWeight,virtualDiagonal,virtualRhs,
    virtualMember:member,dirtyRowCount};
}

function applyVirtualOperator(embedding:SlicePressureEmbedding,prepared:SlicePressureEmbeddingReceipt,
  input:Float32Array,output:Float32Array):void{
  output.fill(0);const {grid}=embedding;
  for(const cell of grid.cells){if(!prepared.virtualMember[cell.id])continue;
    const negative=new Float32Array(3),positive=new Float32Array(3);
    for(const rowId of embedding.incidences[cell.id]??[]){const row=grid.gradientRows[rowId]!;
      if(!prepared.activeRows[rowId]||prepared.theta[rowId]!<=0)continue;
      const own=row.terms.find(term=>term.cellId===cell.id);if(!own)continue;let jump=0;
      if(row.terms.length===2&&f(row.terms[0]!.coefficient)===-f(row.terms[1]!.coefficient)){
        const a=row.terms[0]!,b=row.terms[1]!,pa=prepared.virtualMember[a.cellId]?input[a.cellId]!:0,pb=prepared.virtualMember[b.cellId]?input[b.cellId]!:0;
        jump=mul(b.coefficient,f(pb-pa));
      }else for(const term of row.terms)if(prepared.virtualMember[term.cellId])jump=add(jump,mul(term.coefficient,input[term.cellId]!));
      const contribution=div(mul(prepared.pressureWeight[rowId]!,
        mul(own.coefficient,jump)),prepared.theta[rowId]!);
      (own.coefficient>0?negative:positive)[row.axis]=add((own.coefficient>0?negative:positive)[row.axis]!,contribution);
    }
    output[cell.id]=add(add(add(Math.min(negative[0]!,positive[0]!),Math.max(negative[0]!,positive[0]!)),add(Math.min(negative[1]!,positive[1]!),Math.max(negative[1]!,positive[1]!))),add(Math.min(negative[2]!,positive[2]!),Math.max(negative[2]!,positive[2]!)));
  }
}

export function solveSlicePressureEmbedding(embedding:SlicePressureEmbedding,seed:SliceSceneSeed,
  numerical:SliceNumericalTopology,fields:SliceNumericalFields,maximumIterations:number,
  relativeTolerance:number,preparedInput?:SlicePressureEmbeddingReceipt):SlicePressureEmbeddingSolveReceipt{
  const prepared=preparedInput??prepareSlicePressureEmbedding(embedding,seed,numerical,fields),
    order=embedding.pressureAuthority.executionOrder;
  const solve=solveSlicePressurePCG({diagonal:prepared.virtualDiagonal,rhs:prepared.virtualRhs,
    pressure:embedding.pressure,member:prepared.virtualMember,executionOrder:order,
    maximumIterations,relativeTolerance,apply:(input,output)=>applyVirtualOperator(embedding,prepared,input,output)});
  embedding.pressure.set(solve.pressure);
  for(const cell of numerical.cells){const source=embedding.centreCell[cell.id]!;if(source>=0){fields.pressure[cell.id]=embedding.pressure[source]!;
    fields.pressureDiagonal[cell.id]=prepared.diagonal[cell.id]!;fields.pressureRhs[cell.id]=prepared.rhs[cell.id]!;}}
  return {embedding,prepared,solve};
}

export function projectSlicePressureEmbedding(receipt:SlicePressureEmbeddingSolveReceipt,
  numerical:SliceNumericalTopology,fields:SliceNumericalFields):Float32Array{
  const {embedding,prepared}=receipt,projectedFaceVelocity=new Float32Array(embedding.grid.gradientRows.length);
  for(const row of embedding.grid.gradientRows){const row2=matching2dRow(embedding,numerical,row),base=row.axis===2?0:row2?.id===undefined?0:fields.faceVelocity[row2.id]!;
    if(!prepared.activeRows[row.id]||prepared.theta[row.id]!<=0){projectedFaceVelocity[row.id]=base;continue;}let jump=0;
    for(const term of row.terms)if(prepared.virtualMember[term.cellId])jump=add(jump,mul(term.coefficient,embedding.pressure[term.cellId]!));
    const open=pressureOpenFraction(row,row2,embedding,fields);
    projectedFaceVelocity[row.id]=f(base-mul(open,div(jump,prepared.theta[row.id]!)));
  }
  for(const row of numerical.rows){const source=embedding.centreRow[row.id]!;
    if(source>=0)fields.faceVelocity[row.id]=projectedFaceVelocity[source]!;
  }
  return projectedFaceVelocity;
}
