/** CPU-only census of the production Figure 7 reset; no WebGPU device. */
import {writeFileSync} from 'node:fs';
import {createCm12Figure7} from '../lib/core/cm12-paper-scenes';
import {initialLiquidFractionAtCell} from '../lib/core/initial-fluid';
import {resolveMethodValues} from '../lib/core/method-contract';
import {adaptiveMassMethod, adaptiveMassSolverOptions} from '../lib/methods/adaptive-volume/method';
import {initializeSparseBrickAtlasFromScene, sparseCM12InitialActiveBrickKeys, sparseBrickSpan} from '../lib/methods/adaptive-volume/sparse-brick-atlas';
import {buildSparseAtlasCompositeGrid} from '../lib/methods/adaptive-volume/sparse-atlas-composite-projection';
import {packSparseCM12ResidentTopologyTemplatesForQA, sparseCM12TopologyPagePoolPlan} from '../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident';

const scene = createCm12Figure7();
scene.fluid.initialLiquidVolumes = [{shape:'sphere', center_m:{x:0,y:4.5,z:0},radius_m:0.1}];
const values = resolveMethodValues(adaptiveMassMethod,'balanced', {
  timeStep:'scene',brickFineResolution:'8',presentationPageResolution:'8',selectorMode:'coarse-first',pressureRelativeTolerance:0.194,
});
const options = adaptiveMassSolverOptions(values);
const dimensions = [128,128,128] as const;
const histogram: Record<string,number> = {};
const wetCells: number[][] = [];
let sampledVolume = 0;
for(let z=60;z<68;z++)for(let y=86;y<94;y++)for(let x=60;x<68;x++){
  const rho=initialLiquidFractionAtCell(scene,x,y,z,dimensions,false);
  if(rho>0){histogram[rho]=(histogram[rho]??0)+1; sampledVolume+=rho;wetCells.push([x,y,z,rho]);}
}
const atlas=initializeSparseBrickAtlasFromScene(scene,{
  finestDimensions:dimensions,brickFineResolution:8,
  maximumMacroSpanBricks:options.maximumMacroSpanBricks,surfaceFineRings:options.surfaceFineRings,
  coarseFirstCurvatureTolerance:options.activityPolicy?.curvatureTolerance,
});
const active=sparseCM12InitialActiveBrickKeys(scene,atlas,2);
const grid=buildSparseAtlasCompositeGrid(atlas);
const templates=packSparseCM12ResidentTopologyTemplatesForQA(atlas,grid);
const pool=sparseCM12TopologyPagePoolPlan(Math.max(1,12*active.size-(atlas.bricks.length-active.size)),true,8,1024);
const rows:Record<string,number>={};for(const r of grid.gradientRows) rows[r.kind]=(rows[r.kind]??0)+1;
// At reset every accepted cell is finest-sized, so extension's physical
// opposite-side graph is exactly this six-neighbour grid. Count its eight
// validity sweeps separately from dispatched air lanes.
const neighbours=grid.cells.map(()=>[] as number[]);
for(const r of grid.gradientRows)if(r.terms.length===2){
  const a=r.terms[0]!.cellId,b=r.terms[1]!.cellId;neighbours[a]!.push(b);neighbours[b]!.push(a);
}
let extended=new Set(grid.cells.filter(c=>c.density>0.5).map(c=>c.id));
const extensionValidCells=[extended.size];
for(let i=0;i<8;i++){
  const next=new Set(extended);for(const id of extended)for(const n of neighbours[id]!)next.add(n);
  extended=next;extensionValidCells.push(extended.size);
}
const tracedFaceRows=grid.gradientRows.filter(r=>r.terms.some(t=>extended.has(t.cellId))).length;
const report={scene,values,initial:{geometricFinestCellVolumes:4*Math.PI*2**3/3,sampledVolume,histogram,wetCells,
  extensionValidCells,tracedFaceRows,initialFaceSupportCornerQueries:24*tracedFaceRows,
  brickCount:atlas.bricks.length,activeBrickCount:active.size,cells:grid.cells.length,rows:grid.gradientRows.length,rowKinds:rows,
  activeCells:grid.cells.filter(c=>active.has(c.brickKey)).length,
  wetAcceptedCells:grid.cells.filter(c=>c.density>0).length,
  densityOverHalfCells:grid.cells.filter(c=>c.density>0.5).length,
  incidentToWetRows:grid.gradientRows.filter(r=>r.terms.some(t=>grid.cells[t.cellId]!.density>0)).length,
  terms:grid.gradientRows.reduce((n,r)=>n+r.terms.length,0),
  bricks:atlas.bricks.map(b=>({key:b.key,coordinate:b.coordinate,resolution:b.resolution,span:sparseBrickSpan(b),active:active.has(b.key),wetCells:b.density.filter(v=>v>0).length,mass:[...b.density].reduce((n,v)=>n+v,0)*(8*sparseBrickSpan(b)/b.resolution)**3})),
  allRungTemplateCells:templates.cellCount,allRungTemplateRows:templates.rowCount,pool,
  leafCapacity:atlas.bricks.length+pool.pageCapacity,
  cellCapacity:templates.cellCount+pool.pageCapacity*512,rowCapacity:templates.rowCount+pool.pageCapacity*1728,
}};
writeFileSync('artifacts/cm12-figure-7-radius-01/cpu-census.json',JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({...report.initial,wetCells:undefined,bricks:undefined},null,2));
