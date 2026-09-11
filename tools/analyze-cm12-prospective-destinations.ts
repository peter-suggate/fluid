/** Offline local-velocity candidates tested against captured GPU receivers.
 * This heuristic is a measurement, not a proven characteristic bound.
 */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {performance} from 'node:perf_hooks';
const directory=resolve(process.argv[2]??'artifacts/cm12-prospective-destinations');
const capture=JSON.parse(readFileSync(resolve(directory,'report.json'),'utf8'));
type Cell={id:number;center:number[];width:number[];rho:number;velocity:number[]};
const dt=capture.scene.numerics.fixedDt_s??capture.scene.numerics.maxDt_s;
const reports=capture.reports.map((report:{step:number})=>{
 const frame=JSON.parse(readFileSync(resolve(directory,`frame-${report.step}.json`),'utf8')) as {
  report:Record<string,unknown>;cells:Cell[];backward:number[];forward:number[];gathered:number[]};
 const cells=frame.cells,wet=cells.filter(c=>c.rho>0),byId=new Map(cells.map(c=>[c.id,c]));
 const required=new Set([...frame.backward,...frame.forward,...frame.gathered]);
 const cellKey=(c:Cell)=>c.center.map(v=>Math.floor(v/4)).join(',');
 const residentBricks=new Set(cells.map(c=>c.center.map(v=>Math.floor(v/8)).join(',')));
 const localVariants=[.5,1,2].map(haloWidths=>{
  const begin=performance.now();const packets=new Set<string>();
  for(const c of wet){
   const lo=c.center.map((v,k)=>Math.floor((v-c.width[k]!/2+Math.min(0,c.velocity[k]!*dt)-haloWidths*c.width[k]!)/4));
   const hi=c.center.map((v,k)=>Math.floor((v+c.width[k]!/2+Math.max(0,c.velocity[k]!*dt)+haloWidths*c.width[k]!)/4));
   for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++)packets.add(`${x},${y},${z}`);
  }
  const selectionCpuMs=performance.now()-begin;
  const selected=cells.filter(c=>packets.has(cellKey(c)));
  const missing=(ids:Iterable<number>)=>[...ids].filter(id=>!packets.has(cellKey(byId.get(id)!)));
  const bricks=new Set([...packets].map(p=>p.split(',').map(v=>Math.floor(Number(v)/2)).join(',')));
  const missed=missing(required);
  return {haloWidths,candidatePackets:packets.size,candidateLogicalBricks:bricks.size,
   absentCandidateLogicalBricks:[...bricks].filter(b=>!residentBricks.has(b)).length,
   selectedResidentCells:selected.length,selectionCpuMs,
   missedRequired:missed.length,missedBackward:missing(frame.backward).length,
   missedForward:missing(frame.forward).length,missedGathered:missing(frame.gathered).length,
   firstMiss:missed.length?byId.get(missed[0]!):null};
 });
 assert.equal(new Set(cells.map(c=>c.id)).size,cells.length);
 return {...frame.report,localVariants};
});
const result={scope:'Radius 0.1 Figure 7, four sampled production GPU transport steps. Logical 4-cell bins are candidate coordinates, not native packet IDs. Logical 8-cell bricks are coordinate bins, not physical allocation counts. Local velocity bounds are empirical heuristics. All nonzero source density is retained, with no visual density cutoff.',
 limitations:['No virtual velocity sampling or transport into an unallocated destination is executed.','No GPU selector timing or net speedup is measured; CPU box construction excludes readback and matching.','Receiver coverage does not prove gamma evolution, pressure, boundary, or velocity-extension support can be removed.','Local velocity does not bound an RK2 characteristic in a general deforming field. Only captured receiver containment is established.'],reports};
writeFileSync(resolve(directory,'analysis.json'),JSON.stringify(result,null,2)+'\n');
for(const r of reports)console.log(JSON.stringify(r));
