/** Released geometry/material/timestep transport check. This is an isolated
 * manifold subsystem audit; it does not stand in for native fluid pressure or
 * certify a coupled source scene. Run with node --import tsx. */
import {readFile,writeFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {wetClothWaterSources,sourceClothDocument} from '../lib/cloth/scenes/sources';
import {decodeClothTopology} from '../lib/cloth/source-assets';
import {prepareClothTopology,createClothState} from '../lib/cloth/topology';
import {advanceClothPoreTransport} from '../lib/cloth/porous-flow';

const arg=(name:string)=>process.argv.find(v=>v.startsWith(`--${name}=`))?.slice(name.length+3);
const out=arg('out')??'/tmp/fluid-wetcloth-research/source-transport-audit.json',steps=Number(arg('steps')??1),rows:unknown[]=[],seedFront=process.argv.includes('--seed-front');
if(!Number.isSafeInteger(steps)||steps<1)throw new Error('Transport audit step count must be positive');
for(const source of wetClothWaterSources.filter(s=>s.counts.vertices&&(!arg('scene')||s.id===arg('scene')))){
 try{
  const data=decodeClothTopology(gunzipSync(await readFile(`public${source.asset.url}`))),document=sourceClothDocument(source.id),state=createClothState(prepareClothTopology(data,document.materials));
  if(seedFront){
   // Explicit diagnostic input, never silently called a source initial state.
   const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(let v=0;v<state.waterVolume_m3.length;v++)for(let a=0;a<3;a++){lo[a]=Math.min(lo[a]!,state.positions_m[3*v+a]!);hi[a]=Math.max(hi[a]!,state.positions_m[3*v+a]!);}
   const axis=[0,1,2].sort((a,b)=>(hi[b]!-lo[b]!)-(hi[a]!-lo[a]!))[0]!,cut=lo[axis]!+.2*(hi[axis]!-lo[axis]!);
   for(let v=0;v<state.waterVolume_m3.length;v++)state.waterVolume_m3[v]=state.positions_m[3*v+axis]!<=cut?.8*Math.max(0,state.controlVolume_m3[v]!-state.solidVolume_m3[v]!):0;
  }
  const before=state.waterVolume_m3.reduce((a,b)=>a+b,0),rho=document.materials[0]!.waterDensity_kg_m3;
  const momentum=()=>[0,1,2].map(a=>state.dryMass_kg.reduce((sum,dry,i)=>sum+(dry+rho*state.waterVolume_m3[i]!)*state.velocities_m_s[3*i+a]!,0));
  const originalMomentum=momentum(),reaction=[0,0,0];let movedVolume_m3=0,limitedVolume_m3=0,iterations=0;
  for(let i=0;i<steps;i++){
   const result=advanceClothPoreTransport({state,data,materials:document.materials,physics:document.physics,dt_s:source.dt_s,gravity_m_s2:source.gravity_m_s2});
   movedVolume_m3+=result.movedVolume_m3;limitedVolume_m3+=result.limitedVolume_m3;iterations+=result.iterations;
   for(let a=0;a<3;a++)reaction[a]!+=result.supportReaction_Ns[a]!;
  }
  const volumeError_m3=state.waterVolume_m3.reduce((a,b)=>a+b,0)-before,now=momentum(),momentumError_Ns=Math.hypot(...now.map((x,a)=>x-originalMomentum[a]!-reaction[a]!));
  if(state.waterVolume_m3.some(v=>v<0||!Number.isFinite(v))||Math.abs(volumeError_m3)>Math.max(1e-20,before*1e-10)||momentumError_Ns>1e-10)throw new Error(`Transport ledger defect: water ${volumeError_m3}, momentum ${momentumError_Ns}`);
  rows.push({id:source.id,status:before===0?'dry-source-initial-state-no-transport':seedFront?'diagnostic-front-transport-passed':'isolated-source-transport-passed',steps,initialWater_m3:before,movedVolume_m3,limitedVolume_m3,iterations,volumeError_m3,momentumError_Ns});
 }catch(error){rows.push({id:source.id,status:'failed',error:String(error)});process.exitCode=1;}
 console.log(JSON.stringify(rows.at(-1)));await writeFile(out,JSON.stringify({reproductionValidated:false,mode:'isolated-manifold-with-source-inputs-no-native-pressure',initialWater:seedFront?'diagnostic-80-percent-wet-front':'released-source-initial-water',rows},null,2)+'\n');
}
