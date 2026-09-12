import { productionSceneSliceSeedById } from "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { advanceSlice, createAdvanceSlice, type AdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import type { SliceTransportMicrostepReceipt } from
  "../lib/methods/adaptive-volume/advance-slice/slice-stage-numerics";

const pressureIterations=Number(process.argv.find(v=>v.startsWith("--pressure="))?.split("=")[1]??64);
const diagnosticSymmetry=process.argv.includes("--symmetric-transport-input");
if(!Number.isSafeInteger(pressureIterations)||pressureIterations<0)
  throw new RangeError("--pressure must be a nonnegative integer");

const slice=createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
const width=slice.nx;
const key=(axis:number,x:number,y:number,area:number)=>`${axis}:${x}:${y}:${area}`;

function forceTransportInputSymmetry(s:AdvanceSlice):void{
  const cells=new Map(s.numericalTopology.cells.map(cell=>[
    `${cell.minimum[0]}:${cell.minimum[1]}:${cell.maximum[0]}:${cell.maximum[1]}`,cell.id]));
  for(const cell of s.numericalTopology.cells){
    const mirror=cells.get(`${width-cell.maximum[0]}:${cell.minimum[1]}:${width-cell.minimum[0]}:${cell.maximum[1]}`);
    if(mirror===undefined||mirror<cell.id)continue;
    for(const field of [s.fields.density,s.fields.gamma,s.fields.capacity] as const){
      const value=Math.fround(0.5*(field[cell.id]!+field[mirror]!));field[cell.id]=value;field[mirror]=value;
    }
    const vx=Math.fround(0.5*(s.fields.cellVelocity[2*cell.id]!-s.fields.cellVelocity[2*mirror]!));
    const vy=Math.fround(0.5*(s.fields.cellVelocity[2*cell.id+1]!+s.fields.cellVelocity[2*mirror+1]!));
    s.fields.cellVelocity[2*cell.id]=vx;s.fields.cellVelocity[2*mirror]=-vx;
    s.fields.cellVelocity[2*cell.id+1]=vy;s.fields.cellVelocity[2*mirror+1]=vy;
    const nx=Math.fround(0.5*(s.fields.interfaceNormal[2*cell.id]!-s.fields.interfaceNormal[2*mirror]!));
    const ny=Math.fround(0.5*(s.fields.interfaceNormal[2*cell.id+1]!+s.fields.interfaceNormal[2*mirror+1]!));
    const offset=Math.fround(0.5*(s.fields.interfaceOffset[cell.id]!+s.fields.interfaceOffset[mirror]!));
    s.fields.interfaceNormal[2*cell.id]=nx;s.fields.interfaceNormal[2*mirror]=-nx;
    s.fields.interfaceNormal[2*cell.id+1]=ny;s.fields.interfaceNormal[2*mirror+1]=ny;
    s.fields.interfaceOffset[cell.id]=offset;s.fields.interfaceOffset[mirror]=offset;
  }
  const rows=new Map(s.numericalTopology.rows.map(row=>[key(row.axis,row.center[0],row.center[1],row.area),row.id]));
  for(const row of s.numericalTopology.rows){
    const mirror=rows.get(key(row.axis,width-row.center[0],row.center[1],row.area));
    if(mirror===undefined||mirror<row.id)continue;
    const sign=row.axis===0?-1:1;
    const value=Math.fround(0.5*(s.fields.faceVelocity[row.id]!+sign*s.fields.faceVelocity[mirror]!));
    s.fields.faceVelocity[row.id]=value;s.fields.faceVelocity[mirror]=sign*value;
  }
}

function fluxReport(s:AdvanceSlice,step:number,receipt:SliceTransportMicrostepReceipt){
  const faces=new Map(s.numericalTopology.subfaces.map(face=>[
    key(face.axis,face.center[0],face.center[1],face.area),face.id]));
  const fields=["sweep","initialLowFlux","highFlux","lowFlux","limitedFlux"] as const;
  const result:Record<string,unknown>={diagnosticSymmetry,pressureIterations,
    requestedFrame:6,completedFrame:s.frame,time_s:s.time_s,step};
  for(const name of fields){
    const values=receipt[name];let maximum=0,faceId=-1,mirrorId=-1;
    for(const face of s.numericalTopology.subfaces){
      const mirror=faces.get(key(face.axis,width-face.center[0],face.center[1],face.area));
      if(mirror===undefined)continue;
      const sign=face.axis===0?-1:1,error=Math.abs(values[face.id]!-sign*values[mirror]!);
      if(error>maximum){maximum=error;faceId=face.id;mirrorId=mirror;}
    }
    const face=faceId>=0?s.numericalTopology.subfaces[faceId]:undefined;
    const embedding=s.pressureEmbedding;
    const sourceRow=face&&embedding?embedding.centreRow[face.rowId]:-1;
    const pressureRow=embedding&&sourceRow!==undefined&&sourceRow>=0
      ?embedding.grid.gradientRows[sourceRow]:undefined;
    result[name]={maximum,faceId,mirrorId,
      axis:face?.axis,center:face?.center,rowKind:face&&s.numericalTopology.rows[face.rowId]?.kind,
      negativeCell:face?.negativeCell,
      positiveCell:face?.positiveCell,
      sourceRow,theta:pressureRow&&embedding?.rowTheta[sourceRow!],
      pressureTerms:pressureRow?.terms.filter(term=>embedding!.pressureMember[term.cellId])
        .map(term=>({cellId:term.cellId,coefficient:term.coefficient,
          pressure:embedding!.pressure[term.cellId]})),
      value:faceId>=0?values[faceId]:0,mirror:mirrorId>=0?values[mirrorId]:0};
  }
  console.log(JSON.stringify(result,(_k,v)=>typeof v==="number"&&!Number.isFinite(v)?String(v):v));
}

for(let frame=1;frame<=6;frame++)advanceSlice(slice,{pressureIterations,
  onStageComplete:stage=>{
    if(frame===6&&stage==="velocity-projection"&&diagnosticSymmetry)forceTransportInputSymmetry(slice);
  },
  onTransportMicrostep:(step,s,receipt)=>{if(frame===6)fluxReport(s,step,receipt);}});
let densityError=0;
for(let y=0;y<slice.ny;y++)for(let x=0;x<slice.nx/2;x++)densityError=Math.max(densityError,
  Math.abs(slice.V[y*slice.nx+x]!-slice.V[y*slice.nx+slice.nx-1-x]!));
console.log(JSON.stringify({diagnosticSymmetry,pressureIterations,requestedFrame:6,
  completedFrame:slice.frame,time_s:slice.time_s,finalDensityError:densityError,fault:slice.fault,
  mappingFault:slice.pressureEmbedding?.mappingFault,
  numericalRows:slice.numericalTopology.rows.length,
  mappedRows:slice.pressureEmbedding?.centreRow.reduce((count,row)=>count+(row>=0?1:0),0)}));
