import type { WebGPUAdaptiveMassSolver } from '../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver';
import { readPublishedCM12Field } from './sparse-cm12-published-field';

export interface AnalyticSurfaceColumn {
  readonly x:number;readonly z:number;readonly bottom:number|undefined;
  readonly top:number|undefined;readonly wetLength:number;
}

export function semiImplicitFreeFallSlab(step:number,dt_s:number,gravity_m_s2=-9.81,
  initialBottom_m=.8,initialTop_m=1.2) {
  const displacement=.5*gravity_m_s2*dt_s*dt_s*step*(step+1);
  return {bottom_m:initialBottom_m+displacement,top_m:initialTop_m+displacement,
    contactExpected:initialBottom_m+displacement<=0};
}

/** Shape receipt for the translated 0.8 m by 0.4 m slab. It measures the
 * published contour and does not infer correctness from conserved mass. */
export function assessAnalyticSlabShape(columns:readonly AnalyticSurfaceColumn[],h:number,
  expectedBottom:number,expectedTop:number,seamX=.8) {
  const resolved=columns.filter((column):column is AnalyticSurfaceColumn&{bottom:number;top:number} =>
    column.bottom!==undefined&&column.top!==undefined);
  const interior=resolved.filter(column=>column.x>.4&&column.x<1.2);
  const thicknesses=interior.map(column=>column.top-column.bottom);
  const rms=(values:readonly number[])=>values.length
    ?Math.sqrt(values.reduce((sum,value)=>sum+value*value,0)/values.length):null;
  const zValues=[...new Set(resolved.map(column=>column.z))];
  const widths=zValues.flatMap(z=>{const row=resolved.filter(column=>column.z===z);
    return row.length?[row.at(-1)!.x-row[0]!.x+h]:[];});
  const seamPairs=zValues.flatMap(z=>{
    const row=interior.filter(column=>column.z===z);
    const left=row.filter(column=>column.x<seamX).at(-1);
    const right=row.find(column=>column.x>seamX);
    return left&&right?[{top:Math.abs(left.top-right.top),bottom:Math.abs(left.bottom-right.bottom)}]:[];
  });
  return {expectedBottom_m:expectedBottom,expectedTop_m:expectedTop,
    meanWidth_m:widths.length?widths.reduce((a,b)=>a+b,0)/widths.length:null,
    widthRmsError_m:rms(widths.map(width=>width-.8)),
    meanThickness_m:thicknesses.length?thicknesses.reduce((a,b)=>a+b,0)/thicknesses.length:null,
    thicknessRmsError_m:rms(thicknesses.map(thickness=>thickness-.4)),
    maximumTopSeamJump_m:seamPairs.length?Math.max(...seamPairs.map(pair=>pair.top)):null,
    maximumBottomSeamJump_m:seamPairs.length?Math.max(...seamPairs.map(pair=>pair.bottom)):null,
    minimumBottom_m:resolved.length?Math.min(...resolved.map(column=>column.bottom)):null,
    maximumTop_m:resolved.length?Math.max(...resolved.map(column=>column.top)):null};
}

/** Integrate vertical intervals of the actual published zero contour. This is
 * separate from conserved tracking density: neither metric substitutes for the other.
 */
export async function measureAnalyticMotionPublishedSurface(device:GPUDevice,
  solver:WebGPUAdaptiveMassSolver,h:number,expectedBottom:number,expectedTop:number) {
  const {values}=await readPublishedCM12Field(device,solver);
  const [nx,ny,nz]=[solver.info.nx,solver.info.ny,solver.info.nz];
  let volume=0,topError2=0,bottomError2=0,matchedColumns=0,missingColumns=0,missingSupport=0;
  const negativeSamplesByY=Array.from({length:ny},()=>0);
  const finiteSamplesByY=Array.from({length:ny},()=>0);
  for(let y=0;y<ny;y++)for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
    const value=values[x+nx*(y+ny*z)]!;
    if(Number.isFinite(value)){finiteSamplesByY[y]++;if(value<0)negativeSamplesByY[y]++;}
  }
  const columns:AnalyticSurfaceColumn[]=[];
  for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
    let wetLength=0;const bottoms:number[]=[],tops:number[]=[];
    for(let y=0;y<ny-1;y++){
      const a=values[x+nx*(y+ny*z)]!,b=values[x+nx*(y+1+ny*z)]!;
      if(!Number.isFinite(a)||!Number.isFinite(b)){
        if(a<0||b<0)missingSupport++;continue;
      }
      if(a<0&&b<0)wetLength+=h;
      else if((a<0)!==(b<0)){
        const t=-a/(b-a),crossing=(y+.5+t)*h;
        wetLength+=h*(a<0?t:1-t);
        if(a<0)tops.push(crossing);else bottoms.push(crossing);
      }
    }
    if(values[x+nx*ny*z]!<0){wetLength+=.5*h;bottoms.unshift(0);}
    if(values[x+nx*(ny-1+ny*z)]!<0){wetLength+=.5*h;tops.push(ny*h);}
    volume+=wetLength*h*h;
    const bottom=bottoms[0],top=tops.at(-1);
    if((x+.5)*h>.4&&(x+.5)*h<1.2){
      if(bottom===undefined||top===undefined)missingColumns++;
      else{matchedColumns++;bottomError2+=(bottom-expectedBottom)**2;topError2+=(top-expectedTop)**2;}
    }
    if(wetLength>0)columns.push({x:(x+.5)*h,z:(z+.5)*h,bottom,top,wetLength});
  }
  const wetLayers=negativeSamplesByY.flatMap((count,y)=>count>0?[y]:[]);
  return {volume_m3:volume,relativeVolumeError:Math.abs(volume/.128-1),matchedColumns,missingColumns,missingSupport,
    topRmsError_m:matchedColumns?Math.sqrt(topError2/matchedColumns):null,
    bottomRmsError_m:matchedColumns?Math.sqrt(bottomError2/matchedColumns):null,
    negativeSamplesByY,finiteSamplesByY,
    wetLayerRange:wetLayers.length?[wetLayers[0]!,wetLayers.at(-1)!]:null,
    ceilingNegativeSamples:negativeSamplesByY.slice(-2).reduce((sum,value)=>sum+value,0),
    floorNegativeSamples:negativeSamplesByY.slice(0,2).reduce((sum,value)=>sum+value,0),columns,
    slabShape:assessAnalyticSlabShape(columns,h,expectedBottom,expectedTop)};
}
