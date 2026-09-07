import type { WebGPUAdaptiveMassSolver } from '../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver';
import { readPublishedCM12Field } from './sparse-cm12-published-field';

/** Integrate vertical intervals of the actual published zero contour. This is
 * separate from conserved tracking density: neither metric substitutes for the other.
 */
export async function measureAnalyticMotionPublishedSurface(device:GPUDevice,
  solver:WebGPUAdaptiveMassSolver,h:number,expectedBottom:number,expectedTop:number) {
  const {values}=await readPublishedCM12Field(device,solver);
  const [nx,ny,nz]=[solver.info.nx,solver.info.ny,solver.info.nz];
  let volume=0,topError2=0,bottomError2=0,matchedColumns=0,missingColumns=0,missingSupport=0;
  const columns=[];
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
  return {volume_m3:volume,relativeVolumeError:Math.abs(volume/.128-1),matchedColumns,missingColumns,missingSupport,
    topRmsError_m:matchedColumns?Math.sqrt(topError2/matchedColumns):null,
    bottomRmsError_m:matchedColumns?Math.sqrt(bottomError2/matchedColumns):null,columns};
}
