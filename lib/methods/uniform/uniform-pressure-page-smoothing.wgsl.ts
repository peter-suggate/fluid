/** Exact temporal smoothing of a whole root page. Eight neighbouring
 * octants share one halo; three Jacobi iterations still advance globally
 * between launches. The 8^3 core and radius-three halo fit in 32 KiB. */
export function uniformPressurePageSmoothingWGSL(lanes:number):string{return /* wgsl */`
var<workgroup> pageRegion:array<u32,2744>;
var<workgroup> pageA:array<f32,2744>;
var<workgroup> pageB:array<f32,1728>;
var<workgroup> pageLive:atomic<u32>;
fn pageValue(k:i32,j:i32)->f32{if((k&1)==0){return pageA[j];}return pageB[j];}
fn pageEnd(row:u32,value:f32,to:u32){
 let L=umLevel(0u);var cell=row;if(row>=L.cells){cell=umLOwner(L,row);}
 let job=8u*(cell>>9u)+((cell>>2u)&1u)+((cell>>4u)&2u)+((cell>>6u)&4u);
 if(UM_LISTED&&(atomicLoad(&umList[4u+8u*UM_BLOCKS+job])&4u)==0u){return;}
 umBlockEnd(row,value,to);
}
@compute @workgroup_size(${lanes}) fn smoothPages(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 let R=umLevel(0u);let src=R.p[UM_BSRC];let to=R.p[UM_BSRC^1u];let rhs=R.b[UM_BRHS];let low=R.m[UM_BRHS];
 if(lane==0u){umGo=select(atomicLoad(&umList[UM_COMPACT]),0u,umSlotClosed());}
 let count=workgroupUniformLoad(&umGo);
 for(var job=g.x+n.x*g.y;job<count;job+=n.x*n.y){
  let block=atomicLoad(&umList[UM_COMPACT+1u+job]);
  if(lane==0u){atomicStore(&pageLive,0u);}workgroupBarrier();
  var flags=0u;for(var j=lane;j<512u;j+=${lanes}u){flags|=umLMask(R,512u*block+j);}
  atomicOr(&pageLive,flags);let live=workgroupUniformLoad(&pageLive);
  if((live&1u)==0u){
   for(var j=lane;j<512u;j+=${lanes}u){
    let row=512u*block+j;pageEnd(row,max(native[src+row],native[low+row]),to);
    for(var f=0u;f<6u;f++){let s=umLNear(R,row,f);if(s>=R.cells){pageEnd(s,max(native[src+s],native[low+s]),to);}}
   }
   workgroupBarrier();continue;
  }
  if(lane<27u){umNear[lane]=umPageAt(umBlockPage(block)+vec3i(i32(lane%3u),i32((lane/3u)%3u),i32(lane/9u))-vec3i(1));}
  workgroupBarrier();
  let m=i32(UM_BCOUNT);let s0=8+2*m;
  for(var j=i32(lane);j<s0*s0*s0;j+=${lanes}){
   let at=umRegionRow(R,1,vec3i(j%s0,(j/s0)%s0,j/(s0*s0))-vec3i(m));
   var value=0.0;if(at!=UM_NONE){value=native[src+at];}pageRegion[j]=at;pageA[j]=value;
  }
  workgroupBarrier();
  for(var k=1;k<=m;k++){
   let sp=10+2*(m-k);let sn=sp-2;let last=k==m;let weight=umWeight(UM_BFIRST+u32(k-1));
   for(var j=i32(lane);j<sn*sn*sn;j+=${lanes}){
    let p=vec3i(j%sn,(j/sn)%sn,j/(sn*sn));let before=(p.x+1)+sp*((p.y+1)+sp*(p.z+1));let index=(p.x+k)+s0*((p.y+k)+s0*(p.z+k));
    let old=pageValue(k-1,before);var next=old;let row=pageRegion[index];
    if(row!=UM_NONE){
     let mask=umLMask(R,row);let minimum=native[low+row];next=max(old,minimum);
     if((mask&1u)!=0u){
      let nb=array<f32,6>(pageValue(k-1,before-1),pageValue(k-1,before+1),pageValue(k-1,before-sp),pageValue(k-1,before+sp),pageValue(k-1,before-sp*sp),pageValue(k-1,before+sp*sp));
      next=umLRelax(R,row,mask,old,nb,rhs,minimum,weight);
     }
     if(last){
      pageEnd(row,next,to);
      let ahead=array<i32,6>(-1,1,-sp,sp,-sp*sp,sp*sp);let around=array<i32,6>(-1,1,-s0,s0,-s0*s0,s0*s0);
      for(var f=0u;f<6u;f++){
       let s=pageRegion[index+around[f]];if(s==UM_NONE||s<R.cells){continue;}
       let previous=pageValue(k-1,before+ahead[f]);let bound=native[low+s];let wall=umLMask(R,s);var value=max(previous,bound);
       if((wall&1u)!=0u){var nb:array<f32,6>;nb[f^1u]=old;value=umLRelax(R,s,wall,previous,nb,rhs,bound,weight);}
       pageEnd(s,value,to);
      }
     }
    }
    if(!last){if((k&1)==1){pageB[j]=next;}else{pageA[j]=next;}}
   }
   workgroupBarrier();
  }
 }
}
`;
}
